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

theorem publicBatchWrapper86_f1626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 15) = bselect (a (.wire 159 27)) (a (.virt 17005)) (a (.wire 164 31)) := by
  have c10497 := (publicBatchWrapper86_copies328 a h).2.1
  have c10498 := (publicBatchWrapper86_copies328 a h).2.2.1
  have c10499 := (publicBatchWrapper86_copies328 a h).2.2.2.1
  have c10500 := (publicBatchWrapper86_copies328 a h).2.2.2.2.1
  have c10501 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.1
  have c10502 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.1
  have e_165_2 := arithEq_of_rows h (row := 165) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_2
  simp only [← c10497, ← c10498, ← c10499] at e_165_2
  have e_165_3 := arithEq_of_rows h (row := 165) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_3
  simp only [← c10500, ← c10501, ← c10502] at e_165_3
  have hr := e_165_3
  simp only [e_165_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 23) = bselect (a (.wire 159 27)) (a (.virt 17006)) (a (.wire 164 39)) := by
  have c10503 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.1
  have c10504 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.1
  have c10505 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.1
  have c10506 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10507 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10508 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_4 := arithEq_of_rows h (row := 165) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_4
  simp only [← c10503, ← c10504, ← c10505] at e_165_4
  have e_165_5 := arithEq_of_rows h (row := 165) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_5
  simp only [← c10506, ← c10507, ← c10508] at e_165_5
  have hr := e_165_5
  simp only [e_165_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 31) = bselect (a (.wire 159 27)) (a (.virt 17007)) (a (.wire 164 47)) := by
  have c10509 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10510 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10511 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10512 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10513 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10514 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_6 := arithEq_of_rows h (row := 165) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_6
  simp only [← c10509, ← c10510, ← c10511] at e_165_6
  have e_165_7 := arithEq_of_rows h (row := 165) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_7
  simp only [← c10512, ← c10513, ← c10514] at e_165_7
  have hr := e_165_7
  simp only [e_165_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 39) = bselect (a (.wire 159 27)) (a (.virt 17008)) (a (.wire 164 55)) := by
  have c10515 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10516 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10517 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10518 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10519 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10520 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_8 := arithEq_of_rows h (row := 165) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_8
  simp only [← c10515, ← c10516, ← c10517] at e_165_8
  have e_165_9 := arithEq_of_rows h (row := 165) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_9
  simp only [← c10518, ← c10519, ← c10520] at e_165_9
  have hr := e_165_9
  simp only [e_165_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 47) = bselect (a (.wire 159 27)) (a (.virt 17009)) (a (.wire 164 63)) := by
  have c10521 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10522 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10523 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10524 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10525 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10526 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_10 := arithEq_of_rows h (row := 165) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_10
  simp only [← c10521, ← c10522, ← c10523] at e_165_10
  have e_165_11 := arithEq_of_rows h (row := 165) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_11
  simp only [← c10524, ← c10525, ← c10526] at e_165_11
  have hr := e_165_11
  simp only [e_165_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 55) = bselect (a (.wire 159 27)) (a (.virt 17003)) (a (.wire 164 71)) := by
  have c10527 := (publicBatchWrapper86_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10528 := (publicBatchWrapper86_copies329 a h).1
  have c10529 := (publicBatchWrapper86_copies329 a h).2.1
  have c10530 := (publicBatchWrapper86_copies329 a h).2.2.1
  have c10531 := (publicBatchWrapper86_copies329 a h).2.2.2.1
  have c10532 := (publicBatchWrapper86_copies329 a h).2.2.2.2.1
  have e_165_12 := arithEq_of_rows h (row := 165) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_12
  simp only [← c10527, ← c10528, ← c10529] at e_165_12
  have e_165_13 := arithEq_of_rows h (row := 165) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_13
  simp only [← c10530, ← c10531, ← c10532] at e_165_13
  have hr := e_165_13
  simp only [e_165_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 63) = bselect (a (.wire 159 27)) (a (.virt 17004)) (a (.wire 164 79)) := by
  have c10533 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.1
  have c10534 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.1
  have c10535 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.1
  have c10536 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.1
  have c10537 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.1
  have c10538 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_165_14 := arithEq_of_rows h (row := 165) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_14
  simp only [← c10533, ← c10534, ← c10535] at e_165_14
  have e_165_15 := arithEq_of_rows h (row := 165) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_15
  simp only [← c10536, ← c10537, ← c10538] at e_165_15
  have hr := e_165_15
  simp only [e_165_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 51) = bor (a (.wire 154 47)) (a (.wire 165 3)) := by
  have c10539 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10540 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10541 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10542 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10543 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10544 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_12 := arithEq_of_rows h (row := 154) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_12
  simp only [← c10542, ← c10543, k3, ← c10544] at e_154_12
  have e_162_4 := arithEq_of_rows h (row := 162) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_4
  simp only [← c10539, ← c10540, ← c10541] at e_162_4
  have hr := e_154_12
  simp only [e_162_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1)) (a (.wire 165 55)) (a (.virt 17895)) (a (.virt 17896)) := by
  have c10545 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10546 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10547 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10548 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10549 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10550 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10551 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10552 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10553 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10554 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10555 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10556 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10557 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10558 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10559 := (publicBatchWrapper86_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10560 := (publicBatchWrapper86_copies330 a h).1
  have c10561 := (publicBatchWrapper86_copies330 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_159_7 := arithEq_of_rows h (row := 159) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_7
  simp only [← c10551, ← c10552, ← c10553] at e_159_7
  have e_159_8 := arithEq_of_rows h (row := 159) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_8
  simp only [← c10554, ← c10555, ← c10556] at e_159_8
  have e_165_16 := arithEq_of_rows h (row := 165) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_16
  simp only [← c10545, k3, ← c10546, k3, ← c10547] at e_165_16
  have e_165_17 := arithEq_of_rows h (row := 165) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_17
  simp only [← c10548, ← c10549, k3, ← c10550] at e_165_17
  have e_165_18 := arithEq_of_rows h (row := 165) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_18
  simp only [← c10557, ← c10558, k3, ← c10559] at e_165_18
  refine ⟨?_, ?_⟩
  · have hc := e_159_7
    simp only [e_165_17] at hc
    linear_combination c10560.trans k2 - hc
  · have hc := e_165_18
    simp only [e_165_16, e_159_8, e_165_17] at hc
    linear_combination c10561.trans k2 - hc

theorem publicBatchWrapper86_f1635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 55) = bor (a (.wire 3 27)) (a (.virt 17895)) := by
  have c10562 := (publicBatchWrapper86_copies330 a h).2.2.1
  have c10563 := (publicBatchWrapper86_copies330 a h).2.2.2.1
  have c10564 := (publicBatchWrapper86_copies330 a h).2.2.2.2.1
  have c10565 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.1
  have c10566 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.1
  have c10567 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_13 := arithEq_of_rows h (row := 154) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_13
  simp only [← c10565, ← c10566, k3, ← c10567] at e_154_13
  have e_162_5 := arithEq_of_rows h (row := 162) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_5
  simp only [← c10562, ← c10563, ← c10564] at e_162_5
  have hr := e_154_13
  simp only [e_162_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 55) = a (.virt 201) := by
  have c10568 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.1
  exact c10568

theorem publicBatchWrapper86_f1637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2)) (a (.wire 165 63)) (a (.virt 17897)) (a (.virt 17898)) := by
  have c10569 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.1
  have c10570 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10571 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10572 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10573 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10574 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10575 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10576 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10577 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10578 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10579 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10580 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10581 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10582 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10583 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10584 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10585 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_159_9 := arithEq_of_rows h (row := 159) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_9
  simp only [← c10575, ← c10576, ← c10577] at e_159_9
  have e_159_10 := arithEq_of_rows h (row := 159) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_10
  simp only [← c10578, ← c10579, ← c10580] at e_159_10
  have e_165_19 := arithEq_of_rows h (row := 165) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_19
  simp only [← c10569, k3, ← c10570, k3, ← c10571] at e_165_19
  have e_166_0 := arithEq_of_rows h (row := 166) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_0
  simp only [← c10572, ← c10573, k3, ← c10574] at e_166_0
  have e_166_1 := arithEq_of_rows h (row := 166) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_1
  simp only [← c10581, ← c10582, k3, ← c10583] at e_166_1
  refine ⟨?_, ?_⟩
  · have hc := e_159_9
    simp only [e_166_0] at hc
    linear_combination c10584.trans k2 - hc
  · have hc := e_166_1
    simp only [e_165_19, e_159_10, e_166_0] at hc
    linear_combination c10585.trans k2 - hc

theorem publicBatchWrapper86_f1638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 59) = bor (a (.wire 3 27)) (a (.virt 17897)) := by
  have c10586 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10587 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10588 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10589 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10590 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10591 := (publicBatchWrapper86_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_14 := arithEq_of_rows h (row := 154) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_14
  simp only [← c10589, ← c10590, k3, ← c10591] at e_154_14
  have e_162_6 := arithEq_of_rows h (row := 162) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_6
  simp only [← c10586, ← c10587, ← c10588] at e_162_6
  have hr := e_154_14
  simp only [e_162_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 59) = a (.virt 201) := by
  have c10592 := (publicBatchWrapper86_copies331 a h).1
  exact c10592

theorem publicBatchWrapper86_f1640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3)) (a (.wire 165 15)) (a (.virt 17899)) (a (.virt 17900)) := by
  have c10593 := (publicBatchWrapper86_copies331 a h).2.1
  have c10594 := (publicBatchWrapper86_copies331 a h).2.2.1
  have c10595 := (publicBatchWrapper86_copies331 a h).2.2.2.1
  have c10596 := (publicBatchWrapper86_copies331 a h).2.2.2.2.1
  have c10597 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.1
  have c10598 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.1
  have c10599 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.1
  have c10600 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.1
  have c10601 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.1
  have c10602 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10603 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10604 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10605 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10606 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10607 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10608 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10609 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_159_11 := arithEq_of_rows h (row := 159) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_11
  simp only [← c10599, ← c10600, ← c10601] at e_159_11
  have e_159_12 := arithEq_of_rows h (row := 159) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_12
  simp only [← c10602, ← c10603, ← c10604] at e_159_12
  have e_166_2 := arithEq_of_rows h (row := 166) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_2
  simp only [← c10593, k3, ← c10594, k3, ← c10595] at e_166_2
  have e_166_3 := arithEq_of_rows h (row := 166) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_3
  simp only [← c10596, ← c10597, k3, ← c10598] at e_166_3
  have e_166_4 := arithEq_of_rows h (row := 166) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_4
  simp only [← c10605, ← c10606, k3, ← c10607] at e_166_4
  refine ⟨?_, ?_⟩
  · have hc := e_159_11
    simp only [e_166_3] at hc
    linear_combination c10608.trans k2 - hc
  · have hc := e_166_4
    simp only [e_166_2, e_159_12, e_166_3] at hc
    linear_combination c10609.trans k2 - hc

theorem publicBatchWrapper86_f1641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4)) (a (.wire 165 23)) (a (.virt 17901)) (a (.virt 17902)) := by
  have c10610 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10611 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10612 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10613 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10614 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10615 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10616 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10617 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10618 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10619 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10620 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10621 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10622 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10623 := (publicBatchWrapper86_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10624 := (publicBatchWrapper86_copies332 a h).1
  have c10625 := (publicBatchWrapper86_copies332 a h).2.1
  have c10626 := (publicBatchWrapper86_copies332 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_159_13 := arithEq_of_rows h (row := 159) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_13
  simp only [← c10616, ← c10617, ← c10618] at e_159_13
  have e_159_14 := arithEq_of_rows h (row := 159) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_14
  simp only [← c10619, ← c10620, ← c10621] at e_159_14
  have e_166_5 := arithEq_of_rows h (row := 166) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_5
  simp only [← c10610, k3, ← c10611, k3, ← c10612] at e_166_5
  have e_166_6 := arithEq_of_rows h (row := 166) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_6
  simp only [← c10613, ← c10614, k3, ← c10615] at e_166_6
  have e_166_7 := arithEq_of_rows h (row := 166) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_7
  simp only [← c10622, ← c10623, k3, ← c10624] at e_166_7
  refine ⟨?_, ?_⟩
  · have hc := e_159_13
    simp only [e_166_6] at hc
    linear_combination c10625.trans k2 - hc
  · have hc := e_166_7
    simp only [e_166_5, e_159_14, e_166_6] at hc
    linear_combination c10626.trans k2 - hc

theorem publicBatchWrapper86_f1642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5)) (a (.wire 165 31)) (a (.virt 17903)) (a (.virt 17904)) := by
  have c10627 := (publicBatchWrapper86_copies332 a h).2.2.2.1
  have c10628 := (publicBatchWrapper86_copies332 a h).2.2.2.2.1
  have c10629 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.1
  have c10630 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.1
  have c10631 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.1
  have c10632 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.1
  have c10633 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.1
  have c10634 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10635 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10636 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10637 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10638 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10639 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10640 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10641 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10642 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10643 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_159_15 := arithEq_of_rows h (row := 159) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_15
  simp only [← c10633, ← c10634, ← c10635] at e_159_15
  have e_159_16 := arithEq_of_rows h (row := 159) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_16
  simp only [← c10636, ← c10637, ← c10638] at e_159_16
  have e_166_8 := arithEq_of_rows h (row := 166) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_8
  simp only [← c10627, k3, ← c10628, k3, ← c10629] at e_166_8
  have e_166_9 := arithEq_of_rows h (row := 166) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_9
  simp only [← c10630, ← c10631, k3, ← c10632] at e_166_9
  have e_166_10 := arithEq_of_rows h (row := 166) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_10
  simp only [← c10639, ← c10640, k3, ← c10641] at e_166_10
  refine ⟨?_, ?_⟩
  · have hc := e_159_15
    simp only [e_166_9] at hc
    linear_combination c10642.trans k2 - hc
  · have hc := e_166_10
    simp only [e_166_8, e_159_16, e_166_9] at hc
    linear_combination c10643.trans k2 - hc

theorem publicBatchWrapper86_f1643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6)) (a (.wire 165 39)) (a (.virt 17905)) (a (.virt 17906)) := by
  have c10644 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10645 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10646 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10647 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10648 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10649 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10650 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10651 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10652 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10653 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10654 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10655 := (publicBatchWrapper86_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10656 := (publicBatchWrapper86_copies333 a h).1
  have c10657 := (publicBatchWrapper86_copies333 a h).2.1
  have c10658 := (publicBatchWrapper86_copies333 a h).2.2.1
  have c10659 := (publicBatchWrapper86_copies333 a h).2.2.2.1
  have c10660 := (publicBatchWrapper86_copies333 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_159_17 := arithEq_of_rows h (row := 159) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_17
  simp only [← c10650, ← c10651, ← c10652] at e_159_17
  have e_159_18 := arithEq_of_rows h (row := 159) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_18
  simp only [← c10653, ← c10654, ← c10655] at e_159_18
  have e_166_11 := arithEq_of_rows h (row := 166) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_11
  simp only [← c10644, k3, ← c10645, k3, ← c10646] at e_166_11
  have e_166_12 := arithEq_of_rows h (row := 166) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_12
  simp only [← c10647, ← c10648, k3, ← c10649] at e_166_12
  have e_166_13 := arithEq_of_rows h (row := 166) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_13
  simp only [← c10656, ← c10657, k3, ← c10658] at e_166_13
  refine ⟨?_, ?_⟩
  · have hc := e_159_17
    simp only [e_166_12] at hc
    linear_combination c10659.trans k2 - hc
  · have hc := e_166_13
    simp only [e_166_11, e_159_18, e_166_12] at hc
    linear_combination c10660.trans k2 - hc

theorem publicBatchWrapper86_f1644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 159 79) = band (a (.virt 17899)) (a (.virt 17901)) := by
  have c10661 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.1
  have c10662 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.1
  have c10663 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.1
  have e_159_19 := arithEq_of_rows h (row := 159) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_19
  simp only [← c10661, ← c10662, ← c10663] at e_159_19
  have hr := e_159_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 167 3) = band (a (.virt 17903)) (a (.virt 17905)) := by
  have c10664 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.1
  have c10665 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.1
  have c10666 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_167_0 := arithEq_of_rows h (row := 167) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_0
  simp only [← c10664, ← c10665, ← c10666] at e_167_0
  have hr := e_167_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 167 7) = band (a (.wire 159 79)) (a (.wire 167 3)) := by
  have c10667 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10668 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10669 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_1 := arithEq_of_rows h (row := 167) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_1
  simp only [← c10667, ← c10668, ← c10669] at e_167_1
  have hr := e_167_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 63) = bor (a (.wire 3 27)) (a (.wire 167 7)) := by
  have c10670 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10671 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10672 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10673 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10674 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10675 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_15 := arithEq_of_rows h (row := 154) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_15
  simp only [← c10673, ← c10674, k3, ← c10675] at e_154_15
  have e_162_7 := arithEq_of_rows h (row := 162) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_7
  simp only [← c10670, ← c10671, ← c10672] at e_162_7
  have hr := e_154_15
  simp only [e_162_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 63) = a (.virt 201) := by
  have c10676 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c10676

theorem publicBatchWrapper86_f1649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 203)) (a (.wire 165 55)) (a (.virt 17907)) (a (.virt 17908)) := by
  have c10677 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10678 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10679 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10680 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10681 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10682 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10683 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10684 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10685 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10686 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10687 := (publicBatchWrapper86_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10688 := (publicBatchWrapper86_copies334 a h).1
  have c10689 := (publicBatchWrapper86_copies334 a h).2.1
  have c10690 := (publicBatchWrapper86_copies334 a h).2.2.1
  have c10691 := (publicBatchWrapper86_copies334 a h).2.2.2.1
  have c10692 := (publicBatchWrapper86_copies334 a h).2.2.2.2.1
  have c10693 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_166_14 := arithEq_of_rows h (row := 166) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_14
  simp only [← c10677, k3, ← c10678, k3, ← c10679] at e_166_14
  have e_166_15 := arithEq_of_rows h (row := 166) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_15
  simp only [← c10680, ← c10681, k3, ← c10682] at e_166_15
  have e_166_16 := arithEq_of_rows h (row := 166) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_16
  simp only [← c10689, ← c10690, k3, ← c10691] at e_166_16
  have e_167_2 := arithEq_of_rows h (row := 167) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_2
  simp only [← c10683, ← c10684, ← c10685] at e_167_2
  have e_167_3 := arithEq_of_rows h (row := 167) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_3
  simp only [← c10686, ← c10687, ← c10688] at e_167_3
  refine ⟨?_, ?_⟩
  · have hc := e_167_2
    simp only [e_166_15] at hc
    linear_combination c10692.trans k2 - hc
  · have hc := e_166_16
    simp only [e_166_14, e_167_3, e_166_15] at hc
    linear_combination c10693.trans k2 - hc

theorem publicBatchWrapper86_f1650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 67) = bor (a (.wire 3 71)) (a (.virt 17907)) := by
  have c10694 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.1
  have c10695 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.1
  have c10696 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.1
  have c10697 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.1
  have c10698 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10699 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_16 := arithEq_of_rows h (row := 154) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_16
  simp only [← c10697, ← c10698, k3, ← c10699] at e_154_16
  have e_162_8 := arithEq_of_rows h (row := 162) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_8
  simp only [← c10694, ← c10695, ← c10696] at e_162_8
  have hr := e_154_16
  simp only [e_162_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 67) = a (.virt 201) := by
  have c10700 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c10700

theorem publicBatchWrapper86_f1652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 204)) (a (.wire 165 63)) (a (.virt 17909)) (a (.virt 17910)) := by
  have c10701 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10702 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10703 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10704 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10705 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10706 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10707 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10708 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10709 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10710 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10711 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10712 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10713 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10714 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10715 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10716 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10717 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_166_17 := arithEq_of_rows h (row := 166) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_17
  simp only [← c10701, k3, ← c10702, k3, ← c10703] at e_166_17
  have e_166_18 := arithEq_of_rows h (row := 166) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_18
  simp only [← c10704, ← c10705, k3, ← c10706] at e_166_18
  have e_166_19 := arithEq_of_rows h (row := 166) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_19
  simp only [← c10713, ← c10714, k3, ← c10715] at e_166_19
  have e_167_4 := arithEq_of_rows h (row := 167) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_4
  simp only [← c10707, ← c10708, ← c10709] at e_167_4
  have e_167_5 := arithEq_of_rows h (row := 167) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_5
  simp only [← c10710, ← c10711, ← c10712] at e_167_5
  refine ⟨?_, ?_⟩
  · have hc := e_167_4
    simp only [e_166_18] at hc
    linear_combination c10716.trans k2 - hc
  · have hc := e_166_19
    simp only [e_166_17, e_167_5, e_166_18] at hc
    linear_combination c10717.trans k2 - hc

theorem publicBatchWrapper86_f1653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 71) = bor (a (.wire 3 71)) (a (.virt 17909)) := by
  have c10718 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10719 := (publicBatchWrapper86_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10720 := (publicBatchWrapper86_copies335 a h).1
  have c10721 := (publicBatchWrapper86_copies335 a h).2.1
  have c10722 := (publicBatchWrapper86_copies335 a h).2.2.1
  have c10723 := (publicBatchWrapper86_copies335 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_17 := arithEq_of_rows h (row := 154) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_17
  simp only [← c10721, ← c10722, k3, ← c10723] at e_154_17
  have e_162_9 := arithEq_of_rows h (row := 162) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_9
  simp only [← c10718, ← c10719, ← c10720] at e_162_9
  have hr := e_154_17
  simp only [e_162_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 71) = a (.virt 201) := by
  have c10724 := (publicBatchWrapper86_copies335 a h).2.2.2.2.1
  exact c10724

theorem publicBatchWrapper86_f1655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 205)) (a (.wire 165 15)) (a (.virt 17911)) (a (.virt 17912)) := by
  have c10725 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.1
  have c10726 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.1
  have c10727 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.1
  have c10728 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.1
  have c10729 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.1
  have c10730 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10731 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10732 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10733 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10734 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10735 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10736 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10737 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10738 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10739 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10740 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10741 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_167_6 := arithEq_of_rows h (row := 167) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_6
  simp only [← c10731, ← c10732, ← c10733] at e_167_6
  have e_167_7 := arithEq_of_rows h (row := 167) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_7
  simp only [← c10734, ← c10735, ← c10736] at e_167_7
  have e_168_0 := arithEq_of_rows h (row := 168) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_0
  simp only [← c10725, k3, ← c10726, k3, ← c10727] at e_168_0
  have e_168_1 := arithEq_of_rows h (row := 168) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_1
  simp only [← c10728, ← c10729, k3, ← c10730] at e_168_1
  have e_168_2 := arithEq_of_rows h (row := 168) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_2
  simp only [← c10737, ← c10738, k3, ← c10739] at e_168_2
  refine ⟨?_, ?_⟩
  · have hc := e_167_6
    simp only [e_168_1] at hc
    linear_combination c10740.trans k2 - hc
  · have hc := e_168_2
    simp only [e_168_0, e_167_7, e_168_1] at hc
    linear_combination c10741.trans k2 - hc

theorem publicBatchWrapper86_f1656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 206)) (a (.wire 165 23)) (a (.virt 17913)) (a (.virt 17914)) := by
  have c10742 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10743 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10744 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10745 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10746 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10747 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10748 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10749 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10750 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10751 := (publicBatchWrapper86_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10752 := (publicBatchWrapper86_copies336 a h).1
  have c10753 := (publicBatchWrapper86_copies336 a h).2.1
  have c10754 := (publicBatchWrapper86_copies336 a h).2.2.1
  have c10755 := (publicBatchWrapper86_copies336 a h).2.2.2.1
  have c10756 := (publicBatchWrapper86_copies336 a h).2.2.2.2.1
  have c10757 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.1
  have c10758 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_167_8 := arithEq_of_rows h (row := 167) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_8
  simp only [← c10748, ← c10749, ← c10750] at e_167_8
  have e_167_9 := arithEq_of_rows h (row := 167) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_9
  simp only [← c10751, ← c10752, ← c10753] at e_167_9
  have e_168_3 := arithEq_of_rows h (row := 168) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_3
  simp only [← c10742, k3, ← c10743, k3, ← c10744] at e_168_3
  have e_168_4 := arithEq_of_rows h (row := 168) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_4
  simp only [← c10745, ← c10746, k3, ← c10747] at e_168_4
  have e_168_5 := arithEq_of_rows h (row := 168) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_5
  simp only [← c10754, ← c10755, k3, ← c10756] at e_168_5
  refine ⟨?_, ?_⟩
  · have hc := e_167_8
    simp only [e_168_4] at hc
    linear_combination c10757.trans k2 - hc
  · have hc := e_168_5
    simp only [e_168_3, e_167_9, e_168_4] at hc
    linear_combination c10758.trans k2 - hc

theorem publicBatchWrapper86_f1657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 207)) (a (.wire 165 31)) (a (.virt 17915)) (a (.virt 17916)) := by
  have c10759 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.1
  have c10760 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.1
  have c10761 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.1
  have c10762 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10763 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10764 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10765 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10766 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10767 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10768 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10769 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10770 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10771 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10772 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10773 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10774 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10775 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_167_10 := arithEq_of_rows h (row := 167) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_10
  simp only [← c10765, ← c10766, ← c10767] at e_167_10
  have e_167_11 := arithEq_of_rows h (row := 167) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_11
  simp only [← c10768, ← c10769, ← c10770] at e_167_11
  have e_168_6 := arithEq_of_rows h (row := 168) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_6
  simp only [← c10759, k3, ← c10760, k3, ← c10761] at e_168_6
  have e_168_7 := arithEq_of_rows h (row := 168) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_7
  simp only [← c10762, ← c10763, k3, ← c10764] at e_168_7
  have e_168_8 := arithEq_of_rows h (row := 168) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_8
  simp only [← c10771, ← c10772, k3, ← c10773] at e_168_8
  refine ⟨?_, ?_⟩
  · have hc := e_167_10
    simp only [e_168_7] at hc
    linear_combination c10774.trans k2 - hc
  · have hc := e_168_8
    simp only [e_168_6, e_167_11, e_168_7] at hc
    linear_combination c10775.trans k2 - hc

theorem publicBatchWrapper86_f1658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 208)) (a (.wire 165 39)) (a (.virt 17917)) (a (.virt 17918)) := by
  have c10776 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10777 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10778 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10779 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10780 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10781 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10782 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10783 := (publicBatchWrapper86_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10784 := (publicBatchWrapper86_copies337 a h).1
  have c10785 := (publicBatchWrapper86_copies337 a h).2.1
  have c10786 := (publicBatchWrapper86_copies337 a h).2.2.1
  have c10787 := (publicBatchWrapper86_copies337 a h).2.2.2.1
  have c10788 := (publicBatchWrapper86_copies337 a h).2.2.2.2.1
  have c10789 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.1
  have c10790 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.1
  have c10791 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.1
  have c10792 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_167_12 := arithEq_of_rows h (row := 167) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_12
  simp only [← c10782, ← c10783, ← c10784] at e_167_12
  have e_167_13 := arithEq_of_rows h (row := 167) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_13
  simp only [← c10785, ← c10786, ← c10787] at e_167_13
  have e_168_9 := arithEq_of_rows h (row := 168) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_9
  simp only [← c10776, k3, ← c10777, k3, ← c10778] at e_168_9
  have e_168_10 := arithEq_of_rows h (row := 168) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_10
  simp only [← c10779, ← c10780, k3, ← c10781] at e_168_10
  have e_168_11 := arithEq_of_rows h (row := 168) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_11
  simp only [← c10788, ← c10789, k3, ← c10790] at e_168_11
  refine ⟨?_, ?_⟩
  · have hc := e_167_12
    simp only [e_168_10] at hc
    linear_combination c10791.trans k2 - hc
  · have hc := e_168_11
    simp only [e_168_9, e_167_13, e_168_10] at hc
    linear_combination c10792.trans k2 - hc

theorem publicBatchWrapper86_f1659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 167 59) = band (a (.virt 17911)) (a (.virt 17913)) := by
  have c10793 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.1
  have c10794 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10795 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_14 := arithEq_of_rows h (row := 167) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_14
  simp only [← c10793, ← c10794, ← c10795] at e_167_14
  have hr := e_167_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 167 63) = band (a (.virt 17915)) (a (.virt 17917)) := by
  have c10796 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10797 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10798 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_15 := arithEq_of_rows h (row := 167) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_15
  simp only [← c10796, ← c10797, ← c10798] at e_167_15
  have hr := e_167_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 167 67) = band (a (.wire 167 59)) (a (.wire 167 63)) := by
  have c10799 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10800 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10801 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_16 := arithEq_of_rows h (row := 167) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_16
  simp only [← c10799, ← c10800, ← c10801] at e_167_16
  have hr := e_167_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 75) = bor (a (.wire 3 71)) (a (.wire 167 67)) := by
  have c10802 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10803 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10804 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10805 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10806 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10807 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_18 := arithEq_of_rows h (row := 154) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_18
  simp only [← c10805, ← c10806, k3, ← c10807] at e_154_18
  have e_162_10 := arithEq_of_rows h (row := 162) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_10
  simp only [← c10802, ← c10803, ← c10804] at e_162_10
  have hr := e_154_18
  simp only [e_162_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 75) = a (.virt 201) := by
  have c10808 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c10808

theorem publicBatchWrapper86_f1664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 403)) (a (.wire 165 55)) (a (.virt 17919)) (a (.virt 17920)) := by
  have c10809 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10810 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10811 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10812 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10813 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10814 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10815 := (publicBatchWrapper86_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10816 := (publicBatchWrapper86_copies338 a h).1
  have c10817 := (publicBatchWrapper86_copies338 a h).2.1
  have c10818 := (publicBatchWrapper86_copies338 a h).2.2.1
  have c10819 := (publicBatchWrapper86_copies338 a h).2.2.2.1
  have c10820 := (publicBatchWrapper86_copies338 a h).2.2.2.2.1
  have c10821 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.1
  have c10822 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.1
  have c10823 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.1
  have c10824 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.1
  have c10825 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_167_17 := arithEq_of_rows h (row := 167) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_17
  simp only [← c10815, ← c10816, ← c10817] at e_167_17
  have e_167_18 := arithEq_of_rows h (row := 167) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_18
  simp only [← c10818, ← c10819, ← c10820] at e_167_18
  have e_168_12 := arithEq_of_rows h (row := 168) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_12
  simp only [← c10809, k3, ← c10810, k3, ← c10811] at e_168_12
  have e_168_13 := arithEq_of_rows h (row := 168) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_13
  simp only [← c10812, ← c10813, k3, ← c10814] at e_168_13
  have e_168_14 := arithEq_of_rows h (row := 168) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_14
  simp only [← c10821, ← c10822, k3, ← c10823] at e_168_14
  refine ⟨?_, ?_⟩
  · have hc := e_167_17
    simp only [e_168_13] at hc
    linear_combination c10824.trans k2 - hc
  · have hc := e_168_14
    simp only [e_168_12, e_167_18, e_168_13] at hc
    linear_combination c10825.trans k2 - hc

theorem publicBatchWrapper86_f1665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 79) = bor (a (.wire 4 35)) (a (.virt 17919)) := by
  have c10826 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10827 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10828 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10829 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10830 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10831 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_19 := arithEq_of_rows h (row := 154) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_19
  simp only [← c10829, ← c10830, k3, ← c10831] at e_154_19
  have e_162_11 := arithEq_of_rows h (row := 162) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_11
  simp only [← c10826, ← c10827, ← c10828] at e_162_11
  have hr := e_154_19
  simp only [e_162_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 79) = a (.virt 201) := by
  have c10832 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c10832

theorem publicBatchWrapper86_f1667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 404)) (a (.wire 165 63)) (a (.virt 17921)) (a (.virt 17922)) := by
  have c10833 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10834 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10835 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10836 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10837 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10838 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10839 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10840 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10841 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10842 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10843 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10844 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10845 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10846 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10847 := (publicBatchWrapper86_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10848 := (publicBatchWrapper86_copies339 a h).1
  have c10849 := (publicBatchWrapper86_copies339 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_167_19 := arithEq_of_rows h (row := 167) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_19
  simp only [← c10839, ← c10840, ← c10841] at e_167_19
  have e_168_15 := arithEq_of_rows h (row := 168) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_15
  simp only [← c10833, k3, ← c10834, k3, ← c10835] at e_168_15
  have e_168_16 := arithEq_of_rows h (row := 168) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_16
  simp only [← c10836, ← c10837, k3, ← c10838] at e_168_16
  have e_168_17 := arithEq_of_rows h (row := 168) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_17
  simp only [← c10845, ← c10846, k3, ← c10847] at e_168_17
  have e_169_0 := arithEq_of_rows h (row := 169) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_0
  simp only [← c10842, ← c10843, ← c10844] at e_169_0
  refine ⟨?_, ?_⟩
  · have hc := e_167_19
    simp only [e_168_16] at hc
    linear_combination c10848.trans k2 - hc
  · have hc := e_168_17
    simp only [e_168_15, e_169_0, e_168_16] at hc
    linear_combination c10849.trans k2 - hc

theorem publicBatchWrapper86_f1668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 3) = bor (a (.wire 4 35)) (a (.virt 17921)) := by
  have c10850 := (publicBatchWrapper86_copies339 a h).2.2.1
  have c10851 := (publicBatchWrapper86_copies339 a h).2.2.2.1
  have c10852 := (publicBatchWrapper86_copies339 a h).2.2.2.2.1
  have c10853 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.1
  have c10854 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.1
  have c10855 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_162_12 := arithEq_of_rows h (row := 162) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_12
  simp only [← c10850, ← c10851, ← c10852] at e_162_12
  have e_170_0 := arithEq_of_rows h (row := 170) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_0
  simp only [← c10853, ← c10854, k3, ← c10855] at e_170_0
  have hr := e_170_0
  simp only [e_162_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 3) = a (.virt 201) := by
  have c10856 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.1
  exact c10856

theorem publicBatchWrapper86_f1670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 405)) (a (.wire 165 15)) (a (.virt 17923)) (a (.virt 17924)) := by
  have c10857 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.1
  have c10858 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10859 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10860 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10861 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10862 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10863 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10864 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10865 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10866 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10867 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10868 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10869 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10870 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10871 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10872 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10873 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_168_18 := arithEq_of_rows h (row := 168) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_18
  simp only [← c10857, k3, ← c10858, k3, ← c10859] at e_168_18
  have e_168_19 := arithEq_of_rows h (row := 168) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_19
  simp only [← c10860, ← c10861, k3, ← c10862] at e_168_19
  have e_169_1 := arithEq_of_rows h (row := 169) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_1
  simp only [← c10863, ← c10864, ← c10865] at e_169_1
  have e_169_2 := arithEq_of_rows h (row := 169) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_2
  simp only [← c10866, ← c10867, ← c10868] at e_169_2
  have e_171_0 := arithEq_of_rows h (row := 171) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_0
  simp only [← c10869, ← c10870, k3, ← c10871] at e_171_0
  refine ⟨?_, ?_⟩
  · have hc := e_169_1
    simp only [e_168_19] at hc
    linear_combination c10872.trans k2 - hc
  · have hc := e_171_0
    simp only [e_168_18, e_169_2, e_168_19] at hc
    linear_combination c10873.trans k2 - hc

theorem publicBatchWrapper86_f1671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 406)) (a (.wire 165 23)) (a (.virt 17925)) (a (.virt 17926)) := by
  have c10874 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10875 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10876 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10877 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10878 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10879 := (publicBatchWrapper86_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10880 := (publicBatchWrapper86_copies340 a h).1
  have c10881 := (publicBatchWrapper86_copies340 a h).2.1
  have c10882 := (publicBatchWrapper86_copies340 a h).2.2.1
  have c10883 := (publicBatchWrapper86_copies340 a h).2.2.2.1
  have c10884 := (publicBatchWrapper86_copies340 a h).2.2.2.2.1
  have c10885 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.1
  have c10886 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.1
  have c10887 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.1
  have c10888 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.1
  have c10889 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.1
  have c10890 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_169_3 := arithEq_of_rows h (row := 169) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_3
  simp only [← c10880, ← c10881, ← c10882] at e_169_3
  have e_169_4 := arithEq_of_rows h (row := 169) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_4
  simp only [← c10883, ← c10884, ← c10885] at e_169_4
  have e_171_1 := arithEq_of_rows h (row := 171) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_1
  simp only [← c10874, k3, ← c10875, k3, ← c10876] at e_171_1
  have e_171_2 := arithEq_of_rows h (row := 171) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_2
  simp only [← c10877, ← c10878, k3, ← c10879] at e_171_2
  have e_171_3 := arithEq_of_rows h (row := 171) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_3
  simp only [← c10886, ← c10887, k3, ← c10888] at e_171_3
  refine ⟨?_, ?_⟩
  · have hc := e_169_3
    simp only [e_171_2] at hc
    linear_combination c10889.trans k2 - hc
  · have hc := e_171_3
    simp only [e_171_1, e_169_4, e_171_2] at hc
    linear_combination c10890.trans k2 - hc

theorem publicBatchWrapper86_f1672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 407)) (a (.wire 165 31)) (a (.virt 17927)) (a (.virt 17928)) := by
  have c10891 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10892 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10893 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10894 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10895 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10896 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10897 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10898 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10899 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10900 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10901 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10902 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10903 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10904 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10905 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10906 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10907 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_169_5 := arithEq_of_rows h (row := 169) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_5
  simp only [← c10897, ← c10898, ← c10899] at e_169_5
  have e_169_6 := arithEq_of_rows h (row := 169) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_6
  simp only [← c10900, ← c10901, ← c10902] at e_169_6
  have e_171_4 := arithEq_of_rows h (row := 171) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_4
  simp only [← c10891, k3, ← c10892, k3, ← c10893] at e_171_4
  have e_171_5 := arithEq_of_rows h (row := 171) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_5
  simp only [← c10894, ← c10895, k3, ← c10896] at e_171_5
  have e_171_6 := arithEq_of_rows h (row := 171) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_6
  simp only [← c10903, ← c10904, k3, ← c10905] at e_171_6
  refine ⟨?_, ?_⟩
  · have hc := e_169_5
    simp only [e_171_5] at hc
    linear_combination c10906.trans k2 - hc
  · have hc := e_171_6
    simp only [e_171_4, e_169_6, e_171_5] at hc
    linear_combination c10907.trans k2 - hc

theorem publicBatchWrapper86_f1673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 408)) (a (.wire 165 39)) (a (.virt 17929)) (a (.virt 17930)) := by
  have c10908 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10909 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10910 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10911 := (publicBatchWrapper86_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10912 := (publicBatchWrapper86_copies341 a h).1
  have c10913 := (publicBatchWrapper86_copies341 a h).2.1
  have c10914 := (publicBatchWrapper86_copies341 a h).2.2.1
  have c10915 := (publicBatchWrapper86_copies341 a h).2.2.2.1
  have c10916 := (publicBatchWrapper86_copies341 a h).2.2.2.2.1
  have c10917 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.1
  have c10918 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.1
  have c10919 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.1
  have c10920 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.1
  have c10921 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.1
  have c10922 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10923 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10924 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_169_7 := arithEq_of_rows h (row := 169) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_7
  simp only [← c10914, ← c10915, ← c10916] at e_169_7
  have e_169_8 := arithEq_of_rows h (row := 169) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_8
  simp only [← c10917, ← c10918, ← c10919] at e_169_8
  have e_171_7 := arithEq_of_rows h (row := 171) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_7
  simp only [← c10908, k3, ← c10909, k3, ← c10910] at e_171_7
  have e_171_8 := arithEq_of_rows h (row := 171) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_8
  simp only [← c10911, ← c10912, k3, ← c10913] at e_171_8
  have e_171_9 := arithEq_of_rows h (row := 171) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_9
  simp only [← c10920, ← c10921, k3, ← c10922] at e_171_9
  refine ⟨?_, ?_⟩
  · have hc := e_169_7
    simp only [e_171_8] at hc
    linear_combination c10923.trans k2 - hc
  · have hc := e_171_9
    simp only [e_171_7, e_169_8, e_171_8] at hc
    linear_combination c10924.trans k2 - hc

theorem publicBatchWrapper86_f1674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 169 39) = band (a (.virt 17923)) (a (.virt 17925)) := by
  have c10925 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10926 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10927 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_169_9 := arithEq_of_rows h (row := 169) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_9
  simp only [← c10925, ← c10926, ← c10927] at e_169_9
  have hr := e_169_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 169 43) = band (a (.virt 17927)) (a (.virt 17929)) := by
  have c10928 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10929 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10930 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_169_10 := arithEq_of_rows h (row := 169) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_10
  simp only [← c10928, ← c10929, ← c10930] at e_169_10
  have hr := e_169_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 169 47) = band (a (.wire 169 39)) (a (.wire 169 43)) := by
  have c10931 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10932 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10933 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_169_11 := arithEq_of_rows h (row := 169) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_11
  simp only [← c10931, ← c10932, ← c10933] at e_169_11
  have hr := e_169_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 7) = bor (a (.wire 4 35)) (a (.wire 169 47)) := by
  have c10934 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10935 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10936 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10937 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10938 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10939 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_162_13 := arithEq_of_rows h (row := 162) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_13
  simp only [← c10934, ← c10935, ← c10936] at e_162_13
  have e_170_1 := arithEq_of_rows h (row := 170) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_1
  simp only [← c10937, ← c10938, k3, ← c10939] at e_170_1
  have hr := e_170_1
  simp only [e_162_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 7) = a (.virt 201) := by
  have c10940 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c10940

theorem publicBatchWrapper86_f1679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 603)) (a (.wire 165 55)) (a (.virt 17931)) (a (.virt 17932)) := by
  have c10941 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10942 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10943 := (publicBatchWrapper86_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10944 := (publicBatchWrapper86_copies342 a h).1
  have c10945 := (publicBatchWrapper86_copies342 a h).2.1
  have c10946 := (publicBatchWrapper86_copies342 a h).2.2.1
  have c10947 := (publicBatchWrapper86_copies342 a h).2.2.2.1
  have c10948 := (publicBatchWrapper86_copies342 a h).2.2.2.2.1
  have c10949 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.1
  have c10950 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.1
  have c10951 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.1
  have c10952 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.1
  have c10953 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.1
  have c10954 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10955 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10956 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10957 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_169_12 := arithEq_of_rows h (row := 169) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_12
  simp only [← c10947, ← c10948, ← c10949] at e_169_12
  have e_169_13 := arithEq_of_rows h (row := 169) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_13
  simp only [← c10950, ← c10951, ← c10952] at e_169_13
  have e_171_10 := arithEq_of_rows h (row := 171) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_10
  simp only [← c10941, k3, ← c10942, k3, ← c10943] at e_171_10
  have e_171_11 := arithEq_of_rows h (row := 171) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_11
  simp only [← c10944, ← c10945, k3, ← c10946] at e_171_11
  have e_171_12 := arithEq_of_rows h (row := 171) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_12
  simp only [← c10953, ← c10954, k3, ← c10955] at e_171_12
  refine ⟨?_, ?_⟩
  · have hc := e_169_12
    simp only [e_171_11] at hc
    linear_combination c10956.trans k2 - hc
  · have hc := e_171_12
    simp only [e_171_10, e_169_13, e_171_11] at hc
    linear_combination c10957.trans k2 - hc

theorem publicBatchWrapper86_f1680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 11) = bor (a (.wire 4 79)) (a (.virt 17931)) := by
  have c10958 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10959 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10960 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10961 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10962 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10963 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_162_14 := arithEq_of_rows h (row := 162) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_14
  simp only [← c10958, ← c10959, ← c10960] at e_162_14
  have e_170_2 := arithEq_of_rows h (row := 170) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_2
  simp only [← c10961, ← c10962, k3, ← c10963] at e_170_2
  have hr := e_170_2
  simp only [e_162_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 11) = a (.virt 201) := by
  have c10964 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c10964

theorem publicBatchWrapper86_f1682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 604)) (a (.wire 165 63)) (a (.virt 17933)) (a (.virt 17934)) := by
  have c10965 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10966 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10967 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10968 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10969 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10970 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10971 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10972 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10973 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10974 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10975 := (publicBatchWrapper86_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10976 := (publicBatchWrapper86_copies343 a h).1
  have c10977 := (publicBatchWrapper86_copies343 a h).2.1
  have c10978 := (publicBatchWrapper86_copies343 a h).2.2.1
  have c10979 := (publicBatchWrapper86_copies343 a h).2.2.2.1
  have c10980 := (publicBatchWrapper86_copies343 a h).2.2.2.2.1
  have c10981 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_169_14 := arithEq_of_rows h (row := 169) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_14
  simp only [← c10971, ← c10972, ← c10973] at e_169_14
  have e_169_15 := arithEq_of_rows h (row := 169) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_15
  simp only [← c10974, ← c10975, ← c10976] at e_169_15
  have e_171_13 := arithEq_of_rows h (row := 171) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_13
  simp only [← c10965, k3, ← c10966, k3, ← c10967] at e_171_13
  have e_171_14 := arithEq_of_rows h (row := 171) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_14
  simp only [← c10968, ← c10969, k3, ← c10970] at e_171_14
  have e_171_15 := arithEq_of_rows h (row := 171) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_15
  simp only [← c10977, ← c10978, k3, ← c10979] at e_171_15
  refine ⟨?_, ?_⟩
  · have hc := e_169_14
    simp only [e_171_14] at hc
    linear_combination c10980.trans k2 - hc
  · have hc := e_171_15
    simp only [e_171_13, e_169_15, e_171_14] at hc
    linear_combination c10981.trans k2 - hc

theorem publicBatchWrapper86_f1683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 15) = bor (a (.wire 4 79)) (a (.virt 17933)) := by
  have c10982 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.1
  have c10983 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.1
  have c10984 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.1
  have c10985 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.1
  have c10986 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10987 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_162_15 := arithEq_of_rows h (row := 162) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_15
  simp only [← c10982, ← c10983, ← c10984] at e_162_15
  have e_170_3 := arithEq_of_rows h (row := 170) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_3
  simp only [← c10985, ← c10986, k3, ← c10987] at e_170_3
  have hr := e_170_3
  simp only [e_162_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 15) = a (.virt 201) := by
  have c10988 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c10988

theorem publicBatchWrapper86_f1685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 605)) (a (.wire 165 15)) (a (.virt 17935)) (a (.virt 17936)) := by
  have c10989 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10990 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10991 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10992 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10993 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10994 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10995 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10996 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10997 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10998 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10999 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11000 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11001 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11002 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11003 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11004 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11005 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_169_16 := arithEq_of_rows h (row := 169) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_16
  simp only [← c10995, ← c10996, ← c10997] at e_169_16
  have e_169_17 := arithEq_of_rows h (row := 169) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_17
  simp only [← c10998, ← c10999, ← c11000] at e_169_17
  have e_171_16 := arithEq_of_rows h (row := 171) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_16
  simp only [← c10989, k3, ← c10990, k3, ← c10991] at e_171_16
  have e_171_17 := arithEq_of_rows h (row := 171) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_17
  simp only [← c10992, ← c10993, k3, ← c10994] at e_171_17
  have e_171_18 := arithEq_of_rows h (row := 171) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_18
  simp only [← c11001, ← c11002, k3, ← c11003] at e_171_18
  refine ⟨?_, ?_⟩
  · have hc := e_169_16
    simp only [e_171_17] at hc
    linear_combination c11004.trans k2 - hc
  · have hc := e_171_18
    simp only [e_171_16, e_169_17, e_171_17] at hc
    linear_combination c11005.trans k2 - hc

theorem publicBatchWrapper86_f1686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 606)) (a (.wire 165 23)) (a (.virt 17937)) (a (.virt 17938)) := by
  have c11006 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11007 := (publicBatchWrapper86_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11008 := (publicBatchWrapper86_copies344 a h).1
  have c11009 := (publicBatchWrapper86_copies344 a h).2.1
  have c11010 := (publicBatchWrapper86_copies344 a h).2.2.1
  have c11011 := (publicBatchWrapper86_copies344 a h).2.2.2.1
  have c11012 := (publicBatchWrapper86_copies344 a h).2.2.2.2.1
  have c11013 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.1
  have c11014 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.1
  have c11015 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.1
  have c11016 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.1
  have c11017 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.1
  have c11018 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11019 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11020 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11021 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11022 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_169_18 := arithEq_of_rows h (row := 169) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_18
  simp only [← c11012, ← c11013, ← c11014] at e_169_18
  have e_169_19 := arithEq_of_rows h (row := 169) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_19
  simp only [← c11015, ← c11016, ← c11017] at e_169_19
  have e_171_19 := arithEq_of_rows h (row := 171) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_19
  simp only [← c11006, k3, ← c11007, k3, ← c11008] at e_171_19
  have e_172_0 := arithEq_of_rows h (row := 172) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_0
  simp only [← c11009, ← c11010, k3, ← c11011] at e_172_0
  have e_172_1 := arithEq_of_rows h (row := 172) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_1
  simp only [← c11018, ← c11019, k3, ← c11020] at e_172_1
  refine ⟨?_, ?_⟩
  · have hc := e_169_18
    simp only [e_172_0] at hc
    linear_combination c11021.trans k2 - hc
  · have hc := e_172_1
    simp only [e_171_19, e_169_19, e_172_0] at hc
    linear_combination c11022.trans k2 - hc

theorem publicBatchWrapper86_f1687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 607)) (a (.wire 165 31)) (a (.virt 17939)) (a (.virt 17940)) := by
  have c11023 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11024 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11025 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11026 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11027 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11028 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11029 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11030 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11031 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11032 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11033 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11034 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11035 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11036 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11037 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11038 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11039 := (publicBatchWrapper86_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_172_2 := arithEq_of_rows h (row := 172) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_2
  simp only [← c11023, k3, ← c11024, k3, ← c11025] at e_172_2
  have e_172_3 := arithEq_of_rows h (row := 172) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_3
  simp only [← c11026, ← c11027, k3, ← c11028] at e_172_3
  have e_172_4 := arithEq_of_rows h (row := 172) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_4
  simp only [← c11035, ← c11036, k3, ← c11037] at e_172_4
  have e_173_0 := arithEq_of_rows h (row := 173) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_0
  simp only [← c11029, ← c11030, ← c11031] at e_173_0
  have e_173_1 := arithEq_of_rows h (row := 173) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_1
  simp only [← c11032, ← c11033, ← c11034] at e_173_1
  refine ⟨?_, ?_⟩
  · have hc := e_173_0
    simp only [e_172_3] at hc
    linear_combination c11038.trans k2 - hc
  · have hc := e_172_4
    simp only [e_172_2, e_173_1, e_172_3] at hc
    linear_combination c11039.trans k2 - hc

theorem publicBatchWrapper86_f1688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 608)) (a (.wire 165 39)) (a (.virt 17941)) (a (.virt 17942)) := by
  have c11040 := (publicBatchWrapper86_copies345 a h).1
  have c11041 := (publicBatchWrapper86_copies345 a h).2.1
  have c11042 := (publicBatchWrapper86_copies345 a h).2.2.1
  have c11043 := (publicBatchWrapper86_copies345 a h).2.2.2.1
  have c11044 := (publicBatchWrapper86_copies345 a h).2.2.2.2.1
  have c11045 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.1
  have c11046 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.1
  have c11047 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.1
  have c11048 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.1
  have c11049 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.1
  have c11050 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11051 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11052 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11053 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11054 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11055 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11056 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_172_5 := arithEq_of_rows h (row := 172) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_5
  simp only [← c11040, k3, ← c11041, k3, ← c11042] at e_172_5
  have e_172_6 := arithEq_of_rows h (row := 172) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_6
  simp only [← c11043, ← c11044, k3, ← c11045] at e_172_6
  have e_172_7 := arithEq_of_rows h (row := 172) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_7
  simp only [← c11052, ← c11053, k3, ← c11054] at e_172_7
  have e_173_2 := arithEq_of_rows h (row := 173) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_2
  simp only [← c11046, ← c11047, ← c11048] at e_173_2
  have e_173_3 := arithEq_of_rows h (row := 173) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_3
  simp only [← c11049, ← c11050, ← c11051] at e_173_3
  refine ⟨?_, ?_⟩
  · have hc := e_173_2
    simp only [e_172_6] at hc
    linear_combination c11055.trans k2 - hc
  · have hc := e_172_7
    simp only [e_172_5, e_173_3, e_172_6] at hc
    linear_combination c11056.trans k2 - hc

theorem publicBatchWrapper86_f1689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 173 19) = band (a (.virt 17935)) (a (.virt 17937)) := by
  have c11057 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11058 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11059 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_4 := arithEq_of_rows h (row := 173) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_4
  simp only [← c11057, ← c11058, ← c11059] at e_173_4
  have hr := e_173_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 173 23) = band (a (.virt 17939)) (a (.virt 17941)) := by
  have c11060 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11061 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11062 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_5 := arithEq_of_rows h (row := 173) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_5
  simp only [← c11060, ← c11061, ← c11062] at e_173_5
  have hr := e_173_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 173 27) = band (a (.wire 173 19)) (a (.wire 173 23)) := by
  have c11063 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11064 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11065 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_6 := arithEq_of_rows h (row := 173) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_6
  simp only [← c11063, ← c11064, ← c11065] at e_173_6
  have hr := e_173_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 19) = bor (a (.wire 4 79)) (a (.wire 173 27)) := by
  have c11066 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11067 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11068 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11069 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11070 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11071 := (publicBatchWrapper86_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_162_16 := arithEq_of_rows h (row := 162) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_16
  simp only [← c11066, ← c11067, ← c11068] at e_162_16
  have e_170_4 := arithEq_of_rows h (row := 170) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_4
  simp only [← c11069, ← c11070, k3, ← c11071] at e_170_4
  have hr := e_170_4
  simp only [e_162_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 19) = a (.virt 201) := by
  have c11072 := (publicBatchWrapper86_copies346 a h).1
  exact c11072

theorem publicBatchWrapper86_f1694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 803)) (a (.wire 165 55)) (a (.virt 17943)) (a (.virt 17944)) := by
  have c11073 := (publicBatchWrapper86_copies346 a h).2.1
  have c11074 := (publicBatchWrapper86_copies346 a h).2.2.1
  have c11075 := (publicBatchWrapper86_copies346 a h).2.2.2.1
  have c11076 := (publicBatchWrapper86_copies346 a h).2.2.2.2.1
  have c11077 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.1
  have c11078 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.1
  have c11079 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.1
  have c11080 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.1
  have c11081 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.1
  have c11082 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11083 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11084 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11085 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11086 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11087 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11088 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11089 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_172_8 := arithEq_of_rows h (row := 172) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_8
  simp only [← c11073, k3, ← c11074, k3, ← c11075] at e_172_8
  have e_172_9 := arithEq_of_rows h (row := 172) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_9
  simp only [← c11076, ← c11077, k3, ← c11078] at e_172_9
  have e_172_10 := arithEq_of_rows h (row := 172) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_10
  simp only [← c11085, ← c11086, k3, ← c11087] at e_172_10
  have e_173_7 := arithEq_of_rows h (row := 173) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_7
  simp only [← c11079, ← c11080, ← c11081] at e_173_7
  have e_173_8 := arithEq_of_rows h (row := 173) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_8
  simp only [← c11082, ← c11083, ← c11084] at e_173_8
  refine ⟨?_, ?_⟩
  · have hc := e_173_7
    simp only [e_172_9] at hc
    linear_combination c11088.trans k2 - hc
  · have hc := e_172_10
    simp only [e_172_8, e_173_8, e_172_9] at hc
    linear_combination c11089.trans k2 - hc

theorem publicBatchWrapper86_f1695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 23) = bor (a (.wire 6 43)) (a (.virt 17943)) := by
  have c11090 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11091 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11092 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11093 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11094 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11095 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_162_17 := arithEq_of_rows h (row := 162) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_17
  simp only [← c11090, ← c11091, ← c11092] at e_162_17
  have e_170_5 := arithEq_of_rows h (row := 170) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_5
  simp only [← c11093, ← c11094, k3, ← c11095] at e_170_5
  have hr := e_170_5
  simp only [e_162_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 23) = a (.virt 201) := by
  have c11096 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11096

theorem publicBatchWrapper86_f1697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 804)) (a (.wire 165 63)) (a (.virt 17945)) (a (.virt 17946)) := by
  have c11097 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11098 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11099 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11100 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11101 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11102 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11103 := (publicBatchWrapper86_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11104 := (publicBatchWrapper86_copies347 a h).1
  have c11105 := (publicBatchWrapper86_copies347 a h).2.1
  have c11106 := (publicBatchWrapper86_copies347 a h).2.2.1
  have c11107 := (publicBatchWrapper86_copies347 a h).2.2.2.1
  have c11108 := (publicBatchWrapper86_copies347 a h).2.2.2.2.1
  have c11109 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.1
  have c11110 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.1
  have c11111 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.1
  have c11112 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.1
  have c11113 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_172_11 := arithEq_of_rows h (row := 172) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_11
  simp only [← c11097, k3, ← c11098, k3, ← c11099] at e_172_11
  have e_172_12 := arithEq_of_rows h (row := 172) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_12
  simp only [← c11100, ← c11101, k3, ← c11102] at e_172_12
  have e_172_13 := arithEq_of_rows h (row := 172) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_13
  simp only [← c11109, ← c11110, k3, ← c11111] at e_172_13
  have e_173_9 := arithEq_of_rows h (row := 173) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_9
  simp only [← c11103, ← c11104, ← c11105] at e_173_9
  have e_173_10 := arithEq_of_rows h (row := 173) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_10
  simp only [← c11106, ← c11107, ← c11108] at e_173_10
  refine ⟨?_, ?_⟩
  · have hc := e_173_9
    simp only [e_172_12] at hc
    linear_combination c11112.trans k2 - hc
  · have hc := e_172_13
    simp only [e_172_11, e_173_10, e_172_12] at hc
    linear_combination c11113.trans k2 - hc

theorem publicBatchWrapper86_f1698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 27) = bor (a (.wire 6 43)) (a (.virt 17945)) := by
  have c11114 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11115 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11116 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11117 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11118 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11119 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_162_18 := arithEq_of_rows h (row := 162) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_18
  simp only [← c11114, ← c11115, ← c11116] at e_162_18
  have e_170_6 := arithEq_of_rows h (row := 170) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_6
  simp only [← c11117, ← c11118, k3, ← c11119] at e_170_6
  have hr := e_170_6
  simp only [e_162_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 27) = a (.virt 201) := by
  have c11120 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11120

theorem publicBatchWrapper86_f1700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 805)) (a (.wire 165 15)) (a (.virt 17947)) (a (.virt 17948)) := by
  have c11121 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11122 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11123 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11124 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11125 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11126 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11127 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11128 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11129 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11130 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11131 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11132 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11133 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11134 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11135 := (publicBatchWrapper86_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11136 := (publicBatchWrapper86_copies348 a h).1
  have c11137 := (publicBatchWrapper86_copies348 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_172_14 := arithEq_of_rows h (row := 172) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_14
  simp only [← c11121, k3, ← c11122, k3, ← c11123] at e_172_14
  have e_172_15 := arithEq_of_rows h (row := 172) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_15
  simp only [← c11124, ← c11125, k3, ← c11126] at e_172_15
  have e_172_16 := arithEq_of_rows h (row := 172) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_16
  simp only [← c11133, ← c11134, k3, ← c11135] at e_172_16
  have e_173_11 := arithEq_of_rows h (row := 173) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_11
  simp only [← c11127, ← c11128, ← c11129] at e_173_11
  have e_173_12 := arithEq_of_rows h (row := 173) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_12
  simp only [← c11130, ← c11131, ← c11132] at e_173_12
  refine ⟨?_, ?_⟩
  · have hc := e_173_11
    simp only [e_172_15] at hc
    linear_combination c11136.trans k2 - hc
  · have hc := e_172_16
    simp only [e_172_14, e_173_12, e_172_15] at hc
    linear_combination c11137.trans k2 - hc

theorem publicBatchWrapper86_f1701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 806)) (a (.wire 165 23)) (a (.virt 17949)) (a (.virt 17950)) := by
  have c11138 := (publicBatchWrapper86_copies348 a h).2.2.1
  have c11139 := (publicBatchWrapper86_copies348 a h).2.2.2.1
  have c11140 := (publicBatchWrapper86_copies348 a h).2.2.2.2.1
  have c11141 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.1
  have c11142 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.1
  have c11143 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.1
  have c11144 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.1
  have c11145 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.1
  have c11146 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11147 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11148 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11149 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11150 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11151 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11152 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11153 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11154 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_172_17 := arithEq_of_rows h (row := 172) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_17
  simp only [← c11138, k3, ← c11139, k3, ← c11140] at e_172_17
  have e_172_18 := arithEq_of_rows h (row := 172) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_18
  simp only [← c11141, ← c11142, k3, ← c11143] at e_172_18
  have e_172_19 := arithEq_of_rows h (row := 172) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_19
  simp only [← c11150, ← c11151, k3, ← c11152] at e_172_19
  have e_173_13 := arithEq_of_rows h (row := 173) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_13
  simp only [← c11144, ← c11145, ← c11146] at e_173_13
  have e_173_14 := arithEq_of_rows h (row := 173) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_14
  simp only [← c11147, ← c11148, ← c11149] at e_173_14
  refine ⟨?_, ?_⟩
  · have hc := e_173_13
    simp only [e_172_18] at hc
    linear_combination c11153.trans k2 - hc
  · have hc := e_172_19
    simp only [e_172_17, e_173_14, e_172_18] at hc
    linear_combination c11154.trans k2 - hc

theorem publicBatchWrapper86_f1702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 807)) (a (.wire 165 31)) (a (.virt 17951)) (a (.virt 17952)) := by
  have c11155 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11156 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11157 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11158 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11159 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11160 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11161 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11162 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11163 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11164 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11165 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11166 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11167 := (publicBatchWrapper86_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11168 := (publicBatchWrapper86_copies349 a h).1
  have c11169 := (publicBatchWrapper86_copies349 a h).2.1
  have c11170 := (publicBatchWrapper86_copies349 a h).2.2.1
  have c11171 := (publicBatchWrapper86_copies349 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_173_15 := arithEq_of_rows h (row := 173) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_15
  simp only [← c11161, ← c11162, ← c11163] at e_173_15
  have e_173_16 := arithEq_of_rows h (row := 173) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_16
  simp only [← c11164, ← c11165, ← c11166] at e_173_16
  have e_174_0 := arithEq_of_rows h (row := 174) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_0
  simp only [← c11155, k3, ← c11156, k3, ← c11157] at e_174_0
  have e_174_1 := arithEq_of_rows h (row := 174) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_1
  simp only [← c11158, ← c11159, k3, ← c11160] at e_174_1
  have e_174_2 := arithEq_of_rows h (row := 174) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_2
  simp only [← c11167, ← c11168, k3, ← c11169] at e_174_2
  refine ⟨?_, ?_⟩
  · have hc := e_173_15
    simp only [e_174_1] at hc
    linear_combination c11170.trans k2 - hc
  · have hc := e_174_2
    simp only [e_174_0, e_173_16, e_174_1] at hc
    linear_combination c11171.trans k2 - hc

theorem publicBatchWrapper86_f1703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 808)) (a (.wire 165 39)) (a (.virt 17953)) (a (.virt 17954)) := by
  have c11172 := (publicBatchWrapper86_copies349 a h).2.2.2.2.1
  have c11173 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.1
  have c11174 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.1
  have c11175 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.1
  have c11176 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.1
  have c11177 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.1
  have c11178 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11179 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11180 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11181 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11182 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11183 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11184 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11185 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11186 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11187 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11188 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_173_17 := arithEq_of_rows h (row := 173) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_17
  simp only [← c11178, ← c11179, ← c11180] at e_173_17
  have e_173_18 := arithEq_of_rows h (row := 173) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_18
  simp only [← c11181, ← c11182, ← c11183] at e_173_18
  have e_174_3 := arithEq_of_rows h (row := 174) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_3
  simp only [← c11172, k3, ← c11173, k3, ← c11174] at e_174_3
  have e_174_4 := arithEq_of_rows h (row := 174) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_4
  simp only [← c11175, ← c11176, k3, ← c11177] at e_174_4
  have e_174_5 := arithEq_of_rows h (row := 174) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_5
  simp only [← c11184, ← c11185, k3, ← c11186] at e_174_5
  refine ⟨?_, ?_⟩
  · have hc := e_173_17
    simp only [e_174_4] at hc
    linear_combination c11187.trans k2 - hc
  · have hc := e_174_5
    simp only [e_174_3, e_173_18, e_174_4] at hc
    linear_combination c11188.trans k2 - hc

theorem publicBatchWrapper86_f1704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 173 79) = band (a (.virt 17947)) (a (.virt 17949)) := by
  have c11189 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11190 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11191 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_19 := arithEq_of_rows h (row := 173) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_19
  simp only [← c11189, ← c11190, ← c11191] at e_173_19
  have hr := e_173_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 175 3) = band (a (.virt 17951)) (a (.virt 17953)) := by
  have c11192 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11193 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11194 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_0 := arithEq_of_rows h (row := 175) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_0
  simp only [← c11192, ← c11193, ← c11194] at e_175_0
  have hr := e_175_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 175 7) = band (a (.wire 173 79)) (a (.wire 175 3)) := by
  have c11195 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11196 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11197 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_1 := arithEq_of_rows h (row := 175) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_1
  simp only [← c11195, ← c11196, ← c11197] at e_175_1
  have hr := e_175_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 31) = bor (a (.wire 6 43)) (a (.wire 175 7)) := by
  have c11198 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11199 := (publicBatchWrapper86_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11200 := (publicBatchWrapper86_copies350 a h).1
  have c11201 := (publicBatchWrapper86_copies350 a h).2.1
  have c11202 := (publicBatchWrapper86_copies350 a h).2.2.1
  have c11203 := (publicBatchWrapper86_copies350 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_162_19 := arithEq_of_rows h (row := 162) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_19
  simp only [← c11198, ← c11199, ← c11200] at e_162_19
  have e_170_7 := arithEq_of_rows h (row := 170) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_7
  simp only [← c11201, ← c11202, k3, ← c11203] at e_170_7
  have hr := e_170_7
  simp only [e_162_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 31) = a (.virt 201) := by
  have c11204 := (publicBatchWrapper86_copies350 a h).2.2.2.2.1
  exact c11204

theorem publicBatchWrapper86_f1709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1003)) (a (.wire 165 55)) (a (.virt 17955)) (a (.virt 17956)) := by
  have c11205 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.1
  have c11206 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.1
  have c11207 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.1
  have c11208 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.1
  have c11209 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.1
  have c11210 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11211 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11212 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11213 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11214 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11215 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11216 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11217 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11218 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11219 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11220 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11221 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_174_6 := arithEq_of_rows h (row := 174) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_6
  simp only [← c11205, k3, ← c11206, k3, ← c11207] at e_174_6
  have e_174_7 := arithEq_of_rows h (row := 174) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_7
  simp only [← c11208, ← c11209, k3, ← c11210] at e_174_7
  have e_174_8 := arithEq_of_rows h (row := 174) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_8
  simp only [← c11217, ← c11218, k3, ← c11219] at e_174_8
  have e_175_2 := arithEq_of_rows h (row := 175) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_2
  simp only [← c11211, ← c11212, ← c11213] at e_175_2
  have e_175_3 := arithEq_of_rows h (row := 175) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_3
  simp only [← c11214, ← c11215, ← c11216] at e_175_3
  refine ⟨?_, ?_⟩
  · have hc := e_175_2
    simp only [e_174_7] at hc
    linear_combination c11220.trans k2 - hc
  · have hc := e_174_8
    simp only [e_174_6, e_175_3, e_174_7] at hc
    linear_combination c11221.trans k2 - hc

theorem publicBatchWrapper86_f1710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 35) = bor (a (.wire 8 7)) (a (.virt 17955)) := by
  have c11222 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11223 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11224 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11225 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11226 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11227 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_8 := arithEq_of_rows h (row := 170) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_8
  simp only [← c11225, ← c11226, k3, ← c11227] at e_170_8
  have e_176_0 := arithEq_of_rows h (row := 176) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_0
  simp only [← c11222, ← c11223, ← c11224] at e_176_0
  have hr := e_170_8
  simp only [e_176_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 35) = a (.virt 201) := by
  have c11228 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11228

theorem publicBatchWrapper86_f1712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1004)) (a (.wire 165 63)) (a (.virt 17957)) (a (.virt 17958)) := by
  have c11229 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11230 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11231 := (publicBatchWrapper86_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11232 := (publicBatchWrapper86_copies351 a h).1
  have c11233 := (publicBatchWrapper86_copies351 a h).2.1
  have c11234 := (publicBatchWrapper86_copies351 a h).2.2.1
  have c11235 := (publicBatchWrapper86_copies351 a h).2.2.2.1
  have c11236 := (publicBatchWrapper86_copies351 a h).2.2.2.2.1
  have c11237 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.1
  have c11238 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.1
  have c11239 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.1
  have c11240 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.1
  have c11241 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.1
  have c11242 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11243 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11244 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11245 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_174_9 := arithEq_of_rows h (row := 174) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_9
  simp only [← c11229, k3, ← c11230, k3, ← c11231] at e_174_9
  have e_174_10 := arithEq_of_rows h (row := 174) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_10
  simp only [← c11232, ← c11233, k3, ← c11234] at e_174_10
  have e_174_11 := arithEq_of_rows h (row := 174) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_11
  simp only [← c11241, ← c11242, k3, ← c11243] at e_174_11
  have e_175_4 := arithEq_of_rows h (row := 175) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_4
  simp only [← c11235, ← c11236, ← c11237] at e_175_4
  have e_175_5 := arithEq_of_rows h (row := 175) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_5
  simp only [← c11238, ← c11239, ← c11240] at e_175_5
  refine ⟨?_, ?_⟩
  · have hc := e_175_4
    simp only [e_174_10] at hc
    linear_combination c11244.trans k2 - hc
  · have hc := e_174_11
    simp only [e_174_9, e_175_5, e_174_10] at hc
    linear_combination c11245.trans k2 - hc

theorem publicBatchWrapper86_f1713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 39) = bor (a (.wire 8 7)) (a (.virt 17957)) := by
  have c11246 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11247 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11248 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11249 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11250 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11251 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_9 := arithEq_of_rows h (row := 170) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_9
  simp only [← c11249, ← c11250, k3, ← c11251] at e_170_9
  have e_176_1 := arithEq_of_rows h (row := 176) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_1
  simp only [← c11246, ← c11247, ← c11248] at e_176_1
  have hr := e_170_9
  simp only [e_176_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 39) = a (.virt 201) := by
  have c11252 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11252

theorem publicBatchWrapper86_f1715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1005)) (a (.wire 165 15)) (a (.virt 17959)) (a (.virt 17960)) := by
  have c11253 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11254 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11255 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11256 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11257 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11258 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11259 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11260 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11261 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11262 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11263 := (publicBatchWrapper86_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11264 := (publicBatchWrapper86_copies352 a h).1
  have c11265 := (publicBatchWrapper86_copies352 a h).2.1
  have c11266 := (publicBatchWrapper86_copies352 a h).2.2.1
  have c11267 := (publicBatchWrapper86_copies352 a h).2.2.2.1
  have c11268 := (publicBatchWrapper86_copies352 a h).2.2.2.2.1
  have c11269 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_174_12 := arithEq_of_rows h (row := 174) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_12
  simp only [← c11253, k3, ← c11254, k3, ← c11255] at e_174_12
  have e_174_13 := arithEq_of_rows h (row := 174) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_13
  simp only [← c11256, ← c11257, k3, ← c11258] at e_174_13
  have e_174_14 := arithEq_of_rows h (row := 174) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_14
  simp only [← c11265, ← c11266, k3, ← c11267] at e_174_14
  have e_175_6 := arithEq_of_rows h (row := 175) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_6
  simp only [← c11259, ← c11260, ← c11261] at e_175_6
  have e_175_7 := arithEq_of_rows h (row := 175) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_7
  simp only [← c11262, ← c11263, ← c11264] at e_175_7
  refine ⟨?_, ?_⟩
  · have hc := e_175_6
    simp only [e_174_13] at hc
    linear_combination c11268.trans k2 - hc
  · have hc := e_174_14
    simp only [e_174_12, e_175_7, e_174_13] at hc
    linear_combination c11269.trans k2 - hc

theorem publicBatchWrapper86_f1716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1006)) (a (.wire 165 23)) (a (.virt 17961)) (a (.virt 17962)) := by
  have c11270 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.1
  have c11271 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.1
  have c11272 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.1
  have c11273 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.1
  have c11274 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11275 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11276 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11277 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11278 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11279 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11280 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11281 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11282 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11283 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11284 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11285 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11286 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_174_15 := arithEq_of_rows h (row := 174) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_15
  simp only [← c11270, k3, ← c11271, k3, ← c11272] at e_174_15
  have e_174_16 := arithEq_of_rows h (row := 174) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_16
  simp only [← c11273, ← c11274, k3, ← c11275] at e_174_16
  have e_174_17 := arithEq_of_rows h (row := 174) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_17
  simp only [← c11282, ← c11283, k3, ← c11284] at e_174_17
  have e_175_8 := arithEq_of_rows h (row := 175) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_8
  simp only [← c11276, ← c11277, ← c11278] at e_175_8
  have e_175_9 := arithEq_of_rows h (row := 175) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_9
  simp only [← c11279, ← c11280, ← c11281] at e_175_9
  refine ⟨?_, ?_⟩
  · have hc := e_175_8
    simp only [e_174_16] at hc
    linear_combination c11285.trans k2 - hc
  · have hc := e_174_17
    simp only [e_174_15, e_175_9, e_174_16] at hc
    linear_combination c11286.trans k2 - hc

theorem publicBatchWrapper86_f1717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1007)) (a (.wire 165 31)) (a (.virt 17963)) (a (.virt 17964)) := by
  have c11287 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11288 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11289 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11290 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11291 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11292 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11293 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11294 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11295 := (publicBatchWrapper86_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11296 := (publicBatchWrapper86_copies353 a h).1
  have c11297 := (publicBatchWrapper86_copies353 a h).2.1
  have c11298 := (publicBatchWrapper86_copies353 a h).2.2.1
  have c11299 := (publicBatchWrapper86_copies353 a h).2.2.2.1
  have c11300 := (publicBatchWrapper86_copies353 a h).2.2.2.2.1
  have c11301 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.1
  have c11302 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.1
  have c11303 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_174_18 := arithEq_of_rows h (row := 174) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_18
  simp only [← c11287, k3, ← c11288, k3, ← c11289] at e_174_18
  have e_174_19 := arithEq_of_rows h (row := 174) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_19
  simp only [← c11290, ← c11291, k3, ← c11292] at e_174_19
  have e_175_10 := arithEq_of_rows h (row := 175) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_10
  simp only [← c11293, ← c11294, ← c11295] at e_175_10
  have e_175_11 := arithEq_of_rows h (row := 175) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_11
  simp only [← c11296, ← c11297, ← c11298] at e_175_11
  have e_177_0 := arithEq_of_rows h (row := 177) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_0
  simp only [← c11299, ← c11300, k3, ← c11301] at e_177_0
  refine ⟨?_, ?_⟩
  · have hc := e_175_10
    simp only [e_174_19] at hc
    linear_combination c11302.trans k2 - hc
  · have hc := e_177_0
    simp only [e_174_18, e_175_11, e_174_19] at hc
    linear_combination c11303.trans k2 - hc

theorem publicBatchWrapper86_f1718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1008)) (a (.wire 165 39)) (a (.virt 17965)) (a (.virt 17966)) := by
  have c11304 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.1
  have c11305 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.1
  have c11306 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11307 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11308 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11309 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11310 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11311 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11312 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11313 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11314 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11315 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11316 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11317 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11318 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11319 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11320 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_175_12 := arithEq_of_rows h (row := 175) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_12
  simp only [← c11310, ← c11311, ← c11312] at e_175_12
  have e_175_13 := arithEq_of_rows h (row := 175) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_13
  simp only [← c11313, ← c11314, ← c11315] at e_175_13
  have e_177_1 := arithEq_of_rows h (row := 177) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_1
  simp only [← c11304, k3, ← c11305, k3, ← c11306] at e_177_1
  have e_177_2 := arithEq_of_rows h (row := 177) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_2
  simp only [← c11307, ← c11308, k3, ← c11309] at e_177_2
  have e_177_3 := arithEq_of_rows h (row := 177) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_3
  simp only [← c11316, ← c11317, k3, ← c11318] at e_177_3
  refine ⟨?_, ?_⟩
  · have hc := e_175_12
    simp only [e_177_2] at hc
    linear_combination c11319.trans k2 - hc
  · have hc := e_177_3
    simp only [e_177_1, e_175_13, e_177_2] at hc
    linear_combination c11320.trans k2 - hc

theorem publicBatchWrapper86_f1719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 175 59) = band (a (.virt 17959)) (a (.virt 17961)) := by
  have c11321 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11322 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11323 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_14 := arithEq_of_rows h (row := 175) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_14
  simp only [← c11321, ← c11322, ← c11323] at e_175_14
  have hr := e_175_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 175 63) = band (a (.virt 17963)) (a (.virt 17965)) := by
  have c11324 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11325 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11326 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_175_15 := arithEq_of_rows h (row := 175) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_15
  simp only [← c11324, ← c11325, ← c11326] at e_175_15
  have hr := e_175_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 175 67) = band (a (.wire 175 59)) (a (.wire 175 63)) := by
  have c11327 := (publicBatchWrapper86_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11328 := (publicBatchWrapper86_copies354 a h).1
  have c11329 := (publicBatchWrapper86_copies354 a h).2.1
  have e_175_16 := arithEq_of_rows h (row := 175) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_16
  simp only [← c11327, ← c11328, ← c11329] at e_175_16
  have hr := e_175_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 43) = bor (a (.wire 8 7)) (a (.wire 175 67)) := by
  have c11330 := (publicBatchWrapper86_copies354 a h).2.2.1
  have c11331 := (publicBatchWrapper86_copies354 a h).2.2.2.1
  have c11332 := (publicBatchWrapper86_copies354 a h).2.2.2.2.1
  have c11333 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.1
  have c11334 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.1
  have c11335 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_10 := arithEq_of_rows h (row := 170) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_10
  simp only [← c11333, ← c11334, k3, ← c11335] at e_170_10
  have e_176_2 := arithEq_of_rows h (row := 176) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_2
  simp only [← c11330, ← c11331, ← c11332] at e_176_2
  have hr := e_170_10
  simp only [e_176_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 43) = a (.virt 201) := by
  have c11336 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.1
  exact c11336

theorem publicBatchWrapper86_f1724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1203)) (a (.wire 165 55)) (a (.virt 17967)) (a (.virt 17968)) := by
  have c11337 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.1
  have c11338 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11339 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11340 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11341 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11342 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11343 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11344 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11345 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11346 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11347 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11348 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11349 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11350 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11351 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11352 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11353 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_175_17 := arithEq_of_rows h (row := 175) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_17
  simp only [← c11343, ← c11344, ← c11345] at e_175_17
  have e_175_18 := arithEq_of_rows h (row := 175) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_18
  simp only [← c11346, ← c11347, ← c11348] at e_175_18
  have e_177_4 := arithEq_of_rows h (row := 177) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_4
  simp only [← c11337, k3, ← c11338, k3, ← c11339] at e_177_4
  have e_177_5 := arithEq_of_rows h (row := 177) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_5
  simp only [← c11340, ← c11341, k3, ← c11342] at e_177_5
  have e_177_6 := arithEq_of_rows h (row := 177) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_6
  simp only [← c11349, ← c11350, k3, ← c11351] at e_177_6
  refine ⟨?_, ?_⟩
  · have hc := e_175_17
    simp only [e_177_5] at hc
    linear_combination c11352.trans k2 - hc
  · have hc := e_177_6
    simp only [e_177_4, e_175_18, e_177_5] at hc
    linear_combination c11353.trans k2 - hc

theorem publicBatchWrapper86_f1725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 47) = bor (a (.wire 8 51)) (a (.virt 17967)) := by
  have c11354 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11355 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11356 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11357 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11358 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11359 := (publicBatchWrapper86_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_11 := arithEq_of_rows h (row := 170) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_11
  simp only [← c11357, ← c11358, k3, ← c11359] at e_170_11
  have e_176_3 := arithEq_of_rows h (row := 176) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_3
  simp only [← c11354, ← c11355, ← c11356] at e_176_3
  have hr := e_170_11
  simp only [e_176_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 47) = a (.virt 201) := by
  have c11360 := (publicBatchWrapper86_copies355 a h).1
  exact c11360

theorem publicBatchWrapper86_f1727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1204)) (a (.wire 165 63)) (a (.virt 17969)) (a (.virt 17970)) := by
  have c11361 := (publicBatchWrapper86_copies355 a h).2.1
  have c11362 := (publicBatchWrapper86_copies355 a h).2.2.1
  have c11363 := (publicBatchWrapper86_copies355 a h).2.2.2.1
  have c11364 := (publicBatchWrapper86_copies355 a h).2.2.2.2.1
  have c11365 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.1
  have c11366 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.1
  have c11367 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.1
  have c11368 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.1
  have c11369 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.1
  have c11370 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11371 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11372 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11373 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11374 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11375 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11376 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11377 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_175_19 := arithEq_of_rows h (row := 175) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_19
  simp only [← c11367, ← c11368, ← c11369] at e_175_19
  have e_177_7 := arithEq_of_rows h (row := 177) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_7
  simp only [← c11361, k3, ← c11362, k3, ← c11363] at e_177_7
  have e_177_8 := arithEq_of_rows h (row := 177) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_8
  simp only [← c11364, ← c11365, k3, ← c11366] at e_177_8
  have e_177_9 := arithEq_of_rows h (row := 177) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_9
  simp only [← c11373, ← c11374, k3, ← c11375] at e_177_9
  have e_178_0 := arithEq_of_rows h (row := 178) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_0
  simp only [← c11370, ← c11371, ← c11372] at e_178_0
  refine ⟨?_, ?_⟩
  · have hc := e_175_19
    simp only [e_177_8] at hc
    linear_combination c11376.trans k2 - hc
  · have hc := e_177_9
    simp only [e_177_7, e_178_0, e_177_8] at hc
    linear_combination c11377.trans k2 - hc

theorem publicBatchWrapper86_f1728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 51) = bor (a (.wire 8 51)) (a (.virt 17969)) := by
  have c11378 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11379 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11380 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11381 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11382 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11383 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_12 := arithEq_of_rows h (row := 170) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_12
  simp only [← c11381, ← c11382, k3, ← c11383] at e_170_12
  have e_176_4 := arithEq_of_rows h (row := 176) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_4
  simp only [← c11378, ← c11379, ← c11380] at e_176_4
  have hr := e_170_12
  simp only [e_176_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 51) = a (.virt 201) := by
  have c11384 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11384

theorem publicBatchWrapper86_f1730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1205)) (a (.wire 165 15)) (a (.virt 17971)) (a (.virt 17972)) := by
  have c11385 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11386 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11387 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11388 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11389 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11390 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11391 := (publicBatchWrapper86_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11392 := (publicBatchWrapper86_copies356 a h).1
  have c11393 := (publicBatchWrapper86_copies356 a h).2.1
  have c11394 := (publicBatchWrapper86_copies356 a h).2.2.1
  have c11395 := (publicBatchWrapper86_copies356 a h).2.2.2.1
  have c11396 := (publicBatchWrapper86_copies356 a h).2.2.2.2.1
  have c11397 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.1
  have c11398 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.1
  have c11399 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.1
  have c11400 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.1
  have c11401 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_177_10 := arithEq_of_rows h (row := 177) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_10
  simp only [← c11385, k3, ← c11386, k3, ← c11387] at e_177_10
  have e_177_11 := arithEq_of_rows h (row := 177) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_11
  simp only [← c11388, ← c11389, k3, ← c11390] at e_177_11
  have e_177_12 := arithEq_of_rows h (row := 177) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_12
  simp only [← c11397, ← c11398, k3, ← c11399] at e_177_12
  have e_178_1 := arithEq_of_rows h (row := 178) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_1
  simp only [← c11391, ← c11392, ← c11393] at e_178_1
  have e_178_2 := arithEq_of_rows h (row := 178) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_2
  simp only [← c11394, ← c11395, ← c11396] at e_178_2
  refine ⟨?_, ?_⟩
  · have hc := e_178_1
    simp only [e_177_11] at hc
    linear_combination c11400.trans k2 - hc
  · have hc := e_177_12
    simp only [e_177_10, e_178_2, e_177_11] at hc
    linear_combination c11401.trans k2 - hc

theorem publicBatchWrapper86_f1731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1206)) (a (.wire 165 23)) (a (.virt 17973)) (a (.virt 17974)) := by
  have c11402 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11403 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11404 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11405 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11406 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11407 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11408 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11409 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11410 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11411 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11412 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11413 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11414 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11415 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11416 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11417 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11418 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_177_13 := arithEq_of_rows h (row := 177) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_13
  simp only [← c11402, k3, ← c11403, k3, ← c11404] at e_177_13
  have e_177_14 := arithEq_of_rows h (row := 177) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_14
  simp only [← c11405, ← c11406, k3, ← c11407] at e_177_14
  have e_177_15 := arithEq_of_rows h (row := 177) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_15
  simp only [← c11414, ← c11415, k3, ← c11416] at e_177_15
  have e_178_3 := arithEq_of_rows h (row := 178) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_3
  simp only [← c11408, ← c11409, ← c11410] at e_178_3
  have e_178_4 := arithEq_of_rows h (row := 178) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_4
  simp only [← c11411, ← c11412, ← c11413] at e_178_4
  refine ⟨?_, ?_⟩
  · have hc := e_178_3
    simp only [e_177_14] at hc
    linear_combination c11417.trans k2 - hc
  · have hc := e_177_15
    simp only [e_177_13, e_178_4, e_177_14] at hc
    linear_combination c11418.trans k2 - hc

theorem publicBatchWrapper86_f1732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1207)) (a (.wire 165 31)) (a (.virt 17975)) (a (.virt 17976)) := by
  have c11419 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11420 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11421 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11422 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11423 := (publicBatchWrapper86_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11424 := (publicBatchWrapper86_copies357 a h).1
  have c11425 := (publicBatchWrapper86_copies357 a h).2.1
  have c11426 := (publicBatchWrapper86_copies357 a h).2.2.1
  have c11427 := (publicBatchWrapper86_copies357 a h).2.2.2.1
  have c11428 := (publicBatchWrapper86_copies357 a h).2.2.2.2.1
  have c11429 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.1
  have c11430 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.1
  have c11431 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.1
  have c11432 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.1
  have c11433 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.1
  have c11434 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11435 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_177_16 := arithEq_of_rows h (row := 177) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_16
  simp only [← c11419, k3, ← c11420, k3, ← c11421] at e_177_16
  have e_177_17 := arithEq_of_rows h (row := 177) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_17
  simp only [← c11422, ← c11423, k3, ← c11424] at e_177_17
  have e_177_18 := arithEq_of_rows h (row := 177) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_18
  simp only [← c11431, ← c11432, k3, ← c11433] at e_177_18
  have e_178_5 := arithEq_of_rows h (row := 178) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_5
  simp only [← c11425, ← c11426, ← c11427] at e_178_5
  have e_178_6 := arithEq_of_rows h (row := 178) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_6
  simp only [← c11428, ← c11429, ← c11430] at e_178_6
  refine ⟨?_, ?_⟩
  · have hc := e_178_5
    simp only [e_177_17] at hc
    linear_combination c11434.trans k2 - hc
  · have hc := e_177_18
    simp only [e_177_16, e_178_6, e_177_17] at hc
    linear_combination c11435.trans k2 - hc

theorem publicBatchWrapper86_f1733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1208)) (a (.wire 165 39)) (a (.virt 17977)) (a (.virt 17978)) := by
  have c11436 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11437 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11438 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11439 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11440 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11441 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11442 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11443 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11444 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11445 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11446 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11447 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11448 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11449 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11450 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11451 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11452 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_177_19 := arithEq_of_rows h (row := 177) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_19
  simp only [← c11436, k3, ← c11437, k3, ← c11438] at e_177_19
  have e_178_7 := arithEq_of_rows h (row := 178) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_7
  simp only [← c11442, ← c11443, ← c11444] at e_178_7
  have e_178_8 := arithEq_of_rows h (row := 178) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_8
  simp only [← c11445, ← c11446, ← c11447] at e_178_8
  have e_179_0 := arithEq_of_rows h (row := 179) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_0
  simp only [← c11439, ← c11440, k3, ← c11441] at e_179_0
  have e_179_1 := arithEq_of_rows h (row := 179) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_1
  simp only [← c11448, ← c11449, k3, ← c11450] at e_179_1
  refine ⟨?_, ?_⟩
  · have hc := e_178_7
    simp only [e_179_0] at hc
    linear_combination c11451.trans k2 - hc
  · have hc := e_179_1
    simp only [e_177_19, e_178_8, e_179_0] at hc
    linear_combination c11452.trans k2 - hc

theorem publicBatchWrapper86_f1734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 178 39) = band (a (.virt 17971)) (a (.virt 17973)) := by
  have c11453 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11454 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11455 := (publicBatchWrapper86_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_178_9 := arithEq_of_rows h (row := 178) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_9
  simp only [← c11453, ← c11454, ← c11455] at e_178_9
  have hr := e_178_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 178 43) = band (a (.virt 17975)) (a (.virt 17977)) := by
  have c11456 := (publicBatchWrapper86_copies358 a h).1
  have c11457 := (publicBatchWrapper86_copies358 a h).2.1
  have c11458 := (publicBatchWrapper86_copies358 a h).2.2.1
  have e_178_10 := arithEq_of_rows h (row := 178) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_10
  simp only [← c11456, ← c11457, ← c11458] at e_178_10
  have hr := e_178_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 178 47) = band (a (.wire 178 39)) (a (.wire 178 43)) := by
  have c11459 := (publicBatchWrapper86_copies358 a h).2.2.2.1
  have c11460 := (publicBatchWrapper86_copies358 a h).2.2.2.2.1
  have c11461 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.1
  have e_178_11 := arithEq_of_rows h (row := 178) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_11
  simp only [← c11459, ← c11460, ← c11461] at e_178_11
  have hr := e_178_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 55) = bor (a (.wire 8 51)) (a (.wire 178 47)) := by
  have c11462 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.1
  have c11463 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.1
  have c11464 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.1
  have c11465 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.1
  have c11466 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11467 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_13 := arithEq_of_rows h (row := 170) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_13
  simp only [← c11465, ← c11466, k3, ← c11467] at e_170_13
  have e_176_5 := arithEq_of_rows h (row := 176) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_5
  simp only [← c11462, ← c11463, ← c11464] at e_176_5
  have hr := e_170_13
  simp only [e_176_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 55) = a (.virt 201) := by
  have c11468 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11468

theorem publicBatchWrapper86_f1739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1403)) (a (.wire 165 55)) (a (.virt 17979)) (a (.virt 17980)) := by
  have c11469 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11470 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11471 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11472 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11473 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11474 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11475 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11476 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11477 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11478 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11479 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11480 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11481 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11482 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11483 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11484 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11485 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_178_12 := arithEq_of_rows h (row := 178) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_12
  simp only [← c11475, ← c11476, ← c11477] at e_178_12
  have e_178_13 := arithEq_of_rows h (row := 178) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_13
  simp only [← c11478, ← c11479, ← c11480] at e_178_13
  have e_179_2 := arithEq_of_rows h (row := 179) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_2
  simp only [← c11469, k3, ← c11470, k3, ← c11471] at e_179_2
  have e_179_3 := arithEq_of_rows h (row := 179) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_3
  simp only [← c11472, ← c11473, k3, ← c11474] at e_179_3
  have e_179_4 := arithEq_of_rows h (row := 179) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_4
  simp only [← c11481, ← c11482, k3, ← c11483] at e_179_4
  refine ⟨?_, ?_⟩
  · have hc := e_178_12
    simp only [e_179_3] at hc
    linear_combination c11484.trans k2 - hc
  · have hc := e_179_4
    simp only [e_179_2, e_178_13, e_179_3] at hc
    linear_combination c11485.trans k2 - hc

theorem publicBatchWrapper86_f1740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 59) = bor (a (.wire 10 15)) (a (.virt 17979)) := by
  have c11486 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11487 := (publicBatchWrapper86_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11488 := (publicBatchWrapper86_copies359 a h).1
  have c11489 := (publicBatchWrapper86_copies359 a h).2.1
  have c11490 := (publicBatchWrapper86_copies359 a h).2.2.1
  have c11491 := (publicBatchWrapper86_copies359 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_14 := arithEq_of_rows h (row := 170) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_14
  simp only [← c11489, ← c11490, k3, ← c11491] at e_170_14
  have e_176_6 := arithEq_of_rows h (row := 176) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_6
  simp only [← c11486, ← c11487, ← c11488] at e_176_6
  have hr := e_170_14
  simp only [e_176_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 59) = a (.virt 201) := by
  have c11492 := (publicBatchWrapper86_copies359 a h).2.2.2.2.1
  exact c11492

theorem publicBatchWrapper86_f1742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1404)) (a (.wire 165 63)) (a (.virt 17981)) (a (.virt 17982)) := by
  have c11493 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.1
  have c11494 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.1
  have c11495 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.1
  have c11496 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.1
  have c11497 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.1
  have c11498 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11499 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11500 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11501 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11502 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11503 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11504 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11505 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11506 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11507 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11508 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11509 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_178_14 := arithEq_of_rows h (row := 178) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_14
  simp only [← c11499, ← c11500, ← c11501] at e_178_14
  have e_178_15 := arithEq_of_rows h (row := 178) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_15
  simp only [← c11502, ← c11503, ← c11504] at e_178_15
  have e_179_5 := arithEq_of_rows h (row := 179) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_5
  simp only [← c11493, k3, ← c11494, k3, ← c11495] at e_179_5
  have e_179_6 := arithEq_of_rows h (row := 179) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_6
  simp only [← c11496, ← c11497, k3, ← c11498] at e_179_6
  have e_179_7 := arithEq_of_rows h (row := 179) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_7
  simp only [← c11505, ← c11506, k3, ← c11507] at e_179_7
  refine ⟨?_, ?_⟩
  · have hc := e_178_14
    simp only [e_179_6] at hc
    linear_combination c11508.trans k2 - hc
  · have hc := e_179_7
    simp only [e_179_5, e_178_15, e_179_6] at hc
    linear_combination c11509.trans k2 - hc

theorem publicBatchWrapper86_f1743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 63) = bor (a (.wire 10 15)) (a (.virt 17981)) := by
  have c11510 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11511 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11512 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11513 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11514 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11515 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_15 := arithEq_of_rows h (row := 170) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_15
  simp only [← c11513, ← c11514, k3, ← c11515] at e_170_15
  have e_176_7 := arithEq_of_rows h (row := 176) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_7
  simp only [← c11510, ← c11511, ← c11512] at e_176_7
  have hr := e_170_15
  simp only [e_176_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 63) = a (.virt 201) := by
  have c11516 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11516

theorem publicBatchWrapper86_f1745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1405)) (a (.wire 165 15)) (a (.virt 17983)) (a (.virt 17984)) := by
  have c11517 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11518 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11519 := (publicBatchWrapper86_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11520 := (publicBatchWrapper86_copies360 a h).1
  have c11521 := (publicBatchWrapper86_copies360 a h).2.1
  have c11522 := (publicBatchWrapper86_copies360 a h).2.2.1
  have c11523 := (publicBatchWrapper86_copies360 a h).2.2.2.1
  have c11524 := (publicBatchWrapper86_copies360 a h).2.2.2.2.1
  have c11525 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.1
  have c11526 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.1
  have c11527 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.1
  have c11528 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.1
  have c11529 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.1
  have c11530 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11531 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11532 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11533 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_178_16 := arithEq_of_rows h (row := 178) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_16
  simp only [← c11523, ← c11524, ← c11525] at e_178_16
  have e_178_17 := arithEq_of_rows h (row := 178) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_17
  simp only [← c11526, ← c11527, ← c11528] at e_178_17
  have e_179_8 := arithEq_of_rows h (row := 179) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_8
  simp only [← c11517, k3, ← c11518, k3, ← c11519] at e_179_8
  have e_179_9 := arithEq_of_rows h (row := 179) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_9
  simp only [← c11520, ← c11521, k3, ← c11522] at e_179_9
  have e_179_10 := arithEq_of_rows h (row := 179) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_10
  simp only [← c11529, ← c11530, k3, ← c11531] at e_179_10
  refine ⟨?_, ?_⟩
  · have hc := e_178_16
    simp only [e_179_9] at hc
    linear_combination c11532.trans k2 - hc
  · have hc := e_179_10
    simp only [e_179_8, e_178_17, e_179_9] at hc
    linear_combination c11533.trans k2 - hc

theorem publicBatchWrapper86_f1746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1406)) (a (.wire 165 23)) (a (.virt 17985)) (a (.virt 17986)) := by
  have c11534 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11535 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11536 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11537 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11538 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11539 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11540 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11541 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11542 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11543 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11544 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11545 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11546 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11547 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11548 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11549 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11550 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_178_18 := arithEq_of_rows h (row := 178) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_18
  simp only [← c11540, ← c11541, ← c11542] at e_178_18
  have e_178_19 := arithEq_of_rows h (row := 178) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_19
  simp only [← c11543, ← c11544, ← c11545] at e_178_19
  have e_179_11 := arithEq_of_rows h (row := 179) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_11
  simp only [← c11534, k3, ← c11535, k3, ← c11536] at e_179_11
  have e_179_12 := arithEq_of_rows h (row := 179) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_12
  simp only [← c11537, ← c11538, k3, ← c11539] at e_179_12
  have e_179_13 := arithEq_of_rows h (row := 179) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_13
  simp only [← c11546, ← c11547, k3, ← c11548] at e_179_13
  refine ⟨?_, ?_⟩
  · have hc := e_178_18
    simp only [e_179_12] at hc
    linear_combination c11549.trans k2 - hc
  · have hc := e_179_13
    simp only [e_179_11, e_178_19, e_179_12] at hc
    linear_combination c11550.trans k2 - hc

theorem publicBatchWrapper86_f1747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1407)) (a (.wire 165 31)) (a (.virt 17987)) (a (.virt 17988)) := by
  have c11551 := (publicBatchWrapper86_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11552 := (publicBatchWrapper86_copies361 a h).1
  have c11553 := (publicBatchWrapper86_copies361 a h).2.1
  have c11554 := (publicBatchWrapper86_copies361 a h).2.2.1
  have c11555 := (publicBatchWrapper86_copies361 a h).2.2.2.1
  have c11556 := (publicBatchWrapper86_copies361 a h).2.2.2.2.1
  have c11557 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.1
  have c11558 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.1
  have c11559 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.1
  have c11560 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.1
  have c11561 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.1
  have c11562 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11563 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11564 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11565 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11566 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11567 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_179_14 := arithEq_of_rows h (row := 179) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_14
  simp only [← c11551, k3, ← c11552, k3, ← c11553] at e_179_14
  have e_179_15 := arithEq_of_rows h (row := 179) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_15
  simp only [← c11554, ← c11555, k3, ← c11556] at e_179_15
  have e_179_16 := arithEq_of_rows h (row := 179) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_16
  simp only [← c11563, ← c11564, k3, ← c11565] at e_179_16
  have e_180_0 := arithEq_of_rows h (row := 180) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_0
  simp only [← c11557, ← c11558, ← c11559] at e_180_0
  have e_180_1 := arithEq_of_rows h (row := 180) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_1
  simp only [← c11560, ← c11561, ← c11562] at e_180_1
  refine ⟨?_, ?_⟩
  · have hc := e_180_0
    simp only [e_179_15] at hc
    linear_combination c11566.trans k2 - hc
  · have hc := e_179_16
    simp only [e_179_14, e_180_1, e_179_15] at hc
    linear_combination c11567.trans k2 - hc

theorem publicBatchWrapper86_f1748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1408)) (a (.wire 165 39)) (a (.virt 17989)) (a (.virt 17990)) := by
  have c11568 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11569 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11570 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11571 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11572 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11573 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11574 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11575 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11576 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11577 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11578 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11579 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11580 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11581 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11582 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11583 := (publicBatchWrapper86_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11584 := (publicBatchWrapper86_copies362 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_179_17 := arithEq_of_rows h (row := 179) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_17
  simp only [← c11568, k3, ← c11569, k3, ← c11570] at e_179_17
  have e_179_18 := arithEq_of_rows h (row := 179) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_18
  simp only [← c11571, ← c11572, k3, ← c11573] at e_179_18
  have e_179_19 := arithEq_of_rows h (row := 179) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_19
  simp only [← c11580, ← c11581, k3, ← c11582] at e_179_19
  have e_180_2 := arithEq_of_rows h (row := 180) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_2
  simp only [← c11574, ← c11575, ← c11576] at e_180_2
  have e_180_3 := arithEq_of_rows h (row := 180) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_3
  simp only [← c11577, ← c11578, ← c11579] at e_180_3
  refine ⟨?_, ?_⟩
  · have hc := e_180_2
    simp only [e_179_18] at hc
    linear_combination c11583.trans k2 - hc
  · have hc := e_179_19
    simp only [e_179_17, e_180_3, e_179_18] at hc
    linear_combination c11584.trans k2 - hc

theorem publicBatchWrapper86_f1749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 180 19) = band (a (.virt 17983)) (a (.virt 17985)) := by
  have c11585 := (publicBatchWrapper86_copies362 a h).2.1
  have c11586 := (publicBatchWrapper86_copies362 a h).2.2.1
  have c11587 := (publicBatchWrapper86_copies362 a h).2.2.2.1
  have e_180_4 := arithEq_of_rows h (row := 180) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_4
  simp only [← c11585, ← c11586, ← c11587] at e_180_4
  have hr := e_180_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 180 23) = band (a (.virt 17987)) (a (.virt 17989)) := by
  have c11588 := (publicBatchWrapper86_copies362 a h).2.2.2.2.1
  have c11589 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.1
  have c11590 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.1
  have e_180_5 := arithEq_of_rows h (row := 180) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_5
  simp only [← c11588, ← c11589, ← c11590] at e_180_5
  have hr := e_180_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 180 27) = band (a (.wire 180 19)) (a (.wire 180 23)) := by
  have c11591 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.1
  have c11592 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.1
  have c11593 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.1
  have e_180_6 := arithEq_of_rows h (row := 180) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_6
  simp only [← c11591, ← c11592, ← c11593] at e_180_6
  have hr := e_180_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 67) = bor (a (.wire 10 15)) (a (.wire 180 27)) := by
  have c11594 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11595 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11596 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11597 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11598 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11599 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_16 := arithEq_of_rows h (row := 170) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_16
  simp only [← c11597, ← c11598, k3, ← c11599] at e_170_16
  have e_176_8 := arithEq_of_rows h (row := 176) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_8
  simp only [← c11594, ← c11595, ← c11596] at e_176_8
  have hr := e_170_16
  simp only [e_176_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 67) = a (.virt 201) := by
  have c11600 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11600

theorem publicBatchWrapper86_f1754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1603)) (a (.wire 165 55)) (a (.virt 17991)) (a (.virt 17992)) := by
  have c11601 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11602 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11603 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11604 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11605 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11606 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11607 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11608 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11609 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11610 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11611 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11612 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11613 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11614 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11615 := (publicBatchWrapper86_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11616 := (publicBatchWrapper86_copies363 a h).1
  have c11617 := (publicBatchWrapper86_copies363 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_180_7 := arithEq_of_rows h (row := 180) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_7
  simp only [← c11607, ← c11608, ← c11609] at e_180_7
  have e_180_8 := arithEq_of_rows h (row := 180) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_8
  simp only [← c11610, ← c11611, ← c11612] at e_180_8
  have e_181_0 := arithEq_of_rows h (row := 181) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_0
  simp only [← c11601, k3, ← c11602, k3, ← c11603] at e_181_0
  have e_181_1 := arithEq_of_rows h (row := 181) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_1
  simp only [← c11604, ← c11605, k3, ← c11606] at e_181_1
  have e_181_2 := arithEq_of_rows h (row := 181) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_2
  simp only [← c11613, ← c11614, k3, ← c11615] at e_181_2
  refine ⟨?_, ?_⟩
  · have hc := e_180_7
    simp only [e_181_1] at hc
    linear_combination c11616.trans k2 - hc
  · have hc := e_181_2
    simp only [e_181_0, e_180_8, e_181_1] at hc
    linear_combination c11617.trans k2 - hc

theorem publicBatchWrapper86_f1755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 71) = bor (a (.wire 10 59)) (a (.virt 17991)) := by
  have c11618 := (publicBatchWrapper86_copies363 a h).2.2.1
  have c11619 := (publicBatchWrapper86_copies363 a h).2.2.2.1
  have c11620 := (publicBatchWrapper86_copies363 a h).2.2.2.2.1
  have c11621 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.1
  have c11622 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.1
  have c11623 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_17 := arithEq_of_rows h (row := 170) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_17
  simp only [← c11621, ← c11622, k3, ← c11623] at e_170_17
  have e_176_9 := arithEq_of_rows h (row := 176) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_9
  simp only [← c11618, ← c11619, ← c11620] at e_176_9
  have hr := e_170_17
  simp only [e_176_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 71) = a (.virt 201) := by
  have c11624 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.1
  exact c11624

theorem publicBatchWrapper86_f1757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1604)) (a (.wire 165 63)) (a (.virt 17993)) (a (.virt 17994)) := by
  have c11625 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.1
  have c11626 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11627 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11628 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11629 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11630 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11631 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11632 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11633 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11634 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11635 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11636 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11637 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11638 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11639 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11640 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11641 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_180_9 := arithEq_of_rows h (row := 180) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_9
  simp only [← c11631, ← c11632, ← c11633] at e_180_9
  have e_180_10 := arithEq_of_rows h (row := 180) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_10
  simp only [← c11634, ← c11635, ← c11636] at e_180_10
  have e_181_3 := arithEq_of_rows h (row := 181) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_3
  simp only [← c11625, k3, ← c11626, k3, ← c11627] at e_181_3
  have e_181_4 := arithEq_of_rows h (row := 181) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_4
  simp only [← c11628, ← c11629, k3, ← c11630] at e_181_4
  have e_181_5 := arithEq_of_rows h (row := 181) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_5
  simp only [← c11637, ← c11638, k3, ← c11639] at e_181_5
  refine ⟨?_, ?_⟩
  · have hc := e_180_9
    simp only [e_181_4] at hc
    linear_combination c11640.trans k2 - hc
  · have hc := e_181_5
    simp only [e_181_3, e_180_10, e_181_4] at hc
    linear_combination c11641.trans k2 - hc

theorem publicBatchWrapper86_f1758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 75) = bor (a (.wire 10 59)) (a (.virt 17993)) := by
  have c11642 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11643 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11644 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11645 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11646 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11647 := (publicBatchWrapper86_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_18 := arithEq_of_rows h (row := 170) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_18
  simp only [← c11645, ← c11646, k3, ← c11647] at e_170_18
  have e_176_10 := arithEq_of_rows h (row := 176) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_10
  simp only [← c11642, ← c11643, ← c11644] at e_176_10
  have hr := e_170_18
  simp only [e_176_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 75) = a (.virt 201) := by
  have c11648 := (publicBatchWrapper86_copies364 a h).1
  exact c11648

theorem publicBatchWrapper86_f1760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1605)) (a (.wire 165 15)) (a (.virt 17995)) (a (.virt 17996)) := by
  have c11649 := (publicBatchWrapper86_copies364 a h).2.1
  have c11650 := (publicBatchWrapper86_copies364 a h).2.2.1
  have c11651 := (publicBatchWrapper86_copies364 a h).2.2.2.1
  have c11652 := (publicBatchWrapper86_copies364 a h).2.2.2.2.1
  have c11653 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.1
  have c11654 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.1
  have c11655 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.1
  have c11656 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.1
  have c11657 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.1
  have c11658 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11659 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11660 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11661 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11662 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11663 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11664 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11665 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_180_11 := arithEq_of_rows h (row := 180) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_11
  simp only [← c11655, ← c11656, ← c11657] at e_180_11
  have e_180_12 := arithEq_of_rows h (row := 180) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_12
  simp only [← c11658, ← c11659, ← c11660] at e_180_12
  have e_181_6 := arithEq_of_rows h (row := 181) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_6
  simp only [← c11649, k3, ← c11650, k3, ← c11651] at e_181_6
  have e_181_7 := arithEq_of_rows h (row := 181) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_7
  simp only [← c11652, ← c11653, k3, ← c11654] at e_181_7
  have e_181_8 := arithEq_of_rows h (row := 181) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_8
  simp only [← c11661, ← c11662, k3, ← c11663] at e_181_8
  refine ⟨?_, ?_⟩
  · have hc := e_180_11
    simp only [e_181_7] at hc
    linear_combination c11664.trans k2 - hc
  · have hc := e_181_8
    simp only [e_181_6, e_180_12, e_181_7] at hc
    linear_combination c11665.trans k2 - hc

theorem publicBatchWrapper86_f1761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1606)) (a (.wire 165 23)) (a (.virt 17997)) (a (.virt 17998)) := by
  have c11666 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11667 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11668 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11669 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11670 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11671 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11672 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11673 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11674 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11675 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11676 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11677 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11678 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11679 := (publicBatchWrapper86_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11680 := (publicBatchWrapper86_copies365 a h).1
  have c11681 := (publicBatchWrapper86_copies365 a h).2.1
  have c11682 := (publicBatchWrapper86_copies365 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_180_13 := arithEq_of_rows h (row := 180) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_13
  simp only [← c11672, ← c11673, ← c11674] at e_180_13
  have e_180_14 := arithEq_of_rows h (row := 180) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_14
  simp only [← c11675, ← c11676, ← c11677] at e_180_14
  have e_181_9 := arithEq_of_rows h (row := 181) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_9
  simp only [← c11666, k3, ← c11667, k3, ← c11668] at e_181_9
  have e_181_10 := arithEq_of_rows h (row := 181) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_10
  simp only [← c11669, ← c11670, k3, ← c11671] at e_181_10
  have e_181_11 := arithEq_of_rows h (row := 181) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_11
  simp only [← c11678, ← c11679, k3, ← c11680] at e_181_11
  refine ⟨?_, ?_⟩
  · have hc := e_180_13
    simp only [e_181_10] at hc
    linear_combination c11681.trans k2 - hc
  · have hc := e_181_11
    simp only [e_181_9, e_180_14, e_181_10] at hc
    linear_combination c11682.trans k2 - hc

theorem publicBatchWrapper86_f1762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1607)) (a (.wire 165 31)) (a (.virt 17999)) (a (.virt 18000)) := by
  have c11683 := (publicBatchWrapper86_copies365 a h).2.2.2.1
  have c11684 := (publicBatchWrapper86_copies365 a h).2.2.2.2.1
  have c11685 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.1
  have c11686 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.1
  have c11687 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.1
  have c11688 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.1
  have c11689 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.1
  have c11690 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11691 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11692 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11693 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11694 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11695 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11696 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11697 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11698 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11699 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_180_15 := arithEq_of_rows h (row := 180) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_15
  simp only [← c11689, ← c11690, ← c11691] at e_180_15
  have e_180_16 := arithEq_of_rows h (row := 180) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_16
  simp only [← c11692, ← c11693, ← c11694] at e_180_16
  have e_181_12 := arithEq_of_rows h (row := 181) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_12
  simp only [← c11683, k3, ← c11684, k3, ← c11685] at e_181_12
  have e_181_13 := arithEq_of_rows h (row := 181) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_13
  simp only [← c11686, ← c11687, k3, ← c11688] at e_181_13
  have e_181_14 := arithEq_of_rows h (row := 181) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_14
  simp only [← c11695, ← c11696, k3, ← c11697] at e_181_14
  refine ⟨?_, ?_⟩
  · have hc := e_180_15
    simp only [e_181_13] at hc
    linear_combination c11698.trans k2 - hc
  · have hc := e_181_14
    simp only [e_181_12, e_180_16, e_181_13] at hc
    linear_combination c11699.trans k2 - hc

theorem publicBatchWrapper86_f1763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1608)) (a (.wire 165 39)) (a (.virt 18001)) (a (.virt 18002)) := by
  have c11700 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11701 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11702 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11703 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11704 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11705 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11706 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11707 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11708 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11709 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11710 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11711 := (publicBatchWrapper86_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11712 := (publicBatchWrapper86_copies366 a h).1
  have c11713 := (publicBatchWrapper86_copies366 a h).2.1
  have c11714 := (publicBatchWrapper86_copies366 a h).2.2.1
  have c11715 := (publicBatchWrapper86_copies366 a h).2.2.2.1
  have c11716 := (publicBatchWrapper86_copies366 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_180_17 := arithEq_of_rows h (row := 180) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_17
  simp only [← c11706, ← c11707, ← c11708] at e_180_17
  have e_180_18 := arithEq_of_rows h (row := 180) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_18
  simp only [← c11709, ← c11710, ← c11711] at e_180_18
  have e_181_15 := arithEq_of_rows h (row := 181) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_15
  simp only [← c11700, k3, ← c11701, k3, ← c11702] at e_181_15
  have e_181_16 := arithEq_of_rows h (row := 181) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_16
  simp only [← c11703, ← c11704, k3, ← c11705] at e_181_16
  have e_181_17 := arithEq_of_rows h (row := 181) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_17
  simp only [← c11712, ← c11713, k3, ← c11714] at e_181_17
  refine ⟨?_, ?_⟩
  · have hc := e_180_17
    simp only [e_181_16] at hc
    linear_combination c11715.trans k2 - hc
  · have hc := e_181_17
    simp only [e_181_15, e_180_18, e_181_16] at hc
    linear_combination c11716.trans k2 - hc

theorem publicBatchWrapper86_f1764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 180 79) = band (a (.virt 17995)) (a (.virt 17997)) := by
  have c11717 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.1
  have c11718 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.1
  have c11719 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.1
  have e_180_19 := arithEq_of_rows h (row := 180) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_19
  simp only [← c11717, ← c11718, ← c11719] at e_180_19
  have hr := e_180_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 182 3) = band (a (.virt 17999)) (a (.virt 18001)) := by
  have c11720 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.1
  have c11721 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.1
  have c11722 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_182_0 := arithEq_of_rows h (row := 182) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_0
  simp only [← c11720, ← c11721, ← c11722] at e_182_0
  have hr := e_182_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 182 7) = band (a (.wire 180 79)) (a (.wire 182 3)) := by
  have c11723 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11724 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11725 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_182_1 := arithEq_of_rows h (row := 182) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_1
  simp only [← c11723, ← c11724, ← c11725] at e_182_1
  have hr := e_182_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 79) = bor (a (.wire 10 59)) (a (.wire 182 7)) := by
  have c11726 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11727 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11728 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11729 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11730 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11731 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_170_19 := arithEq_of_rows h (row := 170) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_19
  simp only [← c11729, ← c11730, k3, ← c11731] at e_170_19
  have e_176_11 := arithEq_of_rows h (row := 176) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_11
  simp only [← c11726, ← c11727, ← c11728] at e_176_11
  have hr := e_170_19
  simp only [e_176_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 170 79) = a (.virt 201) := by
  have c11732 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11732

theorem publicBatchWrapper86_f1769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1803)) (a (.wire 165 55)) (a (.virt 18003)) (a (.virt 18004)) := by
  have c11733 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11734 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11735 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11736 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11737 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11738 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11739 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11740 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11741 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11742 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11743 := (publicBatchWrapper86_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11744 := (publicBatchWrapper86_copies367 a h).1
  have c11745 := (publicBatchWrapper86_copies367 a h).2.1
  have c11746 := (publicBatchWrapper86_copies367 a h).2.2.1
  have c11747 := (publicBatchWrapper86_copies367 a h).2.2.2.1
  have c11748 := (publicBatchWrapper86_copies367 a h).2.2.2.2.1
  have c11749 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_181_18 := arithEq_of_rows h (row := 181) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_18
  simp only [← c11733, k3, ← c11734, k3, ← c11735] at e_181_18
  have e_181_19 := arithEq_of_rows h (row := 181) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_19
  simp only [← c11736, ← c11737, k3, ← c11738] at e_181_19
  have e_182_2 := arithEq_of_rows h (row := 182) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_2
  simp only [← c11739, ← c11740, ← c11741] at e_182_2
  have e_182_3 := arithEq_of_rows h (row := 182) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_3
  simp only [← c11742, ← c11743, ← c11744] at e_182_3
  have e_183_0 := arithEq_of_rows h (row := 183) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_0
  simp only [← c11745, ← c11746, k3, ← c11747] at e_183_0
  refine ⟨?_, ?_⟩
  · have hc := e_182_2
    simp only [e_181_19] at hc
    linear_combination c11748.trans k2 - hc
  · have hc := e_183_0
    simp only [e_181_18, e_182_3, e_181_19] at hc
    linear_combination c11749.trans k2 - hc

theorem publicBatchWrapper86_f1770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 3) = bor (a (.wire 11 23)) (a (.virt 18003)) := by
  have c11750 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.1
  have c11751 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.1
  have c11752 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.1
  have c11753 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.1
  have c11754 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11755 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_176_12 := arithEq_of_rows h (row := 176) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_12
  simp only [← c11750, ← c11751, ← c11752] at e_176_12
  have e_184_0 := arithEq_of_rows h (row := 184) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_0
  simp only [← c11753, ← c11754, k3, ← c11755] at e_184_0
  have hr := e_184_0
  simp only [e_176_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 3) = a (.virt 201) := by
  have c11756 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11756

theorem publicBatchWrapper86_f1772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1804)) (a (.wire 165 63)) (a (.virt 18005)) (a (.virt 18006)) := by
  have c11757 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11758 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11759 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11760 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11761 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11762 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11763 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11764 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11765 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11766 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11767 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11768 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11769 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11770 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11771 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11772 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11773 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_182_4 := arithEq_of_rows h (row := 182) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_4
  simp only [← c11763, ← c11764, ← c11765] at e_182_4
  have e_182_5 := arithEq_of_rows h (row := 182) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_5
  simp only [← c11766, ← c11767, ← c11768] at e_182_5
  have e_183_1 := arithEq_of_rows h (row := 183) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_1
  simp only [← c11757, k3, ← c11758, k3, ← c11759] at e_183_1
  have e_183_2 := arithEq_of_rows h (row := 183) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_2
  simp only [← c11760, ← c11761, k3, ← c11762] at e_183_2
  have e_183_3 := arithEq_of_rows h (row := 183) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_3
  simp only [← c11769, ← c11770, k3, ← c11771] at e_183_3
  refine ⟨?_, ?_⟩
  · have hc := e_182_4
    simp only [e_183_2] at hc
    linear_combination c11772.trans k2 - hc
  · have hc := e_183_3
    simp only [e_183_1, e_182_5, e_183_2] at hc
    linear_combination c11773.trans k2 - hc

theorem publicBatchWrapper86_f1773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 7) = bor (a (.wire 11 23)) (a (.virt 18005)) := by
  have c11774 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11775 := (publicBatchWrapper86_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11776 := (publicBatchWrapper86_copies368 a h).1
  have c11777 := (publicBatchWrapper86_copies368 a h).2.1
  have c11778 := (publicBatchWrapper86_copies368 a h).2.2.1
  have c11779 := (publicBatchWrapper86_copies368 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_176_13 := arithEq_of_rows h (row := 176) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_13
  simp only [← c11774, ← c11775, ← c11776] at e_176_13
  have e_184_1 := arithEq_of_rows h (row := 184) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_1
  simp only [← c11777, ← c11778, k3, ← c11779] at e_184_1
  have hr := e_184_1
  simp only [e_176_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 7) = a (.virt 201) := by
  have c11780 := (publicBatchWrapper86_copies368 a h).2.2.2.2.1
  exact c11780

theorem publicBatchWrapper86_f1775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1805)) (a (.wire 165 15)) (a (.virt 18007)) (a (.virt 18008)) := by
  have c11781 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.1
  have c11782 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.1
  have c11783 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.1
  have c11784 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.1
  have c11785 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.1
  have c11786 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11787 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11788 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11789 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11790 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11791 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11792 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11793 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11794 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11795 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11796 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11797 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_182_6 := arithEq_of_rows h (row := 182) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_6
  simp only [← c11787, ← c11788, ← c11789] at e_182_6
  have e_182_7 := arithEq_of_rows h (row := 182) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_7
  simp only [← c11790, ← c11791, ← c11792] at e_182_7
  have e_183_4 := arithEq_of_rows h (row := 183) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_4
  simp only [← c11781, k3, ← c11782, k3, ← c11783] at e_183_4
  have e_183_5 := arithEq_of_rows h (row := 183) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_5
  simp only [← c11784, ← c11785, k3, ← c11786] at e_183_5
  have e_183_6 := arithEq_of_rows h (row := 183) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_6
  simp only [← c11793, ← c11794, k3, ← c11795] at e_183_6
  refine ⟨?_, ?_⟩
  · have hc := e_182_6
    simp only [e_183_5] at hc
    linear_combination c11796.trans k2 - hc
  · have hc := e_183_6
    simp only [e_183_4, e_182_7, e_183_5] at hc
    linear_combination c11797.trans k2 - hc

theorem publicBatchWrapper86_f1776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1806)) (a (.wire 165 23)) (a (.virt 18009)) (a (.virt 18010)) := by
  have c11798 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11799 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11800 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11801 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11802 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11803 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11804 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11805 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11806 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11807 := (publicBatchWrapper86_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11808 := (publicBatchWrapper86_copies369 a h).1
  have c11809 := (publicBatchWrapper86_copies369 a h).2.1
  have c11810 := (publicBatchWrapper86_copies369 a h).2.2.1
  have c11811 := (publicBatchWrapper86_copies369 a h).2.2.2.1
  have c11812 := (publicBatchWrapper86_copies369 a h).2.2.2.2.1
  have c11813 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.1
  have c11814 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_182_8 := arithEq_of_rows h (row := 182) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_8
  simp only [← c11804, ← c11805, ← c11806] at e_182_8
  have e_182_9 := arithEq_of_rows h (row := 182) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_9
  simp only [← c11807, ← c11808, ← c11809] at e_182_9
  have e_183_7 := arithEq_of_rows h (row := 183) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_7
  simp only [← c11798, k3, ← c11799, k3, ← c11800] at e_183_7
  have e_183_8 := arithEq_of_rows h (row := 183) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_8
  simp only [← c11801, ← c11802, k3, ← c11803] at e_183_8
  have e_183_9 := arithEq_of_rows h (row := 183) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_9
  simp only [← c11810, ← c11811, k3, ← c11812] at e_183_9
  refine ⟨?_, ?_⟩
  · have hc := e_182_8
    simp only [e_183_8] at hc
    linear_combination c11813.trans k2 - hc
  · have hc := e_183_9
    simp only [e_183_7, e_182_9, e_183_8] at hc
    linear_combination c11814.trans k2 - hc

theorem publicBatchWrapper86_f1777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1807)) (a (.wire 165 31)) (a (.virt 18011)) (a (.virt 18012)) := by
  have c11815 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.1
  have c11816 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.1
  have c11817 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.1
  have c11818 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11819 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11820 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11821 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11822 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11823 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11824 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11825 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11826 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11827 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11828 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11829 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11830 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11831 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_182_10 := arithEq_of_rows h (row := 182) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_10
  simp only [← c11821, ← c11822, ← c11823] at e_182_10
  have e_182_11 := arithEq_of_rows h (row := 182) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_11
  simp only [← c11824, ← c11825, ← c11826] at e_182_11
  have e_183_10 := arithEq_of_rows h (row := 183) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_10
  simp only [← c11815, k3, ← c11816, k3, ← c11817] at e_183_10
  have e_183_11 := arithEq_of_rows h (row := 183) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_11
  simp only [← c11818, ← c11819, k3, ← c11820] at e_183_11
  have e_183_12 := arithEq_of_rows h (row := 183) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_12
  simp only [← c11827, ← c11828, k3, ← c11829] at e_183_12
  refine ⟨?_, ?_⟩
  · have hc := e_182_10
    simp only [e_183_11] at hc
    linear_combination c11830.trans k2 - hc
  · have hc := e_183_12
    simp only [e_183_10, e_182_11, e_183_11] at hc
    linear_combination c11831.trans k2 - hc

theorem publicBatchWrapper86_f1778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1808)) (a (.wire 165 39)) (a (.virt 18013)) (a (.virt 18014)) := by
  have c11832 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11833 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11834 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11835 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11836 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11837 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11838 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11839 := (publicBatchWrapper86_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11840 := (publicBatchWrapper86_copies370 a h).1
  have c11841 := (publicBatchWrapper86_copies370 a h).2.1
  have c11842 := (publicBatchWrapper86_copies370 a h).2.2.1
  have c11843 := (publicBatchWrapper86_copies370 a h).2.2.2.1
  have c11844 := (publicBatchWrapper86_copies370 a h).2.2.2.2.1
  have c11845 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.1
  have c11846 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.1
  have c11847 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.1
  have c11848 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_182_12 := arithEq_of_rows h (row := 182) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_12
  simp only [← c11838, ← c11839, ← c11840] at e_182_12
  have e_182_13 := arithEq_of_rows h (row := 182) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_13
  simp only [← c11841, ← c11842, ← c11843] at e_182_13
  have e_183_13 := arithEq_of_rows h (row := 183) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_13
  simp only [← c11832, k3, ← c11833, k3, ← c11834] at e_183_13
  have e_183_14 := arithEq_of_rows h (row := 183) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_14
  simp only [← c11835, ← c11836, k3, ← c11837] at e_183_14
  have e_183_15 := arithEq_of_rows h (row := 183) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_15
  simp only [← c11844, ← c11845, k3, ← c11846] at e_183_15
  refine ⟨?_, ?_⟩
  · have hc := e_182_12
    simp only [e_183_14] at hc
    linear_combination c11847.trans k2 - hc
  · have hc := e_183_15
    simp only [e_183_13, e_182_13, e_183_14] at hc
    linear_combination c11848.trans k2 - hc

theorem publicBatchWrapper86_f1779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 182 59) = band (a (.virt 18007)) (a (.virt 18009)) := by
  have c11849 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.1
  have c11850 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11851 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_182_14 := arithEq_of_rows h (row := 182) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_14
  simp only [← c11849, ← c11850, ← c11851] at e_182_14
  have hr := e_182_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 182 63) = band (a (.virt 18011)) (a (.virt 18013)) := by
  have c11852 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11853 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11854 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_182_15 := arithEq_of_rows h (row := 182) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_15
  simp only [← c11852, ← c11853, ← c11854] at e_182_15
  have hr := e_182_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 182 67) = band (a (.wire 182 59)) (a (.wire 182 63)) := by
  have c11855 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11856 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11857 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_182_16 := arithEq_of_rows h (row := 182) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_16
  simp only [← c11855, ← c11856, ← c11857] at e_182_16
  have hr := e_182_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 11) = bor (a (.wire 11 23)) (a (.wire 182 67)) := by
  have c11858 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11859 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11860 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11861 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11862 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11863 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_176_14 := arithEq_of_rows h (row := 176) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_14
  simp only [← c11858, ← c11859, ← c11860] at e_176_14
  have e_184_2 := arithEq_of_rows h (row := 184) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_2
  simp only [← c11861, ← c11862, k3, ← c11863] at e_184_2
  have hr := e_184_2
  simp only [e_176_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 11) = a (.virt 201) := by
  have c11864 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11864

theorem publicBatchWrapper86_f1784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2003)) (a (.wire 165 55)) (a (.virt 18015)) (a (.virt 18016)) := by
  have c11865 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11866 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11867 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11868 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11869 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11870 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11871 := (publicBatchWrapper86_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11872 := (publicBatchWrapper86_copies371 a h).1
  have c11873 := (publicBatchWrapper86_copies371 a h).2.1
  have c11874 := (publicBatchWrapper86_copies371 a h).2.2.1
  have c11875 := (publicBatchWrapper86_copies371 a h).2.2.2.1
  have c11876 := (publicBatchWrapper86_copies371 a h).2.2.2.2.1
  have c11877 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.1
  have c11878 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.1
  have c11879 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.1
  have c11880 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.1
  have c11881 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_182_17 := arithEq_of_rows h (row := 182) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_17
  simp only [← c11871, ← c11872, ← c11873] at e_182_17
  have e_182_18 := arithEq_of_rows h (row := 182) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_18
  simp only [← c11874, ← c11875, ← c11876] at e_182_18
  have e_183_16 := arithEq_of_rows h (row := 183) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_16
  simp only [← c11865, k3, ← c11866, k3, ← c11867] at e_183_16
  have e_183_17 := arithEq_of_rows h (row := 183) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_17
  simp only [← c11868, ← c11869, k3, ← c11870] at e_183_17
  have e_183_18 := arithEq_of_rows h (row := 183) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_18
  simp only [← c11877, ← c11878, k3, ← c11879] at e_183_18
  refine ⟨?_, ?_⟩
  · have hc := e_182_17
    simp only [e_183_17] at hc
    linear_combination c11880.trans k2 - hc
  · have hc := e_183_18
    simp only [e_183_16, e_182_18, e_183_17] at hc
    linear_combination c11881.trans k2 - hc

theorem publicBatchWrapper86_f1785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 15) = bor (a (.wire 11 67)) (a (.virt 18015)) := by
  have c11882 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11883 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11884 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11885 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11886 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11887 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_176_15 := arithEq_of_rows h (row := 176) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_15
  simp only [← c11882, ← c11883, ← c11884] at e_176_15
  have e_184_3 := arithEq_of_rows h (row := 184) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_3
  simp only [← c11885, ← c11886, k3, ← c11887] at e_184_3
  have hr := e_184_3
  simp only [e_176_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 15) = a (.virt 201) := by
  have c11888 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11888

theorem publicBatchWrapper86_f1787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2004)) (a (.wire 165 63)) (a (.virt 18017)) (a (.virt 18018)) := by
  have c11889 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11890 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11891 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11892 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11893 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11894 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11895 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11896 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11897 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11898 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11899 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11900 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11901 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11902 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11903 := (publicBatchWrapper86_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11904 := (publicBatchWrapper86_copies372 a h).1
  have c11905 := (publicBatchWrapper86_copies372 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_182_19 := arithEq_of_rows h (row := 182) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_19
  simp only [← c11895, ← c11896, ← c11897] at e_182_19
  have e_183_19 := arithEq_of_rows h (row := 183) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_19
  simp only [← c11889, k3, ← c11890, k3, ← c11891] at e_183_19
  have e_185_0 := arithEq_of_rows h (row := 185) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_0
  simp only [← c11892, ← c11893, k3, ← c11894] at e_185_0
  have e_185_1 := arithEq_of_rows h (row := 185) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_1
  simp only [← c11901, ← c11902, k3, ← c11903] at e_185_1
  have e_186_0 := arithEq_of_rows h (row := 186) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_0
  simp only [← c11898, ← c11899, ← c11900] at e_186_0
  refine ⟨?_, ?_⟩
  · have hc := e_182_19
    simp only [e_185_0] at hc
    linear_combination c11904.trans k2 - hc
  · have hc := e_185_1
    simp only [e_183_19, e_186_0, e_185_0] at hc
    linear_combination c11905.trans k2 - hc

theorem publicBatchWrapper86_f1788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 19) = bor (a (.wire 11 67)) (a (.virt 18017)) := by
  have c11906 := (publicBatchWrapper86_copies372 a h).2.2.1
  have c11907 := (publicBatchWrapper86_copies372 a h).2.2.2.1
  have c11908 := (publicBatchWrapper86_copies372 a h).2.2.2.2.1
  have c11909 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.1
  have c11910 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.1
  have c11911 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_176_16 := arithEq_of_rows h (row := 176) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_16
  simp only [← c11906, ← c11907, ← c11908] at e_176_16
  have e_184_4 := arithEq_of_rows h (row := 184) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_4
  simp only [← c11909, ← c11910, k3, ← c11911] at e_184_4
  have hr := e_184_4
  simp only [e_176_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 19) = a (.virt 201) := by
  have c11912 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.1
  exact c11912

theorem publicBatchWrapper86_f1790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2005)) (a (.wire 165 15)) (a (.virt 18019)) (a (.virt 18020)) := by
  have c11913 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.1
  have c11914 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11915 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11916 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11917 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11918 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11919 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11920 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11921 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11922 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11923 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11924 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11925 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11926 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11927 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11928 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11929 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_185_2 := arithEq_of_rows h (row := 185) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_2
  simp only [← c11913, k3, ← c11914, k3, ← c11915] at e_185_2
  have e_185_3 := arithEq_of_rows h (row := 185) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_3
  simp only [← c11916, ← c11917, k3, ← c11918] at e_185_3
  have e_185_4 := arithEq_of_rows h (row := 185) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_4
  simp only [← c11925, ← c11926, k3, ← c11927] at e_185_4
  have e_186_1 := arithEq_of_rows h (row := 186) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_1
  simp only [← c11919, ← c11920, ← c11921] at e_186_1
  have e_186_2 := arithEq_of_rows h (row := 186) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_2
  simp only [← c11922, ← c11923, ← c11924] at e_186_2
  refine ⟨?_, ?_⟩
  · have hc := e_186_1
    simp only [e_185_3] at hc
    linear_combination c11928.trans k2 - hc
  · have hc := e_185_4
    simp only [e_185_2, e_186_2, e_185_3] at hc
    linear_combination c11929.trans k2 - hc

theorem publicBatchWrapper86_f1791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2006)) (a (.wire 165 23)) (a (.virt 18021)) (a (.virt 18022)) := by
  have c11930 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11931 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11932 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11933 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11934 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11935 := (publicBatchWrapper86_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11936 := (publicBatchWrapper86_copies373 a h).1
  have c11937 := (publicBatchWrapper86_copies373 a h).2.1
  have c11938 := (publicBatchWrapper86_copies373 a h).2.2.1
  have c11939 := (publicBatchWrapper86_copies373 a h).2.2.2.1
  have c11940 := (publicBatchWrapper86_copies373 a h).2.2.2.2.1
  have c11941 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.1
  have c11942 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.1
  have c11943 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.1
  have c11944 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.1
  have c11945 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.1
  have c11946 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_185_5 := arithEq_of_rows h (row := 185) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_5
  simp only [← c11930, k3, ← c11931, k3, ← c11932] at e_185_5
  have e_185_6 := arithEq_of_rows h (row := 185) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_6
  simp only [← c11933, ← c11934, k3, ← c11935] at e_185_6
  have e_185_7 := arithEq_of_rows h (row := 185) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_7
  simp only [← c11942, ← c11943, k3, ← c11944] at e_185_7
  have e_186_3 := arithEq_of_rows h (row := 186) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_3
  simp only [← c11936, ← c11937, ← c11938] at e_186_3
  have e_186_4 := arithEq_of_rows h (row := 186) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_4
  simp only [← c11939, ← c11940, ← c11941] at e_186_4
  refine ⟨?_, ?_⟩
  · have hc := e_186_3
    simp only [e_185_6] at hc
    linear_combination c11945.trans k2 - hc
  · have hc := e_185_7
    simp only [e_185_5, e_186_4, e_185_6] at hc
    linear_combination c11946.trans k2 - hc

theorem publicBatchWrapper86_f1792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2007)) (a (.wire 165 31)) (a (.virt 18023)) (a (.virt 18024)) := by
  have c11947 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11948 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11949 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11950 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11951 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11952 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11953 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11954 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11955 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11956 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11957 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11958 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11959 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11960 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11961 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11962 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11963 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_185_8 := arithEq_of_rows h (row := 185) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_8
  simp only [← c11947, k3, ← c11948, k3, ← c11949] at e_185_8
  have e_185_9 := arithEq_of_rows h (row := 185) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_9
  simp only [← c11950, ← c11951, k3, ← c11952] at e_185_9
  have e_185_10 := arithEq_of_rows h (row := 185) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_10
  simp only [← c11959, ← c11960, k3, ← c11961] at e_185_10
  have e_186_5 := arithEq_of_rows h (row := 186) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_5
  simp only [← c11953, ← c11954, ← c11955] at e_186_5
  have e_186_6 := arithEq_of_rows h (row := 186) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_6
  simp only [← c11956, ← c11957, ← c11958] at e_186_6
  refine ⟨?_, ?_⟩
  · have hc := e_186_5
    simp only [e_185_9] at hc
    linear_combination c11962.trans k2 - hc
  · have hc := e_185_10
    simp only [e_185_8, e_186_6, e_185_9] at hc
    linear_combination c11963.trans k2 - hc

theorem publicBatchWrapper86_f1793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2008)) (a (.wire 165 39)) (a (.virt 18025)) (a (.virt 18026)) := by
  have c11964 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11965 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11966 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11967 := (publicBatchWrapper86_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11968 := (publicBatchWrapper86_copies374 a h).1
  have c11969 := (publicBatchWrapper86_copies374 a h).2.1
  have c11970 := (publicBatchWrapper86_copies374 a h).2.2.1
  have c11971 := (publicBatchWrapper86_copies374 a h).2.2.2.1
  have c11972 := (publicBatchWrapper86_copies374 a h).2.2.2.2.1
  have c11973 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.1
  have c11974 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.1
  have c11975 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.1
  have c11976 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.1
  have c11977 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.1
  have c11978 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11979 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11980 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_185_11 := arithEq_of_rows h (row := 185) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_11
  simp only [← c11964, k3, ← c11965, k3, ← c11966] at e_185_11
  have e_185_12 := arithEq_of_rows h (row := 185) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_12
  simp only [← c11967, ← c11968, k3, ← c11969] at e_185_12
  have e_185_13 := arithEq_of_rows h (row := 185) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_13
  simp only [← c11976, ← c11977, k3, ← c11978] at e_185_13
  have e_186_7 := arithEq_of_rows h (row := 186) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_7
  simp only [← c11970, ← c11971, ← c11972] at e_186_7
  have e_186_8 := arithEq_of_rows h (row := 186) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_8
  simp only [← c11973, ← c11974, ← c11975] at e_186_8
  refine ⟨?_, ?_⟩
  · have hc := e_186_7
    simp only [e_185_12] at hc
    linear_combination c11979.trans k2 - hc
  · have hc := e_185_13
    simp only [e_185_11, e_186_8, e_185_12] at hc
    linear_combination c11980.trans k2 - hc

theorem publicBatchWrapper86_f1794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 186 39) = band (a (.virt 18019)) (a (.virt 18021)) := by
  have c11981 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11982 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11983 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_186_9 := arithEq_of_rows h (row := 186) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_9
  simp only [← c11981, ← c11982, ← c11983] at e_186_9
  have hr := e_186_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 186 43) = band (a (.virt 18023)) (a (.virt 18025)) := by
  have c11984 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11985 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11986 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_186_10 := arithEq_of_rows h (row := 186) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_10
  simp only [← c11984, ← c11985, ← c11986] at e_186_10
  have hr := e_186_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 186 47) = band (a (.wire 186 39)) (a (.wire 186 43)) := by
  have c11987 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11988 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11989 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_186_11 := arithEq_of_rows h (row := 186) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_11
  simp only [← c11987, ← c11988, ← c11989] at e_186_11
  have hr := e_186_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 23) = bor (a (.wire 11 67)) (a (.wire 186 47)) := by
  have c11990 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11991 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11992 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11993 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11994 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11995 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_176_17 := arithEq_of_rows h (row := 176) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_17
  simp only [← c11990, ← c11991, ← c11992] at e_176_17
  have e_184_5 := arithEq_of_rows h (row := 184) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_5
  simp only [← c11993, ← c11994, k3, ← c11995] at e_184_5
  have hr := e_184_5
  simp only [e_176_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 23) = a (.virt 201) := by
  have c11996 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11996

theorem publicBatchWrapper86_f1799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2203)) (a (.wire 165 55)) (a (.virt 18027)) (a (.virt 18028)) := by
  have c11997 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11998 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11999 := (publicBatchWrapper86_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12000 := (publicBatchWrapper86_copies375 a h).1
  have c12001 := (publicBatchWrapper86_copies375 a h).2.1
  have c12002 := (publicBatchWrapper86_copies375 a h).2.2.1
  have c12003 := (publicBatchWrapper86_copies375 a h).2.2.2.1
  have c12004 := (publicBatchWrapper86_copies375 a h).2.2.2.2.1
  have c12005 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.1
  have c12006 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.1
  have c12007 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.1
  have c12008 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.1
  have c12009 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.1
  have c12010 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12011 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12012 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12013 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_185_14 := arithEq_of_rows h (row := 185) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_14
  simp only [← c11997, k3, ← c11998, k3, ← c11999] at e_185_14
  have e_185_15 := arithEq_of_rows h (row := 185) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_15
  simp only [← c12000, ← c12001, k3, ← c12002] at e_185_15
  have e_185_16 := arithEq_of_rows h (row := 185) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_16
  simp only [← c12009, ← c12010, k3, ← c12011] at e_185_16
  have e_186_12 := arithEq_of_rows h (row := 186) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_12
  simp only [← c12003, ← c12004, ← c12005] at e_186_12
  have e_186_13 := arithEq_of_rows h (row := 186) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_13
  simp only [← c12006, ← c12007, ← c12008] at e_186_13
  refine ⟨?_, ?_⟩
  · have hc := e_186_12
    simp only [e_185_15] at hc
    linear_combination c12012.trans k2 - hc
  · have hc := e_185_16
    simp only [e_185_14, e_186_13, e_185_15] at hc
    linear_combination c12013.trans k2 - hc

theorem publicBatchWrapper86_f1800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 27) = bor (a (.wire 13 31)) (a (.virt 18027)) := by
  have c12014 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12015 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12016 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12017 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12018 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12019 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_176_18 := arithEq_of_rows h (row := 176) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_18
  simp only [← c12014, ← c12015, ← c12016] at e_176_18
  have e_184_6 := arithEq_of_rows h (row := 184) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_6
  simp only [← c12017, ← c12018, k3, ← c12019] at e_184_6
  have hr := e_184_6
  simp only [e_176_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 27) = a (.virt 201) := by
  have c12020 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12020

theorem publicBatchWrapper86_f1802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2204)) (a (.wire 165 63)) (a (.virt 18029)) (a (.virt 18030)) := by
  have c12021 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12022 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12023 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12024 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12025 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12026 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12027 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12028 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12029 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12030 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12031 := (publicBatchWrapper86_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12032 := (publicBatchWrapper86_copies376 a h).1
  have c12033 := (publicBatchWrapper86_copies376 a h).2.1
  have c12034 := (publicBatchWrapper86_copies376 a h).2.2.1
  have c12035 := (publicBatchWrapper86_copies376 a h).2.2.2.1
  have c12036 := (publicBatchWrapper86_copies376 a h).2.2.2.2.1
  have c12037 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_185_17 := arithEq_of_rows h (row := 185) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_17
  simp only [← c12021, k3, ← c12022, k3, ← c12023] at e_185_17
  have e_185_18 := arithEq_of_rows h (row := 185) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_18
  simp only [← c12024, ← c12025, k3, ← c12026] at e_185_18
  have e_185_19 := arithEq_of_rows h (row := 185) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_19
  simp only [← c12033, ← c12034, k3, ← c12035] at e_185_19
  have e_186_14 := arithEq_of_rows h (row := 186) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_14
  simp only [← c12027, ← c12028, ← c12029] at e_186_14
  have e_186_15 := arithEq_of_rows h (row := 186) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_15
  simp only [← c12030, ← c12031, ← c12032] at e_186_15
  refine ⟨?_, ?_⟩
  · have hc := e_186_14
    simp only [e_185_18] at hc
    linear_combination c12036.trans k2 - hc
  · have hc := e_185_19
    simp only [e_185_17, e_186_15, e_185_18] at hc
    linear_combination c12037.trans k2 - hc

theorem publicBatchWrapper86_f1803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 31) = bor (a (.wire 13 31)) (a (.virt 18029)) := by
  have c12038 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.1
  have c12039 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.1
  have c12040 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.1
  have c12041 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.1
  have c12042 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12043 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_176_19 := arithEq_of_rows h (row := 176) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_19
  simp only [← c12038, ← c12039, ← c12040] at e_176_19
  have e_184_7 := arithEq_of_rows h (row := 184) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_7
  simp only [← c12041, ← c12042, k3, ← c12043] at e_184_7
  have hr := e_184_7
  simp only [e_176_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 31) = a (.virt 201) := by
  have c12044 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12044

theorem publicBatchWrapper86_f1805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2205)) (a (.wire 165 15)) (a (.virt 18031)) (a (.virt 18032)) := by
  have c12045 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12046 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12047 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12048 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12049 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12050 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12051 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12052 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12053 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12054 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12055 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12056 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12057 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12058 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12059 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12060 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12061 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_186_16 := arithEq_of_rows h (row := 186) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_16
  simp only [← c12051, ← c12052, ← c12053] at e_186_16
  have e_186_17 := arithEq_of_rows h (row := 186) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_17
  simp only [← c12054, ← c12055, ← c12056] at e_186_17
  have e_187_0 := arithEq_of_rows h (row := 187) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_0
  simp only [← c12045, k3, ← c12046, k3, ← c12047] at e_187_0
  have e_187_1 := arithEq_of_rows h (row := 187) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_1
  simp only [← c12048, ← c12049, k3, ← c12050] at e_187_1
  have e_187_2 := arithEq_of_rows h (row := 187) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_2
  simp only [← c12057, ← c12058, k3, ← c12059] at e_187_2
  refine ⟨?_, ?_⟩
  · have hc := e_186_16
    simp only [e_187_1] at hc
    linear_combination c12060.trans k2 - hc
  · have hc := e_187_2
    simp only [e_187_0, e_186_17, e_187_1] at hc
    linear_combination c12061.trans k2 - hc

theorem publicBatchWrapper86_f1806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2206)) (a (.wire 165 23)) (a (.virt 18033)) (a (.virt 18034)) := by
  have c12062 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12063 := (publicBatchWrapper86_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12064 := (publicBatchWrapper86_copies377 a h).1
  have c12065 := (publicBatchWrapper86_copies377 a h).2.1
  have c12066 := (publicBatchWrapper86_copies377 a h).2.2.1
  have c12067 := (publicBatchWrapper86_copies377 a h).2.2.2.1
  have c12068 := (publicBatchWrapper86_copies377 a h).2.2.2.2.1
  have c12069 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.1
  have c12070 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.1
  have c12071 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.1
  have c12072 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.1
  have c12073 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.1
  have c12074 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12075 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12076 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12077 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12078 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_186_18 := arithEq_of_rows h (row := 186) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_18
  simp only [← c12068, ← c12069, ← c12070] at e_186_18
  have e_186_19 := arithEq_of_rows h (row := 186) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_19
  simp only [← c12071, ← c12072, ← c12073] at e_186_19
  have e_187_3 := arithEq_of_rows h (row := 187) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_3
  simp only [← c12062, k3, ← c12063, k3, ← c12064] at e_187_3
  have e_187_4 := arithEq_of_rows h (row := 187) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_4
  simp only [← c12065, ← c12066, k3, ← c12067] at e_187_4
  have e_187_5 := arithEq_of_rows h (row := 187) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_5
  simp only [← c12074, ← c12075, k3, ← c12076] at e_187_5
  refine ⟨?_, ?_⟩
  · have hc := e_186_18
    simp only [e_187_4] at hc
    linear_combination c12077.trans k2 - hc
  · have hc := e_187_5
    simp only [e_187_3, e_186_19, e_187_4] at hc
    linear_combination c12078.trans k2 - hc

theorem publicBatchWrapper86_f1807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2207)) (a (.wire 165 31)) (a (.virt 18035)) (a (.virt 18036)) := by
  have c12079 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12080 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12081 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12082 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12083 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12084 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12085 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12086 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12087 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12088 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12089 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12090 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12091 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12092 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12093 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12094 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12095 := (publicBatchWrapper86_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_187_6 := arithEq_of_rows h (row := 187) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_6
  simp only [← c12079, k3, ← c12080, k3, ← c12081] at e_187_6
  have e_187_7 := arithEq_of_rows h (row := 187) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_7
  simp only [← c12082, ← c12083, k3, ← c12084] at e_187_7
  have e_187_8 := arithEq_of_rows h (row := 187) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_8
  simp only [← c12091, ← c12092, k3, ← c12093] at e_187_8
  have e_188_0 := arithEq_of_rows h (row := 188) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_0
  simp only [← c12085, ← c12086, ← c12087] at e_188_0
  have e_188_1 := arithEq_of_rows h (row := 188) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_1
  simp only [← c12088, ← c12089, ← c12090] at e_188_1
  refine ⟨?_, ?_⟩
  · have hc := e_188_0
    simp only [e_187_7] at hc
    linear_combination c12094.trans k2 - hc
  · have hc := e_187_8
    simp only [e_187_6, e_188_1, e_187_7] at hc
    linear_combination c12095.trans k2 - hc

theorem publicBatchWrapper86_f1808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2208)) (a (.wire 165 39)) (a (.virt 18037)) (a (.virt 18038)) := by
  have c12096 := (publicBatchWrapper86_copies378 a h).1
  have c12097 := (publicBatchWrapper86_copies378 a h).2.1
  have c12098 := (publicBatchWrapper86_copies378 a h).2.2.1
  have c12099 := (publicBatchWrapper86_copies378 a h).2.2.2.1
  have c12100 := (publicBatchWrapper86_copies378 a h).2.2.2.2.1
  have c12101 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.1
  have c12102 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.1
  have c12103 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.1
  have c12104 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.1
  have c12105 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.1
  have c12106 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12107 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12108 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12109 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12110 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12111 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12112 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_187_9 := arithEq_of_rows h (row := 187) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_9
  simp only [← c12096, k3, ← c12097, k3, ← c12098] at e_187_9
  have e_187_10 := arithEq_of_rows h (row := 187) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_10
  simp only [← c12099, ← c12100, k3, ← c12101] at e_187_10
  have e_187_11 := arithEq_of_rows h (row := 187) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_11
  simp only [← c12108, ← c12109, k3, ← c12110] at e_187_11
  have e_188_2 := arithEq_of_rows h (row := 188) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_2
  simp only [← c12102, ← c12103, ← c12104] at e_188_2
  have e_188_3 := arithEq_of_rows h (row := 188) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_3
  simp only [← c12105, ← c12106, ← c12107] at e_188_3
  refine ⟨?_, ?_⟩
  · have hc := e_188_2
    simp only [e_187_10] at hc
    linear_combination c12111.trans k2 - hc
  · have hc := e_187_11
    simp only [e_187_9, e_188_3, e_187_10] at hc
    linear_combination c12112.trans k2 - hc

theorem publicBatchWrapper86_f1809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 188 19) = band (a (.virt 18031)) (a (.virt 18033)) := by
  have c12113 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12114 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12115 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_4 := arithEq_of_rows h (row := 188) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_4
  simp only [← c12113, ← c12114, ← c12115] at e_188_4
  have hr := e_188_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 188 23) = band (a (.virt 18035)) (a (.virt 18037)) := by
  have c12116 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12117 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12118 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_5 := arithEq_of_rows h (row := 188) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_5
  simp only [← c12116, ← c12117, ← c12118] at e_188_5
  have hr := e_188_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 188 27) = band (a (.wire 188 19)) (a (.wire 188 23)) := by
  have c12119 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12120 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12121 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_6 := arithEq_of_rows h (row := 188) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_6
  simp only [← c12119, ← c12120, ← c12121] at e_188_6
  have hr := e_188_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 35) = bor (a (.wire 13 31)) (a (.wire 188 27)) := by
  have c12122 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12123 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12124 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12125 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12126 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12127 := (publicBatchWrapper86_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_8 := arithEq_of_rows h (row := 184) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_8
  simp only [← c12125, ← c12126, k3, ← c12127] at e_184_8
  have e_189_0 := arithEq_of_rows h (row := 189) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_0
  simp only [← c12122, ← c12123, ← c12124] at e_189_0
  have hr := e_184_8
  simp only [e_189_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 35) = a (.virt 201) := by
  have c12128 := (publicBatchWrapper86_copies379 a h).1
  exact c12128

theorem publicBatchWrapper86_f1814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2403)) (a (.wire 165 55)) (a (.virt 18039)) (a (.virt 18040)) := by
  have c12129 := (publicBatchWrapper86_copies379 a h).2.1
  have c12130 := (publicBatchWrapper86_copies379 a h).2.2.1
  have c12131 := (publicBatchWrapper86_copies379 a h).2.2.2.1
  have c12132 := (publicBatchWrapper86_copies379 a h).2.2.2.2.1
  have c12133 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.1
  have c12134 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.1
  have c12135 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.1
  have c12136 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.1
  have c12137 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.1
  have c12138 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12139 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12140 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12141 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12142 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12143 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12144 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12145 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_187_12 := arithEq_of_rows h (row := 187) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_12
  simp only [← c12129, k3, ← c12130, k3, ← c12131] at e_187_12
  have e_187_13 := arithEq_of_rows h (row := 187) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_13
  simp only [← c12132, ← c12133, k3, ← c12134] at e_187_13
  have e_187_14 := arithEq_of_rows h (row := 187) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_14
  simp only [← c12141, ← c12142, k3, ← c12143] at e_187_14
  have e_188_7 := arithEq_of_rows h (row := 188) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_7
  simp only [← c12135, ← c12136, ← c12137] at e_188_7
  have e_188_8 := arithEq_of_rows h (row := 188) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_8
  simp only [← c12138, ← c12139, ← c12140] at e_188_8
  refine ⟨?_, ?_⟩
  · have hc := e_188_7
    simp only [e_187_13] at hc
    linear_combination c12144.trans k2 - hc
  · have hc := e_187_14
    simp only [e_187_12, e_188_8, e_187_13] at hc
    linear_combination c12145.trans k2 - hc

theorem publicBatchWrapper86_f1815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 39) = bor (a (.wire 13 75)) (a (.virt 18039)) := by
  have c12146 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12147 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12148 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12149 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12150 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12151 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_9 := arithEq_of_rows h (row := 184) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_9
  simp only [← c12149, ← c12150, k3, ← c12151] at e_184_9
  have e_189_1 := arithEq_of_rows h (row := 189) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_1
  simp only [← c12146, ← c12147, ← c12148] at e_189_1
  have hr := e_184_9
  simp only [e_189_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 39) = a (.virt 201) := by
  have c12152 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12152

theorem publicBatchWrapper86_f1817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2404)) (a (.wire 165 63)) (a (.virt 18041)) (a (.virt 18042)) := by
  have c12153 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12154 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12155 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12156 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12157 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12158 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12159 := (publicBatchWrapper86_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12160 := (publicBatchWrapper86_copies380 a h).1
  have c12161 := (publicBatchWrapper86_copies380 a h).2.1
  have c12162 := (publicBatchWrapper86_copies380 a h).2.2.1
  have c12163 := (publicBatchWrapper86_copies380 a h).2.2.2.1
  have c12164 := (publicBatchWrapper86_copies380 a h).2.2.2.2.1
  have c12165 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.1
  have c12166 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.1
  have c12167 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.1
  have c12168 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.1
  have c12169 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_187_15 := arithEq_of_rows h (row := 187) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_15
  simp only [← c12153, k3, ← c12154, k3, ← c12155] at e_187_15
  have e_187_16 := arithEq_of_rows h (row := 187) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_16
  simp only [← c12156, ← c12157, k3, ← c12158] at e_187_16
  have e_187_17 := arithEq_of_rows h (row := 187) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_17
  simp only [← c12165, ← c12166, k3, ← c12167] at e_187_17
  have e_188_9 := arithEq_of_rows h (row := 188) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_9
  simp only [← c12159, ← c12160, ← c12161] at e_188_9
  have e_188_10 := arithEq_of_rows h (row := 188) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_10
  simp only [← c12162, ← c12163, ← c12164] at e_188_10
  refine ⟨?_, ?_⟩
  · have hc := e_188_9
    simp only [e_187_16] at hc
    linear_combination c12168.trans k2 - hc
  · have hc := e_187_17
    simp only [e_187_15, e_188_10, e_187_16] at hc
    linear_combination c12169.trans k2 - hc

theorem publicBatchWrapper86_f1818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 43) = bor (a (.wire 13 75)) (a (.virt 18041)) := by
  have c12170 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12171 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12172 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12173 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12174 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12175 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_10 := arithEq_of_rows h (row := 184) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_10
  simp only [← c12173, ← c12174, k3, ← c12175] at e_184_10
  have e_189_2 := arithEq_of_rows h (row := 189) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_2
  simp only [← c12170, ← c12171, ← c12172] at e_189_2
  have hr := e_184_10
  simp only [e_189_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 43) = a (.virt 201) := by
  have c12176 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12176

theorem publicBatchWrapper86_f1820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2405)) (a (.wire 165 15)) (a (.virt 18043)) (a (.virt 18044)) := by
  have c12177 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12178 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12179 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12180 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12181 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12182 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12183 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12184 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12185 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12186 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12187 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12188 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12189 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12190 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12191 := (publicBatchWrapper86_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12192 := (publicBatchWrapper86_copies381 a h).1
  have c12193 := (publicBatchWrapper86_copies381 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_187_18 := arithEq_of_rows h (row := 187) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_18
  simp only [← c12177, k3, ← c12178, k3, ← c12179] at e_187_18
  have e_187_19 := arithEq_of_rows h (row := 187) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_19
  simp only [← c12180, ← c12181, k3, ← c12182] at e_187_19
  have e_188_11 := arithEq_of_rows h (row := 188) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_11
  simp only [← c12183, ← c12184, ← c12185] at e_188_11
  have e_188_12 := arithEq_of_rows h (row := 188) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_12
  simp only [← c12186, ← c12187, ← c12188] at e_188_12
  have e_190_0 := arithEq_of_rows h (row := 190) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_0
  simp only [← c12189, ← c12190, k3, ← c12191] at e_190_0
  refine ⟨?_, ?_⟩
  · have hc := e_188_11
    simp only [e_187_19] at hc
    linear_combination c12192.trans k2 - hc
  · have hc := e_190_0
    simp only [e_187_18, e_188_12, e_187_19] at hc
    linear_combination c12193.trans k2 - hc

theorem publicBatchWrapper86_f1821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2406)) (a (.wire 165 23)) (a (.virt 18045)) (a (.virt 18046)) := by
  have c12194 := (publicBatchWrapper86_copies381 a h).2.2.1
  have c12195 := (publicBatchWrapper86_copies381 a h).2.2.2.1
  have c12196 := (publicBatchWrapper86_copies381 a h).2.2.2.2.1
  have c12197 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.1
  have c12198 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.1
  have c12199 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.1
  have c12200 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.1
  have c12201 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.1
  have c12202 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12203 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12204 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12205 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12206 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12207 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12208 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12209 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12210 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_188_13 := arithEq_of_rows h (row := 188) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_13
  simp only [← c12200, ← c12201, ← c12202] at e_188_13
  have e_188_14 := arithEq_of_rows h (row := 188) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_14
  simp only [← c12203, ← c12204, ← c12205] at e_188_14
  have e_190_1 := arithEq_of_rows h (row := 190) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_1
  simp only [← c12194, k3, ← c12195, k3, ← c12196] at e_190_1
  have e_190_2 := arithEq_of_rows h (row := 190) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_2
  simp only [← c12197, ← c12198, k3, ← c12199] at e_190_2
  have e_190_3 := arithEq_of_rows h (row := 190) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_3
  simp only [← c12206, ← c12207, k3, ← c12208] at e_190_3
  refine ⟨?_, ?_⟩
  · have hc := e_188_13
    simp only [e_190_2] at hc
    linear_combination c12209.trans k2 - hc
  · have hc := e_190_3
    simp only [e_190_1, e_188_14, e_190_2] at hc
    linear_combination c12210.trans k2 - hc

theorem publicBatchWrapper86_f1822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2407)) (a (.wire 165 31)) (a (.virt 18047)) (a (.virt 18048)) := by
  have c12211 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12212 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12213 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12214 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12215 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12216 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12217 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12218 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12219 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12220 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12221 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12222 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12223 := (publicBatchWrapper86_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12224 := (publicBatchWrapper86_copies382 a h).1
  have c12225 := (publicBatchWrapper86_copies382 a h).2.1
  have c12226 := (publicBatchWrapper86_copies382 a h).2.2.1
  have c12227 := (publicBatchWrapper86_copies382 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_188_15 := arithEq_of_rows h (row := 188) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_15
  simp only [← c12217, ← c12218, ← c12219] at e_188_15
  have e_188_16 := arithEq_of_rows h (row := 188) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_16
  simp only [← c12220, ← c12221, ← c12222] at e_188_16
  have e_190_4 := arithEq_of_rows h (row := 190) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_4
  simp only [← c12211, k3, ← c12212, k3, ← c12213] at e_190_4
  have e_190_5 := arithEq_of_rows h (row := 190) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_5
  simp only [← c12214, ← c12215, k3, ← c12216] at e_190_5
  have e_190_6 := arithEq_of_rows h (row := 190) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_6
  simp only [← c12223, ← c12224, k3, ← c12225] at e_190_6
  refine ⟨?_, ?_⟩
  · have hc := e_188_15
    simp only [e_190_5] at hc
    linear_combination c12226.trans k2 - hc
  · have hc := e_190_6
    simp only [e_190_4, e_188_16, e_190_5] at hc
    linear_combination c12227.trans k2 - hc

theorem publicBatchWrapper86_f1823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2408)) (a (.wire 165 39)) (a (.virt 18049)) (a (.virt 18050)) := by
  have c12228 := (publicBatchWrapper86_copies382 a h).2.2.2.2.1
  have c12229 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.1
  have c12230 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.1
  have c12231 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.1
  have c12232 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.1
  have c12233 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.1
  have c12234 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12235 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12236 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12237 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12238 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12239 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12240 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12241 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12242 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12243 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12244 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_188_17 := arithEq_of_rows h (row := 188) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_17
  simp only [← c12234, ← c12235, ← c12236] at e_188_17
  have e_188_18 := arithEq_of_rows h (row := 188) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_18
  simp only [← c12237, ← c12238, ← c12239] at e_188_18
  have e_190_7 := arithEq_of_rows h (row := 190) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_7
  simp only [← c12228, k3, ← c12229, k3, ← c12230] at e_190_7
  have e_190_8 := arithEq_of_rows h (row := 190) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_8
  simp only [← c12231, ← c12232, k3, ← c12233] at e_190_8
  have e_190_9 := arithEq_of_rows h (row := 190) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_9
  simp only [← c12240, ← c12241, k3, ← c12242] at e_190_9
  refine ⟨?_, ?_⟩
  · have hc := e_188_17
    simp only [e_190_8] at hc
    linear_combination c12243.trans k2 - hc
  · have hc := e_190_9
    simp only [e_190_7, e_188_18, e_190_8] at hc
    linear_combination c12244.trans k2 - hc

theorem publicBatchWrapper86_f1824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 188 79) = band (a (.virt 18043)) (a (.virt 18045)) := by
  have c12245 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12246 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12247 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_19 := arithEq_of_rows h (row := 188) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_19
  simp only [← c12245, ← c12246, ← c12247] at e_188_19
  have hr := e_188_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 191 3) = band (a (.virt 18047)) (a (.virt 18049)) := by
  have c12248 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12249 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12250 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_191_0 := arithEq_of_rows h (row := 191) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_0
  simp only [← c12248, ← c12249, ← c12250] at e_191_0
  have hr := e_191_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 191 7) = band (a (.wire 188 79)) (a (.wire 191 3)) := by
  have c12251 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12252 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12253 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_191_1 := arithEq_of_rows h (row := 191) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_1
  simp only [← c12251, ← c12252, ← c12253] at e_191_1
  have hr := e_191_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 47) = bor (a (.wire 13 75)) (a (.wire 191 7)) := by
  have c12254 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12255 := (publicBatchWrapper86_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12256 := (publicBatchWrapper86_copies383 a h).1
  have c12257 := (publicBatchWrapper86_copies383 a h).2.1
  have c12258 := (publicBatchWrapper86_copies383 a h).2.2.1
  have c12259 := (publicBatchWrapper86_copies383 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_11 := arithEq_of_rows h (row := 184) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_11
  simp only [← c12257, ← c12258, k3, ← c12259] at e_184_11
  have e_189_3 := arithEq_of_rows h (row := 189) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_3
  simp only [← c12254, ← c12255, ← c12256] at e_189_3
  have hr := e_184_11
  simp only [e_189_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 47) = a (.virt 201) := by
  have c12260 := (publicBatchWrapper86_copies383 a h).2.2.2.2.1
  exact c12260

theorem publicBatchWrapper86_f1829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2603)) (a (.wire 165 55)) (a (.virt 18051)) (a (.virt 18052)) := by
  have c12261 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.1
  have c12262 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.1
  have c12263 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.1
  have c12264 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.1
  have c12265 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.1
  have c12266 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12267 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12268 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12269 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12270 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12271 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12272 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12273 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12274 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12275 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12276 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12277 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_190_10 := arithEq_of_rows h (row := 190) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_10
  simp only [← c12261, k3, ← c12262, k3, ← c12263] at e_190_10
  have e_190_11 := arithEq_of_rows h (row := 190) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_11
  simp only [← c12264, ← c12265, k3, ← c12266] at e_190_11
  have e_190_12 := arithEq_of_rows h (row := 190) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_12
  simp only [← c12273, ← c12274, k3, ← c12275] at e_190_12
  have e_191_2 := arithEq_of_rows h (row := 191) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_2
  simp only [← c12267, ← c12268, ← c12269] at e_191_2
  have e_191_3 := arithEq_of_rows h (row := 191) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_3
  simp only [← c12270, ← c12271, ← c12272] at e_191_3
  refine ⟨?_, ?_⟩
  · have hc := e_191_2
    simp only [e_190_11] at hc
    linear_combination c12276.trans k2 - hc
  · have hc := e_190_12
    simp only [e_190_10, e_191_3, e_190_11] at hc
    linear_combination c12277.trans k2 - hc

theorem publicBatchWrapper86_f1830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 51) = bor (a (.wire 15 39)) (a (.virt 18051)) := by
  have c12278 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12279 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12280 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12281 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12282 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12283 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_12 := arithEq_of_rows h (row := 184) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_12
  simp only [← c12281, ← c12282, k3, ← c12283] at e_184_12
  have e_189_4 := arithEq_of_rows h (row := 189) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_4
  simp only [← c12278, ← c12279, ← c12280] at e_189_4
  have hr := e_184_12
  simp only [e_189_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 51) = a (.virt 201) := by
  have c12284 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12284

theorem publicBatchWrapper86_f1832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2604)) (a (.wire 165 63)) (a (.virt 18053)) (a (.virt 18054)) := by
  have c12285 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12286 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12287 := (publicBatchWrapper86_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12288 := (publicBatchWrapper86_copies384 a h).1
  have c12289 := (publicBatchWrapper86_copies384 a h).2.1
  have c12290 := (publicBatchWrapper86_copies384 a h).2.2.1
  have c12291 := (publicBatchWrapper86_copies384 a h).2.2.2.1
  have c12292 := (publicBatchWrapper86_copies384 a h).2.2.2.2.1
  have c12293 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.1
  have c12294 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.1
  have c12295 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.1
  have c12296 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.1
  have c12297 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.1
  have c12298 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12299 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12300 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12301 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_190_13 := arithEq_of_rows h (row := 190) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_13
  simp only [← c12285, k3, ← c12286, k3, ← c12287] at e_190_13
  have e_190_14 := arithEq_of_rows h (row := 190) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_14
  simp only [← c12288, ← c12289, k3, ← c12290] at e_190_14
  have e_190_15 := arithEq_of_rows h (row := 190) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_15
  simp only [← c12297, ← c12298, k3, ← c12299] at e_190_15
  have e_191_4 := arithEq_of_rows h (row := 191) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_4
  simp only [← c12291, ← c12292, ← c12293] at e_191_4
  have e_191_5 := arithEq_of_rows h (row := 191) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_5
  simp only [← c12294, ← c12295, ← c12296] at e_191_5
  refine ⟨?_, ?_⟩
  · have hc := e_191_4
    simp only [e_190_14] at hc
    linear_combination c12300.trans k2 - hc
  · have hc := e_190_15
    simp only [e_190_13, e_191_5, e_190_14] at hc
    linear_combination c12301.trans k2 - hc

theorem publicBatchWrapper86_f1833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 55) = bor (a (.wire 15 39)) (a (.virt 18053)) := by
  have c12302 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12303 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12304 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12305 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12306 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12307 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_13 := arithEq_of_rows h (row := 184) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_13
  simp only [← c12305, ← c12306, k3, ← c12307] at e_184_13
  have e_189_5 := arithEq_of_rows h (row := 189) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_5
  simp only [← c12302, ← c12303, ← c12304] at e_189_5
  have hr := e_184_13
  simp only [e_189_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 55) = a (.virt 201) := by
  have c12308 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12308

theorem publicBatchWrapper86_f1835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2605)) (a (.wire 165 15)) (a (.virt 18055)) (a (.virt 18056)) := by
  have c12309 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12310 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12311 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12312 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12313 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12314 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12315 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12316 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12317 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12318 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12319 := (publicBatchWrapper86_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12320 := (publicBatchWrapper86_copies385 a h).1
  have c12321 := (publicBatchWrapper86_copies385 a h).2.1
  have c12322 := (publicBatchWrapper86_copies385 a h).2.2.1
  have c12323 := (publicBatchWrapper86_copies385 a h).2.2.2.1
  have c12324 := (publicBatchWrapper86_copies385 a h).2.2.2.2.1
  have c12325 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_190_16 := arithEq_of_rows h (row := 190) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_16
  simp only [← c12309, k3, ← c12310, k3, ← c12311] at e_190_16
  have e_190_17 := arithEq_of_rows h (row := 190) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_17
  simp only [← c12312, ← c12313, k3, ← c12314] at e_190_17
  have e_190_18 := arithEq_of_rows h (row := 190) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_18
  simp only [← c12321, ← c12322, k3, ← c12323] at e_190_18
  have e_191_6 := arithEq_of_rows h (row := 191) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_6
  simp only [← c12315, ← c12316, ← c12317] at e_191_6
  have e_191_7 := arithEq_of_rows h (row := 191) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_7
  simp only [← c12318, ← c12319, ← c12320] at e_191_7
  refine ⟨?_, ?_⟩
  · have hc := e_191_6
    simp only [e_190_17] at hc
    linear_combination c12324.trans k2 - hc
  · have hc := e_190_18
    simp only [e_190_16, e_191_7, e_190_17] at hc
    linear_combination c12325.trans k2 - hc

theorem publicBatchWrapper86_f1836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2606)) (a (.wire 165 23)) (a (.virt 18057)) (a (.virt 18058)) := by
  have c12326 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.1
  have c12327 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.1
  have c12328 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.1
  have c12329 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.1
  have c12330 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12331 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12332 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12333 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12334 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12335 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12336 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12337 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12338 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12339 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12340 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12341 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12342 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_190_19 := arithEq_of_rows h (row := 190) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_19
  simp only [← c12326, k3, ← c12327, k3, ← c12328] at e_190_19
  have e_191_8 := arithEq_of_rows h (row := 191) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_8
  simp only [← c12332, ← c12333, ← c12334] at e_191_8
  have e_191_9 := arithEq_of_rows h (row := 191) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_9
  simp only [← c12335, ← c12336, ← c12337] at e_191_9
  have e_192_0 := arithEq_of_rows h (row := 192) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_0
  simp only [← c12329, ← c12330, k3, ← c12331] at e_192_0
  have e_192_1 := arithEq_of_rows h (row := 192) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_1
  simp only [← c12338, ← c12339, k3, ← c12340] at e_192_1
  refine ⟨?_, ?_⟩
  · have hc := e_191_8
    simp only [e_192_0] at hc
    linear_combination c12341.trans k2 - hc
  · have hc := e_192_1
    simp only [e_190_19, e_191_9, e_192_0] at hc
    linear_combination c12342.trans k2 - hc

theorem publicBatchWrapper86_f1837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2607)) (a (.wire 165 31)) (a (.virt 18059)) (a (.virt 18060)) := by
  have c12343 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12344 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12345 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12346 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12347 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12348 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12349 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12350 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12351 := (publicBatchWrapper86_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12352 := (publicBatchWrapper86_copies386 a h).1
  have c12353 := (publicBatchWrapper86_copies386 a h).2.1
  have c12354 := (publicBatchWrapper86_copies386 a h).2.2.1
  have c12355 := (publicBatchWrapper86_copies386 a h).2.2.2.1
  have c12356 := (publicBatchWrapper86_copies386 a h).2.2.2.2.1
  have c12357 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.1
  have c12358 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.1
  have c12359 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_191_10 := arithEq_of_rows h (row := 191) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_10
  simp only [← c12349, ← c12350, ← c12351] at e_191_10
  have e_191_11 := arithEq_of_rows h (row := 191) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_11
  simp only [← c12352, ← c12353, ← c12354] at e_191_11
  have e_192_2 := arithEq_of_rows h (row := 192) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_2
  simp only [← c12343, k3, ← c12344, k3, ← c12345] at e_192_2
  have e_192_3 := arithEq_of_rows h (row := 192) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_3
  simp only [← c12346, ← c12347, k3, ← c12348] at e_192_3
  have e_192_4 := arithEq_of_rows h (row := 192) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_4
  simp only [← c12355, ← c12356, k3, ← c12357] at e_192_4
  refine ⟨?_, ?_⟩
  · have hc := e_191_10
    simp only [e_192_3] at hc
    linear_combination c12358.trans k2 - hc
  · have hc := e_192_4
    simp only [e_192_2, e_191_11, e_192_3] at hc
    linear_combination c12359.trans k2 - hc

theorem publicBatchWrapper86_f1838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2608)) (a (.wire 165 39)) (a (.virt 18061)) (a (.virt 18062)) := by
  have c12360 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.1
  have c12361 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.1
  have c12362 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12363 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12364 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12365 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12366 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12367 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12368 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12369 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12370 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12371 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12372 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12373 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12374 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12375 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12376 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_191_12 := arithEq_of_rows h (row := 191) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_12
  simp only [← c12366, ← c12367, ← c12368] at e_191_12
  have e_191_13 := arithEq_of_rows h (row := 191) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_13
  simp only [← c12369, ← c12370, ← c12371] at e_191_13
  have e_192_5 := arithEq_of_rows h (row := 192) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_5
  simp only [← c12360, k3, ← c12361, k3, ← c12362] at e_192_5
  have e_192_6 := arithEq_of_rows h (row := 192) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_6
  simp only [← c12363, ← c12364, k3, ← c12365] at e_192_6
  have e_192_7 := arithEq_of_rows h (row := 192) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_7
  simp only [← c12372, ← c12373, k3, ← c12374] at e_192_7
  refine ⟨?_, ?_⟩
  · have hc := e_191_12
    simp only [e_192_6] at hc
    linear_combination c12375.trans k2 - hc
  · have hc := e_192_7
    simp only [e_192_5, e_191_13, e_192_6] at hc
    linear_combination c12376.trans k2 - hc

theorem publicBatchWrapper86_f1839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 191 59) = band (a (.virt 18055)) (a (.virt 18057)) := by
  have c12377 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12378 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12379 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_191_14 := arithEq_of_rows h (row := 191) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_14
  simp only [← c12377, ← c12378, ← c12379] at e_191_14
  have hr := e_191_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 191 63) = band (a (.virt 18059)) (a (.virt 18061)) := by
  have c12380 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12381 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12382 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_191_15 := arithEq_of_rows h (row := 191) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_15
  simp only [← c12380, ← c12381, ← c12382] at e_191_15
  have hr := e_191_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 191 67) = band (a (.wire 191 59)) (a (.wire 191 63)) := by
  have c12383 := (publicBatchWrapper86_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12384 := (publicBatchWrapper86_copies387 a h).1
  have c12385 := (publicBatchWrapper86_copies387 a h).2.1
  have e_191_16 := arithEq_of_rows h (row := 191) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_16
  simp only [← c12383, ← c12384, ← c12385] at e_191_16
  have hr := e_191_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 59) = bor (a (.wire 15 39)) (a (.wire 191 67)) := by
  have c12386 := (publicBatchWrapper86_copies387 a h).2.2.1
  have c12387 := (publicBatchWrapper86_copies387 a h).2.2.2.1
  have c12388 := (publicBatchWrapper86_copies387 a h).2.2.2.2.1
  have c12389 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.1
  have c12390 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.1
  have c12391 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_14 := arithEq_of_rows h (row := 184) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_14
  simp only [← c12389, ← c12390, k3, ← c12391] at e_184_14
  have e_189_6 := arithEq_of_rows h (row := 189) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_6
  simp only [← c12386, ← c12387, ← c12388] at e_189_6
  have hr := e_184_14
  simp only [e_189_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 59) = a (.virt 201) := by
  have c12392 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.1
  exact c12392

theorem publicBatchWrapper86_f1844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2803)) (a (.wire 165 55)) (a (.virt 18063)) (a (.virt 18064)) := by
  have c12393 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.1
  have c12394 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12395 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12396 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12397 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12398 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12399 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12400 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12401 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12402 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12403 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12404 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12405 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12406 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12407 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12408 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12409 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_191_17 := arithEq_of_rows h (row := 191) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_17
  simp only [← c12399, ← c12400, ← c12401] at e_191_17
  have e_191_18 := arithEq_of_rows h (row := 191) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_18
  simp only [← c12402, ← c12403, ← c12404] at e_191_18
  have e_192_8 := arithEq_of_rows h (row := 192) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_8
  simp only [← c12393, k3, ← c12394, k3, ← c12395] at e_192_8
  have e_192_9 := arithEq_of_rows h (row := 192) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_9
  simp only [← c12396, ← c12397, k3, ← c12398] at e_192_9
  have e_192_10 := arithEq_of_rows h (row := 192) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_10
  simp only [← c12405, ← c12406, k3, ← c12407] at e_192_10
  refine ⟨?_, ?_⟩
  · have hc := e_191_17
    simp only [e_192_9] at hc
    linear_combination c12408.trans k2 - hc
  · have hc := e_192_10
    simp only [e_192_8, e_191_18, e_192_9] at hc
    linear_combination c12409.trans k2 - hc

theorem publicBatchWrapper86_f1845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 63) = bor (a (.wire 16 3)) (a (.virt 18063)) := by
  have c12410 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12411 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12412 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12413 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12414 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12415 := (publicBatchWrapper86_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_15 := arithEq_of_rows h (row := 184) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_15
  simp only [← c12413, ← c12414, k3, ← c12415] at e_184_15
  have e_189_7 := arithEq_of_rows h (row := 189) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_7
  simp only [← c12410, ← c12411, ← c12412] at e_189_7
  have hr := e_184_15
  simp only [e_189_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 63) = a (.virt 201) := by
  have c12416 := (publicBatchWrapper86_copies388 a h).1
  exact c12416

theorem publicBatchWrapper86_f1847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2804)) (a (.wire 165 63)) (a (.virt 18065)) (a (.virt 18066)) := by
  have c12417 := (publicBatchWrapper86_copies388 a h).2.1
  have c12418 := (publicBatchWrapper86_copies388 a h).2.2.1
  have c12419 := (publicBatchWrapper86_copies388 a h).2.2.2.1
  have c12420 := (publicBatchWrapper86_copies388 a h).2.2.2.2.1
  have c12421 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.1
  have c12422 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.1
  have c12423 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.1
  have c12424 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.1
  have c12425 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.1
  have c12426 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12427 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12428 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12429 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12430 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12431 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12432 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12433 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_191_19 := arithEq_of_rows h (row := 191) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_19
  simp only [← c12423, ← c12424, ← c12425] at e_191_19
  have e_192_11 := arithEq_of_rows h (row := 192) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_11
  simp only [← c12417, k3, ← c12418, k3, ← c12419] at e_192_11
  have e_192_12 := arithEq_of_rows h (row := 192) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_12
  simp only [← c12420, ← c12421, k3, ← c12422] at e_192_12
  have e_192_13 := arithEq_of_rows h (row := 192) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_13
  simp only [← c12429, ← c12430, k3, ← c12431] at e_192_13
  have e_193_0 := arithEq_of_rows h (row := 193) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_0
  simp only [← c12426, ← c12427, ← c12428] at e_193_0
  refine ⟨?_, ?_⟩
  · have hc := e_191_19
    simp only [e_192_12] at hc
    linear_combination c12432.trans k2 - hc
  · have hc := e_192_13
    simp only [e_192_11, e_193_0, e_192_12] at hc
    linear_combination c12433.trans k2 - hc

theorem publicBatchWrapper86_f1848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 67) = bor (a (.wire 16 3)) (a (.virt 18065)) := by
  have c12434 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12435 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12436 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12437 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12438 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12439 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_16 := arithEq_of_rows h (row := 184) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_16
  simp only [← c12437, ← c12438, k3, ← c12439] at e_184_16
  have e_189_8 := arithEq_of_rows h (row := 189) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_8
  simp only [← c12434, ← c12435, ← c12436] at e_189_8
  have hr := e_184_16
  simp only [e_189_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 67) = a (.virt 201) := by
  have c12440 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12440

theorem publicBatchWrapper86_f1850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2805)) (a (.wire 165 15)) (a (.virt 18067)) (a (.virt 18068)) := by
  have c12441 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12442 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12443 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12444 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12445 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12446 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12447 := (publicBatchWrapper86_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12448 := (publicBatchWrapper86_copies389 a h).1
  have c12449 := (publicBatchWrapper86_copies389 a h).2.1
  have c12450 := (publicBatchWrapper86_copies389 a h).2.2.1
  have c12451 := (publicBatchWrapper86_copies389 a h).2.2.2.1
  have c12452 := (publicBatchWrapper86_copies389 a h).2.2.2.2.1
  have c12453 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.1
  have c12454 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.1
  have c12455 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.1
  have c12456 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.1
  have c12457 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_192_14 := arithEq_of_rows h (row := 192) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_14
  simp only [← c12441, k3, ← c12442, k3, ← c12443] at e_192_14
  have e_192_15 := arithEq_of_rows h (row := 192) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_15
  simp only [← c12444, ← c12445, k3, ← c12446] at e_192_15
  have e_192_16 := arithEq_of_rows h (row := 192) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_16
  simp only [← c12453, ← c12454, k3, ← c12455] at e_192_16
  have e_193_1 := arithEq_of_rows h (row := 193) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_1
  simp only [← c12447, ← c12448, ← c12449] at e_193_1
  have e_193_2 := arithEq_of_rows h (row := 193) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_2
  simp only [← c12450, ← c12451, ← c12452] at e_193_2
  refine ⟨?_, ?_⟩
  · have hc := e_193_1
    simp only [e_192_15] at hc
    linear_combination c12456.trans k2 - hc
  · have hc := e_192_16
    simp only [e_192_14, e_193_2, e_192_15] at hc
    linear_combination c12457.trans k2 - hc

theorem publicBatchWrapper86_f1851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2806)) (a (.wire 165 23)) (a (.virt 18069)) (a (.virt 18070)) := by
  have c12458 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12459 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12460 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12461 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12462 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12463 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12464 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12465 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12466 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12467 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12468 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12469 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12470 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12471 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12472 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12473 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12474 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_192_17 := arithEq_of_rows h (row := 192) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_17
  simp only [← c12458, k3, ← c12459, k3, ← c12460] at e_192_17
  have e_192_18 := arithEq_of_rows h (row := 192) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_18
  simp only [← c12461, ← c12462, k3, ← c12463] at e_192_18
  have e_192_19 := arithEq_of_rows h (row := 192) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_19
  simp only [← c12470, ← c12471, k3, ← c12472] at e_192_19
  have e_193_3 := arithEq_of_rows h (row := 193) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_3
  simp only [← c12464, ← c12465, ← c12466] at e_193_3
  have e_193_4 := arithEq_of_rows h (row := 193) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_4
  simp only [← c12467, ← c12468, ← c12469] at e_193_4
  refine ⟨?_, ?_⟩
  · have hc := e_193_3
    simp only [e_192_18] at hc
    linear_combination c12473.trans k2 - hc
  · have hc := e_192_19
    simp only [e_192_17, e_193_4, e_192_18] at hc
    linear_combination c12474.trans k2 - hc

theorem publicBatchWrapper86_f1852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2807)) (a (.wire 165 31)) (a (.virt 18071)) (a (.virt 18072)) := by
  have c12475 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12476 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12477 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12478 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12479 := (publicBatchWrapper86_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12480 := (publicBatchWrapper86_copies390 a h).1
  have c12481 := (publicBatchWrapper86_copies390 a h).2.1
  have c12482 := (publicBatchWrapper86_copies390 a h).2.2.1
  have c12483 := (publicBatchWrapper86_copies390 a h).2.2.2.1
  have c12484 := (publicBatchWrapper86_copies390 a h).2.2.2.2.1
  have c12485 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.1
  have c12486 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.1
  have c12487 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.1
  have c12488 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.1
  have c12489 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.1
  have c12490 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12491 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_193_5 := arithEq_of_rows h (row := 193) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_5
  simp only [← c12481, ← c12482, ← c12483] at e_193_5
  have e_193_6 := arithEq_of_rows h (row := 193) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_6
  simp only [← c12484, ← c12485, ← c12486] at e_193_6
  have e_194_0 := arithEq_of_rows h (row := 194) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_0
  simp only [← c12475, k3, ← c12476, k3, ← c12477] at e_194_0
  have e_194_1 := arithEq_of_rows h (row := 194) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_1
  simp only [← c12478, ← c12479, k3, ← c12480] at e_194_1
  have e_194_2 := arithEq_of_rows h (row := 194) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_2
  simp only [← c12487, ← c12488, k3, ← c12489] at e_194_2
  refine ⟨?_, ?_⟩
  · have hc := e_193_5
    simp only [e_194_1] at hc
    linear_combination c12490.trans k2 - hc
  · have hc := e_194_2
    simp only [e_194_0, e_193_6, e_194_1] at hc
    linear_combination c12491.trans k2 - hc

theorem publicBatchWrapper86_f1853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2808)) (a (.wire 165 39)) (a (.virt 18073)) (a (.virt 18074)) := by
  have c12492 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12493 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12494 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12495 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12496 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12497 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12498 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12499 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12500 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12501 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12502 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12503 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12504 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12505 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12506 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12507 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12508 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_193_7 := arithEq_of_rows h (row := 193) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_7
  simp only [← c12498, ← c12499, ← c12500] at e_193_7
  have e_193_8 := arithEq_of_rows h (row := 193) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_8
  simp only [← c12501, ← c12502, ← c12503] at e_193_8
  have e_194_3 := arithEq_of_rows h (row := 194) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_3
  simp only [← c12492, k3, ← c12493, k3, ← c12494] at e_194_3
  have e_194_4 := arithEq_of_rows h (row := 194) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_4
  simp only [← c12495, ← c12496, k3, ← c12497] at e_194_4
  have e_194_5 := arithEq_of_rows h (row := 194) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_5
  simp only [← c12504, ← c12505, k3, ← c12506] at e_194_5
  refine ⟨?_, ?_⟩
  · have hc := e_193_7
    simp only [e_194_4] at hc
    linear_combination c12507.trans k2 - hc
  · have hc := e_194_5
    simp only [e_194_3, e_193_8, e_194_4] at hc
    linear_combination c12508.trans k2 - hc

theorem publicBatchWrapper86_f1854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 193 39) = band (a (.virt 18067)) (a (.virt 18069)) := by
  have c12509 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12510 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12511 := (publicBatchWrapper86_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_193_9 := arithEq_of_rows h (row := 193) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_9
  simp only [← c12509, ← c12510, ← c12511] at e_193_9
  have hr := e_193_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 193 43) = band (a (.virt 18071)) (a (.virt 18073)) := by
  have c12512 := (publicBatchWrapper86_copies391 a h).1
  have c12513 := (publicBatchWrapper86_copies391 a h).2.1
  have c12514 := (publicBatchWrapper86_copies391 a h).2.2.1
  have e_193_10 := arithEq_of_rows h (row := 193) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_10
  simp only [← c12512, ← c12513, ← c12514] at e_193_10
  have hr := e_193_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 193 47) = band (a (.wire 193 39)) (a (.wire 193 43)) := by
  have c12515 := (publicBatchWrapper86_copies391 a h).2.2.2.1
  have c12516 := (publicBatchWrapper86_copies391 a h).2.2.2.2.1
  have c12517 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.1
  have e_193_11 := arithEq_of_rows h (row := 193) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_11
  simp only [← c12515, ← c12516, ← c12517] at e_193_11
  have hr := e_193_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 71) = bor (a (.wire 16 3)) (a (.wire 193 47)) := by
  have c12518 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.1
  have c12519 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.1
  have c12520 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.1
  have c12521 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.1
  have c12522 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12523 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_17 := arithEq_of_rows h (row := 184) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_17
  simp only [← c12521, ← c12522, k3, ← c12523] at e_184_17
  have e_189_9 := arithEq_of_rows h (row := 189) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_9
  simp only [← c12518, ← c12519, ← c12520] at e_189_9
  have hr := e_184_17
  simp only [e_189_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 71) = a (.virt 201) := by
  have c12524 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12524

theorem publicBatchWrapper86_f1859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3003)) (a (.wire 165 55)) (a (.virt 18075)) (a (.virt 18076)) := by
  have c12525 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12526 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12527 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12528 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12529 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12530 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12531 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12532 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12533 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12534 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12535 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12536 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12537 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12538 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12539 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12540 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12541 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_193_12 := arithEq_of_rows h (row := 193) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_12
  simp only [← c12531, ← c12532, ← c12533] at e_193_12
  have e_193_13 := arithEq_of_rows h (row := 193) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_13
  simp only [← c12534, ← c12535, ← c12536] at e_193_13
  have e_194_6 := arithEq_of_rows h (row := 194) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_6
  simp only [← c12525, k3, ← c12526, k3, ← c12527] at e_194_6
  have e_194_7 := arithEq_of_rows h (row := 194) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_7
  simp only [← c12528, ← c12529, k3, ← c12530] at e_194_7
  have e_194_8 := arithEq_of_rows h (row := 194) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_8
  simp only [← c12537, ← c12538, k3, ← c12539] at e_194_8
  refine ⟨?_, ?_⟩
  · have hc := e_193_12
    simp only [e_194_7] at hc
    linear_combination c12540.trans k2 - hc
  · have hc := e_194_8
    simp only [e_194_6, e_193_13, e_194_7] at hc
    linear_combination c12541.trans k2 - hc

theorem publicBatchWrapper86_f1860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 75) = bor (a (.wire 16 47)) (a (.virt 18075)) := by
  have c12542 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12543 := (publicBatchWrapper86_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12544 := (publicBatchWrapper86_copies392 a h).1
  have c12545 := (publicBatchWrapper86_copies392 a h).2.1
  have c12546 := (publicBatchWrapper86_copies392 a h).2.2.1
  have c12547 := (publicBatchWrapper86_copies392 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_18 := arithEq_of_rows h (row := 184) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_18
  simp only [← c12545, ← c12546, k3, ← c12547] at e_184_18
  have e_189_10 := arithEq_of_rows h (row := 189) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_10
  simp only [← c12542, ← c12543, ← c12544] at e_189_10
  have hr := e_184_18
  simp only [e_189_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 75) = a (.virt 201) := by
  have c12548 := (publicBatchWrapper86_copies392 a h).2.2.2.2.1
  exact c12548

theorem publicBatchWrapper86_f1862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3004)) (a (.wire 165 63)) (a (.virt 18077)) (a (.virt 18078)) := by
  have c12549 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.1
  have c12550 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.1
  have c12551 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.1
  have c12552 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.1
  have c12553 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.1
  have c12554 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12555 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12556 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12557 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12558 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12559 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12560 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12561 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12562 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12563 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12564 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12565 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_193_14 := arithEq_of_rows h (row := 193) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_14
  simp only [← c12555, ← c12556, ← c12557] at e_193_14
  have e_193_15 := arithEq_of_rows h (row := 193) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_15
  simp only [← c12558, ← c12559, ← c12560] at e_193_15
  have e_194_9 := arithEq_of_rows h (row := 194) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_9
  simp only [← c12549, k3, ← c12550, k3, ← c12551] at e_194_9
  have e_194_10 := arithEq_of_rows h (row := 194) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_10
  simp only [← c12552, ← c12553, k3, ← c12554] at e_194_10
  have e_194_11 := arithEq_of_rows h (row := 194) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_11
  simp only [← c12561, ← c12562, k3, ← c12563] at e_194_11
  refine ⟨?_, ?_⟩
  · have hc := e_193_14
    simp only [e_194_10] at hc
    linear_combination c12564.trans k2 - hc
  · have hc := e_194_11
    simp only [e_194_9, e_193_15, e_194_10] at hc
    linear_combination c12565.trans k2 - hc

theorem publicBatchWrapper86_f1863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 79) = bor (a (.wire 16 47)) (a (.virt 18077)) := by
  have c12566 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12567 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12568 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12569 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12570 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12571 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_184_19 := arithEq_of_rows h (row := 184) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_19
  simp only [← c12569, ← c12570, k3, ← c12571] at e_184_19
  have e_189_11 := arithEq_of_rows h (row := 189) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_11
  simp only [← c12566, ← c12567, ← c12568] at e_189_11
  have hr := e_184_19
  simp only [e_189_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 184 79) = a (.virt 201) := by
  have c12572 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12572

theorem publicBatchWrapper86_f1865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3005)) (a (.wire 165 15)) (a (.virt 18079)) (a (.virt 18080)) := by
  have c12573 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12574 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12575 := (publicBatchWrapper86_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12576 := (publicBatchWrapper86_copies393 a h).1
  have c12577 := (publicBatchWrapper86_copies393 a h).2.1
  have c12578 := (publicBatchWrapper86_copies393 a h).2.2.1
  have c12579 := (publicBatchWrapper86_copies393 a h).2.2.2.1
  have c12580 := (publicBatchWrapper86_copies393 a h).2.2.2.2.1
  have c12581 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.1
  have c12582 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.1
  have c12583 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.1
  have c12584 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.1
  have c12585 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.1
  have c12586 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12587 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12588 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12589 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_193_16 := arithEq_of_rows h (row := 193) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_16
  simp only [← c12579, ← c12580, ← c12581] at e_193_16
  have e_193_17 := arithEq_of_rows h (row := 193) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_17
  simp only [← c12582, ← c12583, ← c12584] at e_193_17
  have e_194_12 := arithEq_of_rows h (row := 194) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_12
  simp only [← c12573, k3, ← c12574, k3, ← c12575] at e_194_12
  have e_194_13 := arithEq_of_rows h (row := 194) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_13
  simp only [← c12576, ← c12577, k3, ← c12578] at e_194_13
  have e_194_14 := arithEq_of_rows h (row := 194) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_14
  simp only [← c12585, ← c12586, k3, ← c12587] at e_194_14
  refine ⟨?_, ?_⟩
  · have hc := e_193_16
    simp only [e_194_13] at hc
    linear_combination c12588.trans k2 - hc
  · have hc := e_194_14
    simp only [e_194_12, e_193_17, e_194_13] at hc
    linear_combination c12589.trans k2 - hc

theorem publicBatchWrapper86_f1866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3006)) (a (.wire 165 23)) (a (.virt 18081)) (a (.virt 18082)) := by
  have c12590 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12591 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12592 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12593 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12594 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12595 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12596 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12597 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12598 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12599 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12600 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12601 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12602 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12603 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12604 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12605 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12606 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_193_18 := arithEq_of_rows h (row := 193) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_18
  simp only [← c12596, ← c12597, ← c12598] at e_193_18
  have e_193_19 := arithEq_of_rows h (row := 193) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_19
  simp only [← c12599, ← c12600, ← c12601] at e_193_19
  have e_194_15 := arithEq_of_rows h (row := 194) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_15
  simp only [← c12590, k3, ← c12591, k3, ← c12592] at e_194_15
  have e_194_16 := arithEq_of_rows h (row := 194) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_16
  simp only [← c12593, ← c12594, k3, ← c12595] at e_194_16
  have e_194_17 := arithEq_of_rows h (row := 194) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_17
  simp only [← c12602, ← c12603, k3, ← c12604] at e_194_17
  refine ⟨?_, ?_⟩
  · have hc := e_193_18
    simp only [e_194_16] at hc
    linear_combination c12605.trans k2 - hc
  · have hc := e_194_17
    simp only [e_194_15, e_193_19, e_194_16] at hc
    linear_combination c12606.trans k2 - hc

theorem publicBatchWrapper86_f1867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3007)) (a (.wire 165 31)) (a (.virt 18083)) (a (.virt 18084)) := by
  have c12607 := (publicBatchWrapper86_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12608 := (publicBatchWrapper86_copies394 a h).1
  have c12609 := (publicBatchWrapper86_copies394 a h).2.1
  have c12610 := (publicBatchWrapper86_copies394 a h).2.2.1
  have c12611 := (publicBatchWrapper86_copies394 a h).2.2.2.1
  have c12612 := (publicBatchWrapper86_copies394 a h).2.2.2.2.1
  have c12613 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.1
  have c12614 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.1
  have c12615 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.1
  have c12616 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.1
  have c12617 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.1
  have c12618 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12619 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12620 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12621 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12622 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12623 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_194_18 := arithEq_of_rows h (row := 194) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_18
  simp only [← c12607, k3, ← c12608, k3, ← c12609] at e_194_18
  have e_194_19 := arithEq_of_rows h (row := 194) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_19
  simp only [← c12610, ← c12611, k3, ← c12612] at e_194_19
  have e_195_0 := arithEq_of_rows h (row := 195) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_0
  simp only [← c12613, ← c12614, ← c12615] at e_195_0
  have e_195_1 := arithEq_of_rows h (row := 195) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_1
  simp only [← c12616, ← c12617, ← c12618] at e_195_1
  have e_196_0 := arithEq_of_rows h (row := 196) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_0
  simp only [← c12619, ← c12620, k3, ← c12621] at e_196_0
  refine ⟨?_, ?_⟩
  · have hc := e_195_0
    simp only [e_194_19] at hc
    linear_combination c12622.trans k2 - hc
  · have hc := e_196_0
    simp only [e_194_18, e_195_1, e_194_19] at hc
    linear_combination c12623.trans k2 - hc

theorem publicBatchWrapper86_f1868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3008)) (a (.wire 165 39)) (a (.virt 18085)) (a (.virt 18086)) := by
  have c12624 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12625 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12626 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12627 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12628 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12629 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12630 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12631 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12632 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12633 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12634 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12635 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12636 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12637 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12638 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12639 := (publicBatchWrapper86_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12640 := (publicBatchWrapper86_copies395 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_195_2 := arithEq_of_rows h (row := 195) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_2
  simp only [← c12630, ← c12631, ← c12632] at e_195_2
  have e_195_3 := arithEq_of_rows h (row := 195) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_3
  simp only [← c12633, ← c12634, ← c12635] at e_195_3
  have e_196_1 := arithEq_of_rows h (row := 196) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_1
  simp only [← c12624, k3, ← c12625, k3, ← c12626] at e_196_1
  have e_196_2 := arithEq_of_rows h (row := 196) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_2
  simp only [← c12627, ← c12628, k3, ← c12629] at e_196_2
  have e_196_3 := arithEq_of_rows h (row := 196) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_3
  simp only [← c12636, ← c12637, k3, ← c12638] at e_196_3
  refine ⟨?_, ?_⟩
  · have hc := e_195_2
    simp only [e_196_2] at hc
    linear_combination c12639.trans k2 - hc
  · have hc := e_196_3
    simp only [e_196_1, e_195_3, e_196_2] at hc
    linear_combination c12640.trans k2 - hc

theorem publicBatchWrapper86_f1869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 195 19) = band (a (.virt 18079)) (a (.virt 18081)) := by
  have c12641 := (publicBatchWrapper86_copies395 a h).2.1
  have c12642 := (publicBatchWrapper86_copies395 a h).2.2.1
  have c12643 := (publicBatchWrapper86_copies395 a h).2.2.2.1
  have e_195_4 := arithEq_of_rows h (row := 195) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_4
  simp only [← c12641, ← c12642, ← c12643] at e_195_4
  have hr := e_195_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 195 23) = band (a (.virt 18083)) (a (.virt 18085)) := by
  have c12644 := (publicBatchWrapper86_copies395 a h).2.2.2.2.1
  have c12645 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.1
  have c12646 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.1
  have e_195_5 := arithEq_of_rows h (row := 195) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_5
  simp only [← c12644, ← c12645, ← c12646] at e_195_5
  have hr := e_195_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 195 27) = band (a (.wire 195 19)) (a (.wire 195 23)) := by
  have c12647 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.1
  have c12648 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.1
  have c12649 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.1
  have e_195_6 := arithEq_of_rows h (row := 195) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_6
  simp only [← c12647, ← c12648, ← c12649] at e_195_6
  have hr := e_195_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 3) = bor (a (.wire 16 47)) (a (.wire 195 27)) := by
  have c12650 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12651 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12652 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12653 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12654 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12655 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_189_12 := arithEq_of_rows h (row := 189) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_12
  simp only [← c12650, ← c12651, ← c12652] at e_189_12
  have e_197_0 := arithEq_of_rows h (row := 197) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_0
  simp only [← c12653, ← c12654, k3, ← c12655] at e_197_0
  have hr := e_197_0
  simp only [e_189_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 3) = a (.virt 201) := by
  have c12656 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12656

theorem publicBatchWrapper86_f1874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3203)) (a (.wire 165 55)) (a (.virt 18087)) (a (.virt 18088)) := by
  have c12657 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12658 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12659 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12660 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12661 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12662 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12663 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12664 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12665 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12666 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12667 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12668 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12669 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12670 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12671 := (publicBatchWrapper86_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12672 := (publicBatchWrapper86_copies396 a h).1
  have c12673 := (publicBatchWrapper86_copies396 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_195_7 := arithEq_of_rows h (row := 195) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_7
  simp only [← c12663, ← c12664, ← c12665] at e_195_7
  have e_195_8 := arithEq_of_rows h (row := 195) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_8
  simp only [← c12666, ← c12667, ← c12668] at e_195_8
  have e_196_4 := arithEq_of_rows h (row := 196) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_4
  simp only [← c12657, k3, ← c12658, k3, ← c12659] at e_196_4
  have e_196_5 := arithEq_of_rows h (row := 196) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_5
  simp only [← c12660, ← c12661, k3, ← c12662] at e_196_5
  have e_196_6 := arithEq_of_rows h (row := 196) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_6
  simp only [← c12669, ← c12670, k3, ← c12671] at e_196_6
  refine ⟨?_, ?_⟩
  · have hc := e_195_7
    simp only [e_196_5] at hc
    linear_combination c12672.trans k2 - hc
  · have hc := e_196_6
    simp only [e_196_4, e_195_8, e_196_5] at hc
    linear_combination c12673.trans k2 - hc

theorem publicBatchWrapper86_f1875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 7) = bor (a (.wire 18 11)) (a (.virt 18087)) := by
  have c12674 := (publicBatchWrapper86_copies396 a h).2.2.1
  have c12675 := (publicBatchWrapper86_copies396 a h).2.2.2.1
  have c12676 := (publicBatchWrapper86_copies396 a h).2.2.2.2.1
  have c12677 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.1
  have c12678 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.1
  have c12679 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_189_13 := arithEq_of_rows h (row := 189) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_13
  simp only [← c12674, ← c12675, ← c12676] at e_189_13
  have e_197_1 := arithEq_of_rows h (row := 197) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_1
  simp only [← c12677, ← c12678, k3, ← c12679] at e_197_1
  have hr := e_197_1
  simp only [e_189_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 7) = a (.virt 201) := by
  have c12680 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.1
  exact c12680

theorem publicBatchWrapper86_f1877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3204)) (a (.wire 165 63)) (a (.virt 18089)) (a (.virt 18090)) := by
  have c12681 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.1
  have c12682 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12683 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12684 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12685 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12686 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12687 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12688 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12689 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12690 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12691 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12692 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12693 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12694 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12695 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12696 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12697 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_195_9 := arithEq_of_rows h (row := 195) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_9
  simp only [← c12687, ← c12688, ← c12689] at e_195_9
  have e_195_10 := arithEq_of_rows h (row := 195) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_10
  simp only [← c12690, ← c12691, ← c12692] at e_195_10
  have e_196_7 := arithEq_of_rows h (row := 196) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_7
  simp only [← c12681, k3, ← c12682, k3, ← c12683] at e_196_7
  have e_196_8 := arithEq_of_rows h (row := 196) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_8
  simp only [← c12684, ← c12685, k3, ← c12686] at e_196_8
  have e_196_9 := arithEq_of_rows h (row := 196) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_9
  simp only [← c12693, ← c12694, k3, ← c12695] at e_196_9
  refine ⟨?_, ?_⟩
  · have hc := e_195_9
    simp only [e_196_8] at hc
    linear_combination c12696.trans k2 - hc
  · have hc := e_196_9
    simp only [e_196_7, e_195_10, e_196_8] at hc
    linear_combination c12697.trans k2 - hc

theorem publicBatchWrapper86_f1878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 11) = bor (a (.wire 18 11)) (a (.virt 18089)) := by
  have c12698 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12699 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12700 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12701 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12702 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12703 := (publicBatchWrapper86_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_189_14 := arithEq_of_rows h (row := 189) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_14
  simp only [← c12698, ← c12699, ← c12700] at e_189_14
  have e_197_2 := arithEq_of_rows h (row := 197) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_2
  simp only [← c12701, ← c12702, k3, ← c12703] at e_197_2
  have hr := e_197_2
  simp only [e_189_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 11) = a (.virt 201) := by
  have c12704 := (publicBatchWrapper86_copies397 a h).1
  exact c12704

theorem publicBatchWrapper86_f1880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3205)) (a (.wire 165 15)) (a (.virt 18091)) (a (.virt 18092)) := by
  have c12705 := (publicBatchWrapper86_copies397 a h).2.1
  have c12706 := (publicBatchWrapper86_copies397 a h).2.2.1
  have c12707 := (publicBatchWrapper86_copies397 a h).2.2.2.1
  have c12708 := (publicBatchWrapper86_copies397 a h).2.2.2.2.1
  have c12709 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.1
  have c12710 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.1
  have c12711 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.1
  have c12712 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.1
  have c12713 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.1
  have c12714 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12715 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12716 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12717 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12718 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12719 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12720 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12721 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_195_11 := arithEq_of_rows h (row := 195) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_11
  simp only [← c12711, ← c12712, ← c12713] at e_195_11
  have e_195_12 := arithEq_of_rows h (row := 195) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_12
  simp only [← c12714, ← c12715, ← c12716] at e_195_12
  have e_196_10 := arithEq_of_rows h (row := 196) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_10
  simp only [← c12705, k3, ← c12706, k3, ← c12707] at e_196_10
  have e_196_11 := arithEq_of_rows h (row := 196) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_11
  simp only [← c12708, ← c12709, k3, ← c12710] at e_196_11
  have e_196_12 := arithEq_of_rows h (row := 196) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_12
  simp only [← c12717, ← c12718, k3, ← c12719] at e_196_12
  refine ⟨?_, ?_⟩
  · have hc := e_195_11
    simp only [e_196_11] at hc
    linear_combination c12720.trans k2 - hc
  · have hc := e_196_12
    simp only [e_196_10, e_195_12, e_196_11] at hc
    linear_combination c12721.trans k2 - hc

theorem publicBatchWrapper86_f1881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3206)) (a (.wire 165 23)) (a (.virt 18093)) (a (.virt 18094)) := by
  have c12722 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12723 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12724 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12725 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12726 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12727 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12728 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12729 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12730 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12731 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12732 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12733 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12734 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12735 := (publicBatchWrapper86_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12736 := (publicBatchWrapper86_copies398 a h).1
  have c12737 := (publicBatchWrapper86_copies398 a h).2.1
  have c12738 := (publicBatchWrapper86_copies398 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_195_13 := arithEq_of_rows h (row := 195) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_13
  simp only [← c12728, ← c12729, ← c12730] at e_195_13
  have e_195_14 := arithEq_of_rows h (row := 195) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_14
  simp only [← c12731, ← c12732, ← c12733] at e_195_14
  have e_196_13 := arithEq_of_rows h (row := 196) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_13
  simp only [← c12722, k3, ← c12723, k3, ← c12724] at e_196_13
  have e_196_14 := arithEq_of_rows h (row := 196) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_14
  simp only [← c12725, ← c12726, k3, ← c12727] at e_196_14
  have e_196_15 := arithEq_of_rows h (row := 196) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_15
  simp only [← c12734, ← c12735, k3, ← c12736] at e_196_15
  refine ⟨?_, ?_⟩
  · have hc := e_195_13
    simp only [e_196_14] at hc
    linear_combination c12737.trans k2 - hc
  · have hc := e_196_15
    simp only [e_196_13, e_195_14, e_196_14] at hc
    linear_combination c12738.trans k2 - hc

theorem publicBatchWrapper86_f1882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3207)) (a (.wire 165 31)) (a (.virt 18095)) (a (.virt 18096)) := by
  have c12739 := (publicBatchWrapper86_copies398 a h).2.2.2.1
  have c12740 := (publicBatchWrapper86_copies398 a h).2.2.2.2.1
  have c12741 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.1
  have c12742 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.1
  have c12743 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.1
  have c12744 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.1
  have c12745 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.1
  have c12746 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12747 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12748 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12749 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12750 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12751 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12752 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12753 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12754 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12755 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_195_15 := arithEq_of_rows h (row := 195) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_15
  simp only [← c12745, ← c12746, ← c12747] at e_195_15
  have e_195_16 := arithEq_of_rows h (row := 195) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_16
  simp only [← c12748, ← c12749, ← c12750] at e_195_16
  have e_196_16 := arithEq_of_rows h (row := 196) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_16
  simp only [← c12739, k3, ← c12740, k3, ← c12741] at e_196_16
  have e_196_17 := arithEq_of_rows h (row := 196) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_17
  simp only [← c12742, ← c12743, k3, ← c12744] at e_196_17
  have e_196_18 := arithEq_of_rows h (row := 196) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_18
  simp only [← c12751, ← c12752, k3, ← c12753] at e_196_18
  refine ⟨?_, ?_⟩
  · have hc := e_195_15
    simp only [e_196_17] at hc
    linear_combination c12754.trans k2 - hc
  · have hc := e_196_18
    simp only [e_196_16, e_195_16, e_196_17] at hc
    linear_combination c12755.trans k2 - hc

theorem publicBatchWrapper86_f1883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3208)) (a (.wire 165 39)) (a (.virt 18097)) (a (.virt 18098)) := by
  have c12756 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12757 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12758 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12759 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12760 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12761 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12762 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12763 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12764 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12765 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12766 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12767 := (publicBatchWrapper86_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12768 := (publicBatchWrapper86_copies399 a h).1
  have c12769 := (publicBatchWrapper86_copies399 a h).2.1
  have c12770 := (publicBatchWrapper86_copies399 a h).2.2.1
  have c12771 := (publicBatchWrapper86_copies399 a h).2.2.2.1
  have c12772 := (publicBatchWrapper86_copies399 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_195_17 := arithEq_of_rows h (row := 195) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_17
  simp only [← c12762, ← c12763, ← c12764] at e_195_17
  have e_195_18 := arithEq_of_rows h (row := 195) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_18
  simp only [← c12765, ← c12766, ← c12767] at e_195_18
  have e_196_19 := arithEq_of_rows h (row := 196) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_19
  simp only [← c12756, k3, ← c12757, k3, ← c12758] at e_196_19
  have e_198_0 := arithEq_of_rows h (row := 198) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_0
  simp only [← c12759, ← c12760, k3, ← c12761] at e_198_0
  have e_198_1 := arithEq_of_rows h (row := 198) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_1
  simp only [← c12768, ← c12769, k3, ← c12770] at e_198_1
  refine ⟨?_, ?_⟩
  · have hc := e_195_17
    simp only [e_198_0] at hc
    linear_combination c12771.trans k2 - hc
  · have hc := e_198_1
    simp only [e_196_19, e_195_18, e_198_0] at hc
    linear_combination c12772.trans k2 - hc

theorem publicBatchWrapper86_f1884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 195 79) = band (a (.virt 18091)) (a (.virt 18093)) := by
  have c12773 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.1
  have c12774 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.1
  have c12775 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.1
  have e_195_19 := arithEq_of_rows h (row := 195) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_19
  simp only [← c12773, ← c12774, ← c12775] at e_195_19
  have hr := e_195_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 199 3) = band (a (.virt 18095)) (a (.virt 18097)) := by
  have c12776 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.1
  have c12777 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.1
  have c12778 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_199_0 := arithEq_of_rows h (row := 199) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_0
  simp only [← c12776, ← c12777, ← c12778] at e_199_0
  have hr := e_199_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 199 7) = band (a (.wire 195 79)) (a (.wire 199 3)) := by
  have c12779 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12780 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12781 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_199_1 := arithEq_of_rows h (row := 199) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_1
  simp only [← c12779, ← c12780, ← c12781] at e_199_1
  have hr := e_199_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 15) = bor (a (.wire 18 11)) (a (.wire 199 7)) := by
  have c12782 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12783 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12784 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12785 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12786 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12787 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_189_15 := arithEq_of_rows h (row := 189) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_15
  simp only [← c12782, ← c12783, ← c12784] at e_189_15
  have e_197_3 := arithEq_of_rows h (row := 197) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_3
  simp only [← c12785, ← c12786, k3, ← c12787] at e_197_3
  have hr := e_197_3
  simp only [e_189_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 15) = a (.virt 201) := by
  have c12788 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12788

theorem publicBatchWrapper86_f1889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3403)) (a (.wire 165 55)) (a (.virt 18099)) (a (.virt 18100)) := by
  have c12789 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12790 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12791 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12792 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12793 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12794 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12795 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12796 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12797 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12798 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12799 := (publicBatchWrapper86_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12800 := (publicBatchWrapper86_copies400 a h).1
  have c12801 := (publicBatchWrapper86_copies400 a h).2.1
  have c12802 := (publicBatchWrapper86_copies400 a h).2.2.1
  have c12803 := (publicBatchWrapper86_copies400 a h).2.2.2.1
  have c12804 := (publicBatchWrapper86_copies400 a h).2.2.2.2.1
  have c12805 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_198_2 := arithEq_of_rows h (row := 198) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_2
  simp only [← c12789, k3, ← c12790, k3, ← c12791] at e_198_2
  have e_198_3 := arithEq_of_rows h (row := 198) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_3
  simp only [← c12792, ← c12793, k3, ← c12794] at e_198_3
  have e_198_4 := arithEq_of_rows h (row := 198) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_4
  simp only [← c12801, ← c12802, k3, ← c12803] at e_198_4
  have e_199_2 := arithEq_of_rows h (row := 199) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_2
  simp only [← c12795, ← c12796, ← c12797] at e_199_2
  have e_199_3 := arithEq_of_rows h (row := 199) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_3
  simp only [← c12798, ← c12799, ← c12800] at e_199_3
  refine ⟨?_, ?_⟩
  · have hc := e_199_2
    simp only [e_198_3] at hc
    linear_combination c12804.trans k2 - hc
  · have hc := e_198_4
    simp only [e_198_2, e_199_3, e_198_3] at hc
    linear_combination c12805.trans k2 - hc

theorem publicBatchWrapper86_f1890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 19) = bor (a (.wire 18 55)) (a (.virt 18099)) := by
  have c12806 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.1
  have c12807 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.1
  have c12808 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.1
  have c12809 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.1
  have c12810 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12811 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_189_16 := arithEq_of_rows h (row := 189) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_16
  simp only [← c12806, ← c12807, ← c12808] at e_189_16
  have e_197_4 := arithEq_of_rows h (row := 197) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_4
  simp only [← c12809, ← c12810, k3, ← c12811] at e_197_4
  have hr := e_197_4
  simp only [e_189_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 19) = a (.virt 201) := by
  have c12812 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12812

theorem publicBatchWrapper86_f1892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3404)) (a (.wire 165 63)) (a (.virt 18101)) (a (.virt 18102)) := by
  have c12813 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12814 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12815 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12816 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12817 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12818 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12819 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12820 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12821 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12822 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12823 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12824 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12825 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12826 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12827 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12828 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12829 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_198_5 := arithEq_of_rows h (row := 198) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_5
  simp only [← c12813, k3, ← c12814, k3, ← c12815] at e_198_5
  have e_198_6 := arithEq_of_rows h (row := 198) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_6
  simp only [← c12816, ← c12817, k3, ← c12818] at e_198_6
  have e_198_7 := arithEq_of_rows h (row := 198) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_7
  simp only [← c12825, ← c12826, k3, ← c12827] at e_198_7
  have e_199_4 := arithEq_of_rows h (row := 199) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_4
  simp only [← c12819, ← c12820, ← c12821] at e_199_4
  have e_199_5 := arithEq_of_rows h (row := 199) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_5
  simp only [← c12822, ← c12823, ← c12824] at e_199_5
  refine ⟨?_, ?_⟩
  · have hc := e_199_4
    simp only [e_198_6] at hc
    linear_combination c12828.trans k2 - hc
  · have hc := e_198_7
    simp only [e_198_5, e_199_5, e_198_6] at hc
    linear_combination c12829.trans k2 - hc

theorem publicBatchWrapper86_f1893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 23) = bor (a (.wire 18 55)) (a (.virt 18101)) := by
  have c12830 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12831 := (publicBatchWrapper86_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12832 := (publicBatchWrapper86_copies401 a h).1
  have c12833 := (publicBatchWrapper86_copies401 a h).2.1
  have c12834 := (publicBatchWrapper86_copies401 a h).2.2.1
  have c12835 := (publicBatchWrapper86_copies401 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_189_17 := arithEq_of_rows h (row := 189) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_17
  simp only [← c12830, ← c12831, ← c12832] at e_189_17
  have e_197_5 := arithEq_of_rows h (row := 197) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_5
  simp only [← c12833, ← c12834, k3, ← c12835] at e_197_5
  have hr := e_197_5
  simp only [e_189_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 23) = a (.virt 201) := by
  have c12836 := (publicBatchWrapper86_copies401 a h).2.2.2.2.1
  exact c12836

theorem publicBatchWrapper86_f1895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3405)) (a (.wire 165 15)) (a (.virt 18103)) (a (.virt 18104)) := by
  have c12837 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.1
  have c12838 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.1
  have c12839 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.1
  have c12840 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.1
  have c12841 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.1
  have c12842 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12843 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12844 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12845 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12846 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12847 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12848 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12849 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12850 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12851 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12852 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12853 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_198_8 := arithEq_of_rows h (row := 198) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_8
  simp only [← c12837, k3, ← c12838, k3, ← c12839] at e_198_8
  have e_198_9 := arithEq_of_rows h (row := 198) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_9
  simp only [← c12840, ← c12841, k3, ← c12842] at e_198_9
  have e_198_10 := arithEq_of_rows h (row := 198) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_10
  simp only [← c12849, ← c12850, k3, ← c12851] at e_198_10
  have e_199_6 := arithEq_of_rows h (row := 199) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_6
  simp only [← c12843, ← c12844, ← c12845] at e_199_6
  have e_199_7 := arithEq_of_rows h (row := 199) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_7
  simp only [← c12846, ← c12847, ← c12848] at e_199_7
  refine ⟨?_, ?_⟩
  · have hc := e_199_6
    simp only [e_198_9] at hc
    linear_combination c12852.trans k2 - hc
  · have hc := e_198_10
    simp only [e_198_8, e_199_7, e_198_9] at hc
    linear_combination c12853.trans k2 - hc

theorem publicBatchWrapper86_f1896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3406)) (a (.wire 165 23)) (a (.virt 18105)) (a (.virt 18106)) := by
  have c12854 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12855 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12856 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12857 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12858 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12859 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12860 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12861 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12862 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12863 := (publicBatchWrapper86_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12864 := (publicBatchWrapper86_copies402 a h).1
  have c12865 := (publicBatchWrapper86_copies402 a h).2.1
  have c12866 := (publicBatchWrapper86_copies402 a h).2.2.1
  have c12867 := (publicBatchWrapper86_copies402 a h).2.2.2.1
  have c12868 := (publicBatchWrapper86_copies402 a h).2.2.2.2.1
  have c12869 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.1
  have c12870 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_198_11 := arithEq_of_rows h (row := 198) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_11
  simp only [← c12854, k3, ← c12855, k3, ← c12856] at e_198_11
  have e_198_12 := arithEq_of_rows h (row := 198) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_12
  simp only [← c12857, ← c12858, k3, ← c12859] at e_198_12
  have e_198_13 := arithEq_of_rows h (row := 198) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_13
  simp only [← c12866, ← c12867, k3, ← c12868] at e_198_13
  have e_199_8 := arithEq_of_rows h (row := 199) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_8
  simp only [← c12860, ← c12861, ← c12862] at e_199_8
  have e_199_9 := arithEq_of_rows h (row := 199) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_9
  simp only [← c12863, ← c12864, ← c12865] at e_199_9
  refine ⟨?_, ?_⟩
  · have hc := e_199_8
    simp only [e_198_12] at hc
    linear_combination c12869.trans k2 - hc
  · have hc := e_198_13
    simp only [e_198_11, e_199_9, e_198_12] at hc
    linear_combination c12870.trans k2 - hc

theorem publicBatchWrapper86_f1897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3407)) (a (.wire 165 31)) (a (.virt 18107)) (a (.virt 18108)) := by
  have c12871 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.1
  have c12872 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.1
  have c12873 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.1
  have c12874 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12875 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12876 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12877 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12878 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12879 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12880 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12881 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12882 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12883 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12884 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12885 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12886 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12887 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_198_14 := arithEq_of_rows h (row := 198) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_14
  simp only [← c12871, k3, ← c12872, k3, ← c12873] at e_198_14
  have e_198_15 := arithEq_of_rows h (row := 198) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_15
  simp only [← c12874, ← c12875, k3, ← c12876] at e_198_15
  have e_198_16 := arithEq_of_rows h (row := 198) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_16
  simp only [← c12883, ← c12884, k3, ← c12885] at e_198_16
  have e_199_10 := arithEq_of_rows h (row := 199) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_10
  simp only [← c12877, ← c12878, ← c12879] at e_199_10
  have e_199_11 := arithEq_of_rows h (row := 199) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_11
  simp only [← c12880, ← c12881, ← c12882] at e_199_11
  refine ⟨?_, ?_⟩
  · have hc := e_199_10
    simp only [e_198_15] at hc
    linear_combination c12886.trans k2 - hc
  · have hc := e_198_16
    simp only [e_198_14, e_199_11, e_198_15] at hc
    linear_combination c12887.trans k2 - hc

theorem publicBatchWrapper86_f1898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3408)) (a (.wire 165 39)) (a (.virt 18109)) (a (.virt 18110)) := by
  have c12888 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12889 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12890 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12891 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12892 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12893 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12894 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12895 := (publicBatchWrapper86_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12896 := (publicBatchWrapper86_copies403 a h).1
  have c12897 := (publicBatchWrapper86_copies403 a h).2.1
  have c12898 := (publicBatchWrapper86_copies403 a h).2.2.1
  have c12899 := (publicBatchWrapper86_copies403 a h).2.2.2.1
  have c12900 := (publicBatchWrapper86_copies403 a h).2.2.2.2.1
  have c12901 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.1
  have c12902 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.1
  have c12903 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.1
  have c12904 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_198_17 := arithEq_of_rows h (row := 198) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_17
  simp only [← c12888, k3, ← c12889, k3, ← c12890] at e_198_17
  have e_198_18 := arithEq_of_rows h (row := 198) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_18
  simp only [← c12891, ← c12892, k3, ← c12893] at e_198_18
  have e_198_19 := arithEq_of_rows h (row := 198) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_19
  simp only [← c12900, ← c12901, k3, ← c12902] at e_198_19
  have e_199_12 := arithEq_of_rows h (row := 199) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_12
  simp only [← c12894, ← c12895, ← c12896] at e_199_12
  have e_199_13 := arithEq_of_rows h (row := 199) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_13
  simp only [← c12897, ← c12898, ← c12899] at e_199_13
  refine ⟨?_, ?_⟩
  · have hc := e_199_12
    simp only [e_198_18] at hc
    linear_combination c12903.trans k2 - hc
  · have hc := e_198_19
    simp only [e_198_17, e_199_13, e_198_18] at hc
    linear_combination c12904.trans k2 - hc

theorem publicBatchWrapper86_f1899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 199 59) = band (a (.virt 18103)) (a (.virt 18105)) := by
  have c12905 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.1
  have c12906 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12907 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_199_14 := arithEq_of_rows h (row := 199) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_14
  simp only [← c12905, ← c12906, ← c12907] at e_199_14
  have hr := e_199_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 199 63) = band (a (.virt 18107)) (a (.virt 18109)) := by
  have c12908 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12909 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12910 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_199_15 := arithEq_of_rows h (row := 199) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_15
  simp only [← c12908, ← c12909, ← c12910] at e_199_15
  have hr := e_199_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 199 67) = band (a (.wire 199 59)) (a (.wire 199 63)) := by
  have c12911 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12912 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12913 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_199_16 := arithEq_of_rows h (row := 199) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_16
  simp only [← c12911, ← c12912, ← c12913] at e_199_16
  have hr := e_199_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 27) = bor (a (.wire 18 55)) (a (.wire 199 67)) := by
  have c12914 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12915 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12916 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12917 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12918 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12919 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_189_18 := arithEq_of_rows h (row := 189) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_18
  simp only [← c12914, ← c12915, ← c12916] at e_189_18
  have e_197_6 := arithEq_of_rows h (row := 197) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_6
  simp only [← c12917, ← c12918, k3, ← c12919] at e_197_6
  have hr := e_197_6
  simp only [e_189_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 27) = a (.virt 201) := by
  have c12920 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12920

theorem publicBatchWrapper86_f1904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3603)) (a (.wire 165 55)) (a (.virt 18111)) (a (.virt 18112)) := by
  have c12921 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12922 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12923 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12924 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12925 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12926 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12927 := (publicBatchWrapper86_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12928 := (publicBatchWrapper86_copies404 a h).1
  have c12929 := (publicBatchWrapper86_copies404 a h).2.1
  have c12930 := (publicBatchWrapper86_copies404 a h).2.2.1
  have c12931 := (publicBatchWrapper86_copies404 a h).2.2.2.1
  have c12932 := (publicBatchWrapper86_copies404 a h).2.2.2.2.1
  have c12933 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.1
  have c12934 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.1
  have c12935 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.1
  have c12936 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.1
  have c12937 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_199_17 := arithEq_of_rows h (row := 199) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_17
  simp only [← c12927, ← c12928, ← c12929] at e_199_17
  have e_199_18 := arithEq_of_rows h (row := 199) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_18
  simp only [← c12930, ← c12931, ← c12932] at e_199_18
  have e_200_0 := arithEq_of_rows h (row := 200) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_0
  simp only [← c12921, k3, ← c12922, k3, ← c12923] at e_200_0
  have e_200_1 := arithEq_of_rows h (row := 200) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_1
  simp only [← c12924, ← c12925, k3, ← c12926] at e_200_1
  have e_200_2 := arithEq_of_rows h (row := 200) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_2
  simp only [← c12933, ← c12934, k3, ← c12935] at e_200_2
  refine ⟨?_, ?_⟩
  · have hc := e_199_17
    simp only [e_200_1] at hc
    linear_combination c12936.trans k2 - hc
  · have hc := e_200_2
    simp only [e_200_0, e_199_18, e_200_1] at hc
    linear_combination c12937.trans k2 - hc

theorem publicBatchWrapper86_f1905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 31) = bor (a (.wire 20 19)) (a (.virt 18111)) := by
  have c12938 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12939 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12940 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12941 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12942 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12943 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_189_19 := arithEq_of_rows h (row := 189) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_19
  simp only [← c12938, ← c12939, ← c12940] at e_189_19
  have e_197_7 := arithEq_of_rows h (row := 197) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_7
  simp only [← c12941, ← c12942, k3, ← c12943] at e_197_7
  have hr := e_197_7
  simp only [e_189_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 31) = a (.virt 201) := by
  have c12944 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12944

theorem publicBatchWrapper86_f1907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3604)) (a (.wire 165 63)) (a (.virt 18113)) (a (.virt 18114)) := by
  have c12945 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12946 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12947 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12948 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12949 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12950 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12951 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12952 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12953 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12954 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12955 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12956 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12957 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12958 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12959 := (publicBatchWrapper86_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12960 := (publicBatchWrapper86_copies405 a h).1
  have c12961 := (publicBatchWrapper86_copies405 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_199_19 := arithEq_of_rows h (row := 199) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_19
  simp only [← c12951, ← c12952, ← c12953] at e_199_19
  have e_200_3 := arithEq_of_rows h (row := 200) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_3
  simp only [← c12945, k3, ← c12946, k3, ← c12947] at e_200_3
  have e_200_4 := arithEq_of_rows h (row := 200) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_4
  simp only [← c12948, ← c12949, k3, ← c12950] at e_200_4
  have e_200_5 := arithEq_of_rows h (row := 200) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_5
  simp only [← c12957, ← c12958, k3, ← c12959] at e_200_5
  have e_201_0 := arithEq_of_rows h (row := 201) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_0
  simp only [← c12954, ← c12955, ← c12956] at e_201_0
  refine ⟨?_, ?_⟩
  · have hc := e_199_19
    simp only [e_200_4] at hc
    linear_combination c12960.trans k2 - hc
  · have hc := e_200_5
    simp only [e_200_3, e_201_0, e_200_4] at hc
    linear_combination c12961.trans k2 - hc

theorem publicBatchWrapper86_f1908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 35) = bor (a (.wire 20 19)) (a (.virt 18113)) := by
  have c12962 := (publicBatchWrapper86_copies405 a h).2.2.1
  have c12963 := (publicBatchWrapper86_copies405 a h).2.2.2.1
  have c12964 := (publicBatchWrapper86_copies405 a h).2.2.2.2.1
  have c12965 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.1
  have c12966 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.1
  have c12967 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_8 := arithEq_of_rows h (row := 197) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_8
  simp only [← c12965, ← c12966, k3, ← c12967] at e_197_8
  have e_202_0 := arithEq_of_rows h (row := 202) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_0
  simp only [← c12962, ← c12963, ← c12964] at e_202_0
  have hr := e_197_8
  simp only [e_202_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 35) = a (.virt 201) := by
  have c12968 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.1
  exact c12968

theorem publicBatchWrapper86_f1910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3605)) (a (.wire 165 15)) (a (.virt 18115)) (a (.virt 18116)) := by
  have c12969 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.1
  have c12970 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12971 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12972 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12973 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12974 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12975 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12976 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12977 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12978 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12979 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12980 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12981 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12982 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12983 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12984 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12985 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_200_6 := arithEq_of_rows h (row := 200) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_6
  simp only [← c12969, k3, ← c12970, k3, ← c12971] at e_200_6
  have e_200_7 := arithEq_of_rows h (row := 200) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_7
  simp only [← c12972, ← c12973, k3, ← c12974] at e_200_7
  have e_200_8 := arithEq_of_rows h (row := 200) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_8
  simp only [← c12981, ← c12982, k3, ← c12983] at e_200_8
  have e_201_1 := arithEq_of_rows h (row := 201) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_1
  simp only [← c12975, ← c12976, ← c12977] at e_201_1
  have e_201_2 := arithEq_of_rows h (row := 201) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_2
  simp only [← c12978, ← c12979, ← c12980] at e_201_2
  refine ⟨?_, ?_⟩
  · have hc := e_201_1
    simp only [e_200_7] at hc
    linear_combination c12984.trans k2 - hc
  · have hc := e_200_8
    simp only [e_200_6, e_201_2, e_200_7] at hc
    linear_combination c12985.trans k2 - hc

theorem publicBatchWrapper86_f1911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3606)) (a (.wire 165 23)) (a (.virt 18117)) (a (.virt 18118)) := by
  have c12986 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12987 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12988 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12989 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12990 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12991 := (publicBatchWrapper86_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12992 := (publicBatchWrapper86_copies406 a h).1
  have c12993 := (publicBatchWrapper86_copies406 a h).2.1
  have c12994 := (publicBatchWrapper86_copies406 a h).2.2.1
  have c12995 := (publicBatchWrapper86_copies406 a h).2.2.2.1
  have c12996 := (publicBatchWrapper86_copies406 a h).2.2.2.2.1
  have c12997 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.1
  have c12998 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.1
  have c12999 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.1
  have c13000 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.1
  have c13001 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.1
  have c13002 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_200_9 := arithEq_of_rows h (row := 200) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_9
  simp only [← c12986, k3, ← c12987, k3, ← c12988] at e_200_9
  have e_200_10 := arithEq_of_rows h (row := 200) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_10
  simp only [← c12989, ← c12990, k3, ← c12991] at e_200_10
  have e_200_11 := arithEq_of_rows h (row := 200) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_11
  simp only [← c12998, ← c12999, k3, ← c13000] at e_200_11
  have e_201_3 := arithEq_of_rows h (row := 201) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_3
  simp only [← c12992, ← c12993, ← c12994] at e_201_3
  have e_201_4 := arithEq_of_rows h (row := 201) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_4
  simp only [← c12995, ← c12996, ← c12997] at e_201_4
  refine ⟨?_, ?_⟩
  · have hc := e_201_3
    simp only [e_200_10] at hc
    linear_combination c13001.trans k2 - hc
  · have hc := e_200_11
    simp only [e_200_9, e_201_4, e_200_10] at hc
    linear_combination c13002.trans k2 - hc

theorem publicBatchWrapper86_f1912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3607)) (a (.wire 165 31)) (a (.virt 18119)) (a (.virt 18120)) := by
  have c13003 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13004 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13005 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13006 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13007 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13008 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13009 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13010 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13011 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13012 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13013 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13014 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13015 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13016 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13017 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13018 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13019 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_200_12 := arithEq_of_rows h (row := 200) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_12
  simp only [← c13003, k3, ← c13004, k3, ← c13005] at e_200_12
  have e_200_13 := arithEq_of_rows h (row := 200) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_13
  simp only [← c13006, ← c13007, k3, ← c13008] at e_200_13
  have e_200_14 := arithEq_of_rows h (row := 200) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_14
  simp only [← c13015, ← c13016, k3, ← c13017] at e_200_14
  have e_201_5 := arithEq_of_rows h (row := 201) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_5
  simp only [← c13009, ← c13010, ← c13011] at e_201_5
  have e_201_6 := arithEq_of_rows h (row := 201) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_6
  simp only [← c13012, ← c13013, ← c13014] at e_201_6
  refine ⟨?_, ?_⟩
  · have hc := e_201_5
    simp only [e_200_13] at hc
    linear_combination c13018.trans k2 - hc
  · have hc := e_200_14
    simp only [e_200_12, e_201_6, e_200_13] at hc
    linear_combination c13019.trans k2 - hc

theorem publicBatchWrapper86_f1913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3608)) (a (.wire 165 39)) (a (.virt 18121)) (a (.virt 18122)) := by
  have c13020 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13021 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13022 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13023 := (publicBatchWrapper86_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13024 := (publicBatchWrapper86_copies407 a h).1
  have c13025 := (publicBatchWrapper86_copies407 a h).2.1
  have c13026 := (publicBatchWrapper86_copies407 a h).2.2.1
  have c13027 := (publicBatchWrapper86_copies407 a h).2.2.2.1
  have c13028 := (publicBatchWrapper86_copies407 a h).2.2.2.2.1
  have c13029 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.1
  have c13030 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.1
  have c13031 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.1
  have c13032 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.1
  have c13033 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.1
  have c13034 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13035 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13036 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_200_15 := arithEq_of_rows h (row := 200) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_15
  simp only [← c13020, k3, ← c13021, k3, ← c13022] at e_200_15
  have e_200_16 := arithEq_of_rows h (row := 200) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_16
  simp only [← c13023, ← c13024, k3, ← c13025] at e_200_16
  have e_200_17 := arithEq_of_rows h (row := 200) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_17
  simp only [← c13032, ← c13033, k3, ← c13034] at e_200_17
  have e_201_7 := arithEq_of_rows h (row := 201) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_7
  simp only [← c13026, ← c13027, ← c13028] at e_201_7
  have e_201_8 := arithEq_of_rows h (row := 201) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_8
  simp only [← c13029, ← c13030, ← c13031] at e_201_8
  refine ⟨?_, ?_⟩
  · have hc := e_201_7
    simp only [e_200_16] at hc
    linear_combination c13035.trans k2 - hc
  · have hc := e_200_17
    simp only [e_200_15, e_201_8, e_200_16] at hc
    linear_combination c13036.trans k2 - hc

theorem publicBatchWrapper86_f1914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 201 39) = band (a (.virt 18115)) (a (.virt 18117)) := by
  have c13037 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13038 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13039 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_9 := arithEq_of_rows h (row := 201) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_9
  simp only [← c13037, ← c13038, ← c13039] at e_201_9
  have hr := e_201_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 201 43) = band (a (.virt 18119)) (a (.virt 18121)) := by
  have c13040 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13041 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13042 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_10 := arithEq_of_rows h (row := 201) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_10
  simp only [← c13040, ← c13041, ← c13042] at e_201_10
  have hr := e_201_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 201 47) = band (a (.wire 201 39)) (a (.wire 201 43)) := by
  have c13043 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13044 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13045 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_11 := arithEq_of_rows h (row := 201) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_11
  simp only [← c13043, ← c13044, ← c13045] at e_201_11
  have hr := e_201_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 39) = bor (a (.wire 20 19)) (a (.wire 201 47)) := by
  have c13046 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13047 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13048 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13049 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13050 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13051 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_9 := arithEq_of_rows h (row := 197) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_9
  simp only [← c13049, ← c13050, k3, ← c13051] at e_197_9
  have e_202_1 := arithEq_of_rows h (row := 202) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_1
  simp only [← c13046, ← c13047, ← c13048] at e_202_1
  have hr := e_197_9
  simp only [e_202_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 39) = a (.virt 201) := by
  have c13052 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13052

theorem publicBatchWrapper86_f1919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3803)) (a (.wire 165 55)) (a (.virt 18123)) (a (.virt 18124)) := by
  have c13053 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13054 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13055 := (publicBatchWrapper86_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13056 := (publicBatchWrapper86_copies408 a h).1
  have c13057 := (publicBatchWrapper86_copies408 a h).2.1
  have c13058 := (publicBatchWrapper86_copies408 a h).2.2.1
  have c13059 := (publicBatchWrapper86_copies408 a h).2.2.2.1
  have c13060 := (publicBatchWrapper86_copies408 a h).2.2.2.2.1
  have c13061 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.1
  have c13062 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.1
  have c13063 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.1
  have c13064 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.1
  have c13065 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.1
  have c13066 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13067 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13068 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13069 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_200_18 := arithEq_of_rows h (row := 200) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_18
  simp only [← c13053, k3, ← c13054, k3, ← c13055] at e_200_18
  have e_200_19 := arithEq_of_rows h (row := 200) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_19
  simp only [← c13056, ← c13057, k3, ← c13058] at e_200_19
  have e_201_12 := arithEq_of_rows h (row := 201) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_12
  simp only [← c13059, ← c13060, ← c13061] at e_201_12
  have e_201_13 := arithEq_of_rows h (row := 201) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_13
  simp only [← c13062, ← c13063, ← c13064] at e_201_13
  have e_203_0 := arithEq_of_rows h (row := 203) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_0
  simp only [← c13065, ← c13066, k3, ← c13067] at e_203_0
  refine ⟨?_, ?_⟩
  · have hc := e_201_12
    simp only [e_200_19] at hc
    linear_combination c13068.trans k2 - hc
  · have hc := e_203_0
    simp only [e_200_18, e_201_13, e_200_19] at hc
    linear_combination c13069.trans k2 - hc

theorem publicBatchWrapper86_f1920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 43) = bor (a (.wire 20 63)) (a (.virt 18123)) := by
  have c13070 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13071 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13072 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13073 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13074 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13075 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_10 := arithEq_of_rows h (row := 197) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_10
  simp only [← c13073, ← c13074, k3, ← c13075] at e_197_10
  have e_202_2 := arithEq_of_rows h (row := 202) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_2
  simp only [← c13070, ← c13071, ← c13072] at e_202_2
  have hr := e_197_10
  simp only [e_202_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 43) = a (.virt 201) := by
  have c13076 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13076

theorem publicBatchWrapper86_f1922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3804)) (a (.wire 165 63)) (a (.virt 18125)) (a (.virt 18126)) := by
  have c13077 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13078 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13079 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13080 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13081 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13082 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13083 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13084 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13085 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13086 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13087 := (publicBatchWrapper86_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13088 := (publicBatchWrapper86_copies409 a h).1
  have c13089 := (publicBatchWrapper86_copies409 a h).2.1
  have c13090 := (publicBatchWrapper86_copies409 a h).2.2.1
  have c13091 := (publicBatchWrapper86_copies409 a h).2.2.2.1
  have c13092 := (publicBatchWrapper86_copies409 a h).2.2.2.2.1
  have c13093 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_201_14 := arithEq_of_rows h (row := 201) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_14
  simp only [← c13083, ← c13084, ← c13085] at e_201_14
  have e_201_15 := arithEq_of_rows h (row := 201) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_15
  simp only [← c13086, ← c13087, ← c13088] at e_201_15
  have e_203_1 := arithEq_of_rows h (row := 203) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_1
  simp only [← c13077, k3, ← c13078, k3, ← c13079] at e_203_1
  have e_203_2 := arithEq_of_rows h (row := 203) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_2
  simp only [← c13080, ← c13081, k3, ← c13082] at e_203_2
  have e_203_3 := arithEq_of_rows h (row := 203) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_3
  simp only [← c13089, ← c13090, k3, ← c13091] at e_203_3
  refine ⟨?_, ?_⟩
  · have hc := e_201_14
    simp only [e_203_2] at hc
    linear_combination c13092.trans k2 - hc
  · have hc := e_203_3
    simp only [e_203_1, e_201_15, e_203_2] at hc
    linear_combination c13093.trans k2 - hc

theorem publicBatchWrapper86_f1923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 47) = bor (a (.wire 20 63)) (a (.virt 18125)) := by
  have c13094 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.1
  have c13095 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.1
  have c13096 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.1
  have c13097 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.1
  have c13098 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13099 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_11 := arithEq_of_rows h (row := 197) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_11
  simp only [← c13097, ← c13098, k3, ← c13099] at e_197_11
  have e_202_3 := arithEq_of_rows h (row := 202) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_3
  simp only [← c13094, ← c13095, ← c13096] at e_202_3
  have hr := e_197_11
  simp only [e_202_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 47) = a (.virt 201) := by
  have c13100 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13100

theorem publicBatchWrapper86_f1925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3805)) (a (.wire 165 15)) (a (.virt 18127)) (a (.virt 18128)) := by
  have c13101 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13102 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13103 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13104 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13105 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13106 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13107 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13108 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13109 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13110 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13111 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13112 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13113 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13114 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13115 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13116 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13117 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_201_16 := arithEq_of_rows h (row := 201) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_16
  simp only [← c13107, ← c13108, ← c13109] at e_201_16
  have e_201_17 := arithEq_of_rows h (row := 201) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_17
  simp only [← c13110, ← c13111, ← c13112] at e_201_17
  have e_203_4 := arithEq_of_rows h (row := 203) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_4
  simp only [← c13101, k3, ← c13102, k3, ← c13103] at e_203_4
  have e_203_5 := arithEq_of_rows h (row := 203) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_5
  simp only [← c13104, ← c13105, k3, ← c13106] at e_203_5
  have e_203_6 := arithEq_of_rows h (row := 203) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_6
  simp only [← c13113, ← c13114, k3, ← c13115] at e_203_6
  refine ⟨?_, ?_⟩
  · have hc := e_201_16
    simp only [e_203_5] at hc
    linear_combination c13116.trans k2 - hc
  · have hc := e_203_6
    simp only [e_203_4, e_201_17, e_203_5] at hc
    linear_combination c13117.trans k2 - hc

theorem publicBatchWrapper86_f1926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3806)) (a (.wire 165 23)) (a (.virt 18129)) (a (.virt 18130)) := by
  have c13118 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13119 := (publicBatchWrapper86_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13120 := (publicBatchWrapper86_copies410 a h).1
  have c13121 := (publicBatchWrapper86_copies410 a h).2.1
  have c13122 := (publicBatchWrapper86_copies410 a h).2.2.1
  have c13123 := (publicBatchWrapper86_copies410 a h).2.2.2.1
  have c13124 := (publicBatchWrapper86_copies410 a h).2.2.2.2.1
  have c13125 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.1
  have c13126 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.1
  have c13127 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.1
  have c13128 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.1
  have c13129 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.1
  have c13130 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13131 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13132 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13133 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13134 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_201_18 := arithEq_of_rows h (row := 201) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_18
  simp only [← c13124, ← c13125, ← c13126] at e_201_18
  have e_201_19 := arithEq_of_rows h (row := 201) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_19
  simp only [← c13127, ← c13128, ← c13129] at e_201_19
  have e_203_7 := arithEq_of_rows h (row := 203) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_7
  simp only [← c13118, k3, ← c13119, k3, ← c13120] at e_203_7
  have e_203_8 := arithEq_of_rows h (row := 203) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_8
  simp only [← c13121, ← c13122, k3, ← c13123] at e_203_8
  have e_203_9 := arithEq_of_rows h (row := 203) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_9
  simp only [← c13130, ← c13131, k3, ← c13132] at e_203_9
  refine ⟨?_, ?_⟩
  · have hc := e_201_18
    simp only [e_203_8] at hc
    linear_combination c13133.trans k2 - hc
  · have hc := e_203_9
    simp only [e_203_7, e_201_19, e_203_8] at hc
    linear_combination c13134.trans k2 - hc

theorem publicBatchWrapper86_f1927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3807)) (a (.wire 165 31)) (a (.virt 18131)) (a (.virt 18132)) := by
  have c13135 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13136 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13137 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13138 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13139 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13140 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13141 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13142 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13143 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13144 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13145 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13146 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13147 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13148 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13149 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13150 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13151 := (publicBatchWrapper86_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_203_10 := arithEq_of_rows h (row := 203) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_10
  simp only [← c13135, k3, ← c13136, k3, ← c13137] at e_203_10
  have e_203_11 := arithEq_of_rows h (row := 203) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_11
  simp only [← c13138, ← c13139, k3, ← c13140] at e_203_11
  have e_203_12 := arithEq_of_rows h (row := 203) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_12
  simp only [← c13147, ← c13148, k3, ← c13149] at e_203_12
  have e_204_0 := arithEq_of_rows h (row := 204) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_0
  simp only [← c13141, ← c13142, ← c13143] at e_204_0
  have e_204_1 := arithEq_of_rows h (row := 204) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_1
  simp only [← c13144, ← c13145, ← c13146] at e_204_1
  refine ⟨?_, ?_⟩
  · have hc := e_204_0
    simp only [e_203_11] at hc
    linear_combination c13150.trans k2 - hc
  · have hc := e_203_12
    simp only [e_203_10, e_204_1, e_203_11] at hc
    linear_combination c13151.trans k2 - hc

theorem publicBatchWrapper86_f1928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3808)) (a (.wire 165 39)) (a (.virt 18133)) (a (.virt 18134)) := by
  have c13152 := (publicBatchWrapper86_copies411 a h).1
  have c13153 := (publicBatchWrapper86_copies411 a h).2.1
  have c13154 := (publicBatchWrapper86_copies411 a h).2.2.1
  have c13155 := (publicBatchWrapper86_copies411 a h).2.2.2.1
  have c13156 := (publicBatchWrapper86_copies411 a h).2.2.2.2.1
  have c13157 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.1
  have c13158 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.1
  have c13159 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.1
  have c13160 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.1
  have c13161 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.1
  have c13162 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13163 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13164 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13165 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13166 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13167 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13168 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_203_13 := arithEq_of_rows h (row := 203) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_13
  simp only [← c13152, k3, ← c13153, k3, ← c13154] at e_203_13
  have e_203_14 := arithEq_of_rows h (row := 203) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_14
  simp only [← c13155, ← c13156, k3, ← c13157] at e_203_14
  have e_203_15 := arithEq_of_rows h (row := 203) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_15
  simp only [← c13164, ← c13165, k3, ← c13166] at e_203_15
  have e_204_2 := arithEq_of_rows h (row := 204) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_2
  simp only [← c13158, ← c13159, ← c13160] at e_204_2
  have e_204_3 := arithEq_of_rows h (row := 204) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_3
  simp only [← c13161, ← c13162, ← c13163] at e_204_3
  refine ⟨?_, ?_⟩
  · have hc := e_204_2
    simp only [e_203_14] at hc
    linear_combination c13167.trans k2 - hc
  · have hc := e_203_15
    simp only [e_203_13, e_204_3, e_203_14] at hc
    linear_combination c13168.trans k2 - hc

theorem publicBatchWrapper86_f1929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 204 19) = band (a (.virt 18127)) (a (.virt 18129)) := by
  have c13169 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13170 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13171 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_204_4 := arithEq_of_rows h (row := 204) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_4
  simp only [← c13169, ← c13170, ← c13171] at e_204_4
  have hr := e_204_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 204 23) = band (a (.virt 18131)) (a (.virt 18133)) := by
  have c13172 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13173 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13174 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_204_5 := arithEq_of_rows h (row := 204) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_5
  simp only [← c13172, ← c13173, ← c13174] at e_204_5
  have hr := e_204_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 204 27) = band (a (.wire 204 19)) (a (.wire 204 23)) := by
  have c13175 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13176 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13177 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_204_6 := arithEq_of_rows h (row := 204) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_6
  simp only [← c13175, ← c13176, ← c13177] at e_204_6
  have hr := e_204_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 51) = bor (a (.wire 20 63)) (a (.wire 204 27)) := by
  have c13178 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13179 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13180 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13181 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13182 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13183 := (publicBatchWrapper86_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_12 := arithEq_of_rows h (row := 197) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_12
  simp only [← c13181, ← c13182, k3, ← c13183] at e_197_12
  have e_202_4 := arithEq_of_rows h (row := 202) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_4
  simp only [← c13178, ← c13179, ← c13180] at e_202_4
  have hr := e_197_12
  simp only [e_202_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 51) = a (.virt 201) := by
  have c13184 := (publicBatchWrapper86_copies412 a h).1
  exact c13184

theorem publicBatchWrapper86_f1934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4003)) (a (.wire 165 55)) (a (.virt 18135)) (a (.virt 18136)) := by
  have c13185 := (publicBatchWrapper86_copies412 a h).2.1
  have c13186 := (publicBatchWrapper86_copies412 a h).2.2.1
  have c13187 := (publicBatchWrapper86_copies412 a h).2.2.2.1
  have c13188 := (publicBatchWrapper86_copies412 a h).2.2.2.2.1
  have c13189 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.1
  have c13190 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.1
  have c13191 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.1
  have c13192 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.1
  have c13193 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.1
  have c13194 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13195 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13196 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13197 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13198 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13199 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13200 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13201 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_203_16 := arithEq_of_rows h (row := 203) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_16
  simp only [← c13185, k3, ← c13186, k3, ← c13187] at e_203_16
  have e_203_17 := arithEq_of_rows h (row := 203) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_17
  simp only [← c13188, ← c13189, k3, ← c13190] at e_203_17
  have e_203_18 := arithEq_of_rows h (row := 203) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_18
  simp only [← c13197, ← c13198, k3, ← c13199] at e_203_18
  have e_204_7 := arithEq_of_rows h (row := 204) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_7
  simp only [← c13191, ← c13192, ← c13193] at e_204_7
  have e_204_8 := arithEq_of_rows h (row := 204) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_8
  simp only [← c13194, ← c13195, ← c13196] at e_204_8
  refine ⟨?_, ?_⟩
  · have hc := e_204_7
    simp only [e_203_17] at hc
    linear_combination c13200.trans k2 - hc
  · have hc := e_203_18
    simp only [e_203_16, e_204_8, e_203_17] at hc
    linear_combination c13201.trans k2 - hc

theorem publicBatchWrapper86_f1935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 55) = bor (a (.wire 22 27)) (a (.virt 18135)) := by
  have c13202 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13203 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13204 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13205 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13206 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13207 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_13 := arithEq_of_rows h (row := 197) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_13
  simp only [← c13205, ← c13206, k3, ← c13207] at e_197_13
  have e_202_5 := arithEq_of_rows h (row := 202) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_5
  simp only [← c13202, ← c13203, ← c13204] at e_202_5
  have hr := e_197_13
  simp only [e_202_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 55) = a (.virt 201) := by
  have c13208 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13208

theorem publicBatchWrapper86_f1937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4004)) (a (.wire 165 63)) (a (.virt 18137)) (a (.virt 18138)) := by
  have c13209 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13210 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13211 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13212 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13213 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13214 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13215 := (publicBatchWrapper86_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13216 := (publicBatchWrapper86_copies413 a h).1
  have c13217 := (publicBatchWrapper86_copies413 a h).2.1
  have c13218 := (publicBatchWrapper86_copies413 a h).2.2.1
  have c13219 := (publicBatchWrapper86_copies413 a h).2.2.2.1
  have c13220 := (publicBatchWrapper86_copies413 a h).2.2.2.2.1
  have c13221 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.1
  have c13222 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.1
  have c13223 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.1
  have c13224 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.1
  have c13225 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_203_19 := arithEq_of_rows h (row := 203) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_19
  simp only [← c13209, k3, ← c13210, k3, ← c13211] at e_203_19
  have e_204_9 := arithEq_of_rows h (row := 204) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_9
  simp only [← c13215, ← c13216, ← c13217] at e_204_9
  have e_204_10 := arithEq_of_rows h (row := 204) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_10
  simp only [← c13218, ← c13219, ← c13220] at e_204_10
  have e_205_0 := arithEq_of_rows h (row := 205) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_0
  simp only [← c13212, ← c13213, k3, ← c13214] at e_205_0
  have e_205_1 := arithEq_of_rows h (row := 205) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_1
  simp only [← c13221, ← c13222, k3, ← c13223] at e_205_1
  refine ⟨?_, ?_⟩
  · have hc := e_204_9
    simp only [e_205_0] at hc
    linear_combination c13224.trans k2 - hc
  · have hc := e_205_1
    simp only [e_203_19, e_204_10, e_205_0] at hc
    linear_combination c13225.trans k2 - hc

theorem publicBatchWrapper86_f1938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 59) = bor (a (.wire 22 27)) (a (.virt 18137)) := by
  have c13226 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13227 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13228 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13229 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13230 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13231 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_14 := arithEq_of_rows h (row := 197) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_14
  simp only [← c13229, ← c13230, k3, ← c13231] at e_197_14
  have e_202_6 := arithEq_of_rows h (row := 202) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_6
  simp only [← c13226, ← c13227, ← c13228] at e_202_6
  have hr := e_197_14
  simp only [e_202_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 59) = a (.virt 201) := by
  have c13232 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13232

theorem publicBatchWrapper86_f1940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4005)) (a (.wire 165 15)) (a (.virt 18139)) (a (.virt 18140)) := by
  have c13233 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13234 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13235 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13236 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13237 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13238 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13239 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13240 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13241 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13242 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13243 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13244 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13245 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13246 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13247 := (publicBatchWrapper86_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13248 := (publicBatchWrapper86_copies414 a h).1
  have c13249 := (publicBatchWrapper86_copies414 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_204_11 := arithEq_of_rows h (row := 204) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_11
  simp only [← c13239, ← c13240, ← c13241] at e_204_11
  have e_204_12 := arithEq_of_rows h (row := 204) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_12
  simp only [← c13242, ← c13243, ← c13244] at e_204_12
  have e_205_2 := arithEq_of_rows h (row := 205) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_2
  simp only [← c13233, k3, ← c13234, k3, ← c13235] at e_205_2
  have e_205_3 := arithEq_of_rows h (row := 205) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_3
  simp only [← c13236, ← c13237, k3, ← c13238] at e_205_3
  have e_205_4 := arithEq_of_rows h (row := 205) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_4
  simp only [← c13245, ← c13246, k3, ← c13247] at e_205_4
  refine ⟨?_, ?_⟩
  · have hc := e_204_11
    simp only [e_205_3] at hc
    linear_combination c13248.trans k2 - hc
  · have hc := e_205_4
    simp only [e_205_2, e_204_12, e_205_3] at hc
    linear_combination c13249.trans k2 - hc

theorem publicBatchWrapper86_f1941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4006)) (a (.wire 165 23)) (a (.virt 18141)) (a (.virt 18142)) := by
  have c13250 := (publicBatchWrapper86_copies414 a h).2.2.1
  have c13251 := (publicBatchWrapper86_copies414 a h).2.2.2.1
  have c13252 := (publicBatchWrapper86_copies414 a h).2.2.2.2.1
  have c13253 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.1
  have c13254 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.1
  have c13255 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.1
  have c13256 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.1
  have c13257 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.1
  have c13258 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13259 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13260 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13261 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13262 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13263 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13264 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13265 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13266 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_204_13 := arithEq_of_rows h (row := 204) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_13
  simp only [← c13256, ← c13257, ← c13258] at e_204_13
  have e_204_14 := arithEq_of_rows h (row := 204) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_14
  simp only [← c13259, ← c13260, ← c13261] at e_204_14
  have e_205_5 := arithEq_of_rows h (row := 205) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_5
  simp only [← c13250, k3, ← c13251, k3, ← c13252] at e_205_5
  have e_205_6 := arithEq_of_rows h (row := 205) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_6
  simp only [← c13253, ← c13254, k3, ← c13255] at e_205_6
  have e_205_7 := arithEq_of_rows h (row := 205) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_7
  simp only [← c13262, ← c13263, k3, ← c13264] at e_205_7
  refine ⟨?_, ?_⟩
  · have hc := e_204_13
    simp only [e_205_6] at hc
    linear_combination c13265.trans k2 - hc
  · have hc := e_205_7
    simp only [e_205_5, e_204_14, e_205_6] at hc
    linear_combination c13266.trans k2 - hc

theorem publicBatchWrapper86_f1942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4007)) (a (.wire 165 31)) (a (.virt 18143)) (a (.virt 18144)) := by
  have c13267 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13268 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13269 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13270 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13271 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13272 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13273 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13274 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13275 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13276 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13277 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13278 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13279 := (publicBatchWrapper86_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13280 := (publicBatchWrapper86_copies415 a h).1
  have c13281 := (publicBatchWrapper86_copies415 a h).2.1
  have c13282 := (publicBatchWrapper86_copies415 a h).2.2.1
  have c13283 := (publicBatchWrapper86_copies415 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_204_15 := arithEq_of_rows h (row := 204) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_15
  simp only [← c13273, ← c13274, ← c13275] at e_204_15
  have e_204_16 := arithEq_of_rows h (row := 204) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_16
  simp only [← c13276, ← c13277, ← c13278] at e_204_16
  have e_205_8 := arithEq_of_rows h (row := 205) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_8
  simp only [← c13267, k3, ← c13268, k3, ← c13269] at e_205_8
  have e_205_9 := arithEq_of_rows h (row := 205) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_9
  simp only [← c13270, ← c13271, k3, ← c13272] at e_205_9
  have e_205_10 := arithEq_of_rows h (row := 205) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_10
  simp only [← c13279, ← c13280, k3, ← c13281] at e_205_10
  refine ⟨?_, ?_⟩
  · have hc := e_204_15
    simp only [e_205_9] at hc
    linear_combination c13282.trans k2 - hc
  · have hc := e_205_10
    simp only [e_205_8, e_204_16, e_205_9] at hc
    linear_combination c13283.trans k2 - hc

theorem publicBatchWrapper86_f1943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4008)) (a (.wire 165 39)) (a (.virt 18145)) (a (.virt 18146)) := by
  have c13284 := (publicBatchWrapper86_copies415 a h).2.2.2.2.1
  have c13285 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.1
  have c13286 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.1
  have c13287 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.1
  have c13288 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.1
  have c13289 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.1
  have c13290 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13291 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13292 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13293 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13294 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13295 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13296 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13297 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13298 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13299 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13300 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_204_17 := arithEq_of_rows h (row := 204) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_17
  simp only [← c13290, ← c13291, ← c13292] at e_204_17
  have e_204_18 := arithEq_of_rows h (row := 204) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_18
  simp only [← c13293, ← c13294, ← c13295] at e_204_18
  have e_205_11 := arithEq_of_rows h (row := 205) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_11
  simp only [← c13284, k3, ← c13285, k3, ← c13286] at e_205_11
  have e_205_12 := arithEq_of_rows h (row := 205) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_12
  simp only [← c13287, ← c13288, k3, ← c13289] at e_205_12
  have e_205_13 := arithEq_of_rows h (row := 205) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_13
  simp only [← c13296, ← c13297, k3, ← c13298] at e_205_13
  refine ⟨?_, ?_⟩
  · have hc := e_204_17
    simp only [e_205_12] at hc
    linear_combination c13299.trans k2 - hc
  · have hc := e_205_13
    simp only [e_205_11, e_204_18, e_205_12] at hc
    linear_combination c13300.trans k2 - hc

theorem publicBatchWrapper86_f1944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 204 79) = band (a (.virt 18139)) (a (.virt 18141)) := by
  have c13301 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13302 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13303 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_204_19 := arithEq_of_rows h (row := 204) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_19
  simp only [← c13301, ← c13302, ← c13303] at e_204_19
  have hr := e_204_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 206 3) = band (a (.virt 18143)) (a (.virt 18145)) := by
  have c13304 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13305 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13306 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_206_0 := arithEq_of_rows h (row := 206) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_0
  simp only [← c13304, ← c13305, ← c13306] at e_206_0
  have hr := e_206_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 206 7) = band (a (.wire 204 79)) (a (.wire 206 3)) := by
  have c13307 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13308 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13309 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_206_1 := arithEq_of_rows h (row := 206) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_1
  simp only [← c13307, ← c13308, ← c13309] at e_206_1
  have hr := e_206_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 63) = bor (a (.wire 22 27)) (a (.wire 206 7)) := by
  have c13310 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13311 := (publicBatchWrapper86_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13312 := (publicBatchWrapper86_copies416 a h).1
  have c13313 := (publicBatchWrapper86_copies416 a h).2.1
  have c13314 := (publicBatchWrapper86_copies416 a h).2.2.1
  have c13315 := (publicBatchWrapper86_copies416 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_15 := arithEq_of_rows h (row := 197) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_15
  simp only [← c13313, ← c13314, k3, ← c13315] at e_197_15
  have e_202_7 := arithEq_of_rows h (row := 202) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_7
  simp only [← c13310, ← c13311, ← c13312] at e_202_7
  have hr := e_197_15
  simp only [e_202_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 63) = a (.virt 201) := by
  have c13316 := (publicBatchWrapper86_copies416 a h).2.2.2.2.1
  exact c13316

theorem publicBatchWrapper86_f1949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4203)) (a (.wire 165 55)) (a (.virt 18147)) (a (.virt 18148)) := by
  have c13317 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.1
  have c13318 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.1
  have c13319 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.1
  have c13320 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.1
  have c13321 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.1
  have c13322 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13323 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13324 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13325 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13326 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13327 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13328 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13329 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13330 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13331 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13332 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13333 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_205_14 := arithEq_of_rows h (row := 205) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_14
  simp only [← c13317, k3, ← c13318, k3, ← c13319] at e_205_14
  have e_205_15 := arithEq_of_rows h (row := 205) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_15
  simp only [← c13320, ← c13321, k3, ← c13322] at e_205_15
  have e_205_16 := arithEq_of_rows h (row := 205) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_16
  simp only [← c13329, ← c13330, k3, ← c13331] at e_205_16
  have e_206_2 := arithEq_of_rows h (row := 206) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_2
  simp only [← c13323, ← c13324, ← c13325] at e_206_2
  have e_206_3 := arithEq_of_rows h (row := 206) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_3
  simp only [← c13326, ← c13327, ← c13328] at e_206_3
  refine ⟨?_, ?_⟩
  · have hc := e_206_2
    simp only [e_205_15] at hc
    linear_combination c13332.trans k2 - hc
  · have hc := e_205_16
    simp only [e_205_14, e_206_3, e_205_15] at hc
    linear_combination c13333.trans k2 - hc

theorem publicBatchWrapper86_f1950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 67) = bor (a (.wire 22 71)) (a (.virt 18147)) := by
  have c13334 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13335 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13336 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13337 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13338 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13339 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_16 := arithEq_of_rows h (row := 197) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_16
  simp only [← c13337, ← c13338, k3, ← c13339] at e_197_16
  have e_202_8 := arithEq_of_rows h (row := 202) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_8
  simp only [← c13334, ← c13335, ← c13336] at e_202_8
  have hr := e_197_16
  simp only [e_202_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 67) = a (.virt 201) := by
  have c13340 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13340

theorem publicBatchWrapper86_f1952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4204)) (a (.wire 165 63)) (a (.virt 18149)) (a (.virt 18150)) := by
  have c13341 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13342 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13343 := (publicBatchWrapper86_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13344 := (publicBatchWrapper86_copies417 a h).1
  have c13345 := (publicBatchWrapper86_copies417 a h).2.1
  have c13346 := (publicBatchWrapper86_copies417 a h).2.2.1
  have c13347 := (publicBatchWrapper86_copies417 a h).2.2.2.1
  have c13348 := (publicBatchWrapper86_copies417 a h).2.2.2.2.1
  have c13349 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.1
  have c13350 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.1
  have c13351 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.1
  have c13352 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.1
  have c13353 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.1
  have c13354 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13355 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13356 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13357 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_205_17 := arithEq_of_rows h (row := 205) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_17
  simp only [← c13341, k3, ← c13342, k3, ← c13343] at e_205_17
  have e_205_18 := arithEq_of_rows h (row := 205) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_18
  simp only [← c13344, ← c13345, k3, ← c13346] at e_205_18
  have e_205_19 := arithEq_of_rows h (row := 205) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_19
  simp only [← c13353, ← c13354, k3, ← c13355] at e_205_19
  have e_206_4 := arithEq_of_rows h (row := 206) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_4
  simp only [← c13347, ← c13348, ← c13349] at e_206_4
  have e_206_5 := arithEq_of_rows h (row := 206) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_5
  simp only [← c13350, ← c13351, ← c13352] at e_206_5
  refine ⟨?_, ?_⟩
  · have hc := e_206_4
    simp only [e_205_18] at hc
    linear_combination c13356.trans k2 - hc
  · have hc := e_205_19
    simp only [e_205_17, e_206_5, e_205_18] at hc
    linear_combination c13357.trans k2 - hc

theorem publicBatchWrapper86_f1953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 71) = bor (a (.wire 22 71)) (a (.virt 18149)) := by
  have c13358 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13359 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13360 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13361 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13362 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13363 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_17 := arithEq_of_rows h (row := 197) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_17
  simp only [← c13361, ← c13362, k3, ← c13363] at e_197_17
  have e_202_9 := arithEq_of_rows h (row := 202) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_9
  simp only [← c13358, ← c13359, ← c13360] at e_202_9
  have hr := e_197_17
  simp only [e_202_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 71) = a (.virt 201) := by
  have c13364 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13364

theorem publicBatchWrapper86_f1955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4205)) (a (.wire 165 15)) (a (.virt 18151)) (a (.virt 18152)) := by
  have c13365 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13366 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13367 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13368 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13369 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13370 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13371 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13372 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13373 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13374 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13375 := (publicBatchWrapper86_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13376 := (publicBatchWrapper86_copies418 a h).1
  have c13377 := (publicBatchWrapper86_copies418 a h).2.1
  have c13378 := (publicBatchWrapper86_copies418 a h).2.2.1
  have c13379 := (publicBatchWrapper86_copies418 a h).2.2.2.1
  have c13380 := (publicBatchWrapper86_copies418 a h).2.2.2.2.1
  have c13381 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_206_6 := arithEq_of_rows h (row := 206) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_6
  simp only [← c13371, ← c13372, ← c13373] at e_206_6
  have e_206_7 := arithEq_of_rows h (row := 206) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_7
  simp only [← c13374, ← c13375, ← c13376] at e_206_7
  have e_207_0 := arithEq_of_rows h (row := 207) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_0
  simp only [← c13365, k3, ← c13366, k3, ← c13367] at e_207_0
  have e_207_1 := arithEq_of_rows h (row := 207) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_1
  simp only [← c13368, ← c13369, k3, ← c13370] at e_207_1
  have e_207_2 := arithEq_of_rows h (row := 207) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_2
  simp only [← c13377, ← c13378, k3, ← c13379] at e_207_2
  refine ⟨?_, ?_⟩
  · have hc := e_206_6
    simp only [e_207_1] at hc
    linear_combination c13380.trans k2 - hc
  · have hc := e_207_2
    simp only [e_207_0, e_206_7, e_207_1] at hc
    linear_combination c13381.trans k2 - hc

theorem publicBatchWrapper86_f1956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4206)) (a (.wire 165 23)) (a (.virt 18153)) (a (.virt 18154)) := by
  have c13382 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.1
  have c13383 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.1
  have c13384 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.1
  have c13385 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.1
  have c13386 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13387 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13388 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13389 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13390 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13391 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13392 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13393 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13394 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13395 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13396 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13397 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13398 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_206_8 := arithEq_of_rows h (row := 206) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_8
  simp only [← c13388, ← c13389, ← c13390] at e_206_8
  have e_206_9 := arithEq_of_rows h (row := 206) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_9
  simp only [← c13391, ← c13392, ← c13393] at e_206_9
  have e_207_3 := arithEq_of_rows h (row := 207) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_3
  simp only [← c13382, k3, ← c13383, k3, ← c13384] at e_207_3
  have e_207_4 := arithEq_of_rows h (row := 207) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_4
  simp only [← c13385, ← c13386, k3, ← c13387] at e_207_4
  have e_207_5 := arithEq_of_rows h (row := 207) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_5
  simp only [← c13394, ← c13395, k3, ← c13396] at e_207_5
  refine ⟨?_, ?_⟩
  · have hc := e_206_8
    simp only [e_207_4] at hc
    linear_combination c13397.trans k2 - hc
  · have hc := e_207_5
    simp only [e_207_3, e_206_9, e_207_4] at hc
    linear_combination c13398.trans k2 - hc

theorem publicBatchWrapper86_f1957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4207)) (a (.wire 165 31)) (a (.virt 18155)) (a (.virt 18156)) := by
  have c13399 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13400 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13401 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13402 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13403 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13404 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13405 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13406 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13407 := (publicBatchWrapper86_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13408 := (publicBatchWrapper86_copies419 a h).1
  have c13409 := (publicBatchWrapper86_copies419 a h).2.1
  have c13410 := (publicBatchWrapper86_copies419 a h).2.2.1
  have c13411 := (publicBatchWrapper86_copies419 a h).2.2.2.1
  have c13412 := (publicBatchWrapper86_copies419 a h).2.2.2.2.1
  have c13413 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.1
  have c13414 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.1
  have c13415 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_206_10 := arithEq_of_rows h (row := 206) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_10
  simp only [← c13405, ← c13406, ← c13407] at e_206_10
  have e_206_11 := arithEq_of_rows h (row := 206) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_11
  simp only [← c13408, ← c13409, ← c13410] at e_206_11
  have e_207_6 := arithEq_of_rows h (row := 207) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_6
  simp only [← c13399, k3, ← c13400, k3, ← c13401] at e_207_6
  have e_207_7 := arithEq_of_rows h (row := 207) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_7
  simp only [← c13402, ← c13403, k3, ← c13404] at e_207_7
  have e_207_8 := arithEq_of_rows h (row := 207) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_8
  simp only [← c13411, ← c13412, k3, ← c13413] at e_207_8
  refine ⟨?_, ?_⟩
  · have hc := e_206_10
    simp only [e_207_7] at hc
    linear_combination c13414.trans k2 - hc
  · have hc := e_207_8
    simp only [e_207_6, e_206_11, e_207_7] at hc
    linear_combination c13415.trans k2 - hc

theorem publicBatchWrapper86_f1958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4208)) (a (.wire 165 39)) (a (.virt 18157)) (a (.virt 18158)) := by
  have c13416 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.1
  have c13417 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.1
  have c13418 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13419 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13420 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13421 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13422 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13423 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13424 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13425 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13426 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13427 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13428 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13429 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13430 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13431 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13432 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_206_12 := arithEq_of_rows h (row := 206) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_12
  simp only [← c13422, ← c13423, ← c13424] at e_206_12
  have e_206_13 := arithEq_of_rows h (row := 206) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_13
  simp only [← c13425, ← c13426, ← c13427] at e_206_13
  have e_207_9 := arithEq_of_rows h (row := 207) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_9
  simp only [← c13416, k3, ← c13417, k3, ← c13418] at e_207_9
  have e_207_10 := arithEq_of_rows h (row := 207) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_10
  simp only [← c13419, ← c13420, k3, ← c13421] at e_207_10
  have e_207_11 := arithEq_of_rows h (row := 207) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_11
  simp only [← c13428, ← c13429, k3, ← c13430] at e_207_11
  refine ⟨?_, ?_⟩
  · have hc := e_206_12
    simp only [e_207_10] at hc
    linear_combination c13431.trans k2 - hc
  · have hc := e_207_11
    simp only [e_207_9, e_206_13, e_207_10] at hc
    linear_combination c13432.trans k2 - hc

theorem publicBatchWrapper86_f1959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 206 59) = band (a (.virt 18151)) (a (.virt 18153)) := by
  have c13433 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13434 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13435 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_206_14 := arithEq_of_rows h (row := 206) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_14
  simp only [← c13433, ← c13434, ← c13435] at e_206_14
  have hr := e_206_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 206 63) = band (a (.virt 18155)) (a (.virt 18157)) := by
  have c13436 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13437 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13438 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_206_15 := arithEq_of_rows h (row := 206) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_15
  simp only [← c13436, ← c13437, ← c13438] at e_206_15
  have hr := e_206_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 206 67) = band (a (.wire 206 59)) (a (.wire 206 63)) := by
  have c13439 := (publicBatchWrapper86_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13440 := (publicBatchWrapper86_copies420 a h).1
  have c13441 := (publicBatchWrapper86_copies420 a h).2.1
  have e_206_16 := arithEq_of_rows h (row := 206) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_16
  simp only [← c13439, ← c13440, ← c13441] at e_206_16
  have hr := e_206_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 75) = bor (a (.wire 22 71)) (a (.wire 206 67)) := by
  have c13442 := (publicBatchWrapper86_copies420 a h).2.2.1
  have c13443 := (publicBatchWrapper86_copies420 a h).2.2.2.1
  have c13444 := (publicBatchWrapper86_copies420 a h).2.2.2.2.1
  have c13445 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.1
  have c13446 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.1
  have c13447 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_18 := arithEq_of_rows h (row := 197) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_18
  simp only [← c13445, ← c13446, k3, ← c13447] at e_197_18
  have e_202_10 := arithEq_of_rows h (row := 202) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_10
  simp only [← c13442, ← c13443, ← c13444] at e_202_10
  have hr := e_197_18
  simp only [e_202_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 75) = a (.virt 201) := by
  have c13448 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.1
  exact c13448

theorem publicBatchWrapper86_f1964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4403)) (a (.wire 165 55)) (a (.virt 18159)) (a (.virt 18160)) := by
  have c13449 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.1
  have c13450 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13451 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13452 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13453 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13454 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13455 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13456 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13457 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13458 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13459 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13460 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13461 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13462 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13463 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13464 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13465 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_206_17 := arithEq_of_rows h (row := 206) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_17
  simp only [← c13455, ← c13456, ← c13457] at e_206_17
  have e_206_18 := arithEq_of_rows h (row := 206) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_18
  simp only [← c13458, ← c13459, ← c13460] at e_206_18
  have e_207_12 := arithEq_of_rows h (row := 207) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_12
  simp only [← c13449, k3, ← c13450, k3, ← c13451] at e_207_12
  have e_207_13 := arithEq_of_rows h (row := 207) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_13
  simp only [← c13452, ← c13453, k3, ← c13454] at e_207_13
  have e_207_14 := arithEq_of_rows h (row := 207) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_14
  simp only [← c13461, ← c13462, k3, ← c13463] at e_207_14
  refine ⟨?_, ?_⟩
  · have hc := e_206_17
    simp only [e_207_13] at hc
    linear_combination c13464.trans k2 - hc
  · have hc := e_207_14
    simp only [e_207_12, e_206_18, e_207_13] at hc
    linear_combination c13465.trans k2 - hc

theorem publicBatchWrapper86_f1965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 79) = bor (a (.wire 23 35)) (a (.virt 18159)) := by
  have c13466 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13467 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13468 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13469 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13470 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13471 := (publicBatchWrapper86_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_197_19 := arithEq_of_rows h (row := 197) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_19
  simp only [← c13469, ← c13470, k3, ← c13471] at e_197_19
  have e_202_11 := arithEq_of_rows h (row := 202) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_11
  simp only [← c13466, ← c13467, ← c13468] at e_202_11
  have hr := e_197_19
  simp only [e_202_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 197 79) = a (.virt 201) := by
  have c13472 := (publicBatchWrapper86_copies421 a h).1
  exact c13472

theorem publicBatchWrapper86_f1967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4404)) (a (.wire 165 63)) (a (.virt 18161)) (a (.virt 18162)) := by
  have c13473 := (publicBatchWrapper86_copies421 a h).2.1
  have c13474 := (publicBatchWrapper86_copies421 a h).2.2.1
  have c13475 := (publicBatchWrapper86_copies421 a h).2.2.2.1
  have c13476 := (publicBatchWrapper86_copies421 a h).2.2.2.2.1
  have c13477 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.1
  have c13478 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.1
  have c13479 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.1
  have c13480 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.1
  have c13481 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.1
  have c13482 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13483 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13484 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13485 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13486 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13487 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13488 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13489 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_206_19 := arithEq_of_rows h (row := 206) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_19
  simp only [← c13479, ← c13480, ← c13481] at e_206_19
  have e_207_15 := arithEq_of_rows h (row := 207) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_15
  simp only [← c13473, k3, ← c13474, k3, ← c13475] at e_207_15
  have e_207_16 := arithEq_of_rows h (row := 207) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_16
  simp only [← c13476, ← c13477, k3, ← c13478] at e_207_16
  have e_207_17 := arithEq_of_rows h (row := 207) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_17
  simp only [← c13485, ← c13486, k3, ← c13487] at e_207_17
  have e_208_0 := arithEq_of_rows h (row := 208) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_0
  simp only [← c13482, ← c13483, ← c13484] at e_208_0
  refine ⟨?_, ?_⟩
  · have hc := e_206_19
    simp only [e_207_16] at hc
    linear_combination c13488.trans k2 - hc
  · have hc := e_207_17
    simp only [e_207_15, e_208_0, e_207_16] at hc
    linear_combination c13489.trans k2 - hc

theorem publicBatchWrapper86_f1968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 3) = bor (a (.wire 23 35)) (a (.virt 18161)) := by
  have c13490 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13491 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13492 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13493 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13494 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13495 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_202_12 := arithEq_of_rows h (row := 202) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_12
  simp only [← c13490, ← c13491, ← c13492] at e_202_12
  have e_209_0 := arithEq_of_rows h (row := 209) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_0
  simp only [← c13493, ← c13494, k3, ← c13495] at e_209_0
  have hr := e_209_0
  simp only [e_202_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 3) = a (.virt 201) := by
  have c13496 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13496

theorem publicBatchWrapper86_f1970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4405)) (a (.wire 165 15)) (a (.virt 18163)) (a (.virt 18164)) := by
  have c13497 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13498 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13499 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13500 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13501 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13502 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13503 := (publicBatchWrapper86_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13504 := (publicBatchWrapper86_copies422 a h).1
  have c13505 := (publicBatchWrapper86_copies422 a h).2.1
  have c13506 := (publicBatchWrapper86_copies422 a h).2.2.1
  have c13507 := (publicBatchWrapper86_copies422 a h).2.2.2.1
  have c13508 := (publicBatchWrapper86_copies422 a h).2.2.2.2.1
  have c13509 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.1
  have c13510 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.1
  have c13511 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.1
  have c13512 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.1
  have c13513 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_207_18 := arithEq_of_rows h (row := 207) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_18
  simp only [← c13497, k3, ← c13498, k3, ← c13499] at e_207_18
  have e_207_19 := arithEq_of_rows h (row := 207) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_19
  simp only [← c13500, ← c13501, k3, ← c13502] at e_207_19
  have e_208_1 := arithEq_of_rows h (row := 208) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_1
  simp only [← c13503, ← c13504, ← c13505] at e_208_1
  have e_208_2 := arithEq_of_rows h (row := 208) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_2
  simp only [← c13506, ← c13507, ← c13508] at e_208_2
  have e_210_0 := arithEq_of_rows h (row := 210) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_0
  simp only [← c13509, ← c13510, k3, ← c13511] at e_210_0
  refine ⟨?_, ?_⟩
  · have hc := e_208_1
    simp only [e_207_19] at hc
    linear_combination c13512.trans k2 - hc
  · have hc := e_210_0
    simp only [e_207_18, e_208_2, e_207_19] at hc
    linear_combination c13513.trans k2 - hc

theorem publicBatchWrapper86_f1971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4406)) (a (.wire 165 23)) (a (.virt 18165)) (a (.virt 18166)) := by
  have c13514 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13515 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13516 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13517 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13518 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13519 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13520 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13521 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13522 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13523 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13524 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13525 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13526 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13527 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13528 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13529 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13530 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_208_3 := arithEq_of_rows h (row := 208) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_3
  simp only [← c13520, ← c13521, ← c13522] at e_208_3
  have e_208_4 := arithEq_of_rows h (row := 208) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_4
  simp only [← c13523, ← c13524, ← c13525] at e_208_4
  have e_210_1 := arithEq_of_rows h (row := 210) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_1
  simp only [← c13514, k3, ← c13515, k3, ← c13516] at e_210_1
  have e_210_2 := arithEq_of_rows h (row := 210) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_2
  simp only [← c13517, ← c13518, k3, ← c13519] at e_210_2
  have e_210_3 := arithEq_of_rows h (row := 210) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_3
  simp only [← c13526, ← c13527, k3, ← c13528] at e_210_3
  refine ⟨?_, ?_⟩
  · have hc := e_208_3
    simp only [e_210_2] at hc
    linear_combination c13529.trans k2 - hc
  · have hc := e_210_3
    simp only [e_210_1, e_208_4, e_210_2] at hc
    linear_combination c13530.trans k2 - hc

theorem publicBatchWrapper86_f1972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4407)) (a (.wire 165 31)) (a (.virt 18167)) (a (.virt 18168)) := by
  have c13531 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13532 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13533 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13534 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13535 := (publicBatchWrapper86_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13536 := (publicBatchWrapper86_copies423 a h).1
  have c13537 := (publicBatchWrapper86_copies423 a h).2.1
  have c13538 := (publicBatchWrapper86_copies423 a h).2.2.1
  have c13539 := (publicBatchWrapper86_copies423 a h).2.2.2.1
  have c13540 := (publicBatchWrapper86_copies423 a h).2.2.2.2.1
  have c13541 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.1
  have c13542 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.1
  have c13543 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.1
  have c13544 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.1
  have c13545 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.1
  have c13546 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13547 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_208_5 := arithEq_of_rows h (row := 208) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_5
  simp only [← c13537, ← c13538, ← c13539] at e_208_5
  have e_208_6 := arithEq_of_rows h (row := 208) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_6
  simp only [← c13540, ← c13541, ← c13542] at e_208_6
  have e_210_4 := arithEq_of_rows h (row := 210) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_4
  simp only [← c13531, k3, ← c13532, k3, ← c13533] at e_210_4
  have e_210_5 := arithEq_of_rows h (row := 210) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_5
  simp only [← c13534, ← c13535, k3, ← c13536] at e_210_5
  have e_210_6 := arithEq_of_rows h (row := 210) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_6
  simp only [← c13543, ← c13544, k3, ← c13545] at e_210_6
  refine ⟨?_, ?_⟩
  · have hc := e_208_5
    simp only [e_210_5] at hc
    linear_combination c13546.trans k2 - hc
  · have hc := e_210_6
    simp only [e_210_4, e_208_6, e_210_5] at hc
    linear_combination c13547.trans k2 - hc

theorem publicBatchWrapper86_f1973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4408)) (a (.wire 165 39)) (a (.virt 18169)) (a (.virt 18170)) := by
  have c13548 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13549 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13550 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13551 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13552 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13553 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13554 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13555 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13556 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13557 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13558 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13559 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13560 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13561 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13562 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13563 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13564 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_208_7 := arithEq_of_rows h (row := 208) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_7
  simp only [← c13554, ← c13555, ← c13556] at e_208_7
  have e_208_8 := arithEq_of_rows h (row := 208) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_8
  simp only [← c13557, ← c13558, ← c13559] at e_208_8
  have e_210_7 := arithEq_of_rows h (row := 210) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_7
  simp only [← c13548, k3, ← c13549, k3, ← c13550] at e_210_7
  have e_210_8 := arithEq_of_rows h (row := 210) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_8
  simp only [← c13551, ← c13552, k3, ← c13553] at e_210_8
  have e_210_9 := arithEq_of_rows h (row := 210) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_9
  simp only [← c13560, ← c13561, k3, ← c13562] at e_210_9
  refine ⟨?_, ?_⟩
  · have hc := e_208_7
    simp only [e_210_8] at hc
    linear_combination c13563.trans k2 - hc
  · have hc := e_210_9
    simp only [e_210_7, e_208_8, e_210_8] at hc
    linear_combination c13564.trans k2 - hc

theorem publicBatchWrapper86_f1974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 208 39) = band (a (.virt 18163)) (a (.virt 18165)) := by
  have c13565 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13566 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13567 := (publicBatchWrapper86_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_208_9 := arithEq_of_rows h (row := 208) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_9
  simp only [← c13565, ← c13566, ← c13567] at e_208_9
  have hr := e_208_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 208 43) = band (a (.virt 18167)) (a (.virt 18169)) := by
  have c13568 := (publicBatchWrapper86_copies424 a h).1
  have c13569 := (publicBatchWrapper86_copies424 a h).2.1
  have c13570 := (publicBatchWrapper86_copies424 a h).2.2.1
  have e_208_10 := arithEq_of_rows h (row := 208) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_10
  simp only [← c13568, ← c13569, ← c13570] at e_208_10
  have hr := e_208_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 208 47) = band (a (.wire 208 39)) (a (.wire 208 43)) := by
  have c13571 := (publicBatchWrapper86_copies424 a h).2.2.2.1
  have c13572 := (publicBatchWrapper86_copies424 a h).2.2.2.2.1
  have c13573 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.1
  have e_208_11 := arithEq_of_rows h (row := 208) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_11
  simp only [← c13571, ← c13572, ← c13573] at e_208_11
  have hr := e_208_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 7) = bor (a (.wire 23 35)) (a (.wire 208 47)) := by
  have c13574 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.1
  have c13575 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.1
  have c13576 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.1
  have c13577 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.1
  have c13578 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13579 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_202_13 := arithEq_of_rows h (row := 202) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_13
  simp only [← c13574, ← c13575, ← c13576] at e_202_13
  have e_209_1 := arithEq_of_rows h (row := 209) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_1
  simp only [← c13577, ← c13578, k3, ← c13579] at e_209_1
  have hr := e_209_1
  simp only [e_202_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 7) = a (.virt 201) := by
  have c13580 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13580

theorem publicBatchWrapper86_f1979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4603)) (a (.wire 165 55)) (a (.virt 18171)) (a (.virt 18172)) := by
  have c13581 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13582 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13583 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13584 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13585 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13586 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13587 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13588 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13589 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13590 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13591 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13592 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13593 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13594 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13595 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13596 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13597 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_208_12 := arithEq_of_rows h (row := 208) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_12
  simp only [← c13587, ← c13588, ← c13589] at e_208_12
  have e_208_13 := arithEq_of_rows h (row := 208) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_13
  simp only [← c13590, ← c13591, ← c13592] at e_208_13
  have e_210_10 := arithEq_of_rows h (row := 210) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_10
  simp only [← c13581, k3, ← c13582, k3, ← c13583] at e_210_10
  have e_210_11 := arithEq_of_rows h (row := 210) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_11
  simp only [← c13584, ← c13585, k3, ← c13586] at e_210_11
  have e_210_12 := arithEq_of_rows h (row := 210) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_12
  simp only [← c13593, ← c13594, k3, ← c13595] at e_210_12
  refine ⟨?_, ?_⟩
  · have hc := e_208_12
    simp only [e_210_11] at hc
    linear_combination c13596.trans k2 - hc
  · have hc := e_210_12
    simp only [e_210_10, e_208_13, e_210_11] at hc
    linear_combination c13597.trans k2 - hc

theorem publicBatchWrapper86_f1980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 11) = bor (a (.wire 23 79)) (a (.virt 18171)) := by
  have c13598 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13599 := (publicBatchWrapper86_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13600 := (publicBatchWrapper86_copies425 a h).1
  have c13601 := (publicBatchWrapper86_copies425 a h).2.1
  have c13602 := (publicBatchWrapper86_copies425 a h).2.2.1
  have c13603 := (publicBatchWrapper86_copies425 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_202_14 := arithEq_of_rows h (row := 202) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_14
  simp only [← c13598, ← c13599, ← c13600] at e_202_14
  have e_209_2 := arithEq_of_rows h (row := 209) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_2
  simp only [← c13601, ← c13602, k3, ← c13603] at e_209_2
  have hr := e_209_2
  simp only [e_202_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 11) = a (.virt 201) := by
  have c13604 := (publicBatchWrapper86_copies425 a h).2.2.2.2.1
  exact c13604

theorem publicBatchWrapper86_f1982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4604)) (a (.wire 165 63)) (a (.virt 18173)) (a (.virt 18174)) := by
  have c13605 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.1
  have c13606 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.1
  have c13607 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.1
  have c13608 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.1
  have c13609 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.1
  have c13610 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13611 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13612 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13613 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13614 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13615 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13616 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13617 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13618 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13619 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13620 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13621 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_208_14 := arithEq_of_rows h (row := 208) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_14
  simp only [← c13611, ← c13612, ← c13613] at e_208_14
  have e_208_15 := arithEq_of_rows h (row := 208) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_15
  simp only [← c13614, ← c13615, ← c13616] at e_208_15
  have e_210_13 := arithEq_of_rows h (row := 210) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_13
  simp only [← c13605, k3, ← c13606, k3, ← c13607] at e_210_13
  have e_210_14 := arithEq_of_rows h (row := 210) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_14
  simp only [← c13608, ← c13609, k3, ← c13610] at e_210_14
  have e_210_15 := arithEq_of_rows h (row := 210) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_15
  simp only [← c13617, ← c13618, k3, ← c13619] at e_210_15
  refine ⟨?_, ?_⟩
  · have hc := e_208_14
    simp only [e_210_14] at hc
    linear_combination c13620.trans k2 - hc
  · have hc := e_210_15
    simp only [e_210_13, e_208_15, e_210_14] at hc
    linear_combination c13621.trans k2 - hc

theorem publicBatchWrapper86_f1983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 15) = bor (a (.wire 23 79)) (a (.virt 18173)) := by
  have c13622 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13623 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13624 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13625 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13626 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13627 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_202_15 := arithEq_of_rows h (row := 202) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_15
  simp only [← c13622, ← c13623, ← c13624] at e_202_15
  have e_209_3 := arithEq_of_rows h (row := 209) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_3
  simp only [← c13625, ← c13626, k3, ← c13627] at e_209_3
  have hr := e_209_3
  simp only [e_202_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 15) = a (.virt 201) := by
  have c13628 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13628

theorem publicBatchWrapper86_f1985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4605)) (a (.wire 165 15)) (a (.virt 18175)) (a (.virt 18176)) := by
  have c13629 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13630 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13631 := (publicBatchWrapper86_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13632 := (publicBatchWrapper86_copies426 a h).1
  have c13633 := (publicBatchWrapper86_copies426 a h).2.1
  have c13634 := (publicBatchWrapper86_copies426 a h).2.2.1
  have c13635 := (publicBatchWrapper86_copies426 a h).2.2.2.1
  have c13636 := (publicBatchWrapper86_copies426 a h).2.2.2.2.1
  have c13637 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.1
  have c13638 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.1
  have c13639 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.1
  have c13640 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.1
  have c13641 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.1
  have c13642 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13643 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13644 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13645 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_208_16 := arithEq_of_rows h (row := 208) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_16
  simp only [← c13635, ← c13636, ← c13637] at e_208_16
  have e_208_17 := arithEq_of_rows h (row := 208) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_17
  simp only [← c13638, ← c13639, ← c13640] at e_208_17
  have e_210_16 := arithEq_of_rows h (row := 210) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_16
  simp only [← c13629, k3, ← c13630, k3, ← c13631] at e_210_16
  have e_210_17 := arithEq_of_rows h (row := 210) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_17
  simp only [← c13632, ← c13633, k3, ← c13634] at e_210_17
  have e_210_18 := arithEq_of_rows h (row := 210) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_18
  simp only [← c13641, ← c13642, k3, ← c13643] at e_210_18
  refine ⟨?_, ?_⟩
  · have hc := e_208_16
    simp only [e_210_17] at hc
    linear_combination c13644.trans k2 - hc
  · have hc := e_210_18
    simp only [e_210_16, e_208_17, e_210_17] at hc
    linear_combination c13645.trans k2 - hc

theorem publicBatchWrapper86_f1986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4606)) (a (.wire 165 23)) (a (.virt 18177)) (a (.virt 18178)) := by
  have c13646 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13647 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13648 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13649 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13650 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13651 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13652 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13653 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13654 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13655 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13656 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13657 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13658 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13659 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13660 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13661 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13662 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_208_18 := arithEq_of_rows h (row := 208) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_18
  simp only [← c13652, ← c13653, ← c13654] at e_208_18
  have e_208_19 := arithEq_of_rows h (row := 208) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_19
  simp only [← c13655, ← c13656, ← c13657] at e_208_19
  have e_210_19 := arithEq_of_rows h (row := 210) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_19
  simp only [← c13646, k3, ← c13647, k3, ← c13648] at e_210_19
  have e_211_0 := arithEq_of_rows h (row := 211) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_0
  simp only [← c13649, ← c13650, k3, ← c13651] at e_211_0
  have e_211_1 := arithEq_of_rows h (row := 211) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_1
  simp only [← c13658, ← c13659, k3, ← c13660] at e_211_1
  refine ⟨?_, ?_⟩
  · have hc := e_208_18
    simp only [e_211_0] at hc
    linear_combination c13661.trans k2 - hc
  · have hc := e_211_1
    simp only [e_210_19, e_208_19, e_211_0] at hc
    linear_combination c13662.trans k2 - hc

theorem publicBatchWrapper86_f1987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4607)) (a (.wire 165 31)) (a (.virt 18179)) (a (.virt 18180)) := by
  have c13663 := (publicBatchWrapper86_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13664 := (publicBatchWrapper86_copies427 a h).1
  have c13665 := (publicBatchWrapper86_copies427 a h).2.1
  have c13666 := (publicBatchWrapper86_copies427 a h).2.2.1
  have c13667 := (publicBatchWrapper86_copies427 a h).2.2.2.1
  have c13668 := (publicBatchWrapper86_copies427 a h).2.2.2.2.1
  have c13669 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.1
  have c13670 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.1
  have c13671 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.1
  have c13672 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.1
  have c13673 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.1
  have c13674 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13675 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13676 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13677 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13678 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13679 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_211_2 := arithEq_of_rows h (row := 211) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_2
  simp only [← c13663, k3, ← c13664, k3, ← c13665] at e_211_2
  have e_211_3 := arithEq_of_rows h (row := 211) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_3
  simp only [← c13666, ← c13667, k3, ← c13668] at e_211_3
  have e_211_4 := arithEq_of_rows h (row := 211) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_4
  simp only [← c13675, ← c13676, k3, ← c13677] at e_211_4
  have e_212_0 := arithEq_of_rows h (row := 212) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_0
  simp only [← c13669, ← c13670, ← c13671] at e_212_0
  have e_212_1 := arithEq_of_rows h (row := 212) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_1
  simp only [← c13672, ← c13673, ← c13674] at e_212_1
  refine ⟨?_, ?_⟩
  · have hc := e_212_0
    simp only [e_211_3] at hc
    linear_combination c13678.trans k2 - hc
  · have hc := e_211_4
    simp only [e_211_2, e_212_1, e_211_3] at hc
    linear_combination c13679.trans k2 - hc

theorem publicBatchWrapper86_f1988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4608)) (a (.wire 165 39)) (a (.virt 18181)) (a (.virt 18182)) := by
  have c13680 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13681 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13682 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13683 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13684 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13685 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13686 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13687 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13688 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13689 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13690 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13691 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13692 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13693 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13694 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13695 := (publicBatchWrapper86_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13696 := (publicBatchWrapper86_copies428 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_211_5 := arithEq_of_rows h (row := 211) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_5
  simp only [← c13680, k3, ← c13681, k3, ← c13682] at e_211_5
  have e_211_6 := arithEq_of_rows h (row := 211) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_6
  simp only [← c13683, ← c13684, k3, ← c13685] at e_211_6
  have e_211_7 := arithEq_of_rows h (row := 211) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_7
  simp only [← c13692, ← c13693, k3, ← c13694] at e_211_7
  have e_212_2 := arithEq_of_rows h (row := 212) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_2
  simp only [← c13686, ← c13687, ← c13688] at e_212_2
  have e_212_3 := arithEq_of_rows h (row := 212) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_3
  simp only [← c13689, ← c13690, ← c13691] at e_212_3
  refine ⟨?_, ?_⟩
  · have hc := e_212_2
    simp only [e_211_6] at hc
    linear_combination c13695.trans k2 - hc
  · have hc := e_211_7
    simp only [e_211_5, e_212_3, e_211_6] at hc
    linear_combination c13696.trans k2 - hc

theorem publicBatchWrapper86_f1989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 212 19) = band (a (.virt 18175)) (a (.virt 18177)) := by
  have c13697 := (publicBatchWrapper86_copies428 a h).2.1
  have c13698 := (publicBatchWrapper86_copies428 a h).2.2.1
  have c13699 := (publicBatchWrapper86_copies428 a h).2.2.2.1
  have e_212_4 := arithEq_of_rows h (row := 212) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_4
  simp only [← c13697, ← c13698, ← c13699] at e_212_4
  have hr := e_212_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 212 23) = band (a (.virt 18179)) (a (.virt 18181)) := by
  have c13700 := (publicBatchWrapper86_copies428 a h).2.2.2.2.1
  have c13701 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.1
  have c13702 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.1
  have e_212_5 := arithEq_of_rows h (row := 212) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_5
  simp only [← c13700, ← c13701, ← c13702] at e_212_5
  have hr := e_212_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 212 27) = band (a (.wire 212 19)) (a (.wire 212 23)) := by
  have c13703 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.1
  have c13704 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.1
  have c13705 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.1
  have e_212_6 := arithEq_of_rows h (row := 212) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_6
  simp only [← c13703, ← c13704, ← c13705] at e_212_6
  have hr := e_212_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 19) = bor (a (.wire 23 79)) (a (.wire 212 27)) := by
  have c13706 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13707 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13708 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13709 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13710 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13711 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_202_16 := arithEq_of_rows h (row := 202) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_16
  simp only [← c13706, ← c13707, ← c13708] at e_202_16
  have e_209_4 := arithEq_of_rows h (row := 209) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_4
  simp only [← c13709, ← c13710, k3, ← c13711] at e_209_4
  have hr := e_209_4
  simp only [e_202_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 19) = a (.virt 201) := by
  have c13712 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13712

theorem publicBatchWrapper86_f1994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4803)) (a (.wire 165 55)) (a (.virt 18183)) (a (.virt 18184)) := by
  have c13713 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13714 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13715 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13716 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13717 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13718 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13719 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13720 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13721 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13722 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13723 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13724 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13725 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13726 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13727 := (publicBatchWrapper86_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13728 := (publicBatchWrapper86_copies429 a h).1
  have c13729 := (publicBatchWrapper86_copies429 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_211_8 := arithEq_of_rows h (row := 211) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_8
  simp only [← c13713, k3, ← c13714, k3, ← c13715] at e_211_8
  have e_211_9 := arithEq_of_rows h (row := 211) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_9
  simp only [← c13716, ← c13717, k3, ← c13718] at e_211_9
  have e_211_10 := arithEq_of_rows h (row := 211) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_10
  simp only [← c13725, ← c13726, k3, ← c13727] at e_211_10
  have e_212_7 := arithEq_of_rows h (row := 212) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_7
  simp only [← c13719, ← c13720, ← c13721] at e_212_7
  have e_212_8 := arithEq_of_rows h (row := 212) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_8
  simp only [← c13722, ← c13723, ← c13724] at e_212_8
  refine ⟨?_, ?_⟩
  · have hc := e_212_7
    simp only [e_211_9] at hc
    linear_combination c13728.trans k2 - hc
  · have hc := e_211_10
    simp only [e_211_8, e_212_8, e_211_9] at hc
    linear_combination c13729.trans k2 - hc

theorem publicBatchWrapper86_f1995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 23) = bor (a (.wire 25 43)) (a (.virt 18183)) := by
  have c13730 := (publicBatchWrapper86_copies429 a h).2.2.1
  have c13731 := (publicBatchWrapper86_copies429 a h).2.2.2.1
  have c13732 := (publicBatchWrapper86_copies429 a h).2.2.2.2.1
  have c13733 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.1
  have c13734 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.1
  have c13735 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_202_17 := arithEq_of_rows h (row := 202) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_17
  simp only [← c13730, ← c13731, ← c13732] at e_202_17
  have e_209_5 := arithEq_of_rows h (row := 209) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_5
  simp only [← c13733, ← c13734, k3, ← c13735] at e_209_5
  have hr := e_209_5
  simp only [e_202_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 23) = a (.virt 201) := by
  have c13736 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.1
  exact c13736

theorem publicBatchWrapper86_f1997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4804)) (a (.wire 165 63)) (a (.virt 18185)) (a (.virt 18186)) := by
  have c13737 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.1
  have c13738 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13739 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13740 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13741 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13742 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13743 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13744 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13745 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13746 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13747 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13748 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13749 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13750 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13751 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13752 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13753 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_211_11 := arithEq_of_rows h (row := 211) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_11
  simp only [← c13737, k3, ← c13738, k3, ← c13739] at e_211_11
  have e_211_12 := arithEq_of_rows h (row := 211) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_12
  simp only [← c13740, ← c13741, k3, ← c13742] at e_211_12
  have e_211_13 := arithEq_of_rows h (row := 211) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_13
  simp only [← c13749, ← c13750, k3, ← c13751] at e_211_13
  have e_212_9 := arithEq_of_rows h (row := 212) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_9
  simp only [← c13743, ← c13744, ← c13745] at e_212_9
  have e_212_10 := arithEq_of_rows h (row := 212) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_10
  simp only [← c13746, ← c13747, ← c13748] at e_212_10
  refine ⟨?_, ?_⟩
  · have hc := e_212_9
    simp only [e_211_12] at hc
    linear_combination c13752.trans k2 - hc
  · have hc := e_211_13
    simp only [e_211_11, e_212_10, e_211_12] at hc
    linear_combination c13753.trans k2 - hc

theorem publicBatchWrapper86_f1998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 27) = bor (a (.wire 25 43)) (a (.virt 18185)) := by
  have c13754 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13755 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13756 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13757 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13758 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13759 := (publicBatchWrapper86_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_202_18 := arithEq_of_rows h (row := 202) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_18
  simp only [← c13754, ← c13755, ← c13756] at e_202_18
  have e_209_6 := arithEq_of_rows h (row := 209) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_6
  simp only [← c13757, ← c13758, k3, ← c13759] at e_209_6
  have hr := e_209_6
  simp only [e_202_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 27) = a (.virt 201) := by
  have c13760 := (publicBatchWrapper86_copies430 a h).1
  exact c13760

theorem publicBatchWrapper86_f2000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4805)) (a (.wire 165 15)) (a (.virt 18187)) (a (.virt 18188)) := by
  have c13761 := (publicBatchWrapper86_copies430 a h).2.1
  have c13762 := (publicBatchWrapper86_copies430 a h).2.2.1
  have c13763 := (publicBatchWrapper86_copies430 a h).2.2.2.1
  have c13764 := (publicBatchWrapper86_copies430 a h).2.2.2.2.1
  have c13765 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.1
  have c13766 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.1
  have c13767 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.1
  have c13768 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.1
  have c13769 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.1
  have c13770 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13771 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13772 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13773 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13774 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13775 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13776 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13777 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_211_14 := arithEq_of_rows h (row := 211) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_14
  simp only [← c13761, k3, ← c13762, k3, ← c13763] at e_211_14
  have e_211_15 := arithEq_of_rows h (row := 211) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_15
  simp only [← c13764, ← c13765, k3, ← c13766] at e_211_15
  have e_211_16 := arithEq_of_rows h (row := 211) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_16
  simp only [← c13773, ← c13774, k3, ← c13775] at e_211_16
  have e_212_11 := arithEq_of_rows h (row := 212) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_11
  simp only [← c13767, ← c13768, ← c13769] at e_212_11
  have e_212_12 := arithEq_of_rows h (row := 212) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_12
  simp only [← c13770, ← c13771, ← c13772] at e_212_12
  refine ⟨?_, ?_⟩
  · have hc := e_212_11
    simp only [e_211_15] at hc
    linear_combination c13776.trans k2 - hc
  · have hc := e_211_16
    simp only [e_211_14, e_212_12, e_211_15] at hc
    linear_combination c13777.trans k2 - hc

theorem publicBatchWrapper86_f2001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4806)) (a (.wire 165 23)) (a (.virt 18189)) (a (.virt 18190)) := by
  have c13778 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13779 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13780 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13781 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13782 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13783 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13784 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13785 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13786 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13787 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13788 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13789 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13790 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13791 := (publicBatchWrapper86_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13792 := (publicBatchWrapper86_copies431 a h).1
  have c13793 := (publicBatchWrapper86_copies431 a h).2.1
  have c13794 := (publicBatchWrapper86_copies431 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_211_17 := arithEq_of_rows h (row := 211) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_17
  simp only [← c13778, k3, ← c13779, k3, ← c13780] at e_211_17
  have e_211_18 := arithEq_of_rows h (row := 211) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_18
  simp only [← c13781, ← c13782, k3, ← c13783] at e_211_18
  have e_211_19 := arithEq_of_rows h (row := 211) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_19
  simp only [← c13790, ← c13791, k3, ← c13792] at e_211_19
  have e_212_13 := arithEq_of_rows h (row := 212) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_13
  simp only [← c13784, ← c13785, ← c13786] at e_212_13
  have e_212_14 := arithEq_of_rows h (row := 212) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_14
  simp only [← c13787, ← c13788, ← c13789] at e_212_14
  refine ⟨?_, ?_⟩
  · have hc := e_212_13
    simp only [e_211_18] at hc
    linear_combination c13793.trans k2 - hc
  · have hc := e_211_19
    simp only [e_211_17, e_212_14, e_211_18] at hc
    linear_combination c13794.trans k2 - hc

theorem publicBatchWrapper86_f2002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4807)) (a (.wire 165 31)) (a (.virt 18191)) (a (.virt 18192)) := by
  have c13795 := (publicBatchWrapper86_copies431 a h).2.2.2.1
  have c13796 := (publicBatchWrapper86_copies431 a h).2.2.2.2.1
  have c13797 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.1
  have c13798 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.1
  have c13799 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.1
  have c13800 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.1
  have c13801 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.1
  have c13802 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13803 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13804 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13805 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13806 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13807 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13808 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13809 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13810 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13811 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_212_15 := arithEq_of_rows h (row := 212) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_15
  simp only [← c13801, ← c13802, ← c13803] at e_212_15
  have e_212_16 := arithEq_of_rows h (row := 212) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_16
  simp only [← c13804, ← c13805, ← c13806] at e_212_16
  have e_213_0 := arithEq_of_rows h (row := 213) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_0
  simp only [← c13795, k3, ← c13796, k3, ← c13797] at e_213_0
  have e_213_1 := arithEq_of_rows h (row := 213) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_1
  simp only [← c13798, ← c13799, k3, ← c13800] at e_213_1
  have e_213_2 := arithEq_of_rows h (row := 213) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_2
  simp only [← c13807, ← c13808, k3, ← c13809] at e_213_2
  refine ⟨?_, ?_⟩
  · have hc := e_212_15
    simp only [e_213_1] at hc
    linear_combination c13810.trans k2 - hc
  · have hc := e_213_2
    simp only [e_213_0, e_212_16, e_213_1] at hc
    linear_combination c13811.trans k2 - hc

theorem publicBatchWrapper86_f2003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4808)) (a (.wire 165 39)) (a (.virt 18193)) (a (.virt 18194)) := by
  have c13812 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13813 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13814 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13815 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13816 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13817 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13818 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13819 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13820 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13821 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13822 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13823 := (publicBatchWrapper86_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13824 := (publicBatchWrapper86_copies432 a h).1
  have c13825 := (publicBatchWrapper86_copies432 a h).2.1
  have c13826 := (publicBatchWrapper86_copies432 a h).2.2.1
  have c13827 := (publicBatchWrapper86_copies432 a h).2.2.2.1
  have c13828 := (publicBatchWrapper86_copies432 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_212_17 := arithEq_of_rows h (row := 212) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_17
  simp only [← c13818, ← c13819, ← c13820] at e_212_17
  have e_212_18 := arithEq_of_rows h (row := 212) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_18
  simp only [← c13821, ← c13822, ← c13823] at e_212_18
  have e_213_3 := arithEq_of_rows h (row := 213) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_3
  simp only [← c13812, k3, ← c13813, k3, ← c13814] at e_213_3
  have e_213_4 := arithEq_of_rows h (row := 213) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_4
  simp only [← c13815, ← c13816, k3, ← c13817] at e_213_4
  have e_213_5 := arithEq_of_rows h (row := 213) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_5
  simp only [← c13824, ← c13825, k3, ← c13826] at e_213_5
  refine ⟨?_, ?_⟩
  · have hc := e_212_17
    simp only [e_213_4] at hc
    linear_combination c13827.trans k2 - hc
  · have hc := e_213_5
    simp only [e_213_3, e_212_18, e_213_4] at hc
    linear_combination c13828.trans k2 - hc

theorem publicBatchWrapper86_f2004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 212 79) = band (a (.virt 18187)) (a (.virt 18189)) := by
  have c13829 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.1
  have c13830 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.1
  have c13831 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.1
  have e_212_19 := arithEq_of_rows h (row := 212) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_19
  simp only [← c13829, ← c13830, ← c13831] at e_212_19
  have hr := e_212_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 214 3) = band (a (.virt 18191)) (a (.virt 18193)) := by
  have c13832 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.1
  have c13833 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.1
  have c13834 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_214_0 := arithEq_of_rows h (row := 214) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_0
  simp only [← c13832, ← c13833, ← c13834] at e_214_0
  have hr := e_214_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 214 7) = band (a (.wire 212 79)) (a (.wire 214 3)) := by
  have c13835 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13836 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13837 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_214_1 := arithEq_of_rows h (row := 214) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_1
  simp only [← c13835, ← c13836, ← c13837] at e_214_1
  have hr := e_214_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 31) = bor (a (.wire 25 43)) (a (.wire 214 7)) := by
  have c13838 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13839 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13840 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13841 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13842 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13843 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_202_19 := arithEq_of_rows h (row := 202) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_19
  simp only [← c13838, ← c13839, ← c13840] at e_202_19
  have e_209_7 := arithEq_of_rows h (row := 209) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_7
  simp only [← c13841, ← c13842, k3, ← c13843] at e_209_7
  have hr := e_209_7
  simp only [e_202_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 31) = a (.virt 201) := by
  have c13844 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13844

theorem publicBatchWrapper86_f2009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5003)) (a (.wire 165 55)) (a (.virt 18195)) (a (.virt 18196)) := by
  have c13845 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13846 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13847 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13848 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13849 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13850 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13851 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13852 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13853 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13854 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13855 := (publicBatchWrapper86_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13856 := (publicBatchWrapper86_copies433 a h).1
  have c13857 := (publicBatchWrapper86_copies433 a h).2.1
  have c13858 := (publicBatchWrapper86_copies433 a h).2.2.1
  have c13859 := (publicBatchWrapper86_copies433 a h).2.2.2.1
  have c13860 := (publicBatchWrapper86_copies433 a h).2.2.2.2.1
  have c13861 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_213_6 := arithEq_of_rows h (row := 213) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_6
  simp only [← c13845, k3, ← c13846, k3, ← c13847] at e_213_6
  have e_213_7 := arithEq_of_rows h (row := 213) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_7
  simp only [← c13848, ← c13849, k3, ← c13850] at e_213_7
  have e_213_8 := arithEq_of_rows h (row := 213) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_8
  simp only [← c13857, ← c13858, k3, ← c13859] at e_213_8
  have e_214_2 := arithEq_of_rows h (row := 214) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_2
  simp only [← c13851, ← c13852, ← c13853] at e_214_2
  have e_214_3 := arithEq_of_rows h (row := 214) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_3
  simp only [← c13854, ← c13855, ← c13856] at e_214_3
  refine ⟨?_, ?_⟩
  · have hc := e_214_2
    simp only [e_213_7] at hc
    linear_combination c13860.trans k2 - hc
  · have hc := e_213_8
    simp only [e_213_6, e_214_3, e_213_7] at hc
    linear_combination c13861.trans k2 - hc

theorem publicBatchWrapper86_f2010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 35) = bor (a (.wire 27 7)) (a (.virt 18195)) := by
  have c13862 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.1
  have c13863 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.1
  have c13864 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.1
  have c13865 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.1
  have c13866 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13867 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_8 := arithEq_of_rows h (row := 209) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_8
  simp only [← c13865, ← c13866, k3, ← c13867] at e_209_8
  have e_215_0 := arithEq_of_rows h (row := 215) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_0
  simp only [← c13862, ← c13863, ← c13864] at e_215_0
  have hr := e_209_8
  simp only [e_215_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 35) = a (.virt 201) := by
  have c13868 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13868

theorem publicBatchWrapper86_f2012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5004)) (a (.wire 165 63)) (a (.virt 18197)) (a (.virt 18198)) := by
  have c13869 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13870 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13871 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13872 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13873 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13874 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13875 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13876 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13877 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13878 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13879 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13880 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13881 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13882 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13883 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13884 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13885 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_213_9 := arithEq_of_rows h (row := 213) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_9
  simp only [← c13869, k3, ← c13870, k3, ← c13871] at e_213_9
  have e_213_10 := arithEq_of_rows h (row := 213) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_10
  simp only [← c13872, ← c13873, k3, ← c13874] at e_213_10
  have e_213_11 := arithEq_of_rows h (row := 213) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_11
  simp only [← c13881, ← c13882, k3, ← c13883] at e_213_11
  have e_214_4 := arithEq_of_rows h (row := 214) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_4
  simp only [← c13875, ← c13876, ← c13877] at e_214_4
  have e_214_5 := arithEq_of_rows h (row := 214) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_5
  simp only [← c13878, ← c13879, ← c13880] at e_214_5
  refine ⟨?_, ?_⟩
  · have hc := e_214_4
    simp only [e_213_10] at hc
    linear_combination c13884.trans k2 - hc
  · have hc := e_213_11
    simp only [e_213_9, e_214_5, e_213_10] at hc
    linear_combination c13885.trans k2 - hc

theorem publicBatchWrapper86_f2013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 39) = bor (a (.wire 27 7)) (a (.virt 18197)) := by
  have c13886 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13887 := (publicBatchWrapper86_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13888 := (publicBatchWrapper86_copies434 a h).1
  have c13889 := (publicBatchWrapper86_copies434 a h).2.1
  have c13890 := (publicBatchWrapper86_copies434 a h).2.2.1
  have c13891 := (publicBatchWrapper86_copies434 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_9 := arithEq_of_rows h (row := 209) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_9
  simp only [← c13889, ← c13890, k3, ← c13891] at e_209_9
  have e_215_1 := arithEq_of_rows h (row := 215) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_1
  simp only [← c13886, ← c13887, ← c13888] at e_215_1
  have hr := e_209_9
  simp only [e_215_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 39) = a (.virt 201) := by
  have c13892 := (publicBatchWrapper86_copies434 a h).2.2.2.2.1
  exact c13892

theorem publicBatchWrapper86_f2015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5005)) (a (.wire 165 15)) (a (.virt 18199)) (a (.virt 18200)) := by
  have c13893 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.1
  have c13894 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.1
  have c13895 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.1
  have c13896 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.1
  have c13897 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.1
  have c13898 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13899 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13900 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13901 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13902 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13903 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13904 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13905 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13906 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13907 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13908 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13909 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_213_12 := arithEq_of_rows h (row := 213) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_12
  simp only [← c13893, k3, ← c13894, k3, ← c13895] at e_213_12
  have e_213_13 := arithEq_of_rows h (row := 213) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_13
  simp only [← c13896, ← c13897, k3, ← c13898] at e_213_13
  have e_213_14 := arithEq_of_rows h (row := 213) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_14
  simp only [← c13905, ← c13906, k3, ← c13907] at e_213_14
  have e_214_6 := arithEq_of_rows h (row := 214) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_6
  simp only [← c13899, ← c13900, ← c13901] at e_214_6
  have e_214_7 := arithEq_of_rows h (row := 214) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_7
  simp only [← c13902, ← c13903, ← c13904] at e_214_7
  refine ⟨?_, ?_⟩
  · have hc := e_214_6
    simp only [e_213_13] at hc
    linear_combination c13908.trans k2 - hc
  · have hc := e_213_14
    simp only [e_213_12, e_214_7, e_213_13] at hc
    linear_combination c13909.trans k2 - hc

theorem publicBatchWrapper86_f2016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5006)) (a (.wire 165 23)) (a (.virt 18201)) (a (.virt 18202)) := by
  have c13910 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13911 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13912 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13913 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13914 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13915 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13916 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13917 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13918 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13919 := (publicBatchWrapper86_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13920 := (publicBatchWrapper86_copies435 a h).1
  have c13921 := (publicBatchWrapper86_copies435 a h).2.1
  have c13922 := (publicBatchWrapper86_copies435 a h).2.2.1
  have c13923 := (publicBatchWrapper86_copies435 a h).2.2.2.1
  have c13924 := (publicBatchWrapper86_copies435 a h).2.2.2.2.1
  have c13925 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.1
  have c13926 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_213_15 := arithEq_of_rows h (row := 213) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_15
  simp only [← c13910, k3, ← c13911, k3, ← c13912] at e_213_15
  have e_213_16 := arithEq_of_rows h (row := 213) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_16
  simp only [← c13913, ← c13914, k3, ← c13915] at e_213_16
  have e_213_17 := arithEq_of_rows h (row := 213) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_17
  simp only [← c13922, ← c13923, k3, ← c13924] at e_213_17
  have e_214_8 := arithEq_of_rows h (row := 214) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_8
  simp only [← c13916, ← c13917, ← c13918] at e_214_8
  have e_214_9 := arithEq_of_rows h (row := 214) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_9
  simp only [← c13919, ← c13920, ← c13921] at e_214_9
  refine ⟨?_, ?_⟩
  · have hc := e_214_8
    simp only [e_213_16] at hc
    linear_combination c13925.trans k2 - hc
  · have hc := e_213_17
    simp only [e_213_15, e_214_9, e_213_16] at hc
    linear_combination c13926.trans k2 - hc

theorem publicBatchWrapper86_f2017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5007)) (a (.wire 165 31)) (a (.virt 18203)) (a (.virt 18204)) := by
  have c13927 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.1
  have c13928 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.1
  have c13929 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.1
  have c13930 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13931 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13932 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13933 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13934 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13935 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13936 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13937 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13938 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13939 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13940 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13941 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13942 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13943 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_213_18 := arithEq_of_rows h (row := 213) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_18
  simp only [← c13927, k3, ← c13928, k3, ← c13929] at e_213_18
  have e_213_19 := arithEq_of_rows h (row := 213) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_19
  simp only [← c13930, ← c13931, k3, ← c13932] at e_213_19
  have e_214_10 := arithEq_of_rows h (row := 214) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_10
  simp only [← c13933, ← c13934, ← c13935] at e_214_10
  have e_214_11 := arithEq_of_rows h (row := 214) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_11
  simp only [← c13936, ← c13937, ← c13938] at e_214_11
  have e_216_0 := arithEq_of_rows h (row := 216) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_0
  simp only [← c13939, ← c13940, k3, ← c13941] at e_216_0
  refine ⟨?_, ?_⟩
  · have hc := e_214_10
    simp only [e_213_19] at hc
    linear_combination c13942.trans k2 - hc
  · have hc := e_216_0
    simp only [e_213_18, e_214_11, e_213_19] at hc
    linear_combination c13943.trans k2 - hc

theorem publicBatchWrapper86_f2018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5008)) (a (.wire 165 39)) (a (.virt 18205)) (a (.virt 18206)) := by
  have c13944 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13945 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13946 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13947 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13948 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13949 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13950 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13951 := (publicBatchWrapper86_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13952 := (publicBatchWrapper86_copies436 a h).1
  have c13953 := (publicBatchWrapper86_copies436 a h).2.1
  have c13954 := (publicBatchWrapper86_copies436 a h).2.2.1
  have c13955 := (publicBatchWrapper86_copies436 a h).2.2.2.1
  have c13956 := (publicBatchWrapper86_copies436 a h).2.2.2.2.1
  have c13957 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.1
  have c13958 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.1
  have c13959 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.1
  have c13960 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_214_12 := arithEq_of_rows h (row := 214) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_12
  simp only [← c13950, ← c13951, ← c13952] at e_214_12
  have e_214_13 := arithEq_of_rows h (row := 214) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_13
  simp only [← c13953, ← c13954, ← c13955] at e_214_13
  have e_216_1 := arithEq_of_rows h (row := 216) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_1
  simp only [← c13944, k3, ← c13945, k3, ← c13946] at e_216_1
  have e_216_2 := arithEq_of_rows h (row := 216) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_2
  simp only [← c13947, ← c13948, k3, ← c13949] at e_216_2
  have e_216_3 := arithEq_of_rows h (row := 216) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_3
  simp only [← c13956, ← c13957, k3, ← c13958] at e_216_3
  refine ⟨?_, ?_⟩
  · have hc := e_214_12
    simp only [e_216_2] at hc
    linear_combination c13959.trans k2 - hc
  · have hc := e_216_3
    simp only [e_216_1, e_214_13, e_216_2] at hc
    linear_combination c13960.trans k2 - hc

theorem publicBatchWrapper86_f2019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 214 59) = band (a (.virt 18199)) (a (.virt 18201)) := by
  have c13961 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.1
  have c13962 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13963 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_214_14 := arithEq_of_rows h (row := 214) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_14
  simp only [← c13961, ← c13962, ← c13963] at e_214_14
  have hr := e_214_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 214 63) = band (a (.virt 18203)) (a (.virt 18205)) := by
  have c13964 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13965 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13966 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_214_15 := arithEq_of_rows h (row := 214) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_15
  simp only [← c13964, ← c13965, ← c13966] at e_214_15
  have hr := e_214_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 214 67) = band (a (.wire 214 59)) (a (.wire 214 63)) := by
  have c13967 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13968 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13969 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_214_16 := arithEq_of_rows h (row := 214) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_16
  simp only [← c13967, ← c13968, ← c13969] at e_214_16
  have hr := e_214_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 43) = bor (a (.wire 27 7)) (a (.wire 214 67)) := by
  have c13970 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13971 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13972 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13973 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13974 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13975 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_10 := arithEq_of_rows h (row := 209) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_10
  simp only [← c13973, ← c13974, k3, ← c13975] at e_209_10
  have e_215_2 := arithEq_of_rows h (row := 215) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_2
  simp only [← c13970, ← c13971, ← c13972] at e_215_2
  have hr := e_209_10
  simp only [e_215_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 43) = a (.virt 201) := by
  have c13976 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13976

theorem publicBatchWrapper86_f2024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5203)) (a (.wire 165 55)) (a (.virt 18207)) (a (.virt 18208)) := by
  have c13977 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13978 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13979 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13980 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13981 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13982 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13983 := (publicBatchWrapper86_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13984 := (publicBatchWrapper86_copies437 a h).1
  have c13985 := (publicBatchWrapper86_copies437 a h).2.1
  have c13986 := (publicBatchWrapper86_copies437 a h).2.2.1
  have c13987 := (publicBatchWrapper86_copies437 a h).2.2.2.1
  have c13988 := (publicBatchWrapper86_copies437 a h).2.2.2.2.1
  have c13989 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.1
  have c13990 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.1
  have c13991 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.1
  have c13992 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.1
  have c13993 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_214_17 := arithEq_of_rows h (row := 214) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_17
  simp only [← c13983, ← c13984, ← c13985] at e_214_17
  have e_214_18 := arithEq_of_rows h (row := 214) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_18
  simp only [← c13986, ← c13987, ← c13988] at e_214_18
  have e_216_4 := arithEq_of_rows h (row := 216) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_4
  simp only [← c13977, k3, ← c13978, k3, ← c13979] at e_216_4
  have e_216_5 := arithEq_of_rows h (row := 216) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_5
  simp only [← c13980, ← c13981, k3, ← c13982] at e_216_5
  have e_216_6 := arithEq_of_rows h (row := 216) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_6
  simp only [← c13989, ← c13990, k3, ← c13991] at e_216_6
  refine ⟨?_, ?_⟩
  · have hc := e_214_17
    simp only [e_216_5] at hc
    linear_combination c13992.trans k2 - hc
  · have hc := e_216_6
    simp only [e_216_4, e_214_18, e_216_5] at hc
    linear_combination c13993.trans k2 - hc

theorem publicBatchWrapper86_f2025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 47) = bor (a (.wire 27 51)) (a (.virt 18207)) := by
  have c13994 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13995 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13996 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13997 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13998 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13999 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_11 := arithEq_of_rows h (row := 209) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_11
  simp only [← c13997, ← c13998, k3, ← c13999] at e_209_11
  have e_215_3 := arithEq_of_rows h (row := 215) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_3
  simp only [← c13994, ← c13995, ← c13996] at e_215_3
  have hr := e_209_11
  simp only [e_215_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 47) = a (.virt 201) := by
  have c14000 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14000

theorem publicBatchWrapper86_f2027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5204)) (a (.wire 165 63)) (a (.virt 18209)) (a (.virt 18210)) := by
  have c14001 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14002 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14003 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14004 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14005 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14006 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14007 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14008 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14009 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14010 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14011 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14012 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14013 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14014 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14015 := (publicBatchWrapper86_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14016 := (publicBatchWrapper86_copies438 a h).1
  have c14017 := (publicBatchWrapper86_copies438 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_214_19 := arithEq_of_rows h (row := 214) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_19
  simp only [← c14007, ← c14008, ← c14009] at e_214_19
  have e_216_7 := arithEq_of_rows h (row := 216) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_7
  simp only [← c14001, k3, ← c14002, k3, ← c14003] at e_216_7
  have e_216_8 := arithEq_of_rows h (row := 216) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_8
  simp only [← c14004, ← c14005, k3, ← c14006] at e_216_8
  have e_216_9 := arithEq_of_rows h (row := 216) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_9
  simp only [← c14013, ← c14014, k3, ← c14015] at e_216_9
  have e_217_0 := arithEq_of_rows h (row := 217) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_0
  simp only [← c14010, ← c14011, ← c14012] at e_217_0
  refine ⟨?_, ?_⟩
  · have hc := e_214_19
    simp only [e_216_8] at hc
    linear_combination c14016.trans k2 - hc
  · have hc := e_216_9
    simp only [e_216_7, e_217_0, e_216_8] at hc
    linear_combination c14017.trans k2 - hc

theorem publicBatchWrapper86_f2028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 51) = bor (a (.wire 27 51)) (a (.virt 18209)) := by
  have c14018 := (publicBatchWrapper86_copies438 a h).2.2.1
  have c14019 := (publicBatchWrapper86_copies438 a h).2.2.2.1
  have c14020 := (publicBatchWrapper86_copies438 a h).2.2.2.2.1
  have c14021 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.1
  have c14022 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.1
  have c14023 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_12 := arithEq_of_rows h (row := 209) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_12
  simp only [← c14021, ← c14022, k3, ← c14023] at e_209_12
  have e_215_4 := arithEq_of_rows h (row := 215) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_4
  simp only [← c14018, ← c14019, ← c14020] at e_215_4
  have hr := e_209_12
  simp only [e_215_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 51) = a (.virt 201) := by
  have c14024 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.1
  exact c14024

theorem publicBatchWrapper86_f2030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5205)) (a (.wire 165 15)) (a (.virt 18211)) (a (.virt 18212)) := by
  have c14025 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.1
  have c14026 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14027 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14028 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14029 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14030 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14031 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14032 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14033 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14034 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14035 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14036 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14037 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14038 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14039 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14040 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14041 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_216_10 := arithEq_of_rows h (row := 216) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_10
  simp only [← c14025, k3, ← c14026, k3, ← c14027] at e_216_10
  have e_216_11 := arithEq_of_rows h (row := 216) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_11
  simp only [← c14028, ← c14029, k3, ← c14030] at e_216_11
  have e_216_12 := arithEq_of_rows h (row := 216) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_12
  simp only [← c14037, ← c14038, k3, ← c14039] at e_216_12
  have e_217_1 := arithEq_of_rows h (row := 217) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_1
  simp only [← c14031, ← c14032, ← c14033] at e_217_1
  have e_217_2 := arithEq_of_rows h (row := 217) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_2
  simp only [← c14034, ← c14035, ← c14036] at e_217_2
  refine ⟨?_, ?_⟩
  · have hc := e_217_1
    simp only [e_216_11] at hc
    linear_combination c14040.trans k2 - hc
  · have hc := e_216_12
    simp only [e_216_10, e_217_2, e_216_11] at hc
    linear_combination c14041.trans k2 - hc

theorem publicBatchWrapper86_f2031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5206)) (a (.wire 165 23)) (a (.virt 18213)) (a (.virt 18214)) := by
  have c14042 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14043 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14044 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14045 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14046 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14047 := (publicBatchWrapper86_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14048 := (publicBatchWrapper86_copies439 a h).1
  have c14049 := (publicBatchWrapper86_copies439 a h).2.1
  have c14050 := (publicBatchWrapper86_copies439 a h).2.2.1
  have c14051 := (publicBatchWrapper86_copies439 a h).2.2.2.1
  have c14052 := (publicBatchWrapper86_copies439 a h).2.2.2.2.1
  have c14053 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.1
  have c14054 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.1
  have c14055 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.1
  have c14056 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.1
  have c14057 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.1
  have c14058 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_216_13 := arithEq_of_rows h (row := 216) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_13
  simp only [← c14042, k3, ← c14043, k3, ← c14044] at e_216_13
  have e_216_14 := arithEq_of_rows h (row := 216) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_14
  simp only [← c14045, ← c14046, k3, ← c14047] at e_216_14
  have e_216_15 := arithEq_of_rows h (row := 216) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_15
  simp only [← c14054, ← c14055, k3, ← c14056] at e_216_15
  have e_217_3 := arithEq_of_rows h (row := 217) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_3
  simp only [← c14048, ← c14049, ← c14050] at e_217_3
  have e_217_4 := arithEq_of_rows h (row := 217) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_4
  simp only [← c14051, ← c14052, ← c14053] at e_217_4
  refine ⟨?_, ?_⟩
  · have hc := e_217_3
    simp only [e_216_14] at hc
    linear_combination c14057.trans k2 - hc
  · have hc := e_216_15
    simp only [e_216_13, e_217_4, e_216_14] at hc
    linear_combination c14058.trans k2 - hc

theorem publicBatchWrapper86_f2032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5207)) (a (.wire 165 31)) (a (.virt 18215)) (a (.virt 18216)) := by
  have c14059 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14060 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14061 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14062 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14063 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14064 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14065 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14066 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14067 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14068 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14069 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14070 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14071 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14072 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14073 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14074 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14075 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_216_16 := arithEq_of_rows h (row := 216) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_16
  simp only [← c14059, k3, ← c14060, k3, ← c14061] at e_216_16
  have e_216_17 := arithEq_of_rows h (row := 216) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_17
  simp only [← c14062, ← c14063, k3, ← c14064] at e_216_17
  have e_216_18 := arithEq_of_rows h (row := 216) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_18
  simp only [← c14071, ← c14072, k3, ← c14073] at e_216_18
  have e_217_5 := arithEq_of_rows h (row := 217) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_5
  simp only [← c14065, ← c14066, ← c14067] at e_217_5
  have e_217_6 := arithEq_of_rows h (row := 217) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_6
  simp only [← c14068, ← c14069, ← c14070] at e_217_6
  refine ⟨?_, ?_⟩
  · have hc := e_217_5
    simp only [e_216_17] at hc
    linear_combination c14074.trans k2 - hc
  · have hc := e_216_18
    simp only [e_216_16, e_217_6, e_216_17] at hc
    linear_combination c14075.trans k2 - hc

theorem publicBatchWrapper86_f2033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5208)) (a (.wire 165 39)) (a (.virt 18217)) (a (.virt 18218)) := by
  have c14076 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14077 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14078 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14079 := (publicBatchWrapper86_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14080 := (publicBatchWrapper86_copies440 a h).1
  have c14081 := (publicBatchWrapper86_copies440 a h).2.1
  have c14082 := (publicBatchWrapper86_copies440 a h).2.2.1
  have c14083 := (publicBatchWrapper86_copies440 a h).2.2.2.1
  have c14084 := (publicBatchWrapper86_copies440 a h).2.2.2.2.1
  have c14085 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.1
  have c14086 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.1
  have c14087 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.1
  have c14088 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.1
  have c14089 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.1
  have c14090 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14091 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14092 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_216_19 := arithEq_of_rows h (row := 216) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_19
  simp only [← c14076, k3, ← c14077, k3, ← c14078] at e_216_19
  have e_217_7 := arithEq_of_rows h (row := 217) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_7
  simp only [← c14082, ← c14083, ← c14084] at e_217_7
  have e_217_8 := arithEq_of_rows h (row := 217) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_8
  simp only [← c14085, ← c14086, ← c14087] at e_217_8
  have e_218_0 := arithEq_of_rows h (row := 218) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_0
  simp only [← c14079, ← c14080, k3, ← c14081] at e_218_0
  have e_218_1 := arithEq_of_rows h (row := 218) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_1
  simp only [← c14088, ← c14089, k3, ← c14090] at e_218_1
  refine ⟨?_, ?_⟩
  · have hc := e_217_7
    simp only [e_218_0] at hc
    linear_combination c14091.trans k2 - hc
  · have hc := e_218_1
    simp only [e_216_19, e_217_8, e_218_0] at hc
    linear_combination c14092.trans k2 - hc

theorem publicBatchWrapper86_f2034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 217 39) = band (a (.virt 18211)) (a (.virt 18213)) := by
  have c14093 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14094 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14095 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_217_9 := arithEq_of_rows h (row := 217) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_9
  simp only [← c14093, ← c14094, ← c14095] at e_217_9
  have hr := e_217_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 217 43) = band (a (.virt 18215)) (a (.virt 18217)) := by
  have c14096 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14097 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14098 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_217_10 := arithEq_of_rows h (row := 217) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_10
  simp only [← c14096, ← c14097, ← c14098] at e_217_10
  have hr := e_217_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 217 47) = band (a (.wire 217 39)) (a (.wire 217 43)) := by
  have c14099 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14100 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14101 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_217_11 := arithEq_of_rows h (row := 217) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_11
  simp only [← c14099, ← c14100, ← c14101] at e_217_11
  have hr := e_217_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 55) = bor (a (.wire 27 51)) (a (.wire 217 47)) := by
  have c14102 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14103 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14104 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14105 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14106 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14107 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_13 := arithEq_of_rows h (row := 209) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_13
  simp only [← c14105, ← c14106, k3, ← c14107] at e_209_13
  have e_215_5 := arithEq_of_rows h (row := 215) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_5
  simp only [← c14102, ← c14103, ← c14104] at e_215_5
  have hr := e_209_13
  simp only [e_215_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 55) = a (.virt 201) := by
  have c14108 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14108

theorem publicBatchWrapper86_f2039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5403)) (a (.wire 165 55)) (a (.virt 18219)) (a (.virt 18220)) := by
  have c14109 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14110 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14111 := (publicBatchWrapper86_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14112 := (publicBatchWrapper86_copies441 a h).1
  have c14113 := (publicBatchWrapper86_copies441 a h).2.1
  have c14114 := (publicBatchWrapper86_copies441 a h).2.2.1
  have c14115 := (publicBatchWrapper86_copies441 a h).2.2.2.1
  have c14116 := (publicBatchWrapper86_copies441 a h).2.2.2.2.1
  have c14117 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.1
  have c14118 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.1
  have c14119 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.1
  have c14120 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.1
  have c14121 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.1
  have c14122 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14123 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14124 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14125 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_217_12 := arithEq_of_rows h (row := 217) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_12
  simp only [← c14115, ← c14116, ← c14117] at e_217_12
  have e_217_13 := arithEq_of_rows h (row := 217) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_13
  simp only [← c14118, ← c14119, ← c14120] at e_217_13
  have e_218_2 := arithEq_of_rows h (row := 218) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_2
  simp only [← c14109, k3, ← c14110, k3, ← c14111] at e_218_2
  have e_218_3 := arithEq_of_rows h (row := 218) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_3
  simp only [← c14112, ← c14113, k3, ← c14114] at e_218_3
  have e_218_4 := arithEq_of_rows h (row := 218) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_4
  simp only [← c14121, ← c14122, k3, ← c14123] at e_218_4
  refine ⟨?_, ?_⟩
  · have hc := e_217_12
    simp only [e_218_3] at hc
    linear_combination c14124.trans k2 - hc
  · have hc := e_218_4
    simp only [e_218_2, e_217_13, e_218_3] at hc
    linear_combination c14125.trans k2 - hc

theorem publicBatchWrapper86_f2040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 59) = bor (a (.wire 29 15)) (a (.virt 18219)) := by
  have c14126 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14127 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14128 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14129 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14130 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14131 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_14 := arithEq_of_rows h (row := 209) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_14
  simp only [← c14129, ← c14130, k3, ← c14131] at e_209_14
  have e_215_6 := arithEq_of_rows h (row := 215) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_6
  simp only [← c14126, ← c14127, ← c14128] at e_215_6
  have hr := e_209_14
  simp only [e_215_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 59) = a (.virt 201) := by
  have c14132 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14132

theorem publicBatchWrapper86_f2042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5404)) (a (.wire 165 63)) (a (.virt 18221)) (a (.virt 18222)) := by
  have c14133 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14134 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14135 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14136 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14137 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14138 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14139 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14140 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14141 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14142 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14143 := (publicBatchWrapper86_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14144 := (publicBatchWrapper86_copies442 a h).1
  have c14145 := (publicBatchWrapper86_copies442 a h).2.1
  have c14146 := (publicBatchWrapper86_copies442 a h).2.2.1
  have c14147 := (publicBatchWrapper86_copies442 a h).2.2.2.1
  have c14148 := (publicBatchWrapper86_copies442 a h).2.2.2.2.1
  have c14149 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_217_14 := arithEq_of_rows h (row := 217) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_14
  simp only [← c14139, ← c14140, ← c14141] at e_217_14
  have e_217_15 := arithEq_of_rows h (row := 217) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_15
  simp only [← c14142, ← c14143, ← c14144] at e_217_15
  have e_218_5 := arithEq_of_rows h (row := 218) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_5
  simp only [← c14133, k3, ← c14134, k3, ← c14135] at e_218_5
  have e_218_6 := arithEq_of_rows h (row := 218) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_6
  simp only [← c14136, ← c14137, k3, ← c14138] at e_218_6
  have e_218_7 := arithEq_of_rows h (row := 218) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_7
  simp only [← c14145, ← c14146, k3, ← c14147] at e_218_7
  refine ⟨?_, ?_⟩
  · have hc := e_217_14
    simp only [e_218_6] at hc
    linear_combination c14148.trans k2 - hc
  · have hc := e_218_7
    simp only [e_218_5, e_217_15, e_218_6] at hc
    linear_combination c14149.trans k2 - hc

theorem publicBatchWrapper86_f2043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 63) = bor (a (.wire 29 15)) (a (.virt 18221)) := by
  have c14150 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.1
  have c14151 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.1
  have c14152 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.1
  have c14153 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.1
  have c14154 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14155 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_15 := arithEq_of_rows h (row := 209) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_15
  simp only [← c14153, ← c14154, k3, ← c14155] at e_209_15
  have e_215_7 := arithEq_of_rows h (row := 215) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_7
  simp only [← c14150, ← c14151, ← c14152] at e_215_7
  have hr := e_209_15
  simp only [e_215_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 63) = a (.virt 201) := by
  have c14156 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14156

theorem publicBatchWrapper86_f2045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5405)) (a (.wire 165 15)) (a (.virt 18223)) (a (.virt 18224)) := by
  have c14157 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14158 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14159 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14160 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14161 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14162 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14163 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14164 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14165 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14166 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14167 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14168 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14169 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14170 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14171 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14172 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14173 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_217_16 := arithEq_of_rows h (row := 217) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_16
  simp only [← c14163, ← c14164, ← c14165] at e_217_16
  have e_217_17 := arithEq_of_rows h (row := 217) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_17
  simp only [← c14166, ← c14167, ← c14168] at e_217_17
  have e_218_8 := arithEq_of_rows h (row := 218) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_8
  simp only [← c14157, k3, ← c14158, k3, ← c14159] at e_218_8
  have e_218_9 := arithEq_of_rows h (row := 218) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_9
  simp only [← c14160, ← c14161, k3, ← c14162] at e_218_9
  have e_218_10 := arithEq_of_rows h (row := 218) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_10
  simp only [← c14169, ← c14170, k3, ← c14171] at e_218_10
  refine ⟨?_, ?_⟩
  · have hc := e_217_16
    simp only [e_218_9] at hc
    linear_combination c14172.trans k2 - hc
  · have hc := e_218_10
    simp only [e_218_8, e_217_17, e_218_9] at hc
    linear_combination c14173.trans k2 - hc

theorem publicBatchWrapper86_f2046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5406)) (a (.wire 165 23)) (a (.virt 18225)) (a (.virt 18226)) := by
  have c14174 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14175 := (publicBatchWrapper86_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14176 := (publicBatchWrapper86_copies443 a h).1
  have c14177 := (publicBatchWrapper86_copies443 a h).2.1
  have c14178 := (publicBatchWrapper86_copies443 a h).2.2.1
  have c14179 := (publicBatchWrapper86_copies443 a h).2.2.2.1
  have c14180 := (publicBatchWrapper86_copies443 a h).2.2.2.2.1
  have c14181 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.1
  have c14182 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.1
  have c14183 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.1
  have c14184 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.1
  have c14185 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.1
  have c14186 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14187 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14188 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14189 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14190 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_217_18 := arithEq_of_rows h (row := 217) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_18
  simp only [← c14180, ← c14181, ← c14182] at e_217_18
  have e_217_19 := arithEq_of_rows h (row := 217) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_19
  simp only [← c14183, ← c14184, ← c14185] at e_217_19
  have e_218_11 := arithEq_of_rows h (row := 218) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_11
  simp only [← c14174, k3, ← c14175, k3, ← c14176] at e_218_11
  have e_218_12 := arithEq_of_rows h (row := 218) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_12
  simp only [← c14177, ← c14178, k3, ← c14179] at e_218_12
  have e_218_13 := arithEq_of_rows h (row := 218) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_13
  simp only [← c14186, ← c14187, k3, ← c14188] at e_218_13
  refine ⟨?_, ?_⟩
  · have hc := e_217_18
    simp only [e_218_12] at hc
    linear_combination c14189.trans k2 - hc
  · have hc := e_218_13
    simp only [e_218_11, e_217_19, e_218_12] at hc
    linear_combination c14190.trans k2 - hc

theorem publicBatchWrapper86_f2047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5407)) (a (.wire 165 31)) (a (.virt 18227)) (a (.virt 18228)) := by
  have c14191 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14192 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14193 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14194 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14195 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14196 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14197 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14198 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14199 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14200 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14201 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14202 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14203 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14204 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14205 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14206 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14207 := (publicBatchWrapper86_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_218_14 := arithEq_of_rows h (row := 218) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_14
  simp only [← c14191, k3, ← c14192, k3, ← c14193] at e_218_14
  have e_218_15 := arithEq_of_rows h (row := 218) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_15
  simp only [← c14194, ← c14195, k3, ← c14196] at e_218_15
  have e_218_16 := arithEq_of_rows h (row := 218) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_16
  simp only [← c14203, ← c14204, k3, ← c14205] at e_218_16
  have e_219_0 := arithEq_of_rows h (row := 219) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_0
  simp only [← c14197, ← c14198, ← c14199] at e_219_0
  have e_219_1 := arithEq_of_rows h (row := 219) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_1
  simp only [← c14200, ← c14201, ← c14202] at e_219_1
  refine ⟨?_, ?_⟩
  · have hc := e_219_0
    simp only [e_218_15] at hc
    linear_combination c14206.trans k2 - hc
  · have hc := e_218_16
    simp only [e_218_14, e_219_1, e_218_15] at hc
    linear_combination c14207.trans k2 - hc

theorem publicBatchWrapper86_f2048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5408)) (a (.wire 165 39)) (a (.virt 18229)) (a (.virt 18230)) := by
  have c14208 := (publicBatchWrapper86_copies444 a h).1
  have c14209 := (publicBatchWrapper86_copies444 a h).2.1
  have c14210 := (publicBatchWrapper86_copies444 a h).2.2.1
  have c14211 := (publicBatchWrapper86_copies444 a h).2.2.2.1
  have c14212 := (publicBatchWrapper86_copies444 a h).2.2.2.2.1
  have c14213 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.1
  have c14214 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.1
  have c14215 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.1
  have c14216 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.1
  have c14217 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.1
  have c14218 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14219 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14220 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14221 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14222 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14223 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14224 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_218_17 := arithEq_of_rows h (row := 218) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_17
  simp only [← c14208, k3, ← c14209, k3, ← c14210] at e_218_17
  have e_218_18 := arithEq_of_rows h (row := 218) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_18
  simp only [← c14211, ← c14212, k3, ← c14213] at e_218_18
  have e_218_19 := arithEq_of_rows h (row := 218) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_19
  simp only [← c14220, ← c14221, k3, ← c14222] at e_218_19
  have e_219_2 := arithEq_of_rows h (row := 219) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_2
  simp only [← c14214, ← c14215, ← c14216] at e_219_2
  have e_219_3 := arithEq_of_rows h (row := 219) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_3
  simp only [← c14217, ← c14218, ← c14219] at e_219_3
  refine ⟨?_, ?_⟩
  · have hc := e_219_2
    simp only [e_218_18] at hc
    linear_combination c14223.trans k2 - hc
  · have hc := e_218_19
    simp only [e_218_17, e_219_3, e_218_18] at hc
    linear_combination c14224.trans k2 - hc

theorem publicBatchWrapper86_f2049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 219 19) = band (a (.virt 18223)) (a (.virt 18225)) := by
  have c14225 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14226 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14227 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_219_4 := arithEq_of_rows h (row := 219) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_4
  simp only [← c14225, ← c14226, ← c14227] at e_219_4
  have hr := e_219_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 219 23) = band (a (.virt 18227)) (a (.virt 18229)) := by
  have c14228 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14229 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14230 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_219_5 := arithEq_of_rows h (row := 219) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_5
  simp only [← c14228, ← c14229, ← c14230] at e_219_5
  have hr := e_219_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 219 27) = band (a (.wire 219 19)) (a (.wire 219 23)) := by
  have c14231 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14232 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14233 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_219_6 := arithEq_of_rows h (row := 219) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_6
  simp only [← c14231, ← c14232, ← c14233] at e_219_6
  have hr := e_219_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 67) = bor (a (.wire 29 15)) (a (.wire 219 27)) := by
  have c14234 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14235 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14236 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14237 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14238 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14239 := (publicBatchWrapper86_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_16 := arithEq_of_rows h (row := 209) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_16
  simp only [← c14237, ← c14238, k3, ← c14239] at e_209_16
  have e_215_8 := arithEq_of_rows h (row := 215) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_8
  simp only [← c14234, ← c14235, ← c14236] at e_215_8
  have hr := e_209_16
  simp only [e_215_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 67) = a (.virt 201) := by
  have c14240 := (publicBatchWrapper86_copies445 a h).1
  exact c14240

theorem publicBatchWrapper86_f2054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5603)) (a (.wire 165 55)) (a (.virt 18231)) (a (.virt 18232)) := by
  have c14241 := (publicBatchWrapper86_copies445 a h).2.1
  have c14242 := (publicBatchWrapper86_copies445 a h).2.2.1
  have c14243 := (publicBatchWrapper86_copies445 a h).2.2.2.1
  have c14244 := (publicBatchWrapper86_copies445 a h).2.2.2.2.1
  have c14245 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.1
  have c14246 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.1
  have c14247 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.1
  have c14248 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.1
  have c14249 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.1
  have c14250 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14251 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14252 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14253 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14254 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14255 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14256 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14257 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_219_7 := arithEq_of_rows h (row := 219) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_7
  simp only [← c14247, ← c14248, ← c14249] at e_219_7
  have e_219_8 := arithEq_of_rows h (row := 219) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_8
  simp only [← c14250, ← c14251, ← c14252] at e_219_8
  have e_220_0 := arithEq_of_rows h (row := 220) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_0
  simp only [← c14241, k3, ← c14242, k3, ← c14243] at e_220_0
  have e_220_1 := arithEq_of_rows h (row := 220) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_1
  simp only [← c14244, ← c14245, k3, ← c14246] at e_220_1
  have e_220_2 := arithEq_of_rows h (row := 220) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_2
  simp only [← c14253, ← c14254, k3, ← c14255] at e_220_2
  refine ⟨?_, ?_⟩
  · have hc := e_219_7
    simp only [e_220_1] at hc
    linear_combination c14256.trans k2 - hc
  · have hc := e_220_2
    simp only [e_220_0, e_219_8, e_220_1] at hc
    linear_combination c14257.trans k2 - hc

theorem publicBatchWrapper86_f2055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 71) = bor (a (.wire 29 59)) (a (.virt 18231)) := by
  have c14258 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14259 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14260 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14261 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14262 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14263 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_17 := arithEq_of_rows h (row := 209) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_17
  simp only [← c14261, ← c14262, k3, ← c14263] at e_209_17
  have e_215_9 := arithEq_of_rows h (row := 215) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_9
  simp only [← c14258, ← c14259, ← c14260] at e_215_9
  have hr := e_209_17
  simp only [e_215_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 71) = a (.virt 201) := by
  have c14264 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14264

theorem publicBatchWrapper86_f2057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5604)) (a (.wire 165 63)) (a (.virt 18233)) (a (.virt 18234)) := by
  have c14265 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14266 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14267 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14268 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14269 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14270 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14271 := (publicBatchWrapper86_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14272 := (publicBatchWrapper86_copies446 a h).1
  have c14273 := (publicBatchWrapper86_copies446 a h).2.1
  have c14274 := (publicBatchWrapper86_copies446 a h).2.2.1
  have c14275 := (publicBatchWrapper86_copies446 a h).2.2.2.1
  have c14276 := (publicBatchWrapper86_copies446 a h).2.2.2.2.1
  have c14277 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.1
  have c14278 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.1
  have c14279 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.1
  have c14280 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.1
  have c14281 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_219_9 := arithEq_of_rows h (row := 219) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_9
  simp only [← c14271, ← c14272, ← c14273] at e_219_9
  have e_219_10 := arithEq_of_rows h (row := 219) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_10
  simp only [← c14274, ← c14275, ← c14276] at e_219_10
  have e_220_3 := arithEq_of_rows h (row := 220) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_3
  simp only [← c14265, k3, ← c14266, k3, ← c14267] at e_220_3
  have e_220_4 := arithEq_of_rows h (row := 220) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_4
  simp only [← c14268, ← c14269, k3, ← c14270] at e_220_4
  have e_220_5 := arithEq_of_rows h (row := 220) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_5
  simp only [← c14277, ← c14278, k3, ← c14279] at e_220_5
  refine ⟨?_, ?_⟩
  · have hc := e_219_9
    simp only [e_220_4] at hc
    linear_combination c14280.trans k2 - hc
  · have hc := e_220_5
    simp only [e_220_3, e_219_10, e_220_4] at hc
    linear_combination c14281.trans k2 - hc

theorem publicBatchWrapper86_f2058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 75) = bor (a (.wire 29 59)) (a (.virt 18233)) := by
  have c14282 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14283 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14284 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14285 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14286 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14287 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_18 := arithEq_of_rows h (row := 209) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_18
  simp only [← c14285, ← c14286, k3, ← c14287] at e_209_18
  have e_215_10 := arithEq_of_rows h (row := 215) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_10
  simp only [← c14282, ← c14283, ← c14284] at e_215_10
  have hr := e_209_18
  simp only [e_215_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 75) = a (.virt 201) := by
  have c14288 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14288

theorem publicBatchWrapper86_f2060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5605)) (a (.wire 165 15)) (a (.virt 18235)) (a (.virt 18236)) := by
  have c14289 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14290 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14291 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14292 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14293 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14294 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14295 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14296 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14297 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14298 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14299 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14300 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14301 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14302 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14303 := (publicBatchWrapper86_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14304 := (publicBatchWrapper86_copies447 a h).1
  have c14305 := (publicBatchWrapper86_copies447 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_219_11 := arithEq_of_rows h (row := 219) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_11
  simp only [← c14295, ← c14296, ← c14297] at e_219_11
  have e_219_12 := arithEq_of_rows h (row := 219) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_12
  simp only [← c14298, ← c14299, ← c14300] at e_219_12
  have e_220_6 := arithEq_of_rows h (row := 220) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_6
  simp only [← c14289, k3, ← c14290, k3, ← c14291] at e_220_6
  have e_220_7 := arithEq_of_rows h (row := 220) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_7
  simp only [← c14292, ← c14293, k3, ← c14294] at e_220_7
  have e_220_8 := arithEq_of_rows h (row := 220) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_8
  simp only [← c14301, ← c14302, k3, ← c14303] at e_220_8
  refine ⟨?_, ?_⟩
  · have hc := e_219_11
    simp only [e_220_7] at hc
    linear_combination c14304.trans k2 - hc
  · have hc := e_220_8
    simp only [e_220_6, e_219_12, e_220_7] at hc
    linear_combination c14305.trans k2 - hc

theorem publicBatchWrapper86_f2061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5606)) (a (.wire 165 23)) (a (.virt 18237)) (a (.virt 18238)) := by
  have c14306 := (publicBatchWrapper86_copies447 a h).2.2.1
  have c14307 := (publicBatchWrapper86_copies447 a h).2.2.2.1
  have c14308 := (publicBatchWrapper86_copies447 a h).2.2.2.2.1
  have c14309 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.1
  have c14310 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.1
  have c14311 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.1
  have c14312 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.1
  have c14313 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.1
  have c14314 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14315 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14316 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14317 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14318 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14319 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14320 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14321 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14322 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_219_13 := arithEq_of_rows h (row := 219) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_13
  simp only [← c14312, ← c14313, ← c14314] at e_219_13
  have e_219_14 := arithEq_of_rows h (row := 219) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_14
  simp only [← c14315, ← c14316, ← c14317] at e_219_14
  have e_220_9 := arithEq_of_rows h (row := 220) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_9
  simp only [← c14306, k3, ← c14307, k3, ← c14308] at e_220_9
  have e_220_10 := arithEq_of_rows h (row := 220) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_10
  simp only [← c14309, ← c14310, k3, ← c14311] at e_220_10
  have e_220_11 := arithEq_of_rows h (row := 220) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_11
  simp only [← c14318, ← c14319, k3, ← c14320] at e_220_11
  refine ⟨?_, ?_⟩
  · have hc := e_219_13
    simp only [e_220_10] at hc
    linear_combination c14321.trans k2 - hc
  · have hc := e_220_11
    simp only [e_220_9, e_219_14, e_220_10] at hc
    linear_combination c14322.trans k2 - hc

theorem publicBatchWrapper86_f2062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5607)) (a (.wire 165 31)) (a (.virt 18239)) (a (.virt 18240)) := by
  have c14323 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14324 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14325 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14326 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14327 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14328 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14329 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14330 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14331 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14332 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14333 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14334 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14335 := (publicBatchWrapper86_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14336 := (publicBatchWrapper86_copies448 a h).1
  have c14337 := (publicBatchWrapper86_copies448 a h).2.1
  have c14338 := (publicBatchWrapper86_copies448 a h).2.2.1
  have c14339 := (publicBatchWrapper86_copies448 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_219_15 := arithEq_of_rows h (row := 219) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_15
  simp only [← c14329, ← c14330, ← c14331] at e_219_15
  have e_219_16 := arithEq_of_rows h (row := 219) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_16
  simp only [← c14332, ← c14333, ← c14334] at e_219_16
  have e_220_12 := arithEq_of_rows h (row := 220) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_12
  simp only [← c14323, k3, ← c14324, k3, ← c14325] at e_220_12
  have e_220_13 := arithEq_of_rows h (row := 220) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_13
  simp only [← c14326, ← c14327, k3, ← c14328] at e_220_13
  have e_220_14 := arithEq_of_rows h (row := 220) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_14
  simp only [← c14335, ← c14336, k3, ← c14337] at e_220_14
  refine ⟨?_, ?_⟩
  · have hc := e_219_15
    simp only [e_220_13] at hc
    linear_combination c14338.trans k2 - hc
  · have hc := e_220_14
    simp only [e_220_12, e_219_16, e_220_13] at hc
    linear_combination c14339.trans k2 - hc

theorem publicBatchWrapper86_f2063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5608)) (a (.wire 165 39)) (a (.virt 18241)) (a (.virt 18242)) := by
  have c14340 := (publicBatchWrapper86_copies448 a h).2.2.2.2.1
  have c14341 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.1
  have c14342 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.1
  have c14343 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.1
  have c14344 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.1
  have c14345 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.1
  have c14346 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14347 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14348 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14349 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14350 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14351 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14352 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14353 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14354 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14355 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14356 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_219_17 := arithEq_of_rows h (row := 219) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_17
  simp only [← c14346, ← c14347, ← c14348] at e_219_17
  have e_219_18 := arithEq_of_rows h (row := 219) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_18
  simp only [← c14349, ← c14350, ← c14351] at e_219_18
  have e_220_15 := arithEq_of_rows h (row := 220) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_15
  simp only [← c14340, k3, ← c14341, k3, ← c14342] at e_220_15
  have e_220_16 := arithEq_of_rows h (row := 220) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_16
  simp only [← c14343, ← c14344, k3, ← c14345] at e_220_16
  have e_220_17 := arithEq_of_rows h (row := 220) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_17
  simp only [← c14352, ← c14353, k3, ← c14354] at e_220_17
  refine ⟨?_, ?_⟩
  · have hc := e_219_17
    simp only [e_220_16] at hc
    linear_combination c14355.trans k2 - hc
  · have hc := e_220_17
    simp only [e_220_15, e_219_18, e_220_16] at hc
    linear_combination c14356.trans k2 - hc

theorem publicBatchWrapper86_f2064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 219 79) = band (a (.virt 18235)) (a (.virt 18237)) := by
  have c14357 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14358 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14359 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_219_19 := arithEq_of_rows h (row := 219) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_19
  simp only [← c14357, ← c14358, ← c14359] at e_219_19
  have hr := e_219_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 221 3) = band (a (.virt 18239)) (a (.virt 18241)) := by
  have c14360 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14361 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14362 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_221_0 := arithEq_of_rows h (row := 221) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_0
  simp only [← c14360, ← c14361, ← c14362] at e_221_0
  have hr := e_221_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 221 7) = band (a (.wire 219 79)) (a (.wire 221 3)) := by
  have c14363 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14364 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14365 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_221_1 := arithEq_of_rows h (row := 221) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_1
  simp only [← c14363, ← c14364, ← c14365] at e_221_1
  have hr := e_221_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 79) = bor (a (.wire 29 59)) (a (.wire 221 7)) := by
  have c14366 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14367 := (publicBatchWrapper86_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14368 := (publicBatchWrapper86_copies449 a h).1
  have c14369 := (publicBatchWrapper86_copies449 a h).2.1
  have c14370 := (publicBatchWrapper86_copies449 a h).2.2.1
  have c14371 := (publicBatchWrapper86_copies449 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_209_19 := arithEq_of_rows h (row := 209) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_19
  simp only [← c14369, ← c14370, k3, ← c14371] at e_209_19
  have e_215_11 := arithEq_of_rows h (row := 215) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_11
  simp only [← c14366, ← c14367, ← c14368] at e_215_11
  have hr := e_209_19
  simp only [e_215_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 209 79) = a (.virt 201) := by
  have c14372 := (publicBatchWrapper86_copies449 a h).2.2.2.2.1
  exact c14372

theorem publicBatchWrapper86_f2069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5803)) (a (.wire 165 55)) (a (.virt 18243)) (a (.virt 18244)) := by
  have c14373 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.1
  have c14374 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.1
  have c14375 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.1
  have c14376 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.1
  have c14377 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.1
  have c14378 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14379 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14380 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14381 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14382 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14383 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14384 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14385 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14386 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14387 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14388 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14389 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_220_18 := arithEq_of_rows h (row := 220) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_18
  simp only [← c14373, k3, ← c14374, k3, ← c14375] at e_220_18
  have e_220_19 := arithEq_of_rows h (row := 220) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_19
  simp only [← c14376, ← c14377, k3, ← c14378] at e_220_19
  have e_221_2 := arithEq_of_rows h (row := 221) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_2
  simp only [← c14379, ← c14380, ← c14381] at e_221_2
  have e_221_3 := arithEq_of_rows h (row := 221) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_3
  simp only [← c14382, ← c14383, ← c14384] at e_221_3
  have e_222_0 := arithEq_of_rows h (row := 222) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_0
  simp only [← c14385, ← c14386, k3, ← c14387] at e_222_0
  refine ⟨?_, ?_⟩
  · have hc := e_221_2
    simp only [e_220_19] at hc
    linear_combination c14388.trans k2 - hc
  · have hc := e_222_0
    simp only [e_220_18, e_221_3, e_220_19] at hc
    linear_combination c14389.trans k2 - hc

theorem publicBatchWrapper86_f2070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 3) = bor (a (.wire 30 23)) (a (.virt 18243)) := by
  have c14390 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14391 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14392 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14393 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14394 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14395 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_215_12 := arithEq_of_rows h (row := 215) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_12
  simp only [← c14390, ← c14391, ← c14392] at e_215_12
  have e_223_0 := arithEq_of_rows h (row := 223) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_0
  simp only [← c14393, ← c14394, k3, ← c14395] at e_223_0
  have hr := e_223_0
  simp only [e_215_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 3) = a (.virt 201) := by
  have c14396 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14396

theorem publicBatchWrapper86_f2072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5804)) (a (.wire 165 63)) (a (.virt 18245)) (a (.virt 18246)) := by
  have c14397 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14398 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14399 := (publicBatchWrapper86_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14400 := (publicBatchWrapper86_copies450 a h).1
  have c14401 := (publicBatchWrapper86_copies450 a h).2.1
  have c14402 := (publicBatchWrapper86_copies450 a h).2.2.1
  have c14403 := (publicBatchWrapper86_copies450 a h).2.2.2.1
  have c14404 := (publicBatchWrapper86_copies450 a h).2.2.2.2.1
  have c14405 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.1
  have c14406 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.1
  have c14407 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.1
  have c14408 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.1
  have c14409 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.1
  have c14410 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14411 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14412 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14413 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_221_4 := arithEq_of_rows h (row := 221) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_4
  simp only [← c14403, ← c14404, ← c14405] at e_221_4
  have e_221_5 := arithEq_of_rows h (row := 221) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_5
  simp only [← c14406, ← c14407, ← c14408] at e_221_5
  have e_222_1 := arithEq_of_rows h (row := 222) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_1
  simp only [← c14397, k3, ← c14398, k3, ← c14399] at e_222_1
  have e_222_2 := arithEq_of_rows h (row := 222) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_2
  simp only [← c14400, ← c14401, k3, ← c14402] at e_222_2
  have e_222_3 := arithEq_of_rows h (row := 222) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_3
  simp only [← c14409, ← c14410, k3, ← c14411] at e_222_3
  refine ⟨?_, ?_⟩
  · have hc := e_221_4
    simp only [e_222_2] at hc
    linear_combination c14412.trans k2 - hc
  · have hc := e_222_3
    simp only [e_222_1, e_221_5, e_222_2] at hc
    linear_combination c14413.trans k2 - hc

theorem publicBatchWrapper86_f2073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 7) = bor (a (.wire 30 23)) (a (.virt 18245)) := by
  have c14414 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14415 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14416 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14417 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14418 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14419 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_215_13 := arithEq_of_rows h (row := 215) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_13
  simp only [← c14414, ← c14415, ← c14416] at e_215_13
  have e_223_1 := arithEq_of_rows h (row := 223) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_1
  simp only [← c14417, ← c14418, k3, ← c14419] at e_223_1
  have hr := e_223_1
  simp only [e_215_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 7) = a (.virt 201) := by
  have c14420 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14420

theorem publicBatchWrapper86_f2075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5805)) (a (.wire 165 15)) (a (.virt 18247)) (a (.virt 18248)) := by
  have c14421 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14422 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14423 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14424 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14425 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14426 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14427 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14428 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14429 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14430 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14431 := (publicBatchWrapper86_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14432 := (publicBatchWrapper86_copies451 a h).1
  have c14433 := (publicBatchWrapper86_copies451 a h).2.1
  have c14434 := (publicBatchWrapper86_copies451 a h).2.2.1
  have c14435 := (publicBatchWrapper86_copies451 a h).2.2.2.1
  have c14436 := (publicBatchWrapper86_copies451 a h).2.2.2.2.1
  have c14437 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_221_6 := arithEq_of_rows h (row := 221) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_6
  simp only [← c14427, ← c14428, ← c14429] at e_221_6
  have e_221_7 := arithEq_of_rows h (row := 221) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_7
  simp only [← c14430, ← c14431, ← c14432] at e_221_7
  have e_222_4 := arithEq_of_rows h (row := 222) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_4
  simp only [← c14421, k3, ← c14422, k3, ← c14423] at e_222_4
  have e_222_5 := arithEq_of_rows h (row := 222) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_5
  simp only [← c14424, ← c14425, k3, ← c14426] at e_222_5
  have e_222_6 := arithEq_of_rows h (row := 222) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_6
  simp only [← c14433, ← c14434, k3, ← c14435] at e_222_6
  refine ⟨?_, ?_⟩
  · have hc := e_221_6
    simp only [e_222_5] at hc
    linear_combination c14436.trans k2 - hc
  · have hc := e_222_6
    simp only [e_222_4, e_221_7, e_222_5] at hc
    linear_combination c14437.trans k2 - hc

theorem publicBatchWrapper86_f2076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5806)) (a (.wire 165 23)) (a (.virt 18249)) (a (.virt 18250)) := by
  have c14438 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.1
  have c14439 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.1
  have c14440 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.1
  have c14441 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.1
  have c14442 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14443 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14444 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14445 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14446 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14447 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14448 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14449 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14450 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14451 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14452 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14453 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14454 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_221_8 := arithEq_of_rows h (row := 221) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_8
  simp only [← c14444, ← c14445, ← c14446] at e_221_8
  have e_221_9 := arithEq_of_rows h (row := 221) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_9
  simp only [← c14447, ← c14448, ← c14449] at e_221_9
  have e_222_7 := arithEq_of_rows h (row := 222) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_7
  simp only [← c14438, k3, ← c14439, k3, ← c14440] at e_222_7
  have e_222_8 := arithEq_of_rows h (row := 222) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_8
  simp only [← c14441, ← c14442, k3, ← c14443] at e_222_8
  have e_222_9 := arithEq_of_rows h (row := 222) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_9
  simp only [← c14450, ← c14451, k3, ← c14452] at e_222_9
  refine ⟨?_, ?_⟩
  · have hc := e_221_8
    simp only [e_222_8] at hc
    linear_combination c14453.trans k2 - hc
  · have hc := e_222_9
    simp only [e_222_7, e_221_9, e_222_8] at hc
    linear_combination c14454.trans k2 - hc

theorem publicBatchWrapper86_f2077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5807)) (a (.wire 165 31)) (a (.virt 18251)) (a (.virt 18252)) := by
  have c14455 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14456 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14457 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14458 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14459 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14460 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14461 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14462 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14463 := (publicBatchWrapper86_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14464 := (publicBatchWrapper86_copies452 a h).1
  have c14465 := (publicBatchWrapper86_copies452 a h).2.1
  have c14466 := (publicBatchWrapper86_copies452 a h).2.2.1
  have c14467 := (publicBatchWrapper86_copies452 a h).2.2.2.1
  have c14468 := (publicBatchWrapper86_copies452 a h).2.2.2.2.1
  have c14469 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.1
  have c14470 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.1
  have c14471 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_221_10 := arithEq_of_rows h (row := 221) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_10
  simp only [← c14461, ← c14462, ← c14463] at e_221_10
  have e_221_11 := arithEq_of_rows h (row := 221) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_11
  simp only [← c14464, ← c14465, ← c14466] at e_221_11
  have e_222_10 := arithEq_of_rows h (row := 222) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_10
  simp only [← c14455, k3, ← c14456, k3, ← c14457] at e_222_10
  have e_222_11 := arithEq_of_rows h (row := 222) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_11
  simp only [← c14458, ← c14459, k3, ← c14460] at e_222_11
  have e_222_12 := arithEq_of_rows h (row := 222) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_12
  simp only [← c14467, ← c14468, k3, ← c14469] at e_222_12
  refine ⟨?_, ?_⟩
  · have hc := e_221_10
    simp only [e_222_11] at hc
    linear_combination c14470.trans k2 - hc
  · have hc := e_222_12
    simp only [e_222_10, e_221_11, e_222_11] at hc
    linear_combination c14471.trans k2 - hc

theorem publicBatchWrapper86_f2078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5808)) (a (.wire 165 39)) (a (.virt 18253)) (a (.virt 18254)) := by
  have c14472 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.1
  have c14473 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.1
  have c14474 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14475 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14476 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14477 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14478 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14479 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14480 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14481 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14482 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14483 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14484 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14485 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14486 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14487 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14488 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_221_12 := arithEq_of_rows h (row := 221) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_12
  simp only [← c14478, ← c14479, ← c14480] at e_221_12
  have e_221_13 := arithEq_of_rows h (row := 221) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_13
  simp only [← c14481, ← c14482, ← c14483] at e_221_13
  have e_222_13 := arithEq_of_rows h (row := 222) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_13
  simp only [← c14472, k3, ← c14473, k3, ← c14474] at e_222_13
  have e_222_14 := arithEq_of_rows h (row := 222) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_14
  simp only [← c14475, ← c14476, k3, ← c14477] at e_222_14
  have e_222_15 := arithEq_of_rows h (row := 222) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_15
  simp only [← c14484, ← c14485, k3, ← c14486] at e_222_15
  refine ⟨?_, ?_⟩
  · have hc := e_221_12
    simp only [e_222_14] at hc
    linear_combination c14487.trans k2 - hc
  · have hc := e_222_15
    simp only [e_222_13, e_221_13, e_222_14] at hc
    linear_combination c14488.trans k2 - hc

theorem publicBatchWrapper86_f2079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 221 59) = band (a (.virt 18247)) (a (.virt 18249)) := by
  have c14489 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14490 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14491 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_221_14 := arithEq_of_rows h (row := 221) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_14
  simp only [← c14489, ← c14490, ← c14491] at e_221_14
  have hr := e_221_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 221 63) = band (a (.virt 18251)) (a (.virt 18253)) := by
  have c14492 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14493 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14494 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_221_15 := arithEq_of_rows h (row := 221) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_15
  simp only [← c14492, ← c14493, ← c14494] at e_221_15
  have hr := e_221_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 221 67) = band (a (.wire 221 59)) (a (.wire 221 63)) := by
  have c14495 := (publicBatchWrapper86_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14496 := (publicBatchWrapper86_copies453 a h).1
  have c14497 := (publicBatchWrapper86_copies453 a h).2.1
  have e_221_16 := arithEq_of_rows h (row := 221) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_16
  simp only [← c14495, ← c14496, ← c14497] at e_221_16
  have hr := e_221_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 11) = bor (a (.wire 30 23)) (a (.wire 221 67)) := by
  have c14498 := (publicBatchWrapper86_copies453 a h).2.2.1
  have c14499 := (publicBatchWrapper86_copies453 a h).2.2.2.1
  have c14500 := (publicBatchWrapper86_copies453 a h).2.2.2.2.1
  have c14501 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.1
  have c14502 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.1
  have c14503 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_215_14 := arithEq_of_rows h (row := 215) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_14
  simp only [← c14498, ← c14499, ← c14500] at e_215_14
  have e_223_2 := arithEq_of_rows h (row := 223) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_2
  simp only [← c14501, ← c14502, k3, ← c14503] at e_223_2
  have hr := e_223_2
  simp only [e_215_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 11) = a (.virt 201) := by
  have c14504 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.1
  exact c14504

theorem publicBatchWrapper86_f2084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6003)) (a (.wire 165 55)) (a (.virt 18255)) (a (.virt 18256)) := by
  have c14505 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.1
  have c14506 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14507 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14508 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14509 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14510 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14511 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14512 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14513 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14514 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14515 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14516 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14517 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14518 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14519 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14520 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14521 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_221_17 := arithEq_of_rows h (row := 221) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_17
  simp only [← c14511, ← c14512, ← c14513] at e_221_17
  have e_221_18 := arithEq_of_rows h (row := 221) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_18
  simp only [← c14514, ← c14515, ← c14516] at e_221_18
  have e_222_16 := arithEq_of_rows h (row := 222) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_16
  simp only [← c14505, k3, ← c14506, k3, ← c14507] at e_222_16
  have e_222_17 := arithEq_of_rows h (row := 222) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_17
  simp only [← c14508, ← c14509, k3, ← c14510] at e_222_17
  have e_222_18 := arithEq_of_rows h (row := 222) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_18
  simp only [← c14517, ← c14518, k3, ← c14519] at e_222_18
  refine ⟨?_, ?_⟩
  · have hc := e_221_17
    simp only [e_222_17] at hc
    linear_combination c14520.trans k2 - hc
  · have hc := e_222_18
    simp only [e_222_16, e_221_18, e_222_17] at hc
    linear_combination c14521.trans k2 - hc

theorem publicBatchWrapper86_f2085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 15) = bor (a (.wire 30 67)) (a (.virt 18255)) := by
  have c14522 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14523 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14524 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14525 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14526 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14527 := (publicBatchWrapper86_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_215_15 := arithEq_of_rows h (row := 215) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_15
  simp only [← c14522, ← c14523, ← c14524] at e_215_15
  have e_223_3 := arithEq_of_rows h (row := 223) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_3
  simp only [← c14525, ← c14526, k3, ← c14527] at e_223_3
  have hr := e_223_3
  simp only [e_215_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 15) = a (.virt 201) := by
  have c14528 := (publicBatchWrapper86_copies454 a h).1
  exact c14528

theorem publicBatchWrapper86_f2087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6004)) (a (.wire 165 63)) (a (.virt 18257)) (a (.virt 18258)) := by
  have c14529 := (publicBatchWrapper86_copies454 a h).2.1
  have c14530 := (publicBatchWrapper86_copies454 a h).2.2.1
  have c14531 := (publicBatchWrapper86_copies454 a h).2.2.2.1
  have c14532 := (publicBatchWrapper86_copies454 a h).2.2.2.2.1
  have c14533 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.1
  have c14534 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.1
  have c14535 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.1
  have c14536 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.1
  have c14537 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.1
  have c14538 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14539 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14540 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14541 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14542 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14543 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14544 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14545 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_221_19 := arithEq_of_rows h (row := 221) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_19
  simp only [← c14535, ← c14536, ← c14537] at e_221_19
  have e_222_19 := arithEq_of_rows h (row := 222) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_19
  simp only [← c14529, k3, ← c14530, k3, ← c14531] at e_222_19
  have e_224_0 := arithEq_of_rows h (row := 224) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_0
  simp only [← c14532, ← c14533, k3, ← c14534] at e_224_0
  have e_224_1 := arithEq_of_rows h (row := 224) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_1
  simp only [← c14541, ← c14542, k3, ← c14543] at e_224_1
  have e_225_0 := arithEq_of_rows h (row := 225) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_0
  simp only [← c14538, ← c14539, ← c14540] at e_225_0
  refine ⟨?_, ?_⟩
  · have hc := e_221_19
    simp only [e_224_0] at hc
    linear_combination c14544.trans k2 - hc
  · have hc := e_224_1
    simp only [e_222_19, e_225_0, e_224_0] at hc
    linear_combination c14545.trans k2 - hc

theorem publicBatchWrapper86_f2088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 19) = bor (a (.wire 30 67)) (a (.virt 18257)) := by
  have c14546 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14547 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14548 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14549 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14550 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14551 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_215_16 := arithEq_of_rows h (row := 215) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_16
  simp only [← c14546, ← c14547, ← c14548] at e_215_16
  have e_223_4 := arithEq_of_rows h (row := 223) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_4
  simp only [← c14549, ← c14550, k3, ← c14551] at e_223_4
  have hr := e_223_4
  simp only [e_215_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 19) = a (.virt 201) := by
  have c14552 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14552

theorem publicBatchWrapper86_f2090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6005)) (a (.wire 165 15)) (a (.virt 18259)) (a (.virt 18260)) := by
  have c14553 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14554 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14555 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14556 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14557 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14558 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14559 := (publicBatchWrapper86_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14560 := (publicBatchWrapper86_copies455 a h).1
  have c14561 := (publicBatchWrapper86_copies455 a h).2.1
  have c14562 := (publicBatchWrapper86_copies455 a h).2.2.1
  have c14563 := (publicBatchWrapper86_copies455 a h).2.2.2.1
  have c14564 := (publicBatchWrapper86_copies455 a h).2.2.2.2.1
  have c14565 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.1
  have c14566 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.1
  have c14567 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.1
  have c14568 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.1
  have c14569 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_224_2 := arithEq_of_rows h (row := 224) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_2
  simp only [← c14553, k3, ← c14554, k3, ← c14555] at e_224_2
  have e_224_3 := arithEq_of_rows h (row := 224) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_3
  simp only [← c14556, ← c14557, k3, ← c14558] at e_224_3
  have e_224_4 := arithEq_of_rows h (row := 224) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_4
  simp only [← c14565, ← c14566, k3, ← c14567] at e_224_4
  have e_225_1 := arithEq_of_rows h (row := 225) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_1
  simp only [← c14559, ← c14560, ← c14561] at e_225_1
  have e_225_2 := arithEq_of_rows h (row := 225) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_2
  simp only [← c14562, ← c14563, ← c14564] at e_225_2
  refine ⟨?_, ?_⟩
  · have hc := e_225_1
    simp only [e_224_3] at hc
    linear_combination c14568.trans k2 - hc
  · have hc := e_224_4
    simp only [e_224_2, e_225_2, e_224_3] at hc
    linear_combination c14569.trans k2 - hc

theorem publicBatchWrapper86_f2091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6006)) (a (.wire 165 23)) (a (.virt 18261)) (a (.virt 18262)) := by
  have c14570 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14571 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14572 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14573 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14574 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14575 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14576 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14577 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14578 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14579 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14580 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14581 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14582 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14583 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14584 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14585 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14586 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_224_5 := arithEq_of_rows h (row := 224) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_5
  simp only [← c14570, k3, ← c14571, k3, ← c14572] at e_224_5
  have e_224_6 := arithEq_of_rows h (row := 224) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_6
  simp only [← c14573, ← c14574, k3, ← c14575] at e_224_6
  have e_224_7 := arithEq_of_rows h (row := 224) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_7
  simp only [← c14582, ← c14583, k3, ← c14584] at e_224_7
  have e_225_3 := arithEq_of_rows h (row := 225) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_3
  simp only [← c14576, ← c14577, ← c14578] at e_225_3
  have e_225_4 := arithEq_of_rows h (row := 225) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_4
  simp only [← c14579, ← c14580, ← c14581] at e_225_4
  refine ⟨?_, ?_⟩
  · have hc := e_225_3
    simp only [e_224_6] at hc
    linear_combination c14585.trans k2 - hc
  · have hc := e_224_7
    simp only [e_224_5, e_225_4, e_224_6] at hc
    linear_combination c14586.trans k2 - hc

theorem publicBatchWrapper86_f2092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6007)) (a (.wire 165 31)) (a (.virt 18263)) (a (.virt 18264)) := by
  have c14587 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14588 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14589 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14590 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14591 := (publicBatchWrapper86_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14592 := (publicBatchWrapper86_copies456 a h).1
  have c14593 := (publicBatchWrapper86_copies456 a h).2.1
  have c14594 := (publicBatchWrapper86_copies456 a h).2.2.1
  have c14595 := (publicBatchWrapper86_copies456 a h).2.2.2.1
  have c14596 := (publicBatchWrapper86_copies456 a h).2.2.2.2.1
  have c14597 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.1
  have c14598 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.1
  have c14599 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.1
  have c14600 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.1
  have c14601 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.1
  have c14602 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14603 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_224_8 := arithEq_of_rows h (row := 224) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_8
  simp only [← c14587, k3, ← c14588, k3, ← c14589] at e_224_8
  have e_224_9 := arithEq_of_rows h (row := 224) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_9
  simp only [← c14590, ← c14591, k3, ← c14592] at e_224_9
  have e_224_10 := arithEq_of_rows h (row := 224) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_10
  simp only [← c14599, ← c14600, k3, ← c14601] at e_224_10
  have e_225_5 := arithEq_of_rows h (row := 225) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_5
  simp only [← c14593, ← c14594, ← c14595] at e_225_5
  have e_225_6 := arithEq_of_rows h (row := 225) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_6
  simp only [← c14596, ← c14597, ← c14598] at e_225_6
  refine ⟨?_, ?_⟩
  · have hc := e_225_5
    simp only [e_224_9] at hc
    linear_combination c14602.trans k2 - hc
  · have hc := e_224_10
    simp only [e_224_8, e_225_6, e_224_9] at hc
    linear_combination c14603.trans k2 - hc

theorem publicBatchWrapper86_f2093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6008)) (a (.wire 165 39)) (a (.virt 18265)) (a (.virt 18266)) := by
  have c14604 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14605 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14606 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14607 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14608 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14609 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14610 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14611 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14612 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14613 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14614 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14615 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14616 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14617 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14618 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14619 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14620 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_224_11 := arithEq_of_rows h (row := 224) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_11
  simp only [← c14604, k3, ← c14605, k3, ← c14606] at e_224_11
  have e_224_12 := arithEq_of_rows h (row := 224) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_12
  simp only [← c14607, ← c14608, k3, ← c14609] at e_224_12
  have e_224_13 := arithEq_of_rows h (row := 224) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_13
  simp only [← c14616, ← c14617, k3, ← c14618] at e_224_13
  have e_225_7 := arithEq_of_rows h (row := 225) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_7
  simp only [← c14610, ← c14611, ← c14612] at e_225_7
  have e_225_8 := arithEq_of_rows h (row := 225) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_8
  simp only [← c14613, ← c14614, ← c14615] at e_225_8
  refine ⟨?_, ?_⟩
  · have hc := e_225_7
    simp only [e_224_12] at hc
    linear_combination c14619.trans k2 - hc
  · have hc := e_224_13
    simp only [e_224_11, e_225_8, e_224_12] at hc
    linear_combination c14620.trans k2 - hc

theorem publicBatchWrapper86_f2094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 225 39) = band (a (.virt 18259)) (a (.virt 18261)) := by
  have c14621 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14622 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14623 := (publicBatchWrapper86_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_225_9 := arithEq_of_rows h (row := 225) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_9
  simp only [← c14621, ← c14622, ← c14623] at e_225_9
  have hr := e_225_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 225 43) = band (a (.virt 18263)) (a (.virt 18265)) := by
  have c14624 := (publicBatchWrapper86_copies457 a h).1
  have c14625 := (publicBatchWrapper86_copies457 a h).2.1
  have c14626 := (publicBatchWrapper86_copies457 a h).2.2.1
  have e_225_10 := arithEq_of_rows h (row := 225) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_10
  simp only [← c14624, ← c14625, ← c14626] at e_225_10
  have hr := e_225_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 225 47) = band (a (.wire 225 39)) (a (.wire 225 43)) := by
  have c14627 := (publicBatchWrapper86_copies457 a h).2.2.2.1
  have c14628 := (publicBatchWrapper86_copies457 a h).2.2.2.2.1
  have c14629 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.1
  have e_225_11 := arithEq_of_rows h (row := 225) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_11
  simp only [← c14627, ← c14628, ← c14629] at e_225_11
  have hr := e_225_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 23) = bor (a (.wire 30 67)) (a (.wire 225 47)) := by
  have c14630 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.1
  have c14631 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.1
  have c14632 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.1
  have c14633 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.1
  have c14634 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14635 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_215_17 := arithEq_of_rows h (row := 215) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_17
  simp only [← c14630, ← c14631, ← c14632] at e_215_17
  have e_223_5 := arithEq_of_rows h (row := 223) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_5
  simp only [← c14633, ← c14634, k3, ← c14635] at e_223_5
  have hr := e_223_5
  simp only [e_215_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 23) = a (.virt 201) := by
  have c14636 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14636

theorem publicBatchWrapper86_f2099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6203)) (a (.wire 165 55)) (a (.virt 18267)) (a (.virt 18268)) := by
  have c14637 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14638 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14639 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14640 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14641 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14642 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14643 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14644 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14645 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14646 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14647 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14648 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14649 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14650 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14651 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14652 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14653 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_224_14 := arithEq_of_rows h (row := 224) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_14
  simp only [← c14637, k3, ← c14638, k3, ← c14639] at e_224_14
  have e_224_15 := arithEq_of_rows h (row := 224) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_15
  simp only [← c14640, ← c14641, k3, ← c14642] at e_224_15
  have e_224_16 := arithEq_of_rows h (row := 224) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_16
  simp only [← c14649, ← c14650, k3, ← c14651] at e_224_16
  have e_225_12 := arithEq_of_rows h (row := 225) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_12
  simp only [← c14643, ← c14644, ← c14645] at e_225_12
  have e_225_13 := arithEq_of_rows h (row := 225) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_13
  simp only [← c14646, ← c14647, ← c14648] at e_225_13
  refine ⟨?_, ?_⟩
  · have hc := e_225_12
    simp only [e_224_15] at hc
    linear_combination c14652.trans k2 - hc
  · have hc := e_224_16
    simp only [e_224_14, e_225_13, e_224_15] at hc
    linear_combination c14653.trans k2 - hc

theorem publicBatchWrapper86_f2100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 27) = bor (a (.wire 32 31)) (a (.virt 18267)) := by
  have c14654 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14655 := (publicBatchWrapper86_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14656 := (publicBatchWrapper86_copies458 a h).1
  have c14657 := (publicBatchWrapper86_copies458 a h).2.1
  have c14658 := (publicBatchWrapper86_copies458 a h).2.2.1
  have c14659 := (publicBatchWrapper86_copies458 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_215_18 := arithEq_of_rows h (row := 215) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_18
  simp only [← c14654, ← c14655, ← c14656] at e_215_18
  have e_223_6 := arithEq_of_rows h (row := 223) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_6
  simp only [← c14657, ← c14658, k3, ← c14659] at e_223_6
  have hr := e_223_6
  simp only [e_215_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 27) = a (.virt 201) := by
  have c14660 := (publicBatchWrapper86_copies458 a h).2.2.2.2.1
  exact c14660

theorem publicBatchWrapper86_f2102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6204)) (a (.wire 165 63)) (a (.virt 18269)) (a (.virt 18270)) := by
  have c14661 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.1
  have c14662 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.1
  have c14663 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.1
  have c14664 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.1
  have c14665 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.1
  have c14666 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14667 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14668 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14669 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14670 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14671 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14672 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14673 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14674 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14675 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14676 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14677 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_224_17 := arithEq_of_rows h (row := 224) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_17
  simp only [← c14661, k3, ← c14662, k3, ← c14663] at e_224_17
  have e_224_18 := arithEq_of_rows h (row := 224) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_18
  simp only [← c14664, ← c14665, k3, ← c14666] at e_224_18
  have e_224_19 := arithEq_of_rows h (row := 224) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_19
  simp only [← c14673, ← c14674, k3, ← c14675] at e_224_19
  have e_225_14 := arithEq_of_rows h (row := 225) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_14
  simp only [← c14667, ← c14668, ← c14669] at e_225_14
  have e_225_15 := arithEq_of_rows h (row := 225) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_15
  simp only [← c14670, ← c14671, ← c14672] at e_225_15
  refine ⟨?_, ?_⟩
  · have hc := e_225_14
    simp only [e_224_18] at hc
    linear_combination c14676.trans k2 - hc
  · have hc := e_224_19
    simp only [e_224_17, e_225_15, e_224_18] at hc
    linear_combination c14677.trans k2 - hc

theorem publicBatchWrapper86_f2103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 31) = bor (a (.wire 32 31)) (a (.virt 18269)) := by
  have c14678 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14679 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14680 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14681 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14682 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14683 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_215_19 := arithEq_of_rows h (row := 215) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_19
  simp only [← c14678, ← c14679, ← c14680] at e_215_19
  have e_223_7 := arithEq_of_rows h (row := 223) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_7
  simp only [← c14681, ← c14682, k3, ← c14683] at e_223_7
  have hr := e_223_7
  simp only [e_215_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 31) = a (.virt 201) := by
  have c14684 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14684

theorem publicBatchWrapper86_f2105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6205)) (a (.wire 165 15)) (a (.virt 18271)) (a (.virt 18272)) := by
  have c14685 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14686 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14687 := (publicBatchWrapper86_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14688 := (publicBatchWrapper86_copies459 a h).1
  have c14689 := (publicBatchWrapper86_copies459 a h).2.1
  have c14690 := (publicBatchWrapper86_copies459 a h).2.2.1
  have c14691 := (publicBatchWrapper86_copies459 a h).2.2.2.1
  have c14692 := (publicBatchWrapper86_copies459 a h).2.2.2.2.1
  have c14693 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.1
  have c14694 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.1
  have c14695 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.1
  have c14696 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.1
  have c14697 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.1
  have c14698 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14699 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14700 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14701 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_225_16 := arithEq_of_rows h (row := 225) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_16
  simp only [← c14691, ← c14692, ← c14693] at e_225_16
  have e_225_17 := arithEq_of_rows h (row := 225) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_17
  simp only [← c14694, ← c14695, ← c14696] at e_225_17
  have e_226_0 := arithEq_of_rows h (row := 226) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_0
  simp only [← c14685, k3, ← c14686, k3, ← c14687] at e_226_0
  have e_226_1 := arithEq_of_rows h (row := 226) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_1
  simp only [← c14688, ← c14689, k3, ← c14690] at e_226_1
  have e_226_2 := arithEq_of_rows h (row := 226) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_2
  simp only [← c14697, ← c14698, k3, ← c14699] at e_226_2
  refine ⟨?_, ?_⟩
  · have hc := e_225_16
    simp only [e_226_1] at hc
    linear_combination c14700.trans k2 - hc
  · have hc := e_226_2
    simp only [e_226_0, e_225_17, e_226_1] at hc
    linear_combination c14701.trans k2 - hc

theorem publicBatchWrapper86_f2106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6206)) (a (.wire 165 23)) (a (.virt 18273)) (a (.virt 18274)) := by
  have c14702 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14703 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14704 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14705 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14706 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14707 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14708 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14709 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14710 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14711 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14712 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14713 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14714 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14715 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14716 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14717 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14718 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_225_18 := arithEq_of_rows h (row := 225) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_18
  simp only [← c14708, ← c14709, ← c14710] at e_225_18
  have e_225_19 := arithEq_of_rows h (row := 225) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_19
  simp only [← c14711, ← c14712, ← c14713] at e_225_19
  have e_226_3 := arithEq_of_rows h (row := 226) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_3
  simp only [← c14702, k3, ← c14703, k3, ← c14704] at e_226_3
  have e_226_4 := arithEq_of_rows h (row := 226) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_4
  simp only [← c14705, ← c14706, k3, ← c14707] at e_226_4
  have e_226_5 := arithEq_of_rows h (row := 226) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_5
  simp only [← c14714, ← c14715, k3, ← c14716] at e_226_5
  refine ⟨?_, ?_⟩
  · have hc := e_225_18
    simp only [e_226_4] at hc
    linear_combination c14717.trans k2 - hc
  · have hc := e_226_5
    simp only [e_226_3, e_225_19, e_226_4] at hc
    linear_combination c14718.trans k2 - hc

theorem publicBatchWrapper86_f2107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6207)) (a (.wire 165 31)) (a (.virt 18275)) (a (.virt 18276)) := by
  have c14719 := (publicBatchWrapper86_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14720 := (publicBatchWrapper86_copies460 a h).1
  have c14721 := (publicBatchWrapper86_copies460 a h).2.1
  have c14722 := (publicBatchWrapper86_copies460 a h).2.2.1
  have c14723 := (publicBatchWrapper86_copies460 a h).2.2.2.1
  have c14724 := (publicBatchWrapper86_copies460 a h).2.2.2.2.1
  have c14725 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.1
  have c14726 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.1
  have c14727 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.1
  have c14728 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.1
  have c14729 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.1
  have c14730 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14731 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14732 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14733 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14734 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14735 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_226_6 := arithEq_of_rows h (row := 226) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_6
  simp only [← c14719, k3, ← c14720, k3, ← c14721] at e_226_6
  have e_226_7 := arithEq_of_rows h (row := 226) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_7
  simp only [← c14722, ← c14723, k3, ← c14724] at e_226_7
  have e_226_8 := arithEq_of_rows h (row := 226) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_8
  simp only [← c14731, ← c14732, k3, ← c14733] at e_226_8
  have e_227_0 := arithEq_of_rows h (row := 227) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_0
  simp only [← c14725, ← c14726, ← c14727] at e_227_0
  have e_227_1 := arithEq_of_rows h (row := 227) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_1
  simp only [← c14728, ← c14729, ← c14730] at e_227_1
  refine ⟨?_, ?_⟩
  · have hc := e_227_0
    simp only [e_226_7] at hc
    linear_combination c14734.trans k2 - hc
  · have hc := e_226_8
    simp only [e_226_6, e_227_1, e_226_7] at hc
    linear_combination c14735.trans k2 - hc

theorem publicBatchWrapper86_f2108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6208)) (a (.wire 165 39)) (a (.virt 18277)) (a (.virt 18278)) := by
  have c14736 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14737 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14738 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14739 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14740 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14741 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14742 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14743 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14744 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14745 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14746 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14747 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14748 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14749 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14750 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14751 := (publicBatchWrapper86_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14752 := (publicBatchWrapper86_copies461 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_226_9 := arithEq_of_rows h (row := 226) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_9
  simp only [← c14736, k3, ← c14737, k3, ← c14738] at e_226_9
  have e_226_10 := arithEq_of_rows h (row := 226) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_10
  simp only [← c14739, ← c14740, k3, ← c14741] at e_226_10
  have e_226_11 := arithEq_of_rows h (row := 226) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_11
  simp only [← c14748, ← c14749, k3, ← c14750] at e_226_11
  have e_227_2 := arithEq_of_rows h (row := 227) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_2
  simp only [← c14742, ← c14743, ← c14744] at e_227_2
  have e_227_3 := arithEq_of_rows h (row := 227) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_3
  simp only [← c14745, ← c14746, ← c14747] at e_227_3
  refine ⟨?_, ?_⟩
  · have hc := e_227_2
    simp only [e_226_10] at hc
    linear_combination c14751.trans k2 - hc
  · have hc := e_226_11
    simp only [e_226_9, e_227_3, e_226_10] at hc
    linear_combination c14752.trans k2 - hc

theorem publicBatchWrapper86_f2109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 227 19) = band (a (.virt 18271)) (a (.virt 18273)) := by
  have c14753 := (publicBatchWrapper86_copies461 a h).2.1
  have c14754 := (publicBatchWrapper86_copies461 a h).2.2.1
  have c14755 := (publicBatchWrapper86_copies461 a h).2.2.2.1
  have e_227_4 := arithEq_of_rows h (row := 227) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_4
  simp only [← c14753, ← c14754, ← c14755] at e_227_4
  have hr := e_227_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 227 23) = band (a (.virt 18275)) (a (.virt 18277)) := by
  have c14756 := (publicBatchWrapper86_copies461 a h).2.2.2.2.1
  have c14757 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.1
  have c14758 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.1
  have e_227_5 := arithEq_of_rows h (row := 227) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_5
  simp only [← c14756, ← c14757, ← c14758] at e_227_5
  have hr := e_227_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 227 27) = band (a (.wire 227 19)) (a (.wire 227 23)) := by
  have c14759 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.1
  have c14760 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.1
  have c14761 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.1
  have e_227_6 := arithEq_of_rows h (row := 227) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_6
  simp only [← c14759, ← c14760, ← c14761] at e_227_6
  have hr := e_227_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 35) = bor (a (.wire 32 31)) (a (.wire 227 27)) := by
  have c14762 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14763 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14764 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14765 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14766 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14767 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_8 := arithEq_of_rows h (row := 223) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_8
  simp only [← c14765, ← c14766, k3, ← c14767] at e_223_8
  have e_228_0 := arithEq_of_rows h (row := 228) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_0
  simp only [← c14762, ← c14763, ← c14764] at e_228_0
  have hr := e_223_8
  simp only [e_228_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 35) = a (.virt 201) := by
  have c14768 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14768

theorem publicBatchWrapper86_f2114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6403)) (a (.wire 165 55)) (a (.virt 18279)) (a (.virt 18280)) := by
  have c14769 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14770 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14771 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14772 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14773 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14774 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14775 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14776 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14777 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14778 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14779 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14780 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14781 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14782 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14783 := (publicBatchWrapper86_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14784 := (publicBatchWrapper86_copies462 a h).1
  have c14785 := (publicBatchWrapper86_copies462 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_226_12 := arithEq_of_rows h (row := 226) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_12
  simp only [← c14769, k3, ← c14770, k3, ← c14771] at e_226_12
  have e_226_13 := arithEq_of_rows h (row := 226) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_13
  simp only [← c14772, ← c14773, k3, ← c14774] at e_226_13
  have e_226_14 := arithEq_of_rows h (row := 226) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_14
  simp only [← c14781, ← c14782, k3, ← c14783] at e_226_14
  have e_227_7 := arithEq_of_rows h (row := 227) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_7
  simp only [← c14775, ← c14776, ← c14777] at e_227_7
  have e_227_8 := arithEq_of_rows h (row := 227) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_8
  simp only [← c14778, ← c14779, ← c14780] at e_227_8
  refine ⟨?_, ?_⟩
  · have hc := e_227_7
    simp only [e_226_13] at hc
    linear_combination c14784.trans k2 - hc
  · have hc := e_226_14
    simp only [e_226_12, e_227_8, e_226_13] at hc
    linear_combination c14785.trans k2 - hc

theorem publicBatchWrapper86_f2115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 39) = bor (a (.wire 32 75)) (a (.virt 18279)) := by
  have c14786 := (publicBatchWrapper86_copies462 a h).2.2.1
  have c14787 := (publicBatchWrapper86_copies462 a h).2.2.2.1
  have c14788 := (publicBatchWrapper86_copies462 a h).2.2.2.2.1
  have c14789 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.1
  have c14790 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.1
  have c14791 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_9 := arithEq_of_rows h (row := 223) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_9
  simp only [← c14789, ← c14790, k3, ← c14791] at e_223_9
  have e_228_1 := arithEq_of_rows h (row := 228) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_1
  simp only [← c14786, ← c14787, ← c14788] at e_228_1
  have hr := e_223_9
  simp only [e_228_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 39) = a (.virt 201) := by
  have c14792 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.1
  exact c14792

theorem publicBatchWrapper86_f2117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6404)) (a (.wire 165 63)) (a (.virt 18281)) (a (.virt 18282)) := by
  have c14793 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.1
  have c14794 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14795 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14796 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14797 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14798 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14799 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14800 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14801 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14802 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14803 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14804 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14805 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14806 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14807 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14808 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14809 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_226_15 := arithEq_of_rows h (row := 226) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_15
  simp only [← c14793, k3, ← c14794, k3, ← c14795] at e_226_15
  have e_226_16 := arithEq_of_rows h (row := 226) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_16
  simp only [← c14796, ← c14797, k3, ← c14798] at e_226_16
  have e_226_17 := arithEq_of_rows h (row := 226) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_17
  simp only [← c14805, ← c14806, k3, ← c14807] at e_226_17
  have e_227_9 := arithEq_of_rows h (row := 227) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_9
  simp only [← c14799, ← c14800, ← c14801] at e_227_9
  have e_227_10 := arithEq_of_rows h (row := 227) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_10
  simp only [← c14802, ← c14803, ← c14804] at e_227_10
  refine ⟨?_, ?_⟩
  · have hc := e_227_9
    simp only [e_226_16] at hc
    linear_combination c14808.trans k2 - hc
  · have hc := e_226_17
    simp only [e_226_15, e_227_10, e_226_16] at hc
    linear_combination c14809.trans k2 - hc

theorem publicBatchWrapper86_f2118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 43) = bor (a (.wire 32 75)) (a (.virt 18281)) := by
  have c14810 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14811 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14812 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14813 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14814 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14815 := (publicBatchWrapper86_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_10 := arithEq_of_rows h (row := 223) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_10
  simp only [← c14813, ← c14814, k3, ← c14815] at e_223_10
  have e_228_2 := arithEq_of_rows h (row := 228) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_2
  simp only [← c14810, ← c14811, ← c14812] at e_228_2
  have hr := e_223_10
  simp only [e_228_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 43) = a (.virt 201) := by
  have c14816 := (publicBatchWrapper86_copies463 a h).1
  exact c14816

theorem publicBatchWrapper86_f2120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6405)) (a (.wire 165 15)) (a (.virt 18283)) (a (.virt 18284)) := by
  have c14817 := (publicBatchWrapper86_copies463 a h).2.1
  have c14818 := (publicBatchWrapper86_copies463 a h).2.2.1
  have c14819 := (publicBatchWrapper86_copies463 a h).2.2.2.1
  have c14820 := (publicBatchWrapper86_copies463 a h).2.2.2.2.1
  have c14821 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.1
  have c14822 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.1
  have c14823 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.1
  have c14824 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.1
  have c14825 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.1
  have c14826 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14827 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14828 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14829 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14830 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14831 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14832 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14833 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_226_18 := arithEq_of_rows h (row := 226) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_18
  simp only [← c14817, k3, ← c14818, k3, ← c14819] at e_226_18
  have e_226_19 := arithEq_of_rows h (row := 226) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_19
  simp only [← c14820, ← c14821, k3, ← c14822] at e_226_19
  have e_227_11 := arithEq_of_rows h (row := 227) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_11
  simp only [← c14823, ← c14824, ← c14825] at e_227_11
  have e_227_12 := arithEq_of_rows h (row := 227) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_12
  simp only [← c14826, ← c14827, ← c14828] at e_227_12
  have e_229_0 := arithEq_of_rows h (row := 229) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_0
  simp only [← c14829, ← c14830, k3, ← c14831] at e_229_0
  refine ⟨?_, ?_⟩
  · have hc := e_227_11
    simp only [e_226_19] at hc
    linear_combination c14832.trans k2 - hc
  · have hc := e_229_0
    simp only [e_226_18, e_227_12, e_226_19] at hc
    linear_combination c14833.trans k2 - hc

theorem publicBatchWrapper86_f2121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6406)) (a (.wire 165 23)) (a (.virt 18285)) (a (.virt 18286)) := by
  have c14834 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14835 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14836 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14837 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14838 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14839 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14840 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14841 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14842 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14843 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14844 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14845 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14846 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14847 := (publicBatchWrapper86_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14848 := (publicBatchWrapper86_copies464 a h).1
  have c14849 := (publicBatchWrapper86_copies464 a h).2.1
  have c14850 := (publicBatchWrapper86_copies464 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_227_13 := arithEq_of_rows h (row := 227) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_13
  simp only [← c14840, ← c14841, ← c14842] at e_227_13
  have e_227_14 := arithEq_of_rows h (row := 227) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_14
  simp only [← c14843, ← c14844, ← c14845] at e_227_14
  have e_229_1 := arithEq_of_rows h (row := 229) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_1
  simp only [← c14834, k3, ← c14835, k3, ← c14836] at e_229_1
  have e_229_2 := arithEq_of_rows h (row := 229) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_2
  simp only [← c14837, ← c14838, k3, ← c14839] at e_229_2
  have e_229_3 := arithEq_of_rows h (row := 229) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_3
  simp only [← c14846, ← c14847, k3, ← c14848] at e_229_3
  refine ⟨?_, ?_⟩
  · have hc := e_227_13
    simp only [e_229_2] at hc
    linear_combination c14849.trans k2 - hc
  · have hc := e_229_3
    simp only [e_229_1, e_227_14, e_229_2] at hc
    linear_combination c14850.trans k2 - hc

theorem publicBatchWrapper86_f2122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6407)) (a (.wire 165 31)) (a (.virt 18287)) (a (.virt 18288)) := by
  have c14851 := (publicBatchWrapper86_copies464 a h).2.2.2.1
  have c14852 := (publicBatchWrapper86_copies464 a h).2.2.2.2.1
  have c14853 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.1
  have c14854 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.1
  have c14855 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.1
  have c14856 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.1
  have c14857 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.1
  have c14858 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14859 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14860 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14861 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14862 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14863 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14864 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14865 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14866 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14867 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_227_15 := arithEq_of_rows h (row := 227) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_15
  simp only [← c14857, ← c14858, ← c14859] at e_227_15
  have e_227_16 := arithEq_of_rows h (row := 227) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_16
  simp only [← c14860, ← c14861, ← c14862] at e_227_16
  have e_229_4 := arithEq_of_rows h (row := 229) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_4
  simp only [← c14851, k3, ← c14852, k3, ← c14853] at e_229_4
  have e_229_5 := arithEq_of_rows h (row := 229) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_5
  simp only [← c14854, ← c14855, k3, ← c14856] at e_229_5
  have e_229_6 := arithEq_of_rows h (row := 229) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_6
  simp only [← c14863, ← c14864, k3, ← c14865] at e_229_6
  refine ⟨?_, ?_⟩
  · have hc := e_227_15
    simp only [e_229_5] at hc
    linear_combination c14866.trans k2 - hc
  · have hc := e_229_6
    simp only [e_229_4, e_227_16, e_229_5] at hc
    linear_combination c14867.trans k2 - hc

theorem publicBatchWrapper86_f2123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6408)) (a (.wire 165 39)) (a (.virt 18289)) (a (.virt 18290)) := by
  have c14868 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14869 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14870 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14871 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14872 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14873 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14874 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14875 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14876 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14877 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14878 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14879 := (publicBatchWrapper86_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14880 := (publicBatchWrapper86_copies465 a h).1
  have c14881 := (publicBatchWrapper86_copies465 a h).2.1
  have c14882 := (publicBatchWrapper86_copies465 a h).2.2.1
  have c14883 := (publicBatchWrapper86_copies465 a h).2.2.2.1
  have c14884 := (publicBatchWrapper86_copies465 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_227_17 := arithEq_of_rows h (row := 227) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_17
  simp only [← c14874, ← c14875, ← c14876] at e_227_17
  have e_227_18 := arithEq_of_rows h (row := 227) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_18
  simp only [← c14877, ← c14878, ← c14879] at e_227_18
  have e_229_7 := arithEq_of_rows h (row := 229) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_7
  simp only [← c14868, k3, ← c14869, k3, ← c14870] at e_229_7
  have e_229_8 := arithEq_of_rows h (row := 229) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_8
  simp only [← c14871, ← c14872, k3, ← c14873] at e_229_8
  have e_229_9 := arithEq_of_rows h (row := 229) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_9
  simp only [← c14880, ← c14881, k3, ← c14882] at e_229_9
  refine ⟨?_, ?_⟩
  · have hc := e_227_17
    simp only [e_229_8] at hc
    linear_combination c14883.trans k2 - hc
  · have hc := e_229_9
    simp only [e_229_7, e_227_18, e_229_8] at hc
    linear_combination c14884.trans k2 - hc

theorem publicBatchWrapper86_f2124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 227 79) = band (a (.virt 18283)) (a (.virt 18285)) := by
  have c14885 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.1
  have c14886 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.1
  have c14887 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.1
  have e_227_19 := arithEq_of_rows h (row := 227) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_19
  simp only [← c14885, ← c14886, ← c14887] at e_227_19
  have hr := e_227_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 230 3) = band (a (.virt 18287)) (a (.virt 18289)) := by
  have c14888 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.1
  have c14889 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.1
  have c14890 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_230_0 := arithEq_of_rows h (row := 230) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_0
  simp only [← c14888, ← c14889, ← c14890] at e_230_0
  have hr := e_230_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 230 7) = band (a (.wire 227 79)) (a (.wire 230 3)) := by
  have c14891 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14892 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14893 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_230_1 := arithEq_of_rows h (row := 230) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_1
  simp only [← c14891, ← c14892, ← c14893] at e_230_1
  have hr := e_230_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 47) = bor (a (.wire 32 75)) (a (.wire 230 7)) := by
  have c14894 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14895 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14896 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14897 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14898 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14899 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_11 := arithEq_of_rows h (row := 223) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_11
  simp only [← c14897, ← c14898, k3, ← c14899] at e_223_11
  have e_228_3 := arithEq_of_rows h (row := 228) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_3
  simp only [← c14894, ← c14895, ← c14896] at e_228_3
  have hr := e_223_11
  simp only [e_228_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 47) = a (.virt 201) := by
  have c14900 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14900

theorem publicBatchWrapper86_f2129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6603)) (a (.wire 165 55)) (a (.virt 18291)) (a (.virt 18292)) := by
  have c14901 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14902 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14903 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14904 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14905 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14906 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14907 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14908 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14909 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14910 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14911 := (publicBatchWrapper86_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14912 := (publicBatchWrapper86_copies466 a h).1
  have c14913 := (publicBatchWrapper86_copies466 a h).2.1
  have c14914 := (publicBatchWrapper86_copies466 a h).2.2.1
  have c14915 := (publicBatchWrapper86_copies466 a h).2.2.2.1
  have c14916 := (publicBatchWrapper86_copies466 a h).2.2.2.2.1
  have c14917 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_229_10 := arithEq_of_rows h (row := 229) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_10
  simp only [← c14901, k3, ← c14902, k3, ← c14903] at e_229_10
  have e_229_11 := arithEq_of_rows h (row := 229) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_11
  simp only [← c14904, ← c14905, k3, ← c14906] at e_229_11
  have e_229_12 := arithEq_of_rows h (row := 229) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_12
  simp only [← c14913, ← c14914, k3, ← c14915] at e_229_12
  have e_230_2 := arithEq_of_rows h (row := 230) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_2
  simp only [← c14907, ← c14908, ← c14909] at e_230_2
  have e_230_3 := arithEq_of_rows h (row := 230) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_3
  simp only [← c14910, ← c14911, ← c14912] at e_230_3
  refine ⟨?_, ?_⟩
  · have hc := e_230_2
    simp only [e_229_11] at hc
    linear_combination c14916.trans k2 - hc
  · have hc := e_229_12
    simp only [e_229_10, e_230_3, e_229_11] at hc
    linear_combination c14917.trans k2 - hc

theorem publicBatchWrapper86_f2130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 51) = bor (a (.wire 34 39)) (a (.virt 18291)) := by
  have c14918 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.1
  have c14919 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.1
  have c14920 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.1
  have c14921 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.1
  have c14922 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14923 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_12 := arithEq_of_rows h (row := 223) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_12
  simp only [← c14921, ← c14922, k3, ← c14923] at e_223_12
  have e_228_4 := arithEq_of_rows h (row := 228) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_4
  simp only [← c14918, ← c14919, ← c14920] at e_228_4
  have hr := e_223_12
  simp only [e_228_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 51) = a (.virt 201) := by
  have c14924 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14924

theorem publicBatchWrapper86_f2132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6604)) (a (.wire 165 63)) (a (.virt 18293)) (a (.virt 18294)) := by
  have c14925 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14926 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14927 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14928 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14929 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14930 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14931 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14932 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14933 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14934 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14935 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14936 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14937 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14938 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14939 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14940 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14941 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_229_13 := arithEq_of_rows h (row := 229) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_13
  simp only [← c14925, k3, ← c14926, k3, ← c14927] at e_229_13
  have e_229_14 := arithEq_of_rows h (row := 229) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_14
  simp only [← c14928, ← c14929, k3, ← c14930] at e_229_14
  have e_229_15 := arithEq_of_rows h (row := 229) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_15
  simp only [← c14937, ← c14938, k3, ← c14939] at e_229_15
  have e_230_4 := arithEq_of_rows h (row := 230) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_4
  simp only [← c14931, ← c14932, ← c14933] at e_230_4
  have e_230_5 := arithEq_of_rows h (row := 230) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_5
  simp only [← c14934, ← c14935, ← c14936] at e_230_5
  refine ⟨?_, ?_⟩
  · have hc := e_230_4
    simp only [e_229_14] at hc
    linear_combination c14940.trans k2 - hc
  · have hc := e_229_15
    simp only [e_229_13, e_230_5, e_229_14] at hc
    linear_combination c14941.trans k2 - hc

theorem publicBatchWrapper86_f2133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 55) = bor (a (.wire 34 39)) (a (.virt 18293)) := by
  have c14942 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14943 := (publicBatchWrapper86_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14944 := (publicBatchWrapper86_copies467 a h).1
  have c14945 := (publicBatchWrapper86_copies467 a h).2.1
  have c14946 := (publicBatchWrapper86_copies467 a h).2.2.1
  have c14947 := (publicBatchWrapper86_copies467 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_13 := arithEq_of_rows h (row := 223) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_13
  simp only [← c14945, ← c14946, k3, ← c14947] at e_223_13
  have e_228_5 := arithEq_of_rows h (row := 228) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_5
  simp only [← c14942, ← c14943, ← c14944] at e_228_5
  have hr := e_223_13
  simp only [e_228_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 55) = a (.virt 201) := by
  have c14948 := (publicBatchWrapper86_copies467 a h).2.2.2.2.1
  exact c14948

theorem publicBatchWrapper86_f2135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6605)) (a (.wire 165 15)) (a (.virt 18295)) (a (.virt 18296)) := by
  have c14949 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.1
  have c14950 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.1
  have c14951 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.1
  have c14952 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.1
  have c14953 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.1
  have c14954 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14955 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14956 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14957 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14958 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14959 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14960 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14961 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14962 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14963 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14964 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14965 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_229_16 := arithEq_of_rows h (row := 229) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_16
  simp only [← c14949, k3, ← c14950, k3, ← c14951] at e_229_16
  have e_229_17 := arithEq_of_rows h (row := 229) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_17
  simp only [← c14952, ← c14953, k3, ← c14954] at e_229_17
  have e_229_18 := arithEq_of_rows h (row := 229) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_18
  simp only [← c14961, ← c14962, k3, ← c14963] at e_229_18
  have e_230_6 := arithEq_of_rows h (row := 230) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_6
  simp only [← c14955, ← c14956, ← c14957] at e_230_6
  have e_230_7 := arithEq_of_rows h (row := 230) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_7
  simp only [← c14958, ← c14959, ← c14960] at e_230_7
  refine ⟨?_, ?_⟩
  · have hc := e_230_6
    simp only [e_229_17] at hc
    linear_combination c14964.trans k2 - hc
  · have hc := e_229_18
    simp only [e_229_16, e_230_7, e_229_17] at hc
    linear_combination c14965.trans k2 - hc

theorem publicBatchWrapper86_f2136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6606)) (a (.wire 165 23)) (a (.virt 18297)) (a (.virt 18298)) := by
  have c14966 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14967 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14968 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14969 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14970 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14971 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14972 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14973 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14974 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14975 := (publicBatchWrapper86_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14976 := (publicBatchWrapper86_copies468 a h).1
  have c14977 := (publicBatchWrapper86_copies468 a h).2.1
  have c14978 := (publicBatchWrapper86_copies468 a h).2.2.1
  have c14979 := (publicBatchWrapper86_copies468 a h).2.2.2.1
  have c14980 := (publicBatchWrapper86_copies468 a h).2.2.2.2.1
  have c14981 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.1
  have c14982 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_229_19 := arithEq_of_rows h (row := 229) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_19
  simp only [← c14966, k3, ← c14967, k3, ← c14968] at e_229_19
  have e_230_8 := arithEq_of_rows h (row := 230) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_8
  simp only [← c14972, ← c14973, ← c14974] at e_230_8
  have e_230_9 := arithEq_of_rows h (row := 230) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_9
  simp only [← c14975, ← c14976, ← c14977] at e_230_9
  have e_231_0 := arithEq_of_rows h (row := 231) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_0
  simp only [← c14969, ← c14970, k3, ← c14971] at e_231_0
  have e_231_1 := arithEq_of_rows h (row := 231) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_1
  simp only [← c14978, ← c14979, k3, ← c14980] at e_231_1
  refine ⟨?_, ?_⟩
  · have hc := e_230_8
    simp only [e_231_0] at hc
    linear_combination c14981.trans k2 - hc
  · have hc := e_231_1
    simp only [e_229_19, e_230_9, e_231_0] at hc
    linear_combination c14982.trans k2 - hc

theorem publicBatchWrapper86_f2137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6607)) (a (.wire 165 31)) (a (.virt 18299)) (a (.virt 18300)) := by
  have c14983 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.1
  have c14984 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.1
  have c14985 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.1
  have c14986 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14987 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14988 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14989 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14990 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14991 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14992 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14993 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14994 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14995 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14996 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14997 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14998 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14999 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_230_10 := arithEq_of_rows h (row := 230) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_10
  simp only [← c14989, ← c14990, ← c14991] at e_230_10
  have e_230_11 := arithEq_of_rows h (row := 230) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_11
  simp only [← c14992, ← c14993, ← c14994] at e_230_11
  have e_231_2 := arithEq_of_rows h (row := 231) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_2
  simp only [← c14983, k3, ← c14984, k3, ← c14985] at e_231_2
  have e_231_3 := arithEq_of_rows h (row := 231) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_3
  simp only [← c14986, ← c14987, k3, ← c14988] at e_231_3
  have e_231_4 := arithEq_of_rows h (row := 231) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_4
  simp only [← c14995, ← c14996, k3, ← c14997] at e_231_4
  refine ⟨?_, ?_⟩
  · have hc := e_230_10
    simp only [e_231_3] at hc
    linear_combination c14998.trans k2 - hc
  · have hc := e_231_4
    simp only [e_231_2, e_230_11, e_231_3] at hc
    linear_combination c14999.trans k2 - hc

theorem publicBatchWrapper86_f2138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6608)) (a (.wire 165 39)) (a (.virt 18301)) (a (.virt 18302)) := by
  have c15000 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15001 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15002 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15003 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15004 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15005 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15006 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15007 := (publicBatchWrapper86_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15008 := (publicBatchWrapper86_copies469 a h).1
  have c15009 := (publicBatchWrapper86_copies469 a h).2.1
  have c15010 := (publicBatchWrapper86_copies469 a h).2.2.1
  have c15011 := (publicBatchWrapper86_copies469 a h).2.2.2.1
  have c15012 := (publicBatchWrapper86_copies469 a h).2.2.2.2.1
  have c15013 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.1
  have c15014 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.1
  have c15015 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.1
  have c15016 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_230_12 := arithEq_of_rows h (row := 230) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_12
  simp only [← c15006, ← c15007, ← c15008] at e_230_12
  have e_230_13 := arithEq_of_rows h (row := 230) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_13
  simp only [← c15009, ← c15010, ← c15011] at e_230_13
  have e_231_5 := arithEq_of_rows h (row := 231) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_5
  simp only [← c15000, k3, ← c15001, k3, ← c15002] at e_231_5
  have e_231_6 := arithEq_of_rows h (row := 231) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_6
  simp only [← c15003, ← c15004, k3, ← c15005] at e_231_6
  have e_231_7 := arithEq_of_rows h (row := 231) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_7
  simp only [← c15012, ← c15013, k3, ← c15014] at e_231_7
  refine ⟨?_, ?_⟩
  · have hc := e_230_12
    simp only [e_231_6] at hc
    linear_combination c15015.trans k2 - hc
  · have hc := e_231_7
    simp only [e_231_5, e_230_13, e_231_6] at hc
    linear_combination c15016.trans k2 - hc

theorem publicBatchWrapper86_f2139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 230 59) = band (a (.virt 18295)) (a (.virt 18297)) := by
  have c15017 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.1
  have c15018 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15019 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_230_14 := arithEq_of_rows h (row := 230) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_14
  simp only [← c15017, ← c15018, ← c15019] at e_230_14
  have hr := e_230_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 230 63) = band (a (.virt 18299)) (a (.virt 18301)) := by
  have c15020 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15021 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15022 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_230_15 := arithEq_of_rows h (row := 230) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_15
  simp only [← c15020, ← c15021, ← c15022] at e_230_15
  have hr := e_230_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 230 67) = band (a (.wire 230 59)) (a (.wire 230 63)) := by
  have c15023 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15024 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15025 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_230_16 := arithEq_of_rows h (row := 230) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_16
  simp only [← c15023, ← c15024, ← c15025] at e_230_16
  have hr := e_230_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 59) = bor (a (.wire 34 39)) (a (.wire 230 67)) := by
  have c15026 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15027 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15028 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15029 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15030 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15031 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_14 := arithEq_of_rows h (row := 223) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_14
  simp only [← c15029, ← c15030, k3, ← c15031] at e_223_14
  have e_228_6 := arithEq_of_rows h (row := 228) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_6
  simp only [← c15026, ← c15027, ← c15028] at e_228_6
  have hr := e_223_14
  simp only [e_228_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 59) = a (.virt 201) := by
  have c15032 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15032

theorem publicBatchWrapper86_f2144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6803)) (a (.wire 165 55)) (a (.virt 18303)) (a (.virt 18304)) := by
  have c15033 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15034 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15035 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15036 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15037 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15038 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15039 := (publicBatchWrapper86_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15040 := (publicBatchWrapper86_copies470 a h).1
  have c15041 := (publicBatchWrapper86_copies470 a h).2.1
  have c15042 := (publicBatchWrapper86_copies470 a h).2.2.1
  have c15043 := (publicBatchWrapper86_copies470 a h).2.2.2.1
  have c15044 := (publicBatchWrapper86_copies470 a h).2.2.2.2.1
  have c15045 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.1
  have c15046 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.1
  have c15047 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.1
  have c15048 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.1
  have c15049 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_230_17 := arithEq_of_rows h (row := 230) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_17
  simp only [← c15039, ← c15040, ← c15041] at e_230_17
  have e_230_18 := arithEq_of_rows h (row := 230) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_18
  simp only [← c15042, ← c15043, ← c15044] at e_230_18
  have e_231_8 := arithEq_of_rows h (row := 231) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_8
  simp only [← c15033, k3, ← c15034, k3, ← c15035] at e_231_8
  have e_231_9 := arithEq_of_rows h (row := 231) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_9
  simp only [← c15036, ← c15037, k3, ← c15038] at e_231_9
  have e_231_10 := arithEq_of_rows h (row := 231) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_10
  simp only [← c15045, ← c15046, k3, ← c15047] at e_231_10
  refine ⟨?_, ?_⟩
  · have hc := e_230_17
    simp only [e_231_9] at hc
    linear_combination c15048.trans k2 - hc
  · have hc := e_231_10
    simp only [e_231_8, e_230_18, e_231_9] at hc
    linear_combination c15049.trans k2 - hc

theorem publicBatchWrapper86_f2145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 63) = bor (a (.wire 35 3)) (a (.virt 18303)) := by
  have c15050 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15051 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15052 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15053 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15054 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15055 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_15 := arithEq_of_rows h (row := 223) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_15
  simp only [← c15053, ← c15054, k3, ← c15055] at e_223_15
  have e_228_7 := arithEq_of_rows h (row := 228) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_7
  simp only [← c15050, ← c15051, ← c15052] at e_228_7
  have hr := e_223_15
  simp only [e_228_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 63) = a (.virt 201) := by
  have c15056 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15056

theorem publicBatchWrapper86_f2147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6804)) (a (.wire 165 63)) (a (.virt 18305)) (a (.virt 18306)) := by
  have c15057 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15058 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15059 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15060 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15061 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15062 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15063 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15064 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15065 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15066 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15067 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15068 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15069 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15070 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15071 := (publicBatchWrapper86_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15072 := (publicBatchWrapper86_copies471 a h).1
  have c15073 := (publicBatchWrapper86_copies471 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_230_19 := arithEq_of_rows h (row := 230) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_19
  simp only [← c15063, ← c15064, ← c15065] at e_230_19
  have e_231_11 := arithEq_of_rows h (row := 231) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_11
  simp only [← c15057, k3, ← c15058, k3, ← c15059] at e_231_11
  have e_231_12 := arithEq_of_rows h (row := 231) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_12
  simp only [← c15060, ← c15061, k3, ← c15062] at e_231_12
  have e_231_13 := arithEq_of_rows h (row := 231) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_13
  simp only [← c15069, ← c15070, k3, ← c15071] at e_231_13
  have e_232_0 := arithEq_of_rows h (row := 232) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_0
  simp only [← c15066, ← c15067, ← c15068] at e_232_0
  refine ⟨?_, ?_⟩
  · have hc := e_230_19
    simp only [e_231_12] at hc
    linear_combination c15072.trans k2 - hc
  · have hc := e_231_13
    simp only [e_231_11, e_232_0, e_231_12] at hc
    linear_combination c15073.trans k2 - hc

theorem publicBatchWrapper86_f2148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 67) = bor (a (.wire 35 3)) (a (.virt 18305)) := by
  have c15074 := (publicBatchWrapper86_copies471 a h).2.2.1
  have c15075 := (publicBatchWrapper86_copies471 a h).2.2.2.1
  have c15076 := (publicBatchWrapper86_copies471 a h).2.2.2.2.1
  have c15077 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.1
  have c15078 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.1
  have c15079 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_16 := arithEq_of_rows h (row := 223) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_16
  simp only [← c15077, ← c15078, k3, ← c15079] at e_223_16
  have e_228_8 := arithEq_of_rows h (row := 228) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_8
  simp only [← c15074, ← c15075, ← c15076] at e_228_8
  have hr := e_223_16
  simp only [e_228_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 67) = a (.virt 201) := by
  have c15080 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.1
  exact c15080

theorem publicBatchWrapper86_f2150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6805)) (a (.wire 165 15)) (a (.virt 18307)) (a (.virt 18308)) := by
  have c15081 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.1
  have c15082 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15083 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15084 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15085 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15086 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15087 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15088 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15089 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15090 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15091 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15092 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15093 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15094 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15095 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15096 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15097 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_231_14 := arithEq_of_rows h (row := 231) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_14
  simp only [← c15081, k3, ← c15082, k3, ← c15083] at e_231_14
  have e_231_15 := arithEq_of_rows h (row := 231) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_15
  simp only [← c15084, ← c15085, k3, ← c15086] at e_231_15
  have e_231_16 := arithEq_of_rows h (row := 231) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_16
  simp only [← c15093, ← c15094, k3, ← c15095] at e_231_16
  have e_232_1 := arithEq_of_rows h (row := 232) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_1
  simp only [← c15087, ← c15088, ← c15089] at e_232_1
  have e_232_2 := arithEq_of_rows h (row := 232) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_2
  simp only [← c15090, ← c15091, ← c15092] at e_232_2
  refine ⟨?_, ?_⟩
  · have hc := e_232_1
    simp only [e_231_15] at hc
    linear_combination c15096.trans k2 - hc
  · have hc := e_231_16
    simp only [e_231_14, e_232_2, e_231_15] at hc
    linear_combination c15097.trans k2 - hc

theorem publicBatchWrapper86_f2151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6806)) (a (.wire 165 23)) (a (.virt 18309)) (a (.virt 18310)) := by
  have c15098 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15099 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15100 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15101 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15102 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15103 := (publicBatchWrapper86_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15104 := (publicBatchWrapper86_copies472 a h).1
  have c15105 := (publicBatchWrapper86_copies472 a h).2.1
  have c15106 := (publicBatchWrapper86_copies472 a h).2.2.1
  have c15107 := (publicBatchWrapper86_copies472 a h).2.2.2.1
  have c15108 := (publicBatchWrapper86_copies472 a h).2.2.2.2.1
  have c15109 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.1
  have c15110 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.1
  have c15111 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.1
  have c15112 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.1
  have c15113 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.1
  have c15114 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_231_17 := arithEq_of_rows h (row := 231) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_17
  simp only [← c15098, k3, ← c15099, k3, ← c15100] at e_231_17
  have e_231_18 := arithEq_of_rows h (row := 231) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_18
  simp only [← c15101, ← c15102, k3, ← c15103] at e_231_18
  have e_231_19 := arithEq_of_rows h (row := 231) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_19
  simp only [← c15110, ← c15111, k3, ← c15112] at e_231_19
  have e_232_3 := arithEq_of_rows h (row := 232) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_3
  simp only [← c15104, ← c15105, ← c15106] at e_232_3
  have e_232_4 := arithEq_of_rows h (row := 232) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_4
  simp only [← c15107, ← c15108, ← c15109] at e_232_4
  refine ⟨?_, ?_⟩
  · have hc := e_232_3
    simp only [e_231_18] at hc
    linear_combination c15113.trans k2 - hc
  · have hc := e_231_19
    simp only [e_231_17, e_232_4, e_231_18] at hc
    linear_combination c15114.trans k2 - hc

theorem publicBatchWrapper86_f2152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6807)) (a (.wire 165 31)) (a (.virt 18311)) (a (.virt 18312)) := by
  have c15115 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15116 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15117 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15118 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15119 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15120 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15121 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15122 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15123 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15124 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15125 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15126 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15127 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15128 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15129 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15130 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15131 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_232_5 := arithEq_of_rows h (row := 232) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_5
  simp only [← c15121, ← c15122, ← c15123] at e_232_5
  have e_232_6 := arithEq_of_rows h (row := 232) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_6
  simp only [← c15124, ← c15125, ← c15126] at e_232_6
  have e_233_0 := arithEq_of_rows h (row := 233) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_0
  simp only [← c15115, k3, ← c15116, k3, ← c15117] at e_233_0
  have e_233_1 := arithEq_of_rows h (row := 233) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_1
  simp only [← c15118, ← c15119, k3, ← c15120] at e_233_1
  have e_233_2 := arithEq_of_rows h (row := 233) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_2
  simp only [← c15127, ← c15128, k3, ← c15129] at e_233_2
  refine ⟨?_, ?_⟩
  · have hc := e_232_5
    simp only [e_233_1] at hc
    linear_combination c15130.trans k2 - hc
  · have hc := e_233_2
    simp only [e_233_0, e_232_6, e_233_1] at hc
    linear_combination c15131.trans k2 - hc

theorem publicBatchWrapper86_f2153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6808)) (a (.wire 165 39)) (a (.virt 18313)) (a (.virt 18314)) := by
  have c15132 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15133 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15134 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15135 := (publicBatchWrapper86_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15136 := (publicBatchWrapper86_copies473 a h).1
  have c15137 := (publicBatchWrapper86_copies473 a h).2.1
  have c15138 := (publicBatchWrapper86_copies473 a h).2.2.1
  have c15139 := (publicBatchWrapper86_copies473 a h).2.2.2.1
  have c15140 := (publicBatchWrapper86_copies473 a h).2.2.2.2.1
  have c15141 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.1
  have c15142 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.1
  have c15143 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.1
  have c15144 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.1
  have c15145 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.1
  have c15146 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15147 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15148 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_232_7 := arithEq_of_rows h (row := 232) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_7
  simp only [← c15138, ← c15139, ← c15140] at e_232_7
  have e_232_8 := arithEq_of_rows h (row := 232) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_8
  simp only [← c15141, ← c15142, ← c15143] at e_232_8
  have e_233_3 := arithEq_of_rows h (row := 233) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_3
  simp only [← c15132, k3, ← c15133, k3, ← c15134] at e_233_3
  have e_233_4 := arithEq_of_rows h (row := 233) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_4
  simp only [← c15135, ← c15136, k3, ← c15137] at e_233_4
  have e_233_5 := arithEq_of_rows h (row := 233) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_5
  simp only [← c15144, ← c15145, k3, ← c15146] at e_233_5
  refine ⟨?_, ?_⟩
  · have hc := e_232_7
    simp only [e_233_4] at hc
    linear_combination c15147.trans k2 - hc
  · have hc := e_233_5
    simp only [e_233_3, e_232_8, e_233_4] at hc
    linear_combination c15148.trans k2 - hc

theorem publicBatchWrapper86_f2154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 232 39) = band (a (.virt 18307)) (a (.virt 18309)) := by
  have c15149 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15150 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15151 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_232_9 := arithEq_of_rows h (row := 232) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_9
  simp only [← c15149, ← c15150, ← c15151] at e_232_9
  have hr := e_232_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 232 43) = band (a (.virt 18311)) (a (.virt 18313)) := by
  have c15152 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15153 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15154 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_232_10 := arithEq_of_rows h (row := 232) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_10
  simp only [← c15152, ← c15153, ← c15154] at e_232_10
  have hr := e_232_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 232 47) = band (a (.wire 232 39)) (a (.wire 232 43)) := by
  have c15155 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15156 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15157 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_232_11 := arithEq_of_rows h (row := 232) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_11
  simp only [← c15155, ← c15156, ← c15157] at e_232_11
  have hr := e_232_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 71) = bor (a (.wire 35 3)) (a (.wire 232 47)) := by
  have c15158 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15159 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15160 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15161 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15162 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15163 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_17 := arithEq_of_rows h (row := 223) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_17
  simp only [← c15161, ← c15162, k3, ← c15163] at e_223_17
  have e_228_9 := arithEq_of_rows h (row := 228) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_9
  simp only [← c15158, ← c15159, ← c15160] at e_228_9
  have hr := e_223_17
  simp only [e_228_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 71) = a (.virt 201) := by
  have c15164 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15164

theorem publicBatchWrapper86_f2159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7003)) (a (.wire 165 55)) (a (.virt 18315)) (a (.virt 18316)) := by
  have c15165 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15166 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15167 := (publicBatchWrapper86_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15168 := (publicBatchWrapper86_copies474 a h).1
  have c15169 := (publicBatchWrapper86_copies474 a h).2.1
  have c15170 := (publicBatchWrapper86_copies474 a h).2.2.1
  have c15171 := (publicBatchWrapper86_copies474 a h).2.2.2.1
  have c15172 := (publicBatchWrapper86_copies474 a h).2.2.2.2.1
  have c15173 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.1
  have c15174 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.1
  have c15175 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.1
  have c15176 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.1
  have c15177 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.1
  have c15178 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15179 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15180 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15181 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_232_12 := arithEq_of_rows h (row := 232) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_12
  simp only [← c15171, ← c15172, ← c15173] at e_232_12
  have e_232_13 := arithEq_of_rows h (row := 232) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_13
  simp only [← c15174, ← c15175, ← c15176] at e_232_13
  have e_233_6 := arithEq_of_rows h (row := 233) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_6
  simp only [← c15165, k3, ← c15166, k3, ← c15167] at e_233_6
  have e_233_7 := arithEq_of_rows h (row := 233) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_7
  simp only [← c15168, ← c15169, k3, ← c15170] at e_233_7
  have e_233_8 := arithEq_of_rows h (row := 233) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_8
  simp only [← c15177, ← c15178, k3, ← c15179] at e_233_8
  refine ⟨?_, ?_⟩
  · have hc := e_232_12
    simp only [e_233_7] at hc
    linear_combination c15180.trans k2 - hc
  · have hc := e_233_8
    simp only [e_233_6, e_232_13, e_233_7] at hc
    linear_combination c15181.trans k2 - hc

theorem publicBatchWrapper86_f2160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 75) = bor (a (.wire 35 47)) (a (.virt 18315)) := by
  have c15182 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15183 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15184 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15185 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15186 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15187 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_18 := arithEq_of_rows h (row := 223) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_18
  simp only [← c15185, ← c15186, k3, ← c15187] at e_223_18
  have e_228_10 := arithEq_of_rows h (row := 228) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_10
  simp only [← c15182, ← c15183, ← c15184] at e_228_10
  have hr := e_223_18
  simp only [e_228_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 75) = a (.virt 201) := by
  have c15188 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15188

theorem publicBatchWrapper86_f2162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7004)) (a (.wire 165 63)) (a (.virt 18317)) (a (.virt 18318)) := by
  have c15189 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15190 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15191 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15192 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15193 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15194 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15195 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15196 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15197 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15198 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15199 := (publicBatchWrapper86_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15200 := (publicBatchWrapper86_copies475 a h).1
  have c15201 := (publicBatchWrapper86_copies475 a h).2.1
  have c15202 := (publicBatchWrapper86_copies475 a h).2.2.1
  have c15203 := (publicBatchWrapper86_copies475 a h).2.2.2.1
  have c15204 := (publicBatchWrapper86_copies475 a h).2.2.2.2.1
  have c15205 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_232_14 := arithEq_of_rows h (row := 232) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_14
  simp only [← c15195, ← c15196, ← c15197] at e_232_14
  have e_232_15 := arithEq_of_rows h (row := 232) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_15
  simp only [← c15198, ← c15199, ← c15200] at e_232_15
  have e_233_9 := arithEq_of_rows h (row := 233) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_9
  simp only [← c15189, k3, ← c15190, k3, ← c15191] at e_233_9
  have e_233_10 := arithEq_of_rows h (row := 233) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_10
  simp only [← c15192, ← c15193, k3, ← c15194] at e_233_10
  have e_233_11 := arithEq_of_rows h (row := 233) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_11
  simp only [← c15201, ← c15202, k3, ← c15203] at e_233_11
  refine ⟨?_, ?_⟩
  · have hc := e_232_14
    simp only [e_233_10] at hc
    linear_combination c15204.trans k2 - hc
  · have hc := e_233_11
    simp only [e_233_9, e_232_15, e_233_10] at hc
    linear_combination c15205.trans k2 - hc

theorem publicBatchWrapper86_f2163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 79) = bor (a (.wire 35 47)) (a (.virt 18317)) := by
  have c15206 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.1
  have c15207 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.1
  have c15208 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.1
  have c15209 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.1
  have c15210 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15211 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_223_19 := arithEq_of_rows h (row := 223) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_19
  simp only [← c15209, ← c15210, k3, ← c15211] at e_223_19
  have e_228_11 := arithEq_of_rows h (row := 228) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_11
  simp only [← c15206, ← c15207, ← c15208] at e_228_11
  have hr := e_223_19
  simp only [e_228_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 223 79) = a (.virt 201) := by
  have c15212 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15212

theorem publicBatchWrapper86_f2165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7005)) (a (.wire 165 15)) (a (.virt 18319)) (a (.virt 18320)) := by
  have c15213 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15214 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15215 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15216 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15217 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15218 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15219 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15220 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15221 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15222 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15223 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15224 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15225 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15226 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15227 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15228 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15229 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_232_16 := arithEq_of_rows h (row := 232) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_16
  simp only [← c15219, ← c15220, ← c15221] at e_232_16
  have e_232_17 := arithEq_of_rows h (row := 232) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_17
  simp only [← c15222, ← c15223, ← c15224] at e_232_17
  have e_233_12 := arithEq_of_rows h (row := 233) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_12
  simp only [← c15213, k3, ← c15214, k3, ← c15215] at e_233_12
  have e_233_13 := arithEq_of_rows h (row := 233) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_13
  simp only [← c15216, ← c15217, k3, ← c15218] at e_233_13
  have e_233_14 := arithEq_of_rows h (row := 233) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_14
  simp only [← c15225, ← c15226, k3, ← c15227] at e_233_14
  refine ⟨?_, ?_⟩
  · have hc := e_232_16
    simp only [e_233_13] at hc
    linear_combination c15228.trans k2 - hc
  · have hc := e_233_14
    simp only [e_233_12, e_232_17, e_233_13] at hc
    linear_combination c15229.trans k2 - hc

theorem publicBatchWrapper86_f2166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7006)) (a (.wire 165 23)) (a (.virt 18321)) (a (.virt 18322)) := by
  have c15230 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15231 := (publicBatchWrapper86_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15232 := (publicBatchWrapper86_copies476 a h).1
  have c15233 := (publicBatchWrapper86_copies476 a h).2.1
  have c15234 := (publicBatchWrapper86_copies476 a h).2.2.1
  have c15235 := (publicBatchWrapper86_copies476 a h).2.2.2.1
  have c15236 := (publicBatchWrapper86_copies476 a h).2.2.2.2.1
  have c15237 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.1
  have c15238 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.1
  have c15239 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.1
  have c15240 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.1
  have c15241 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.1
  have c15242 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15243 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15244 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15245 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15246 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_232_18 := arithEq_of_rows h (row := 232) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_18
  simp only [← c15236, ← c15237, ← c15238] at e_232_18
  have e_232_19 := arithEq_of_rows h (row := 232) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_19
  simp only [← c15239, ← c15240, ← c15241] at e_232_19
  have e_233_15 := arithEq_of_rows h (row := 233) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_15
  simp only [← c15230, k3, ← c15231, k3, ← c15232] at e_233_15
  have e_233_16 := arithEq_of_rows h (row := 233) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_16
  simp only [← c15233, ← c15234, k3, ← c15235] at e_233_16
  have e_233_17 := arithEq_of_rows h (row := 233) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_17
  simp only [← c15242, ← c15243, k3, ← c15244] at e_233_17
  refine ⟨?_, ?_⟩
  · have hc := e_232_18
    simp only [e_233_16] at hc
    linear_combination c15245.trans k2 - hc
  · have hc := e_233_17
    simp only [e_233_15, e_232_19, e_233_16] at hc
    linear_combination c15246.trans k2 - hc

theorem publicBatchWrapper86_f2167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7007)) (a (.wire 165 31)) (a (.virt 18323)) (a (.virt 18324)) := by
  have c15247 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15248 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15249 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15250 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15251 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15252 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15253 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15254 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15255 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15256 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15257 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15258 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15259 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15260 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15261 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15262 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15263 := (publicBatchWrapper86_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_233_18 := arithEq_of_rows h (row := 233) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_18
  simp only [← c15247, k3, ← c15248, k3, ← c15249] at e_233_18
  have e_233_19 := arithEq_of_rows h (row := 233) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_19
  simp only [← c15250, ← c15251, k3, ← c15252] at e_233_19
  have e_234_0 := arithEq_of_rows h (row := 234) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_0
  simp only [← c15253, ← c15254, ← c15255] at e_234_0
  have e_234_1 := arithEq_of_rows h (row := 234) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_1
  simp only [← c15256, ← c15257, ← c15258] at e_234_1
  have e_235_0 := arithEq_of_rows h (row := 235) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_0
  simp only [← c15259, ← c15260, k3, ← c15261] at e_235_0
  refine ⟨?_, ?_⟩
  · have hc := e_234_0
    simp only [e_233_19] at hc
    linear_combination c15262.trans k2 - hc
  · have hc := e_235_0
    simp only [e_233_18, e_234_1, e_233_19] at hc
    linear_combination c15263.trans k2 - hc

theorem publicBatchWrapper86_f2168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7008)) (a (.wire 165 39)) (a (.virt 18325)) (a (.virt 18326)) := by
  have c15264 := (publicBatchWrapper86_copies477 a h).1
  have c15265 := (publicBatchWrapper86_copies477 a h).2.1
  have c15266 := (publicBatchWrapper86_copies477 a h).2.2.1
  have c15267 := (publicBatchWrapper86_copies477 a h).2.2.2.1
  have c15268 := (publicBatchWrapper86_copies477 a h).2.2.2.2.1
  have c15269 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.1
  have c15270 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.1
  have c15271 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.1
  have c15272 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.1
  have c15273 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.1
  have c15274 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15275 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15276 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15277 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15278 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15279 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15280 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_234_2 := arithEq_of_rows h (row := 234) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_2
  simp only [← c15270, ← c15271, ← c15272] at e_234_2
  have e_234_3 := arithEq_of_rows h (row := 234) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_3
  simp only [← c15273, ← c15274, ← c15275] at e_234_3
  have e_235_1 := arithEq_of_rows h (row := 235) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_1
  simp only [← c15264, k3, ← c15265, k3, ← c15266] at e_235_1
  have e_235_2 := arithEq_of_rows h (row := 235) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_2
  simp only [← c15267, ← c15268, k3, ← c15269] at e_235_2
  have e_235_3 := arithEq_of_rows h (row := 235) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_3
  simp only [← c15276, ← c15277, k3, ← c15278] at e_235_3
  refine ⟨?_, ?_⟩
  · have hc := e_234_2
    simp only [e_235_2] at hc
    linear_combination c15279.trans k2 - hc
  · have hc := e_235_3
    simp only [e_235_1, e_234_3, e_235_2] at hc
    linear_combination c15280.trans k2 - hc

theorem publicBatchWrapper86_f2169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 234 19) = band (a (.virt 18319)) (a (.virt 18321)) := by
  have c15281 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15282 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15283 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_234_4 := arithEq_of_rows h (row := 234) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_4
  simp only [← c15281, ← c15282, ← c15283] at e_234_4
  have hr := e_234_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 234 23) = band (a (.virt 18323)) (a (.virt 18325)) := by
  have c15284 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15285 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15286 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_234_5 := arithEq_of_rows h (row := 234) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_5
  simp only [← c15284, ← c15285, ← c15286] at e_234_5
  have hr := e_234_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 234 27) = band (a (.wire 234 19)) (a (.wire 234 23)) := by
  have c15287 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15288 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15289 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_234_6 := arithEq_of_rows h (row := 234) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_6
  simp only [← c15287, ← c15288, ← c15289] at e_234_6
  have hr := e_234_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 3) = bor (a (.wire 35 47)) (a (.wire 234 27)) := by
  have c15290 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15291 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15292 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15293 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15294 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15295 := (publicBatchWrapper86_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_228_12 := arithEq_of_rows h (row := 228) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_12
  simp only [← c15290, ← c15291, ← c15292] at e_228_12
  have e_236_0 := arithEq_of_rows h (row := 236) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_0
  simp only [← c15293, ← c15294, k3, ← c15295] at e_236_0
  have hr := e_236_0
  simp only [e_228_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 3) = a (.virt 201) := by
  have c15296 := (publicBatchWrapper86_copies478 a h).1
  exact c15296

theorem publicBatchWrapper86_f2174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7203)) (a (.wire 165 55)) (a (.virt 18327)) (a (.virt 18328)) := by
  have c15297 := (publicBatchWrapper86_copies478 a h).2.1
  have c15298 := (publicBatchWrapper86_copies478 a h).2.2.1
  have c15299 := (publicBatchWrapper86_copies478 a h).2.2.2.1
  have c15300 := (publicBatchWrapper86_copies478 a h).2.2.2.2.1
  have c15301 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.1
  have c15302 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.1
  have c15303 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.1
  have c15304 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.1
  have c15305 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.1
  have c15306 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15307 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15308 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15309 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15310 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15311 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15312 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15313 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_234_7 := arithEq_of_rows h (row := 234) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_7
  simp only [← c15303, ← c15304, ← c15305] at e_234_7
  have e_234_8 := arithEq_of_rows h (row := 234) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_8
  simp only [← c15306, ← c15307, ← c15308] at e_234_8
  have e_235_4 := arithEq_of_rows h (row := 235) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_4
  simp only [← c15297, k3, ← c15298, k3, ← c15299] at e_235_4
  have e_235_5 := arithEq_of_rows h (row := 235) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_5
  simp only [← c15300, ← c15301, k3, ← c15302] at e_235_5
  have e_235_6 := arithEq_of_rows h (row := 235) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_6
  simp only [← c15309, ← c15310, k3, ← c15311] at e_235_6
  refine ⟨?_, ?_⟩
  · have hc := e_234_7
    simp only [e_235_5] at hc
    linear_combination c15312.trans k2 - hc
  · have hc := e_235_6
    simp only [e_235_4, e_234_8, e_235_5] at hc
    linear_combination c15313.trans k2 - hc

theorem publicBatchWrapper86_f2175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 7) = bor (a (.wire 37 11)) (a (.virt 18327)) := by
  have c15314 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15315 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15316 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15317 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15318 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15319 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_228_13 := arithEq_of_rows h (row := 228) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_13
  simp only [← c15314, ← c15315, ← c15316] at e_228_13
  have e_236_1 := arithEq_of_rows h (row := 236) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_1
  simp only [← c15317, ← c15318, k3, ← c15319] at e_236_1
  have hr := e_236_1
  simp only [e_228_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 7) = a (.virt 201) := by
  have c15320 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15320

theorem publicBatchWrapper86_f2177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7204)) (a (.wire 165 63)) (a (.virt 18329)) (a (.virt 18330)) := by
  have c15321 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15322 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15323 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15324 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15325 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15326 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15327 := (publicBatchWrapper86_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15328 := (publicBatchWrapper86_copies479 a h).1
  have c15329 := (publicBatchWrapper86_copies479 a h).2.1
  have c15330 := (publicBatchWrapper86_copies479 a h).2.2.1
  have c15331 := (publicBatchWrapper86_copies479 a h).2.2.2.1
  have c15332 := (publicBatchWrapper86_copies479 a h).2.2.2.2.1
  have c15333 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.1
  have c15334 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.1
  have c15335 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.1
  have c15336 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.1
  have c15337 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_234_9 := arithEq_of_rows h (row := 234) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_9
  simp only [← c15327, ← c15328, ← c15329] at e_234_9
  have e_234_10 := arithEq_of_rows h (row := 234) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_10
  simp only [← c15330, ← c15331, ← c15332] at e_234_10
  have e_235_7 := arithEq_of_rows h (row := 235) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_7
  simp only [← c15321, k3, ← c15322, k3, ← c15323] at e_235_7
  have e_235_8 := arithEq_of_rows h (row := 235) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_8
  simp only [← c15324, ← c15325, k3, ← c15326] at e_235_8
  have e_235_9 := arithEq_of_rows h (row := 235) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_9
  simp only [← c15333, ← c15334, k3, ← c15335] at e_235_9
  refine ⟨?_, ?_⟩
  · have hc := e_234_9
    simp only [e_235_8] at hc
    linear_combination c15336.trans k2 - hc
  · have hc := e_235_9
    simp only [e_235_7, e_234_10, e_235_8] at hc
    linear_combination c15337.trans k2 - hc

theorem publicBatchWrapper86_f2178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 11) = bor (a (.wire 37 11)) (a (.virt 18329)) := by
  have c15338 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15339 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15340 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15341 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15342 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15343 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_228_14 := arithEq_of_rows h (row := 228) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_14
  simp only [← c15338, ← c15339, ← c15340] at e_228_14
  have e_236_2 := arithEq_of_rows h (row := 236) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_2
  simp only [← c15341, ← c15342, k3, ← c15343] at e_236_2
  have hr := e_236_2
  simp only [e_228_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 11) = a (.virt 201) := by
  have c15344 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15344

theorem publicBatchWrapper86_f2180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7205)) (a (.wire 165 15)) (a (.virt 18331)) (a (.virt 18332)) := by
  have c15345 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15346 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15347 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15348 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15349 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15350 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15351 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15352 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15353 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15354 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15355 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15356 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15357 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15358 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15359 := (publicBatchWrapper86_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15360 := (publicBatchWrapper86_copies480 a h).1
  have c15361 := (publicBatchWrapper86_copies480 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_234_11 := arithEq_of_rows h (row := 234) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_11
  simp only [← c15351, ← c15352, ← c15353] at e_234_11
  have e_234_12 := arithEq_of_rows h (row := 234) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_12
  simp only [← c15354, ← c15355, ← c15356] at e_234_12
  have e_235_10 := arithEq_of_rows h (row := 235) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_10
  simp only [← c15345, k3, ← c15346, k3, ← c15347] at e_235_10
  have e_235_11 := arithEq_of_rows h (row := 235) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_11
  simp only [← c15348, ← c15349, k3, ← c15350] at e_235_11
  have e_235_12 := arithEq_of_rows h (row := 235) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_12
  simp only [← c15357, ← c15358, k3, ← c15359] at e_235_12
  refine ⟨?_, ?_⟩
  · have hc := e_234_11
    simp only [e_235_11] at hc
    linear_combination c15360.trans k2 - hc
  · have hc := e_235_12
    simp only [e_235_10, e_234_12, e_235_11] at hc
    linear_combination c15361.trans k2 - hc

theorem publicBatchWrapper86_f2181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7206)) (a (.wire 165 23)) (a (.virt 18333)) (a (.virt 18334)) := by
  have c15362 := (publicBatchWrapper86_copies480 a h).2.2.1
  have c15363 := (publicBatchWrapper86_copies480 a h).2.2.2.1
  have c15364 := (publicBatchWrapper86_copies480 a h).2.2.2.2.1
  have c15365 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.1
  have c15366 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.1
  have c15367 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.1
  have c15368 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.1
  have c15369 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.1
  have c15370 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15371 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15372 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15373 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15374 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15375 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15376 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15377 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15378 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_234_13 := arithEq_of_rows h (row := 234) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_13
  simp only [← c15368, ← c15369, ← c15370] at e_234_13
  have e_234_14 := arithEq_of_rows h (row := 234) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_14
  simp only [← c15371, ← c15372, ← c15373] at e_234_14
  have e_235_13 := arithEq_of_rows h (row := 235) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_13
  simp only [← c15362, k3, ← c15363, k3, ← c15364] at e_235_13
  have e_235_14 := arithEq_of_rows h (row := 235) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_14
  simp only [← c15365, ← c15366, k3, ← c15367] at e_235_14
  have e_235_15 := arithEq_of_rows h (row := 235) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_15
  simp only [← c15374, ← c15375, k3, ← c15376] at e_235_15
  refine ⟨?_, ?_⟩
  · have hc := e_234_13
    simp only [e_235_14] at hc
    linear_combination c15377.trans k2 - hc
  · have hc := e_235_15
    simp only [e_235_13, e_234_14, e_235_14] at hc
    linear_combination c15378.trans k2 - hc

theorem publicBatchWrapper86_f2182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7207)) (a (.wire 165 31)) (a (.virt 18335)) (a (.virt 18336)) := by
  have c15379 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15380 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15381 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15382 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15383 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15384 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15385 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15386 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15387 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15388 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15389 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15390 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15391 := (publicBatchWrapper86_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15392 := (publicBatchWrapper86_copies481 a h).1
  have c15393 := (publicBatchWrapper86_copies481 a h).2.1
  have c15394 := (publicBatchWrapper86_copies481 a h).2.2.1
  have c15395 := (publicBatchWrapper86_copies481 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_234_15 := arithEq_of_rows h (row := 234) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_15
  simp only [← c15385, ← c15386, ← c15387] at e_234_15
  have e_234_16 := arithEq_of_rows h (row := 234) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_16
  simp only [← c15388, ← c15389, ← c15390] at e_234_16
  have e_235_16 := arithEq_of_rows h (row := 235) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_16
  simp only [← c15379, k3, ← c15380, k3, ← c15381] at e_235_16
  have e_235_17 := arithEq_of_rows h (row := 235) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_17
  simp only [← c15382, ← c15383, k3, ← c15384] at e_235_17
  have e_235_18 := arithEq_of_rows h (row := 235) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_18
  simp only [← c15391, ← c15392, k3, ← c15393] at e_235_18
  refine ⟨?_, ?_⟩
  · have hc := e_234_15
    simp only [e_235_17] at hc
    linear_combination c15394.trans k2 - hc
  · have hc := e_235_18
    simp only [e_235_16, e_234_16, e_235_17] at hc
    linear_combination c15395.trans k2 - hc

theorem publicBatchWrapper86_f2183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7208)) (a (.wire 165 39)) (a (.virt 18337)) (a (.virt 18338)) := by
  have c15396 := (publicBatchWrapper86_copies481 a h).2.2.2.2.1
  have c15397 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.1
  have c15398 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.1
  have c15399 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.1
  have c15400 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.1
  have c15401 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.1
  have c15402 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15403 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15404 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15405 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15406 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15407 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15408 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15409 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15410 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15411 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15412 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_234_17 := arithEq_of_rows h (row := 234) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_17
  simp only [← c15402, ← c15403, ← c15404] at e_234_17
  have e_234_18 := arithEq_of_rows h (row := 234) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_18
  simp only [← c15405, ← c15406, ← c15407] at e_234_18
  have e_235_19 := arithEq_of_rows h (row := 235) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_19
  simp only [← c15396, k3, ← c15397, k3, ← c15398] at e_235_19
  have e_237_0 := arithEq_of_rows h (row := 237) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_0
  simp only [← c15399, ← c15400, k3, ← c15401] at e_237_0
  have e_237_1 := arithEq_of_rows h (row := 237) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_1
  simp only [← c15408, ← c15409, k3, ← c15410] at e_237_1
  refine ⟨?_, ?_⟩
  · have hc := e_234_17
    simp only [e_237_0] at hc
    linear_combination c15411.trans k2 - hc
  · have hc := e_237_1
    simp only [e_235_19, e_234_18, e_237_0] at hc
    linear_combination c15412.trans k2 - hc

theorem publicBatchWrapper86_f2184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 234 79) = band (a (.virt 18331)) (a (.virt 18333)) := by
  have c15413 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15414 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15415 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_234_19 := arithEq_of_rows h (row := 234) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_19
  simp only [← c15413, ← c15414, ← c15415] at e_234_19
  have hr := e_234_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 238 3) = band (a (.virt 18335)) (a (.virt 18337)) := by
  have c15416 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15417 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15418 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_238_0 := arithEq_of_rows h (row := 238) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_0
  simp only [← c15416, ← c15417, ← c15418] at e_238_0
  have hr := e_238_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 238 7) = band (a (.wire 234 79)) (a (.wire 238 3)) := by
  have c15419 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15420 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15421 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_238_1 := arithEq_of_rows h (row := 238) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_1
  simp only [← c15419, ← c15420, ← c15421] at e_238_1
  have hr := e_238_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 15) = bor (a (.wire 37 11)) (a (.wire 238 7)) := by
  have c15422 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15423 := (publicBatchWrapper86_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15424 := (publicBatchWrapper86_copies482 a h).1
  have c15425 := (publicBatchWrapper86_copies482 a h).2.1
  have c15426 := (publicBatchWrapper86_copies482 a h).2.2.1
  have c15427 := (publicBatchWrapper86_copies482 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_228_15 := arithEq_of_rows h (row := 228) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_15
  simp only [← c15422, ← c15423, ← c15424] at e_228_15
  have e_236_3 := arithEq_of_rows h (row := 236) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_3
  simp only [← c15425, ← c15426, k3, ← c15427] at e_236_3
  have hr := e_236_3
  simp only [e_228_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 15) = a (.virt 201) := by
  have c15428 := (publicBatchWrapper86_copies482 a h).2.2.2.2.1
  exact c15428

theorem publicBatchWrapper86_f2189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7403)) (a (.wire 165 55)) (a (.virt 18339)) (a (.virt 18340)) := by
  have c15429 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.1
  have c15430 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.1
  have c15431 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.1
  have c15432 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.1
  have c15433 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.1
  have c15434 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15435 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15436 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15437 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15438 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15439 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15440 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15441 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15442 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15443 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15444 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15445 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_237_2 := arithEq_of_rows h (row := 237) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_2
  simp only [← c15429, k3, ← c15430, k3, ← c15431] at e_237_2
  have e_237_3 := arithEq_of_rows h (row := 237) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_3
  simp only [← c15432, ← c15433, k3, ← c15434] at e_237_3
  have e_237_4 := arithEq_of_rows h (row := 237) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_4
  simp only [← c15441, ← c15442, k3, ← c15443] at e_237_4
  have e_238_2 := arithEq_of_rows h (row := 238) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_2
  simp only [← c15435, ← c15436, ← c15437] at e_238_2
  have e_238_3 := arithEq_of_rows h (row := 238) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_3
  simp only [← c15438, ← c15439, ← c15440] at e_238_3
  refine ⟨?_, ?_⟩
  · have hc := e_238_2
    simp only [e_237_3] at hc
    linear_combination c15444.trans k2 - hc
  · have hc := e_237_4
    simp only [e_237_2, e_238_3, e_237_3] at hc
    linear_combination c15445.trans k2 - hc

theorem publicBatchWrapper86_f2190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 19) = bor (a (.wire 37 55)) (a (.virt 18339)) := by
  have c15446 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15447 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15448 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15449 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15450 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15451 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_228_16 := arithEq_of_rows h (row := 228) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_16
  simp only [← c15446, ← c15447, ← c15448] at e_228_16
  have e_236_4 := arithEq_of_rows h (row := 236) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_4
  simp only [← c15449, ← c15450, k3, ← c15451] at e_236_4
  have hr := e_236_4
  simp only [e_228_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 19) = a (.virt 201) := by
  have c15452 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15452

theorem publicBatchWrapper86_f2192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7404)) (a (.wire 165 63)) (a (.virt 18341)) (a (.virt 18342)) := by
  have c15453 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15454 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15455 := (publicBatchWrapper86_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15456 := (publicBatchWrapper86_copies483 a h).1
  have c15457 := (publicBatchWrapper86_copies483 a h).2.1
  have c15458 := (publicBatchWrapper86_copies483 a h).2.2.1
  have c15459 := (publicBatchWrapper86_copies483 a h).2.2.2.1
  have c15460 := (publicBatchWrapper86_copies483 a h).2.2.2.2.1
  have c15461 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.1
  have c15462 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.1
  have c15463 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.1
  have c15464 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.1
  have c15465 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.1
  have c15466 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15467 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15468 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15469 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_237_5 := arithEq_of_rows h (row := 237) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_5
  simp only [← c15453, k3, ← c15454, k3, ← c15455] at e_237_5
  have e_237_6 := arithEq_of_rows h (row := 237) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_6
  simp only [← c15456, ← c15457, k3, ← c15458] at e_237_6
  have e_237_7 := arithEq_of_rows h (row := 237) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_7
  simp only [← c15465, ← c15466, k3, ← c15467] at e_237_7
  have e_238_4 := arithEq_of_rows h (row := 238) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_4
  simp only [← c15459, ← c15460, ← c15461] at e_238_4
  have e_238_5 := arithEq_of_rows h (row := 238) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_5
  simp only [← c15462, ← c15463, ← c15464] at e_238_5
  refine ⟨?_, ?_⟩
  · have hc := e_238_4
    simp only [e_237_6] at hc
    linear_combination c15468.trans k2 - hc
  · have hc := e_237_7
    simp only [e_237_5, e_238_5, e_237_6] at hc
    linear_combination c15469.trans k2 - hc

theorem publicBatchWrapper86_f2193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 23) = bor (a (.wire 37 55)) (a (.virt 18341)) := by
  have c15470 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15471 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15472 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15473 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15474 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15475 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_228_17 := arithEq_of_rows h (row := 228) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_17
  simp only [← c15470, ← c15471, ← c15472] at e_228_17
  have e_236_5 := arithEq_of_rows h (row := 236) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_5
  simp only [← c15473, ← c15474, k3, ← c15475] at e_236_5
  have hr := e_236_5
  simp only [e_228_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 23) = a (.virt 201) := by
  have c15476 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15476

theorem publicBatchWrapper86_f2195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7405)) (a (.wire 165 15)) (a (.virt 18343)) (a (.virt 18344)) := by
  have c15477 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15478 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15479 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15480 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15481 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15482 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15483 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15484 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15485 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15486 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15487 := (publicBatchWrapper86_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15488 := (publicBatchWrapper86_copies484 a h).1
  have c15489 := (publicBatchWrapper86_copies484 a h).2.1
  have c15490 := (publicBatchWrapper86_copies484 a h).2.2.1
  have c15491 := (publicBatchWrapper86_copies484 a h).2.2.2.1
  have c15492 := (publicBatchWrapper86_copies484 a h).2.2.2.2.1
  have c15493 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_237_8 := arithEq_of_rows h (row := 237) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_8
  simp only [← c15477, k3, ← c15478, k3, ← c15479] at e_237_8
  have e_237_9 := arithEq_of_rows h (row := 237) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_9
  simp only [← c15480, ← c15481, k3, ← c15482] at e_237_9
  have e_237_10 := arithEq_of_rows h (row := 237) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_10
  simp only [← c15489, ← c15490, k3, ← c15491] at e_237_10
  have e_238_6 := arithEq_of_rows h (row := 238) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_6
  simp only [← c15483, ← c15484, ← c15485] at e_238_6
  have e_238_7 := arithEq_of_rows h (row := 238) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_7
  simp only [← c15486, ← c15487, ← c15488] at e_238_7
  refine ⟨?_, ?_⟩
  · have hc := e_238_6
    simp only [e_237_9] at hc
    linear_combination c15492.trans k2 - hc
  · have hc := e_237_10
    simp only [e_237_8, e_238_7, e_237_9] at hc
    linear_combination c15493.trans k2 - hc

theorem publicBatchWrapper86_f2196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7406)) (a (.wire 165 23)) (a (.virt 18345)) (a (.virt 18346)) := by
  have c15494 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.1
  have c15495 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.1
  have c15496 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.1
  have c15497 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.1
  have c15498 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15499 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15500 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15501 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15502 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15503 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15504 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15505 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15506 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15507 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15508 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15509 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15510 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_237_11 := arithEq_of_rows h (row := 237) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_11
  simp only [← c15494, k3, ← c15495, k3, ← c15496] at e_237_11
  have e_237_12 := arithEq_of_rows h (row := 237) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_12
  simp only [← c15497, ← c15498, k3, ← c15499] at e_237_12
  have e_237_13 := arithEq_of_rows h (row := 237) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_13
  simp only [← c15506, ← c15507, k3, ← c15508] at e_237_13
  have e_238_8 := arithEq_of_rows h (row := 238) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_8
  simp only [← c15500, ← c15501, ← c15502] at e_238_8
  have e_238_9 := arithEq_of_rows h (row := 238) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_9
  simp only [← c15503, ← c15504, ← c15505] at e_238_9
  refine ⟨?_, ?_⟩
  · have hc := e_238_8
    simp only [e_237_12] at hc
    linear_combination c15509.trans k2 - hc
  · have hc := e_237_13
    simp only [e_237_11, e_238_9, e_237_12] at hc
    linear_combination c15510.trans k2 - hc

theorem publicBatchWrapper86_f2197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7407)) (a (.wire 165 31)) (a (.virt 18347)) (a (.virt 18348)) := by
  have c15511 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15512 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15513 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15514 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15515 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15516 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15517 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15518 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15519 := (publicBatchWrapper86_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15520 := (publicBatchWrapper86_copies485 a h).1
  have c15521 := (publicBatchWrapper86_copies485 a h).2.1
  have c15522 := (publicBatchWrapper86_copies485 a h).2.2.1
  have c15523 := (publicBatchWrapper86_copies485 a h).2.2.2.1
  have c15524 := (publicBatchWrapper86_copies485 a h).2.2.2.2.1
  have c15525 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.1
  have c15526 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.1
  have c15527 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_237_14 := arithEq_of_rows h (row := 237) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_14
  simp only [← c15511, k3, ← c15512, k3, ← c15513] at e_237_14
  have e_237_15 := arithEq_of_rows h (row := 237) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_15
  simp only [← c15514, ← c15515, k3, ← c15516] at e_237_15
  have e_237_16 := arithEq_of_rows h (row := 237) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_16
  simp only [← c15523, ← c15524, k3, ← c15525] at e_237_16
  have e_238_10 := arithEq_of_rows h (row := 238) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_10
  simp only [← c15517, ← c15518, ← c15519] at e_238_10
  have e_238_11 := arithEq_of_rows h (row := 238) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_11
  simp only [← c15520, ← c15521, ← c15522] at e_238_11
  refine ⟨?_, ?_⟩
  · have hc := e_238_10
    simp only [e_237_15] at hc
    linear_combination c15526.trans k2 - hc
  · have hc := e_237_16
    simp only [e_237_14, e_238_11, e_237_15] at hc
    linear_combination c15527.trans k2 - hc

theorem publicBatchWrapper86_f2198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7408)) (a (.wire 165 39)) (a (.virt 18349)) (a (.virt 18350)) := by
  have c15528 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.1
  have c15529 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.1
  have c15530 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15531 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15532 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15533 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15534 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15535 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15536 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15537 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15538 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15539 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15540 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15541 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15542 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15543 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15544 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_237_17 := arithEq_of_rows h (row := 237) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_17
  simp only [← c15528, k3, ← c15529, k3, ← c15530] at e_237_17
  have e_237_18 := arithEq_of_rows h (row := 237) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_18
  simp only [← c15531, ← c15532, k3, ← c15533] at e_237_18
  have e_237_19 := arithEq_of_rows h (row := 237) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_19
  simp only [← c15540, ← c15541, k3, ← c15542] at e_237_19
  have e_238_12 := arithEq_of_rows h (row := 238) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_12
  simp only [← c15534, ← c15535, ← c15536] at e_238_12
  have e_238_13 := arithEq_of_rows h (row := 238) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_13
  simp only [← c15537, ← c15538, ← c15539] at e_238_13
  refine ⟨?_, ?_⟩
  · have hc := e_238_12
    simp only [e_237_18] at hc
    linear_combination c15543.trans k2 - hc
  · have hc := e_237_19
    simp only [e_237_17, e_238_13, e_237_18] at hc
    linear_combination c15544.trans k2 - hc

theorem publicBatchWrapper86_f2199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 238 59) = band (a (.virt 18343)) (a (.virt 18345)) := by
  have c15545 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15546 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15547 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_238_14 := arithEq_of_rows h (row := 238) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_14
  simp only [← c15545, ← c15546, ← c15547] at e_238_14
  have hr := e_238_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 238 63) = band (a (.virt 18347)) (a (.virt 18349)) := by
  have c15548 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15549 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15550 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_238_15 := arithEq_of_rows h (row := 238) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_15
  simp only [← c15548, ← c15549, ← c15550] at e_238_15
  have hr := e_238_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 238 67) = band (a (.wire 238 59)) (a (.wire 238 63)) := by
  have c15551 := (publicBatchWrapper86_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15552 := (publicBatchWrapper86_copies486 a h).1
  have c15553 := (publicBatchWrapper86_copies486 a h).2.1
  have e_238_16 := arithEq_of_rows h (row := 238) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_16
  simp only [← c15551, ← c15552, ← c15553] at e_238_16
  have hr := e_238_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 27) = bor (a (.wire 37 55)) (a (.wire 238 67)) := by
  have c15554 := (publicBatchWrapper86_copies486 a h).2.2.1
  have c15555 := (publicBatchWrapper86_copies486 a h).2.2.2.1
  have c15556 := (publicBatchWrapper86_copies486 a h).2.2.2.2.1
  have c15557 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.1
  have c15558 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.1
  have c15559 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_228_18 := arithEq_of_rows h (row := 228) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_18
  simp only [← c15554, ← c15555, ← c15556] at e_228_18
  have e_236_6 := arithEq_of_rows h (row := 236) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_6
  simp only [← c15557, ← c15558, k3, ← c15559] at e_236_6
  have hr := e_236_6
  simp only [e_228_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 27) = a (.virt 201) := by
  have c15560 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.1
  exact c15560

theorem publicBatchWrapper86_f2204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7603)) (a (.wire 165 55)) (a (.virt 18351)) (a (.virt 18352)) := by
  have c15561 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.1
  have c15562 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15563 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15564 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15565 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15566 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15567 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15568 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15569 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15570 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15571 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15572 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15573 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15574 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15575 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15576 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15577 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_238_17 := arithEq_of_rows h (row := 238) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_17
  simp only [← c15567, ← c15568, ← c15569] at e_238_17
  have e_238_18 := arithEq_of_rows h (row := 238) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_18
  simp only [← c15570, ← c15571, ← c15572] at e_238_18
  have e_239_0 := arithEq_of_rows h (row := 239) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_0
  simp only [← c15561, k3, ← c15562, k3, ← c15563] at e_239_0
  have e_239_1 := arithEq_of_rows h (row := 239) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_1
  simp only [← c15564, ← c15565, k3, ← c15566] at e_239_1
  have e_239_2 := arithEq_of_rows h (row := 239) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_2
  simp only [← c15573, ← c15574, k3, ← c15575] at e_239_2
  refine ⟨?_, ?_⟩
  · have hc := e_238_17
    simp only [e_239_1] at hc
    linear_combination c15576.trans k2 - hc
  · have hc := e_239_2
    simp only [e_239_0, e_238_18, e_239_1] at hc
    linear_combination c15577.trans k2 - hc

theorem publicBatchWrapper86_f2205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 31) = bor (a (.wire 39 19)) (a (.virt 18351)) := by
  have c15578 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15579 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15580 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15581 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15582 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15583 := (publicBatchWrapper86_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_228_19 := arithEq_of_rows h (row := 228) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_19
  simp only [← c15578, ← c15579, ← c15580] at e_228_19
  have e_236_7 := arithEq_of_rows h (row := 236) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_7
  simp only [← c15581, ← c15582, k3, ← c15583] at e_236_7
  have hr := e_236_7
  simp only [e_228_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 31) = a (.virt 201) := by
  have c15584 := (publicBatchWrapper86_copies487 a h).1
  exact c15584

theorem publicBatchWrapper86_f2207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7604)) (a (.wire 165 63)) (a (.virt 18353)) (a (.virt 18354)) := by
  have c15585 := (publicBatchWrapper86_copies487 a h).2.1
  have c15586 := (publicBatchWrapper86_copies487 a h).2.2.1
  have c15587 := (publicBatchWrapper86_copies487 a h).2.2.2.1
  have c15588 := (publicBatchWrapper86_copies487 a h).2.2.2.2.1
  have c15589 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.1
  have c15590 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.1
  have c15591 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.1
  have c15592 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.1
  have c15593 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.1
  have c15594 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15595 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15596 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15597 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15598 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15599 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15600 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15601 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_238_19 := arithEq_of_rows h (row := 238) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_19
  simp only [← c15591, ← c15592, ← c15593] at e_238_19
  have e_239_3 := arithEq_of_rows h (row := 239) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_3
  simp only [← c15585, k3, ← c15586, k3, ← c15587] at e_239_3
  have e_239_4 := arithEq_of_rows h (row := 239) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_4
  simp only [← c15588, ← c15589, k3, ← c15590] at e_239_4
  have e_239_5 := arithEq_of_rows h (row := 239) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_5
  simp only [← c15597, ← c15598, k3, ← c15599] at e_239_5
  have e_240_0 := arithEq_of_rows h (row := 240) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_0
  simp only [← c15594, ← c15595, ← c15596] at e_240_0
  refine ⟨?_, ?_⟩
  · have hc := e_238_19
    simp only [e_239_4] at hc
    linear_combination c15600.trans k2 - hc
  · have hc := e_239_5
    simp only [e_239_3, e_240_0, e_239_4] at hc
    linear_combination c15601.trans k2 - hc

theorem publicBatchWrapper86_f2208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 35) = bor (a (.wire 39 19)) (a (.virt 18353)) := by
  have c15602 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15603 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15604 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15605 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15606 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15607 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_8 := arithEq_of_rows h (row := 236) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_8
  simp only [← c15605, ← c15606, k3, ← c15607] at e_236_8
  have e_241_0 := arithEq_of_rows h (row := 241) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_0
  simp only [← c15602, ← c15603, ← c15604] at e_241_0
  have hr := e_236_8
  simp only [e_241_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 35) = a (.virt 201) := by
  have c15608 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15608

theorem publicBatchWrapper86_f2210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7605)) (a (.wire 165 15)) (a (.virt 18355)) (a (.virt 18356)) := by
  have c15609 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15610 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15611 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15612 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15613 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15614 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15615 := (publicBatchWrapper86_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15616 := (publicBatchWrapper86_copies488 a h).1
  have c15617 := (publicBatchWrapper86_copies488 a h).2.1
  have c15618 := (publicBatchWrapper86_copies488 a h).2.2.1
  have c15619 := (publicBatchWrapper86_copies488 a h).2.2.2.1
  have c15620 := (publicBatchWrapper86_copies488 a h).2.2.2.2.1
  have c15621 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.1
  have c15622 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.1
  have c15623 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.1
  have c15624 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.1
  have c15625 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_239_6 := arithEq_of_rows h (row := 239) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_6
  simp only [← c15609, k3, ← c15610, k3, ← c15611] at e_239_6
  have e_239_7 := arithEq_of_rows h (row := 239) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_7
  simp only [← c15612, ← c15613, k3, ← c15614] at e_239_7
  have e_239_8 := arithEq_of_rows h (row := 239) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_8
  simp only [← c15621, ← c15622, k3, ← c15623] at e_239_8
  have e_240_1 := arithEq_of_rows h (row := 240) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_1
  simp only [← c15615, ← c15616, ← c15617] at e_240_1
  have e_240_2 := arithEq_of_rows h (row := 240) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_2
  simp only [← c15618, ← c15619, ← c15620] at e_240_2
  refine ⟨?_, ?_⟩
  · have hc := e_240_1
    simp only [e_239_7] at hc
    linear_combination c15624.trans k2 - hc
  · have hc := e_239_8
    simp only [e_239_6, e_240_2, e_239_7] at hc
    linear_combination c15625.trans k2 - hc

theorem publicBatchWrapper86_f2211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7606)) (a (.wire 165 23)) (a (.virt 18357)) (a (.virt 18358)) := by
  have c15626 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15627 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15628 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15629 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15630 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15631 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15632 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15633 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15634 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15635 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15636 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15637 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15638 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15639 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15640 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15641 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15642 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_239_9 := arithEq_of_rows h (row := 239) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_9
  simp only [← c15626, k3, ← c15627, k3, ← c15628] at e_239_9
  have e_239_10 := arithEq_of_rows h (row := 239) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_10
  simp only [← c15629, ← c15630, k3, ← c15631] at e_239_10
  have e_239_11 := arithEq_of_rows h (row := 239) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_11
  simp only [← c15638, ← c15639, k3, ← c15640] at e_239_11
  have e_240_3 := arithEq_of_rows h (row := 240) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_3
  simp only [← c15632, ← c15633, ← c15634] at e_240_3
  have e_240_4 := arithEq_of_rows h (row := 240) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_4
  simp only [← c15635, ← c15636, ← c15637] at e_240_4
  refine ⟨?_, ?_⟩
  · have hc := e_240_3
    simp only [e_239_10] at hc
    linear_combination c15641.trans k2 - hc
  · have hc := e_239_11
    simp only [e_239_9, e_240_4, e_239_10] at hc
    linear_combination c15642.trans k2 - hc

theorem publicBatchWrapper86_f2212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7607)) (a (.wire 165 31)) (a (.virt 18359)) (a (.virt 18360)) := by
  have c15643 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15644 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15645 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15646 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15647 := (publicBatchWrapper86_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15648 := (publicBatchWrapper86_copies489 a h).1
  have c15649 := (publicBatchWrapper86_copies489 a h).2.1
  have c15650 := (publicBatchWrapper86_copies489 a h).2.2.1
  have c15651 := (publicBatchWrapper86_copies489 a h).2.2.2.1
  have c15652 := (publicBatchWrapper86_copies489 a h).2.2.2.2.1
  have c15653 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.1
  have c15654 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.1
  have c15655 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.1
  have c15656 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.1
  have c15657 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.1
  have c15658 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15659 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_239_12 := arithEq_of_rows h (row := 239) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_12
  simp only [← c15643, k3, ← c15644, k3, ← c15645] at e_239_12
  have e_239_13 := arithEq_of_rows h (row := 239) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_13
  simp only [← c15646, ← c15647, k3, ← c15648] at e_239_13
  have e_239_14 := arithEq_of_rows h (row := 239) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_14
  simp only [← c15655, ← c15656, k3, ← c15657] at e_239_14
  have e_240_5 := arithEq_of_rows h (row := 240) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_5
  simp only [← c15649, ← c15650, ← c15651] at e_240_5
  have e_240_6 := arithEq_of_rows h (row := 240) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_6
  simp only [← c15652, ← c15653, ← c15654] at e_240_6
  refine ⟨?_, ?_⟩
  · have hc := e_240_5
    simp only [e_239_13] at hc
    linear_combination c15658.trans k2 - hc
  · have hc := e_239_14
    simp only [e_239_12, e_240_6, e_239_13] at hc
    linear_combination c15659.trans k2 - hc

theorem publicBatchWrapper86_f2213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7608)) (a (.wire 165 39)) (a (.virt 18361)) (a (.virt 18362)) := by
  have c15660 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15661 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15662 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15663 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15664 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15665 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15666 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15667 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15668 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15669 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15670 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15671 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15672 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15673 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15674 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15675 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15676 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_239_15 := arithEq_of_rows h (row := 239) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_15
  simp only [← c15660, k3, ← c15661, k3, ← c15662] at e_239_15
  have e_239_16 := arithEq_of_rows h (row := 239) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_16
  simp only [← c15663, ← c15664, k3, ← c15665] at e_239_16
  have e_239_17 := arithEq_of_rows h (row := 239) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_17
  simp only [← c15672, ← c15673, k3, ← c15674] at e_239_17
  have e_240_7 := arithEq_of_rows h (row := 240) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_7
  simp only [← c15666, ← c15667, ← c15668] at e_240_7
  have e_240_8 := arithEq_of_rows h (row := 240) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_8
  simp only [← c15669, ← c15670, ← c15671] at e_240_8
  refine ⟨?_, ?_⟩
  · have hc := e_240_7
    simp only [e_239_16] at hc
    linear_combination c15675.trans k2 - hc
  · have hc := e_239_17
    simp only [e_239_15, e_240_8, e_239_16] at hc
    linear_combination c15676.trans k2 - hc

theorem publicBatchWrapper86_f2214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 240 39) = band (a (.virt 18355)) (a (.virt 18357)) := by
  have c15677 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15678 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15679 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_240_9 := arithEq_of_rows h (row := 240) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_9
  simp only [← c15677, ← c15678, ← c15679] at e_240_9
  have hr := e_240_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 240 43) = band (a (.virt 18359)) (a (.virt 18361)) := by
  have c15680 := (publicBatchWrapper86_copies490 a h).1
  have c15681 := (publicBatchWrapper86_copies490 a h).2.1
  have c15682 := (publicBatchWrapper86_copies490 a h).2.2.1
  have e_240_10 := arithEq_of_rows h (row := 240) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_10
  simp only [← c15680, ← c15681, ← c15682] at e_240_10
  have hr := e_240_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 240 47) = band (a (.wire 240 39)) (a (.wire 240 43)) := by
  have c15683 := (publicBatchWrapper86_copies490 a h).2.2.2.1
  have c15684 := (publicBatchWrapper86_copies490 a h).2.2.2.2.1
  have c15685 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.1
  have e_240_11 := arithEq_of_rows h (row := 240) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_11
  simp only [← c15683, ← c15684, ← c15685] at e_240_11
  have hr := e_240_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 39) = bor (a (.wire 39 19)) (a (.wire 240 47)) := by
  have c15686 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.1
  have c15687 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.1
  have c15688 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.1
  have c15689 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.1
  have c15690 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15691 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_9 := arithEq_of_rows h (row := 236) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_9
  simp only [← c15689, ← c15690, k3, ← c15691] at e_236_9
  have e_241_1 := arithEq_of_rows h (row := 241) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_1
  simp only [← c15686, ← c15687, ← c15688] at e_241_1
  have hr := e_236_9
  simp only [e_241_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 39) = a (.virt 201) := by
  have c15692 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15692

theorem publicBatchWrapper86_f2219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7803)) (a (.wire 165 55)) (a (.virt 18363)) (a (.virt 18364)) := by
  have c15693 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15694 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15695 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15696 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15697 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15698 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15699 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15700 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15701 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15702 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15703 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15704 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15705 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15706 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15707 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15708 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15709 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_239_18 := arithEq_of_rows h (row := 239) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_18
  simp only [← c15693, k3, ← c15694, k3, ← c15695] at e_239_18
  have e_239_19 := arithEq_of_rows h (row := 239) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_19
  simp only [← c15696, ← c15697, k3, ← c15698] at e_239_19
  have e_240_12 := arithEq_of_rows h (row := 240) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_12
  simp only [← c15699, ← c15700, ← c15701] at e_240_12
  have e_240_13 := arithEq_of_rows h (row := 240) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_13
  simp only [← c15702, ← c15703, ← c15704] at e_240_13
  have e_242_0 := arithEq_of_rows h (row := 242) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_0
  simp only [← c15705, ← c15706, k3, ← c15707] at e_242_0
  refine ⟨?_, ?_⟩
  · have hc := e_240_12
    simp only [e_239_19] at hc
    linear_combination c15708.trans k2 - hc
  · have hc := e_242_0
    simp only [e_239_18, e_240_13, e_239_19] at hc
    linear_combination c15709.trans k2 - hc

theorem publicBatchWrapper86_f2220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 43) = bor (a (.wire 39 63)) (a (.virt 18363)) := by
  have c15710 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15711 := (publicBatchWrapper86_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15712 := (publicBatchWrapper86_copies491 a h).1
  have c15713 := (publicBatchWrapper86_copies491 a h).2.1
  have c15714 := (publicBatchWrapper86_copies491 a h).2.2.1
  have c15715 := (publicBatchWrapper86_copies491 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_10 := arithEq_of_rows h (row := 236) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_10
  simp only [← c15713, ← c15714, k3, ← c15715] at e_236_10
  have e_241_2 := arithEq_of_rows h (row := 241) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_2
  simp only [← c15710, ← c15711, ← c15712] at e_241_2
  have hr := e_236_10
  simp only [e_241_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 43) = a (.virt 201) := by
  have c15716 := (publicBatchWrapper86_copies491 a h).2.2.2.2.1
  exact c15716

theorem publicBatchWrapper86_f2222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7804)) (a (.wire 165 63)) (a (.virt 18365)) (a (.virt 18366)) := by
  have c15717 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.1
  have c15718 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.1
  have c15719 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.1
  have c15720 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.1
  have c15721 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.1
  have c15722 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15723 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15724 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15725 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15726 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15727 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15728 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15729 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15730 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15731 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15732 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15733 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_240_14 := arithEq_of_rows h (row := 240) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_14
  simp only [← c15723, ← c15724, ← c15725] at e_240_14
  have e_240_15 := arithEq_of_rows h (row := 240) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_15
  simp only [← c15726, ← c15727, ← c15728] at e_240_15
  have e_242_1 := arithEq_of_rows h (row := 242) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_1
  simp only [← c15717, k3, ← c15718, k3, ← c15719] at e_242_1
  have e_242_2 := arithEq_of_rows h (row := 242) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_2
  simp only [← c15720, ← c15721, k3, ← c15722] at e_242_2
  have e_242_3 := arithEq_of_rows h (row := 242) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_3
  simp only [← c15729, ← c15730, k3, ← c15731] at e_242_3
  refine ⟨?_, ?_⟩
  · have hc := e_240_14
    simp only [e_242_2] at hc
    linear_combination c15732.trans k2 - hc
  · have hc := e_242_3
    simp only [e_242_1, e_240_15, e_242_2] at hc
    linear_combination c15733.trans k2 - hc

theorem publicBatchWrapper86_f2223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 47) = bor (a (.wire 39 63)) (a (.virt 18365)) := by
  have c15734 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15735 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15736 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15737 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15738 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15739 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_11 := arithEq_of_rows h (row := 236) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_11
  simp only [← c15737, ← c15738, k3, ← c15739] at e_236_11
  have e_241_3 := arithEq_of_rows h (row := 241) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_3
  simp only [← c15734, ← c15735, ← c15736] at e_241_3
  have hr := e_236_11
  simp only [e_241_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 47) = a (.virt 201) := by
  have c15740 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15740

theorem publicBatchWrapper86_f2225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7805)) (a (.wire 165 15)) (a (.virt 18367)) (a (.virt 18368)) := by
  have c15741 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15742 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15743 := (publicBatchWrapper86_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15744 := (publicBatchWrapper86_copies492 a h).1
  have c15745 := (publicBatchWrapper86_copies492 a h).2.1
  have c15746 := (publicBatchWrapper86_copies492 a h).2.2.1
  have c15747 := (publicBatchWrapper86_copies492 a h).2.2.2.1
  have c15748 := (publicBatchWrapper86_copies492 a h).2.2.2.2.1
  have c15749 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.1
  have c15750 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.1
  have c15751 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.1
  have c15752 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.1
  have c15753 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.1
  have c15754 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15755 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15756 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15757 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_240_16 := arithEq_of_rows h (row := 240) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_16
  simp only [← c15747, ← c15748, ← c15749] at e_240_16
  have e_240_17 := arithEq_of_rows h (row := 240) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_17
  simp only [← c15750, ← c15751, ← c15752] at e_240_17
  have e_242_4 := arithEq_of_rows h (row := 242) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_4
  simp only [← c15741, k3, ← c15742, k3, ← c15743] at e_242_4
  have e_242_5 := arithEq_of_rows h (row := 242) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_5
  simp only [← c15744, ← c15745, k3, ← c15746] at e_242_5
  have e_242_6 := arithEq_of_rows h (row := 242) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_6
  simp only [← c15753, ← c15754, k3, ← c15755] at e_242_6
  refine ⟨?_, ?_⟩
  · have hc := e_240_16
    simp only [e_242_5] at hc
    linear_combination c15756.trans k2 - hc
  · have hc := e_242_6
    simp only [e_242_4, e_240_17, e_242_5] at hc
    linear_combination c15757.trans k2 - hc

theorem publicBatchWrapper86_f2226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7806)) (a (.wire 165 23)) (a (.virt 18369)) (a (.virt 18370)) := by
  have c15758 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15759 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15760 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15761 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15762 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15763 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15764 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15765 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15766 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15767 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15768 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15769 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15770 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15771 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15772 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15773 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15774 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_240_18 := arithEq_of_rows h (row := 240) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_18
  simp only [← c15764, ← c15765, ← c15766] at e_240_18
  have e_240_19 := arithEq_of_rows h (row := 240) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_19
  simp only [← c15767, ← c15768, ← c15769] at e_240_19
  have e_242_7 := arithEq_of_rows h (row := 242) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_7
  simp only [← c15758, k3, ← c15759, k3, ← c15760] at e_242_7
  have e_242_8 := arithEq_of_rows h (row := 242) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_8
  simp only [← c15761, ← c15762, k3, ← c15763] at e_242_8
  have e_242_9 := arithEq_of_rows h (row := 242) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_9
  simp only [← c15770, ← c15771, k3, ← c15772] at e_242_9
  refine ⟨?_, ?_⟩
  · have hc := e_240_18
    simp only [e_242_8] at hc
    linear_combination c15773.trans k2 - hc
  · have hc := e_242_9
    simp only [e_242_7, e_240_19, e_242_8] at hc
    linear_combination c15774.trans k2 - hc

theorem publicBatchWrapper86_f2227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7807)) (a (.wire 165 31)) (a (.virt 18371)) (a (.virt 18372)) := by
  have c15775 := (publicBatchWrapper86_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15776 := (publicBatchWrapper86_copies493 a h).1
  have c15777 := (publicBatchWrapper86_copies493 a h).2.1
  have c15778 := (publicBatchWrapper86_copies493 a h).2.2.1
  have c15779 := (publicBatchWrapper86_copies493 a h).2.2.2.1
  have c15780 := (publicBatchWrapper86_copies493 a h).2.2.2.2.1
  have c15781 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.1
  have c15782 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.1
  have c15783 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.1
  have c15784 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.1
  have c15785 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.1
  have c15786 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15787 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15788 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15789 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15790 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15791 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_242_10 := arithEq_of_rows h (row := 242) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_10
  simp only [← c15775, k3, ← c15776, k3, ← c15777] at e_242_10
  have e_242_11 := arithEq_of_rows h (row := 242) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_11
  simp only [← c15778, ← c15779, k3, ← c15780] at e_242_11
  have e_242_12 := arithEq_of_rows h (row := 242) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_12
  simp only [← c15787, ← c15788, k3, ← c15789] at e_242_12
  have e_243_0 := arithEq_of_rows h (row := 243) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_0
  simp only [← c15781, ← c15782, ← c15783] at e_243_0
  have e_243_1 := arithEq_of_rows h (row := 243) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_1
  simp only [← c15784, ← c15785, ← c15786] at e_243_1
  refine ⟨?_, ?_⟩
  · have hc := e_243_0
    simp only [e_242_11] at hc
    linear_combination c15790.trans k2 - hc
  · have hc := e_242_12
    simp only [e_242_10, e_243_1, e_242_11] at hc
    linear_combination c15791.trans k2 - hc

theorem publicBatchWrapper86_f2228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7808)) (a (.wire 165 39)) (a (.virt 18373)) (a (.virt 18374)) := by
  have c15792 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15793 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15794 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15795 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15796 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15797 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15798 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15799 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15800 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15801 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15802 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15803 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15804 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15805 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15806 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15807 := (publicBatchWrapper86_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15808 := (publicBatchWrapper86_copies494 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_242_13 := arithEq_of_rows h (row := 242) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_13
  simp only [← c15792, k3, ← c15793, k3, ← c15794] at e_242_13
  have e_242_14 := arithEq_of_rows h (row := 242) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_14
  simp only [← c15795, ← c15796, k3, ← c15797] at e_242_14
  have e_242_15 := arithEq_of_rows h (row := 242) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_15
  simp only [← c15804, ← c15805, k3, ← c15806] at e_242_15
  have e_243_2 := arithEq_of_rows h (row := 243) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_2
  simp only [← c15798, ← c15799, ← c15800] at e_243_2
  have e_243_3 := arithEq_of_rows h (row := 243) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_3
  simp only [← c15801, ← c15802, ← c15803] at e_243_3
  refine ⟨?_, ?_⟩
  · have hc := e_243_2
    simp only [e_242_14] at hc
    linear_combination c15807.trans k2 - hc
  · have hc := e_242_15
    simp only [e_242_13, e_243_3, e_242_14] at hc
    linear_combination c15808.trans k2 - hc

theorem publicBatchWrapper86_f2229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 243 19) = band (a (.virt 18367)) (a (.virt 18369)) := by
  have c15809 := (publicBatchWrapper86_copies494 a h).2.1
  have c15810 := (publicBatchWrapper86_copies494 a h).2.2.1
  have c15811 := (publicBatchWrapper86_copies494 a h).2.2.2.1
  have e_243_4 := arithEq_of_rows h (row := 243) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_4
  simp only [← c15809, ← c15810, ← c15811] at e_243_4
  have hr := e_243_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 243 23) = band (a (.virt 18371)) (a (.virt 18373)) := by
  have c15812 := (publicBatchWrapper86_copies494 a h).2.2.2.2.1
  have c15813 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.1
  have c15814 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.1
  have e_243_5 := arithEq_of_rows h (row := 243) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_5
  simp only [← c15812, ← c15813, ← c15814] at e_243_5
  have hr := e_243_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 243 27) = band (a (.wire 243 19)) (a (.wire 243 23)) := by
  have c15815 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.1
  have c15816 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.1
  have c15817 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.1
  have e_243_6 := arithEq_of_rows h (row := 243) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_6
  simp only [← c15815, ← c15816, ← c15817] at e_243_6
  have hr := e_243_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 51) = bor (a (.wire 39 63)) (a (.wire 243 27)) := by
  have c15818 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15819 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15820 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15821 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15822 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15823 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_12 := arithEq_of_rows h (row := 236) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_12
  simp only [← c15821, ← c15822, k3, ← c15823] at e_236_12
  have e_241_4 := arithEq_of_rows h (row := 241) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_4
  simp only [← c15818, ← c15819, ← c15820] at e_241_4
  have hr := e_236_12
  simp only [e_241_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 51) = a (.virt 201) := by
  have c15824 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15824

theorem publicBatchWrapper86_f2234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8003)) (a (.wire 165 55)) (a (.virt 18375)) (a (.virt 18376)) := by
  have c15825 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15826 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15827 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15828 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15829 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15830 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15831 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15832 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15833 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15834 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15835 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15836 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15837 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15838 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15839 := (publicBatchWrapper86_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15840 := (publicBatchWrapper86_copies495 a h).1
  have c15841 := (publicBatchWrapper86_copies495 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_242_16 := arithEq_of_rows h (row := 242) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_16
  simp only [← c15825, k3, ← c15826, k3, ← c15827] at e_242_16
  have e_242_17 := arithEq_of_rows h (row := 242) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_17
  simp only [← c15828, ← c15829, k3, ← c15830] at e_242_17
  have e_242_18 := arithEq_of_rows h (row := 242) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_18
  simp only [← c15837, ← c15838, k3, ← c15839] at e_242_18
  have e_243_7 := arithEq_of_rows h (row := 243) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_7
  simp only [← c15831, ← c15832, ← c15833] at e_243_7
  have e_243_8 := arithEq_of_rows h (row := 243) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_8
  simp only [← c15834, ← c15835, ← c15836] at e_243_8
  refine ⟨?_, ?_⟩
  · have hc := e_243_7
    simp only [e_242_17] at hc
    linear_combination c15840.trans k2 - hc
  · have hc := e_242_18
    simp only [e_242_16, e_243_8, e_242_17] at hc
    linear_combination c15841.trans k2 - hc

theorem publicBatchWrapper86_f2235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 55) = bor (a (.wire 41 27)) (a (.virt 18375)) := by
  have c15842 := (publicBatchWrapper86_copies495 a h).2.2.1
  have c15843 := (publicBatchWrapper86_copies495 a h).2.2.2.1
  have c15844 := (publicBatchWrapper86_copies495 a h).2.2.2.2.1
  have c15845 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.1
  have c15846 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.1
  have c15847 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_13 := arithEq_of_rows h (row := 236) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_13
  simp only [← c15845, ← c15846, k3, ← c15847] at e_236_13
  have e_241_5 := arithEq_of_rows h (row := 241) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_5
  simp only [← c15842, ← c15843, ← c15844] at e_241_5
  have hr := e_236_13
  simp only [e_241_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 55) = a (.virt 201) := by
  have c15848 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.1
  exact c15848

theorem publicBatchWrapper86_f2237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8004)) (a (.wire 165 63)) (a (.virt 18377)) (a (.virt 18378)) := by
  have c15849 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.1
  have c15850 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15851 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15852 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15853 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15854 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15855 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15856 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15857 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15858 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15859 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15860 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15861 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15862 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15863 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15864 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15865 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_242_19 := arithEq_of_rows h (row := 242) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_19
  simp only [← c15849, k3, ← c15850, k3, ← c15851] at e_242_19
  have e_243_9 := arithEq_of_rows h (row := 243) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_9
  simp only [← c15855, ← c15856, ← c15857] at e_243_9
  have e_243_10 := arithEq_of_rows h (row := 243) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_10
  simp only [← c15858, ← c15859, ← c15860] at e_243_10
  have e_244_0 := arithEq_of_rows h (row := 244) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_0
  simp only [← c15852, ← c15853, k3, ← c15854] at e_244_0
  have e_244_1 := arithEq_of_rows h (row := 244) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_1
  simp only [← c15861, ← c15862, k3, ← c15863] at e_244_1
  refine ⟨?_, ?_⟩
  · have hc := e_243_9
    simp only [e_244_0] at hc
    linear_combination c15864.trans k2 - hc
  · have hc := e_244_1
    simp only [e_242_19, e_243_10, e_244_0] at hc
    linear_combination c15865.trans k2 - hc

theorem publicBatchWrapper86_f2238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 59) = bor (a (.wire 41 27)) (a (.virt 18377)) := by
  have c15866 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15867 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15868 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15869 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15870 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15871 := (publicBatchWrapper86_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_14 := arithEq_of_rows h (row := 236) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_14
  simp only [← c15869, ← c15870, k3, ← c15871] at e_236_14
  have e_241_6 := arithEq_of_rows h (row := 241) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_6
  simp only [← c15866, ← c15867, ← c15868] at e_241_6
  have hr := e_236_14
  simp only [e_241_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 59) = a (.virt 201) := by
  have c15872 := (publicBatchWrapper86_copies496 a h).1
  exact c15872

theorem publicBatchWrapper86_f2240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8005)) (a (.wire 165 15)) (a (.virt 18379)) (a (.virt 18380)) := by
  have c15873 := (publicBatchWrapper86_copies496 a h).2.1
  have c15874 := (publicBatchWrapper86_copies496 a h).2.2.1
  have c15875 := (publicBatchWrapper86_copies496 a h).2.2.2.1
  have c15876 := (publicBatchWrapper86_copies496 a h).2.2.2.2.1
  have c15877 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.1
  have c15878 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.1
  have c15879 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.1
  have c15880 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.1
  have c15881 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.1
  have c15882 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15883 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15884 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15885 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15886 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15887 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15888 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15889 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_243_11 := arithEq_of_rows h (row := 243) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_11
  simp only [← c15879, ← c15880, ← c15881] at e_243_11
  have e_243_12 := arithEq_of_rows h (row := 243) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_12
  simp only [← c15882, ← c15883, ← c15884] at e_243_12
  have e_244_2 := arithEq_of_rows h (row := 244) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_2
  simp only [← c15873, k3, ← c15874, k3, ← c15875] at e_244_2
  have e_244_3 := arithEq_of_rows h (row := 244) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_3
  simp only [← c15876, ← c15877, k3, ← c15878] at e_244_3
  have e_244_4 := arithEq_of_rows h (row := 244) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_4
  simp only [← c15885, ← c15886, k3, ← c15887] at e_244_4
  refine ⟨?_, ?_⟩
  · have hc := e_243_11
    simp only [e_244_3] at hc
    linear_combination c15888.trans k2 - hc
  · have hc := e_244_4
    simp only [e_244_2, e_243_12, e_244_3] at hc
    linear_combination c15889.trans k2 - hc

theorem publicBatchWrapper86_f2241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8006)) (a (.wire 165 23)) (a (.virt 18381)) (a (.virt 18382)) := by
  have c15890 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15891 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15892 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15893 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15894 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15895 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15896 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15897 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15898 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15899 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15900 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15901 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15902 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15903 := (publicBatchWrapper86_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15904 := (publicBatchWrapper86_copies497 a h).1
  have c15905 := (publicBatchWrapper86_copies497 a h).2.1
  have c15906 := (publicBatchWrapper86_copies497 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_243_13 := arithEq_of_rows h (row := 243) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_13
  simp only [← c15896, ← c15897, ← c15898] at e_243_13
  have e_243_14 := arithEq_of_rows h (row := 243) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_14
  simp only [← c15899, ← c15900, ← c15901] at e_243_14
  have e_244_5 := arithEq_of_rows h (row := 244) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_5
  simp only [← c15890, k3, ← c15891, k3, ← c15892] at e_244_5
  have e_244_6 := arithEq_of_rows h (row := 244) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_6
  simp only [← c15893, ← c15894, k3, ← c15895] at e_244_6
  have e_244_7 := arithEq_of_rows h (row := 244) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_7
  simp only [← c15902, ← c15903, k3, ← c15904] at e_244_7
  refine ⟨?_, ?_⟩
  · have hc := e_243_13
    simp only [e_244_6] at hc
    linear_combination c15905.trans k2 - hc
  · have hc := e_244_7
    simp only [e_244_5, e_243_14, e_244_6] at hc
    linear_combination c15906.trans k2 - hc

theorem publicBatchWrapper86_f2242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8007)) (a (.wire 165 31)) (a (.virt 18383)) (a (.virt 18384)) := by
  have c15907 := (publicBatchWrapper86_copies497 a h).2.2.2.1
  have c15908 := (publicBatchWrapper86_copies497 a h).2.2.2.2.1
  have c15909 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.1
  have c15910 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.1
  have c15911 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.1
  have c15912 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.1
  have c15913 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.1
  have c15914 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15915 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15916 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15917 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15918 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15919 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15920 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15921 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15922 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15923 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_243_15 := arithEq_of_rows h (row := 243) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_15
  simp only [← c15913, ← c15914, ← c15915] at e_243_15
  have e_243_16 := arithEq_of_rows h (row := 243) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_16
  simp only [← c15916, ← c15917, ← c15918] at e_243_16
  have e_244_8 := arithEq_of_rows h (row := 244) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_8
  simp only [← c15907, k3, ← c15908, k3, ← c15909] at e_244_8
  have e_244_9 := arithEq_of_rows h (row := 244) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_9
  simp only [← c15910, ← c15911, k3, ← c15912] at e_244_9
  have e_244_10 := arithEq_of_rows h (row := 244) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_10
  simp only [← c15919, ← c15920, k3, ← c15921] at e_244_10
  refine ⟨?_, ?_⟩
  · have hc := e_243_15
    simp only [e_244_9] at hc
    linear_combination c15922.trans k2 - hc
  · have hc := e_244_10
    simp only [e_244_8, e_243_16, e_244_9] at hc
    linear_combination c15923.trans k2 - hc

theorem publicBatchWrapper86_f2243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8008)) (a (.wire 165 39)) (a (.virt 18385)) (a (.virt 18386)) := by
  have c15924 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15925 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15926 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15927 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15928 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15929 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15930 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15931 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15932 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15933 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15934 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15935 := (publicBatchWrapper86_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15936 := (publicBatchWrapper86_copies498 a h).1
  have c15937 := (publicBatchWrapper86_copies498 a h).2.1
  have c15938 := (publicBatchWrapper86_copies498 a h).2.2.1
  have c15939 := (publicBatchWrapper86_copies498 a h).2.2.2.1
  have c15940 := (publicBatchWrapper86_copies498 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_243_17 := arithEq_of_rows h (row := 243) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_17
  simp only [← c15930, ← c15931, ← c15932] at e_243_17
  have e_243_18 := arithEq_of_rows h (row := 243) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_18
  simp only [← c15933, ← c15934, ← c15935] at e_243_18
  have e_244_11 := arithEq_of_rows h (row := 244) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_11
  simp only [← c15924, k3, ← c15925, k3, ← c15926] at e_244_11
  have e_244_12 := arithEq_of_rows h (row := 244) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_12
  simp only [← c15927, ← c15928, k3, ← c15929] at e_244_12
  have e_244_13 := arithEq_of_rows h (row := 244) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_13
  simp only [← c15936, ← c15937, k3, ← c15938] at e_244_13
  refine ⟨?_, ?_⟩
  · have hc := e_243_17
    simp only [e_244_12] at hc
    linear_combination c15939.trans k2 - hc
  · have hc := e_244_13
    simp only [e_244_11, e_243_18, e_244_12] at hc
    linear_combination c15940.trans k2 - hc

theorem publicBatchWrapper86_f2244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 243 79) = band (a (.virt 18379)) (a (.virt 18381)) := by
  have c15941 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.1
  have c15942 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.1
  have c15943 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.1
  have e_243_19 := arithEq_of_rows h (row := 243) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_19
  simp only [← c15941, ← c15942, ← c15943] at e_243_19
  have hr := e_243_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 245 3) = band (a (.virt 18383)) (a (.virt 18385)) := by
  have c15944 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.1
  have c15945 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.1
  have c15946 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_245_0 := arithEq_of_rows h (row := 245) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_0
  simp only [← c15944, ← c15945, ← c15946] at e_245_0
  have hr := e_245_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 245 7) = band (a (.wire 243 79)) (a (.wire 245 3)) := by
  have c15947 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15948 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15949 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_245_1 := arithEq_of_rows h (row := 245) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_1
  simp only [← c15947, ← c15948, ← c15949] at e_245_1
  have hr := e_245_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 63) = bor (a (.wire 41 27)) (a (.wire 245 7)) := by
  have c15950 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15951 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15952 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15953 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15954 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15955 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_15 := arithEq_of_rows h (row := 236) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_15
  simp only [← c15953, ← c15954, k3, ← c15955] at e_236_15
  have e_241_7 := arithEq_of_rows h (row := 241) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_7
  simp only [← c15950, ← c15951, ← c15952] at e_241_7
  have hr := e_236_15
  simp only [e_241_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 63) = a (.virt 201) := by
  have c15956 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15956

end Plonky2Spec.Generated
