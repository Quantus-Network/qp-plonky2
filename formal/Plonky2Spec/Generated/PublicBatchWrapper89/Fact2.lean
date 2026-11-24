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

theorem publicBatchWrapper89_f1614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 67) = bnot (a (.wire 79 67)) := by
  have c10515 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10516 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10517 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_163_16 := arithEq_of_rows h (row := 163) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_16
  simp only [← c10515, k3, ← c10516, k3, ← c10517] at e_163_16
  have hr := e_163_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 71) = bnot (a (.wire 153 51)) := by
  have c10518 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10519 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10520 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_163_17 := arithEq_of_rows h (row := 163) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_17
  simp only [← c10518, k3, ← c10519, k3, ← c10520] at e_163_17
  have hr := e_163_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 19) = band (a (.wire 163 67)) (a (.wire 163 71)) := by
  have c10521 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10522 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10523 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_4 := arithEq_of_rows h (row := 160) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_4
  simp only [← c10521, ← c10522, ← c10523] at e_160_4
  have hr := e_160_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 79) = bselect (a (.wire 160 19)) (a (.virt 16405)) (a (.wire 163 15)) := by
  have c10524 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10525 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10526 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10527 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10528 := (publicBatchWrapper89_copies329 a h).1
  have c10529 := (publicBatchWrapper89_copies329 a h).2.1
  have e_163_18 := arithEq_of_rows h (row := 163) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_18
  simp only [← c10524, ← c10525, ← c10526] at e_163_18
  have e_163_19 := arithEq_of_rows h (row := 163) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_19
  simp only [← c10527, ← c10528, ← c10529] at e_163_19
  have hr := e_163_19
  simp only [e_163_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 7) = bselect (a (.wire 160 19)) (a (.virt 16406)) (a (.wire 163 23)) := by
  have c10530 := (publicBatchWrapper89_copies329 a h).2.2.1
  have c10531 := (publicBatchWrapper89_copies329 a h).2.2.2.1
  have c10532 := (publicBatchWrapper89_copies329 a h).2.2.2.2.1
  have c10533 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.1
  have c10534 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.1
  have c10535 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.1
  have e_165_0 := arithEq_of_rows h (row := 165) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_0
  simp only [← c10530, ← c10531, ← c10532] at e_165_0
  have e_165_1 := arithEq_of_rows h (row := 165) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_1
  simp only [← c10533, ← c10534, ← c10535] at e_165_1
  have hr := e_165_1
  simp only [e_165_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 15) = bselect (a (.wire 160 19)) (a (.virt 16407)) (a (.wire 163 31)) := by
  have c10536 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.1
  have c10537 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.1
  have c10538 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10539 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10540 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10541 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_2 := arithEq_of_rows h (row := 165) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_2
  simp only [← c10536, ← c10537, ← c10538] at e_165_2
  have e_165_3 := arithEq_of_rows h (row := 165) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_3
  simp only [← c10539, ← c10540, ← c10541] at e_165_3
  have hr := e_165_3
  simp only [e_165_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 23) = bselect (a (.wire 160 19)) (a (.virt 16408)) (a (.wire 163 39)) := by
  have c10542 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10543 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10544 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10545 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10546 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10547 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_4 := arithEq_of_rows h (row := 165) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_4
  simp only [← c10542, ← c10543, ← c10544] at e_165_4
  have e_165_5 := arithEq_of_rows h (row := 165) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_5
  simp only [← c10545, ← c10546, ← c10547] at e_165_5
  have hr := e_165_5
  simp only [e_165_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 31) = bselect (a (.wire 160 19)) (a (.virt 16409)) (a (.wire 163 47)) := by
  have c10548 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10549 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10550 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10551 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10552 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10553 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_6 := arithEq_of_rows h (row := 165) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_6
  simp only [← c10548, ← c10549, ← c10550] at e_165_6
  have e_165_7 := arithEq_of_rows h (row := 165) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_7
  simp only [← c10551, ← c10552, ← c10553] at e_165_7
  have hr := e_165_7
  simp only [e_165_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 39) = bselect (a (.wire 160 19)) (a (.virt 16403)) (a (.wire 163 55)) := by
  have c10554 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10555 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10556 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10557 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10558 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10559 := (publicBatchWrapper89_copies329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_165_8 := arithEq_of_rows h (row := 165) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_8
  simp only [← c10554, ← c10555, ← c10556] at e_165_8
  have e_165_9 := arithEq_of_rows h (row := 165) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_9
  simp only [← c10557, ← c10558, ← c10559] at e_165_9
  have hr := e_165_9
  simp only [e_165_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 47) = bselect (a (.wire 160 19)) (a (.virt 16404)) (a (.wire 163 63)) := by
  have c10560 := (publicBatchWrapper89_copies330 a h).1
  have c10561 := (publicBatchWrapper89_copies330 a h).2.1
  have c10562 := (publicBatchWrapper89_copies330 a h).2.2.1
  have c10563 := (publicBatchWrapper89_copies330 a h).2.2.2.1
  have c10564 := (publicBatchWrapper89_copies330 a h).2.2.2.2.1
  have c10565 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.1
  have e_165_10 := arithEq_of_rows h (row := 165) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_10
  simp only [← c10560, ← c10561, ← c10562] at e_165_10
  have e_165_11 := arithEq_of_rows h (row := 165) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_11
  simp only [← c10563, ← c10564, ← c10565] at e_165_11
  have hr := e_165_11
  simp only [e_165_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 55) = bor (a (.wire 153 51)) (a (.wire 163 67)) := by
  have c10566 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.1
  have c10567 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.1
  have c10568 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.1
  have c10569 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.1
  have c10570 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10571 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_153_13 := arithEq_of_rows h (row := 153) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_13
  simp only [← c10569, ← c10570, k3, ← c10571] at e_153_13
  have e_164_1 := arithEq_of_rows h (row := 164) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_1
  simp only [← c10566, ← c10567, ← c10568] at e_164_1
  have hr := e_153_13
  simp only [e_164_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 51) = bnot (a (.wire 81 31)) := by
  have c10572 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10573 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10574 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_165_12 := arithEq_of_rows h (row := 165) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_12
  simp only [← c10572, k3, ← c10573, k3, ← c10574] at e_165_12
  have hr := e_165_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 55) = bnot (a (.wire 153 55)) := by
  have c10575 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10576 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10577 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_165_13 := arithEq_of_rows h (row := 165) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_13
  simp only [← c10575, k3, ← c10576, k3, ← c10577] at e_165_13
  have hr := e_165_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 23) = band (a (.wire 165 51)) (a (.wire 165 55)) := by
  have c10578 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10579 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10580 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_5 := arithEq_of_rows h (row := 160) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_5
  simp only [← c10578, ← c10579, ← c10580] at e_160_5
  have hr := e_160_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 63) = bselect (a (.wire 160 23)) (a (.virt 16605)) (a (.wire 163 79)) := by
  have c10581 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10582 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10583 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10584 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10585 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10586 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_165_14 := arithEq_of_rows h (row := 165) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_14
  simp only [← c10581, ← c10582, ← c10583] at e_165_14
  have e_165_15 := arithEq_of_rows h (row := 165) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_15
  simp only [← c10584, ← c10585, ← c10586] at e_165_15
  have hr := e_165_15
  simp only [e_165_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 71) = bselect (a (.wire 160 23)) (a (.virt 16606)) (a (.wire 165 7)) := by
  have c10587 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10588 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10589 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10590 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10591 := (publicBatchWrapper89_copies330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10592 := (publicBatchWrapper89_copies331 a h).1
  have e_165_16 := arithEq_of_rows h (row := 165) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_16
  simp only [← c10587, ← c10588, ← c10589] at e_165_16
  have e_165_17 := arithEq_of_rows h (row := 165) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_17
  simp only [← c10590, ← c10591, ← c10592] at e_165_17
  have hr := e_165_17
  simp only [e_165_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 165 79) = bselect (a (.wire 160 23)) (a (.virt 16607)) (a (.wire 165 15)) := by
  have c10593 := (publicBatchWrapper89_copies331 a h).2.1
  have c10594 := (publicBatchWrapper89_copies331 a h).2.2.1
  have c10595 := (publicBatchWrapper89_copies331 a h).2.2.2.1
  have c10596 := (publicBatchWrapper89_copies331 a h).2.2.2.2.1
  have c10597 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.1
  have c10598 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.1
  have e_165_18 := arithEq_of_rows h (row := 165) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_18
  simp only [← c10593, ← c10594, ← c10595] at e_165_18
  have e_165_19 := arithEq_of_rows h (row := 165) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_19
  simp only [← c10596, ← c10597, ← c10598] at e_165_19
  have hr := e_165_19
  simp only [e_165_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 7) = bselect (a (.wire 160 23)) (a (.virt 16608)) (a (.wire 165 23)) := by
  have c10599 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.1
  have c10600 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.1
  have c10601 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.1
  have c10602 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10603 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10604 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_166_0 := arithEq_of_rows h (row := 166) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_0
  simp only [← c10599, ← c10600, ← c10601] at e_166_0
  have e_166_1 := arithEq_of_rows h (row := 166) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_1
  simp only [← c10602, ← c10603, ← c10604] at e_166_1
  have hr := e_166_1
  simp only [e_166_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 15) = bselect (a (.wire 160 23)) (a (.virt 16609)) (a (.wire 165 31)) := by
  have c10605 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10606 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10607 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10608 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10609 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10610 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_166_2 := arithEq_of_rows h (row := 166) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_2
  simp only [← c10605, ← c10606, ← c10607] at e_166_2
  have e_166_3 := arithEq_of_rows h (row := 166) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_3
  simp only [← c10608, ← c10609, ← c10610] at e_166_3
  have hr := e_166_3
  simp only [e_166_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 23) = bselect (a (.wire 160 23)) (a (.virt 16603)) (a (.wire 165 39)) := by
  have c10611 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10612 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10613 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10614 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10615 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10616 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_166_4 := arithEq_of_rows h (row := 166) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_4
  simp only [← c10611, ← c10612, ← c10613] at e_166_4
  have e_166_5 := arithEq_of_rows h (row := 166) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_5
  simp only [← c10614, ← c10615, ← c10616] at e_166_5
  have hr := e_166_5
  simp only [e_166_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 31) = bselect (a (.wire 160 23)) (a (.virt 16604)) (a (.wire 165 47)) := by
  have c10617 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10618 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10619 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10620 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10621 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10622 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_166_6 := arithEq_of_rows h (row := 166) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_6
  simp only [← c10617, ← c10618, ← c10619] at e_166_6
  have e_166_7 := arithEq_of_rows h (row := 166) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_7
  simp only [← c10620, ← c10621, ← c10622] at e_166_7
  have hr := e_166_7
  simp only [e_166_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 59) = bor (a (.wire 153 55)) (a (.wire 165 51)) := by
  have c10623 := (publicBatchWrapper89_copies331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10624 := (publicBatchWrapper89_copies332 a h).1
  have c10625 := (publicBatchWrapper89_copies332 a h).2.1
  have c10626 := (publicBatchWrapper89_copies332 a h).2.2.1
  have c10627 := (publicBatchWrapper89_copies332 a h).2.2.2.1
  have c10628 := (publicBatchWrapper89_copies332 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_153_14 := arithEq_of_rows h (row := 153) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_14
  simp only [← c10626, ← c10627, k3, ← c10628] at e_153_14
  have e_164_2 := arithEq_of_rows h (row := 164) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_2
  simp only [← c10623, ← c10624, ← c10625] at e_164_2
  have hr := e_153_14
  simp only [e_164_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 35) = bnot (a (.wire 81 75)) := by
  have c10629 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.1
  have c10630 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.1
  have c10631 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_166_8 := arithEq_of_rows h (row := 166) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_8
  simp only [← c10629, k3, ← c10630, k3, ← c10631] at e_166_8
  have hr := e_166_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 39) = bnot (a (.wire 153 59)) := by
  have c10632 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.1
  have c10633 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.1
  have c10634 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_166_9 := arithEq_of_rows h (row := 166) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_9
  simp only [← c10632, k3, ← c10633, k3, ← c10634] at e_166_9
  have hr := e_166_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 27) = band (a (.wire 166 35)) (a (.wire 166 39)) := by
  have c10635 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10636 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10637 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_6 := arithEq_of_rows h (row := 160) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_6
  simp only [← c10635, ← c10636, ← c10637] at e_160_6
  have hr := e_160_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 47) = bselect (a (.wire 160 27)) (a (.virt 16805)) (a (.wire 165 63)) := by
  have c10638 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10639 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10640 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10641 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10642 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10643 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_166_10 := arithEq_of_rows h (row := 166) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_10
  simp only [← c10638, ← c10639, ← c10640] at e_166_10
  have e_166_11 := arithEq_of_rows h (row := 166) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_11
  simp only [← c10641, ← c10642, ← c10643] at e_166_11
  have hr := e_166_11
  simp only [e_166_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 55) = bselect (a (.wire 160 27)) (a (.virt 16806)) (a (.wire 165 71)) := by
  have c10644 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10645 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10646 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10647 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10648 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10649 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_166_12 := arithEq_of_rows h (row := 166) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_12
  simp only [← c10644, ← c10645, ← c10646] at e_166_12
  have e_166_13 := arithEq_of_rows h (row := 166) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_13
  simp only [← c10647, ← c10648, ← c10649] at e_166_13
  have hr := e_166_13
  simp only [e_166_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 63) = bselect (a (.wire 160 27)) (a (.virt 16807)) (a (.wire 165 79)) := by
  have c10650 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10651 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10652 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10653 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10654 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10655 := (publicBatchWrapper89_copies332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_166_14 := arithEq_of_rows h (row := 166) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_14
  simp only [← c10650, ← c10651, ← c10652] at e_166_14
  have e_166_15 := arithEq_of_rows h (row := 166) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_15
  simp only [← c10653, ← c10654, ← c10655] at e_166_15
  have hr := e_166_15
  simp only [e_166_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 71) = bselect (a (.wire 160 27)) (a (.virt 16808)) (a (.wire 166 7)) := by
  have c10656 := (publicBatchWrapper89_copies333 a h).1
  have c10657 := (publicBatchWrapper89_copies333 a h).2.1
  have c10658 := (publicBatchWrapper89_copies333 a h).2.2.1
  have c10659 := (publicBatchWrapper89_copies333 a h).2.2.2.1
  have c10660 := (publicBatchWrapper89_copies333 a h).2.2.2.2.1
  have c10661 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.1
  have e_166_16 := arithEq_of_rows h (row := 166) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_16
  simp only [← c10656, ← c10657, ← c10658] at e_166_16
  have e_166_17 := arithEq_of_rows h (row := 166) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_17
  simp only [← c10659, ← c10660, ← c10661] at e_166_17
  have hr := e_166_17
  simp only [e_166_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 166 79) = bselect (a (.wire 160 27)) (a (.virt 16809)) (a (.wire 166 15)) := by
  have c10662 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.1
  have c10663 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.1
  have c10664 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.1
  have c10665 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.1
  have c10666 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10667 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_166_18 := arithEq_of_rows h (row := 166) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_18
  simp only [← c10662, ← c10663, ← c10664] at e_166_18
  have e_166_19 := arithEq_of_rows h (row := 166) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_166_19
  simp only [← c10665, ← c10666, ← c10667] at e_166_19
  have hr := e_166_19
  simp only [e_166_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 7) = bselect (a (.wire 160 27)) (a (.virt 16803)) (a (.wire 166 23)) := by
  have c10668 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10669 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10670 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10671 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10672 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10673 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_0 := arithEq_of_rows h (row := 167) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_0
  simp only [← c10668, ← c10669, ← c10670] at e_167_0
  have e_167_1 := arithEq_of_rows h (row := 167) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_1
  simp only [← c10671, ← c10672, ← c10673] at e_167_1
  have hr := e_167_1
  simp only [e_167_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 15) = bselect (a (.wire 160 27)) (a (.virt 16804)) (a (.wire 166 31)) := by
  have c10674 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10675 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10676 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10677 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10678 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10679 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_2 := arithEq_of_rows h (row := 167) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_2
  simp only [← c10674, ← c10675, ← c10676] at e_167_2
  have e_167_3 := arithEq_of_rows h (row := 167) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_3
  simp only [← c10677, ← c10678, ← c10679] at e_167_3
  have hr := e_167_3
  simp only [e_167_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 63) = bor (a (.wire 153 59)) (a (.wire 166 35)) := by
  have c10680 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10681 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10682 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10683 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10684 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10685 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_153_15 := arithEq_of_rows h (row := 153) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_15
  simp only [← c10683, ← c10684, k3, ← c10685] at e_153_15
  have e_164_3 := arithEq_of_rows h (row := 164) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_3
  simp only [← c10680, ← c10681, ← c10682] at e_164_3
  have hr := e_153_15
  simp only [e_164_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 19) = bnot (a (.wire 83 39)) := by
  have c10686 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10687 := (publicBatchWrapper89_copies333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10688 := (publicBatchWrapper89_copies334 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_167_4 := arithEq_of_rows h (row := 167) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_4
  simp only [← c10686, k3, ← c10687, k3, ← c10688] at e_167_4
  have hr := e_167_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 23) = bnot (a (.wire 153 63)) := by
  have c10689 := (publicBatchWrapper89_copies334 a h).2.1
  have c10690 := (publicBatchWrapper89_copies334 a h).2.2.1
  have c10691 := (publicBatchWrapper89_copies334 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_167_5 := arithEq_of_rows h (row := 167) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_5
  simp only [← c10689, k3, ← c10690, k3, ← c10691] at e_167_5
  have hr := e_167_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 31) = band (a (.wire 167 19)) (a (.wire 167 23)) := by
  have c10692 := (publicBatchWrapper89_copies334 a h).2.2.2.2.1
  have c10693 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.1
  have c10694 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.1
  have e_160_7 := arithEq_of_rows h (row := 160) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_7
  simp only [← c10692, ← c10693, ← c10694] at e_160_7
  have hr := e_160_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 31) = bselect (a (.wire 160 31)) (a (.virt 17005)) (a (.wire 166 47)) := by
  have c10695 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.1
  have c10696 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.1
  have c10697 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.1
  have c10698 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10699 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10700 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_6 := arithEq_of_rows h (row := 167) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_6
  simp only [← c10695, ← c10696, ← c10697] at e_167_6
  have e_167_7 := arithEq_of_rows h (row := 167) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_7
  simp only [← c10698, ← c10699, ← c10700] at e_167_7
  have hr := e_167_7
  simp only [e_167_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 39) = bselect (a (.wire 160 31)) (a (.virt 17006)) (a (.wire 166 55)) := by
  have c10701 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10702 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10703 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10704 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10705 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10706 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_8 := arithEq_of_rows h (row := 167) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_8
  simp only [← c10701, ← c10702, ← c10703] at e_167_8
  have e_167_9 := arithEq_of_rows h (row := 167) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_9
  simp only [← c10704, ← c10705, ← c10706] at e_167_9
  have hr := e_167_9
  simp only [e_167_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 47) = bselect (a (.wire 160 31)) (a (.virt 17007)) (a (.wire 166 63)) := by
  have c10707 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10708 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10709 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10710 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10711 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10712 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_10 := arithEq_of_rows h (row := 167) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_10
  simp only [← c10707, ← c10708, ← c10709] at e_167_10
  have e_167_11 := arithEq_of_rows h (row := 167) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_11
  simp only [← c10710, ← c10711, ← c10712] at e_167_11
  have hr := e_167_11
  simp only [e_167_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 55) = bselect (a (.wire 160 31)) (a (.virt 17008)) (a (.wire 166 71)) := by
  have c10713 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10714 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10715 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10716 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10717 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10718 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_12 := arithEq_of_rows h (row := 167) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_12
  simp only [← c10713, ← c10714, ← c10715] at e_167_12
  have e_167_13 := arithEq_of_rows h (row := 167) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_13
  simp only [← c10716, ← c10717, ← c10718] at e_167_13
  have hr := e_167_13
  simp only [e_167_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 63) = bselect (a (.wire 160 31)) (a (.virt 17009)) (a (.wire 166 79)) := by
  have c10719 := (publicBatchWrapper89_copies334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10720 := (publicBatchWrapper89_copies335 a h).1
  have c10721 := (publicBatchWrapper89_copies335 a h).2.1
  have c10722 := (publicBatchWrapper89_copies335 a h).2.2.1
  have c10723 := (publicBatchWrapper89_copies335 a h).2.2.2.1
  have c10724 := (publicBatchWrapper89_copies335 a h).2.2.2.2.1
  have e_167_14 := arithEq_of_rows h (row := 167) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_14
  simp only [← c10719, ← c10720, ← c10721] at e_167_14
  have e_167_15 := arithEq_of_rows h (row := 167) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_15
  simp only [← c10722, ← c10723, ← c10724] at e_167_15
  have hr := e_167_15
  simp only [e_167_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 71) = bselect (a (.wire 160 31)) (a (.virt 17003)) (a (.wire 167 7)) := by
  have c10725 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.1
  have c10726 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.1
  have c10727 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.1
  have c10728 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.1
  have c10729 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.1
  have c10730 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_167_16 := arithEq_of_rows h (row := 167) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_16
  simp only [← c10725, ← c10726, ← c10727] at e_167_16
  have e_167_17 := arithEq_of_rows h (row := 167) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_17
  simp only [← c10728, ← c10729, ← c10730] at e_167_17
  have hr := e_167_17
  simp only [e_167_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 167 79) = bselect (a (.wire 160 31)) (a (.virt 17004)) (a (.wire 167 15)) := by
  have c10731 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10732 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10733 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10734 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10735 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10736 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_167_18 := arithEq_of_rows h (row := 167) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_18
  simp only [← c10731, ← c10732, ← c10733] at e_167_18
  have e_167_19 := arithEq_of_rows h (row := 167) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_167_19
  simp only [← c10734, ← c10735, ← c10736] at e_167_19
  have hr := e_167_19
  simp only [e_167_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 67) = bor (a (.wire 153 63)) (a (.wire 167 19)) := by
  have c10737 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10738 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10739 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10740 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10741 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10742 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_153_16 := arithEq_of_rows h (row := 153) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_16
  simp only [← c10740, ← c10741, k3, ← c10742] at e_153_16
  have e_164_4 := arithEq_of_rows h (row := 164) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_4
  simp only [← c10737, ← c10738, ← c10739] at e_164_4
  have hr := e_153_16
  simp only [e_164_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 3) = bnot (a (.wire 84 3)) := by
  have c10743 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10744 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10745 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_168_0 := arithEq_of_rows h (row := 168) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_0
  simp only [← c10743, k3, ← c10744, k3, ← c10745] at e_168_0
  have hr := e_168_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 7) = bnot (a (.wire 153 67)) := by
  have c10746 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10747 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10748 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_168_1 := arithEq_of_rows h (row := 168) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_1
  simp only [← c10746, k3, ← c10747, k3, ← c10748] at e_168_1
  have hr := e_168_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 35) = band (a (.wire 168 3)) (a (.wire 168 7)) := by
  have c10749 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10750 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10751 := (publicBatchWrapper89_copies335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_160_8 := arithEq_of_rows h (row := 160) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_8
  simp only [← c10749, ← c10750, ← c10751] at e_160_8
  have hr := e_160_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 15) = bselect (a (.wire 160 35)) (a (.virt 17205)) (a (.wire 167 31)) := by
  have c10752 := (publicBatchWrapper89_copies336 a h).1
  have c10753 := (publicBatchWrapper89_copies336 a h).2.1
  have c10754 := (publicBatchWrapper89_copies336 a h).2.2.1
  have c10755 := (publicBatchWrapper89_copies336 a h).2.2.2.1
  have c10756 := (publicBatchWrapper89_copies336 a h).2.2.2.2.1
  have c10757 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.1
  have e_168_2 := arithEq_of_rows h (row := 168) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_2
  simp only [← c10752, ← c10753, ← c10754] at e_168_2
  have e_168_3 := arithEq_of_rows h (row := 168) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_3
  simp only [← c10755, ← c10756, ← c10757] at e_168_3
  have hr := e_168_3
  simp only [e_168_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 23) = bselect (a (.wire 160 35)) (a (.virt 17206)) (a (.wire 167 39)) := by
  have c10758 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.1
  have c10759 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.1
  have c10760 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.1
  have c10761 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.1
  have c10762 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10763 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_168_4 := arithEq_of_rows h (row := 168) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_4
  simp only [← c10758, ← c10759, ← c10760] at e_168_4
  have e_168_5 := arithEq_of_rows h (row := 168) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_5
  simp only [← c10761, ← c10762, ← c10763] at e_168_5
  have hr := e_168_5
  simp only [e_168_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 31) = bselect (a (.wire 160 35)) (a (.virt 17207)) (a (.wire 167 47)) := by
  have c10764 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10765 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10766 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10767 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10768 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10769 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_168_6 := arithEq_of_rows h (row := 168) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_6
  simp only [← c10764, ← c10765, ← c10766] at e_168_6
  have e_168_7 := arithEq_of_rows h (row := 168) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_7
  simp only [← c10767, ← c10768, ← c10769] at e_168_7
  have hr := e_168_7
  simp only [e_168_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 39) = bselect (a (.wire 160 35)) (a (.virt 17208)) (a (.wire 167 55)) := by
  have c10770 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10771 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10772 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10773 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10774 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10775 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_168_8 := arithEq_of_rows h (row := 168) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_8
  simp only [← c10770, ← c10771, ← c10772] at e_168_8
  have e_168_9 := arithEq_of_rows h (row := 168) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_9
  simp only [← c10773, ← c10774, ← c10775] at e_168_9
  have hr := e_168_9
  simp only [e_168_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 47) = bselect (a (.wire 160 35)) (a (.virt 17209)) (a (.wire 167 63)) := by
  have c10776 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10777 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10778 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10779 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10780 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10781 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_168_10 := arithEq_of_rows h (row := 168) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_10
  simp only [← c10776, ← c10777, ← c10778] at e_168_10
  have e_168_11 := arithEq_of_rows h (row := 168) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_11
  simp only [← c10779, ← c10780, ← c10781] at e_168_11
  have hr := e_168_11
  simp only [e_168_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 55) = bselect (a (.wire 160 35)) (a (.virt 17203)) (a (.wire 167 71)) := by
  have c10782 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10783 := (publicBatchWrapper89_copies336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10784 := (publicBatchWrapper89_copies337 a h).1
  have c10785 := (publicBatchWrapper89_copies337 a h).2.1
  have c10786 := (publicBatchWrapper89_copies337 a h).2.2.1
  have c10787 := (publicBatchWrapper89_copies337 a h).2.2.2.1
  have e_168_12 := arithEq_of_rows h (row := 168) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_12
  simp only [← c10782, ← c10783, ← c10784] at e_168_12
  have e_168_13 := arithEq_of_rows h (row := 168) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_13
  simp only [← c10785, ← c10786, ← c10787] at e_168_13
  have hr := e_168_13
  simp only [e_168_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 63) = bselect (a (.wire 160 35)) (a (.virt 17204)) (a (.wire 167 79)) := by
  have c10788 := (publicBatchWrapper89_copies337 a h).2.2.2.2.1
  have c10789 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.1
  have c10790 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.1
  have c10791 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.1
  have c10792 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.1
  have c10793 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.1
  have e_168_14 := arithEq_of_rows h (row := 168) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_14
  simp only [← c10788, ← c10789, ← c10790] at e_168_14
  have e_168_15 := arithEq_of_rows h (row := 168) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_15
  simp only [← c10791, ← c10792, ← c10793] at e_168_15
  have hr := e_168_15
  simp only [e_168_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 71) = bor (a (.wire 153 67)) (a (.wire 168 3)) := by
  have c10794 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10795 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10796 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10797 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10798 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10799 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_153_17 := arithEq_of_rows h (row := 153) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_17
  simp only [← c10797, ← c10798, k3, ← c10799] at e_153_17
  have e_164_5 := arithEq_of_rows h (row := 164) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_5
  simp only [← c10794, ← c10795, ← c10796] at e_164_5
  have hr := e_153_17
  simp only [e_164_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 67) = bnot (a (.wire 84 47)) := by
  have c10800 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10801 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10802 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_168_16 := arithEq_of_rows h (row := 168) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_16
  simp only [← c10800, k3, ← c10801, k3, ← c10802] at e_168_16
  have hr := e_168_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 71) = bnot (a (.wire 153 71)) := by
  have c10803 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10804 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10805 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_168_17 := arithEq_of_rows h (row := 168) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_17
  simp only [← c10803, k3, ← c10804, k3, ← c10805] at e_168_17
  have hr := e_168_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 39) = band (a (.wire 168 67)) (a (.wire 168 71)) := by
  have c10806 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10807 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10808 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_9 := arithEq_of_rows h (row := 160) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_9
  simp only [← c10806, ← c10807, ← c10808] at e_160_9
  have hr := e_160_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 168 79) = bselect (a (.wire 160 39)) (a (.virt 17405)) (a (.wire 168 15)) := by
  have c10809 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10810 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10811 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10812 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10813 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10814 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_168_18 := arithEq_of_rows h (row := 168) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_18
  simp only [← c10809, ← c10810, ← c10811] at e_168_18
  have e_168_19 := arithEq_of_rows h (row := 168) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_168_19
  simp only [← c10812, ← c10813, ← c10814] at e_168_19
  have hr := e_168_19
  simp only [e_168_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 7) = bselect (a (.wire 160 39)) (a (.virt 17406)) (a (.wire 168 23)) := by
  have c10815 := (publicBatchWrapper89_copies337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10816 := (publicBatchWrapper89_copies338 a h).1
  have c10817 := (publicBatchWrapper89_copies338 a h).2.1
  have c10818 := (publicBatchWrapper89_copies338 a h).2.2.1
  have c10819 := (publicBatchWrapper89_copies338 a h).2.2.2.1
  have c10820 := (publicBatchWrapper89_copies338 a h).2.2.2.2.1
  have e_169_0 := arithEq_of_rows h (row := 169) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_0
  simp only [← c10815, ← c10816, ← c10817] at e_169_0
  have e_169_1 := arithEq_of_rows h (row := 169) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_1
  simp only [← c10818, ← c10819, ← c10820] at e_169_1
  have hr := e_169_1
  simp only [e_169_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 15) = bselect (a (.wire 160 39)) (a (.virt 17407)) (a (.wire 168 31)) := by
  have c10821 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.1
  have c10822 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.1
  have c10823 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.1
  have c10824 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.1
  have c10825 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.1
  have c10826 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_169_2 := arithEq_of_rows h (row := 169) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_2
  simp only [← c10821, ← c10822, ← c10823] at e_169_2
  have e_169_3 := arithEq_of_rows h (row := 169) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_3
  simp only [← c10824, ← c10825, ← c10826] at e_169_3
  have hr := e_169_3
  simp only [e_169_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 23) = bselect (a (.wire 160 39)) (a (.virt 17408)) (a (.wire 168 39)) := by
  have c10827 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10828 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10829 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10830 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10831 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10832 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_169_4 := arithEq_of_rows h (row := 169) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_4
  simp only [← c10827, ← c10828, ← c10829] at e_169_4
  have e_169_5 := arithEq_of_rows h (row := 169) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_5
  simp only [← c10830, ← c10831, ← c10832] at e_169_5
  have hr := e_169_5
  simp only [e_169_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 31) = bselect (a (.wire 160 39)) (a (.virt 17409)) (a (.wire 168 47)) := by
  have c10833 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10834 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10835 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10836 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10837 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10838 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_169_6 := arithEq_of_rows h (row := 169) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_6
  simp only [← c10833, ← c10834, ← c10835] at e_169_6
  have e_169_7 := arithEq_of_rows h (row := 169) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_7
  simp only [← c10836, ← c10837, ← c10838] at e_169_7
  have hr := e_169_7
  simp only [e_169_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 39) = bselect (a (.wire 160 39)) (a (.virt 17403)) (a (.wire 168 55)) := by
  have c10839 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10840 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10841 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10842 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10843 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10844 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_169_8 := arithEq_of_rows h (row := 169) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_8
  simp only [← c10839, ← c10840, ← c10841] at e_169_8
  have e_169_9 := arithEq_of_rows h (row := 169) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_9
  simp only [← c10842, ← c10843, ← c10844] at e_169_9
  have hr := e_169_9
  simp only [e_169_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 47) = bselect (a (.wire 160 39)) (a (.virt 17404)) (a (.wire 168 63)) := by
  have c10845 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10846 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10847 := (publicBatchWrapper89_copies338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10848 := (publicBatchWrapper89_copies339 a h).1
  have c10849 := (publicBatchWrapper89_copies339 a h).2.1
  have c10850 := (publicBatchWrapper89_copies339 a h).2.2.1
  have e_169_10 := arithEq_of_rows h (row := 169) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_10
  simp only [← c10845, ← c10846, ← c10847] at e_169_10
  have e_169_11 := arithEq_of_rows h (row := 169) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_11
  simp only [← c10848, ← c10849, ← c10850] at e_169_11
  have hr := e_169_11
  simp only [e_169_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 75) = bor (a (.wire 153 71)) (a (.wire 168 67)) := by
  have c10851 := (publicBatchWrapper89_copies339 a h).2.2.2.1
  have c10852 := (publicBatchWrapper89_copies339 a h).2.2.2.2.1
  have c10853 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.1
  have c10854 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.1
  have c10855 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.1
  have c10856 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_153_18 := arithEq_of_rows h (row := 153) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_18
  simp only [← c10854, ← c10855, k3, ← c10856] at e_153_18
  have e_164_6 := arithEq_of_rows h (row := 164) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_6
  simp only [← c10851, ← c10852, ← c10853] at e_164_6
  have hr := e_153_18
  simp only [e_164_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 51) = bnot (a (.wire 86 11)) := by
  have c10857 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.1
  have c10858 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10859 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_169_12 := arithEq_of_rows h (row := 169) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_12
  simp only [← c10857, k3, ← c10858, k3, ← c10859] at e_169_12
  have hr := e_169_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 55) = bnot (a (.wire 153 75)) := by
  have c10860 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10861 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10862 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_169_13 := arithEq_of_rows h (row := 169) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_13
  simp only [← c10860, k3, ← c10861, k3, ← c10862] at e_169_13
  have hr := e_169_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 43) = band (a (.wire 169 51)) (a (.wire 169 55)) := by
  have c10863 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10864 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10865 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_10 := arithEq_of_rows h (row := 160) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_10
  simp only [← c10863, ← c10864, ← c10865] at e_160_10
  have hr := e_160_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 63) = bselect (a (.wire 160 43)) (a (.virt 17605)) (a (.wire 168 79)) := by
  have c10866 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10867 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10868 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10869 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10870 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10871 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_169_14 := arithEq_of_rows h (row := 169) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_14
  simp only [← c10866, ← c10867, ← c10868] at e_169_14
  have e_169_15 := arithEq_of_rows h (row := 169) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_15
  simp only [← c10869, ← c10870, ← c10871] at e_169_15
  have hr := e_169_15
  simp only [e_169_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 71) = bselect (a (.wire 160 43)) (a (.virt 17606)) (a (.wire 169 7)) := by
  have c10872 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10873 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10874 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10875 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10876 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10877 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_169_16 := arithEq_of_rows h (row := 169) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_16
  simp only [← c10872, ← c10873, ← c10874] at e_169_16
  have e_169_17 := arithEq_of_rows h (row := 169) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_17
  simp only [← c10875, ← c10876, ← c10877] at e_169_17
  have hr := e_169_17
  simp only [e_169_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 169 79) = bselect (a (.wire 160 43)) (a (.virt 17607)) (a (.wire 169 15)) := by
  have c10878 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10879 := (publicBatchWrapper89_copies339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10880 := (publicBatchWrapper89_copies340 a h).1
  have c10881 := (publicBatchWrapper89_copies340 a h).2.1
  have c10882 := (publicBatchWrapper89_copies340 a h).2.2.1
  have c10883 := (publicBatchWrapper89_copies340 a h).2.2.2.1
  have e_169_18 := arithEq_of_rows h (row := 169) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_18
  simp only [← c10878, ← c10879, ← c10880] at e_169_18
  have e_169_19 := arithEq_of_rows h (row := 169) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_169_19
  simp only [← c10881, ← c10882, ← c10883] at e_169_19
  have hr := e_169_19
  simp only [e_169_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 170 7) = bselect (a (.wire 160 43)) (a (.virt 17608)) (a (.wire 169 23)) := by
  have c10884 := (publicBatchWrapper89_copies340 a h).2.2.2.2.1
  have c10885 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.1
  have c10886 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.1
  have c10887 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.1
  have c10888 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.1
  have c10889 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.1
  have e_170_0 := arithEq_of_rows h (row := 170) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_0
  simp only [← c10884, ← c10885, ← c10886] at e_170_0
  have e_170_1 := arithEq_of_rows h (row := 170) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_1
  simp only [← c10887, ← c10888, ← c10889] at e_170_1
  have hr := e_170_1
  simp only [e_170_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 170 15) = bselect (a (.wire 160 43)) (a (.virt 17609)) (a (.wire 169 31)) := by
  have c10890 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10891 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10892 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10893 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10894 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10895 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_170_2 := arithEq_of_rows h (row := 170) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_2
  simp only [← c10890, ← c10891, ← c10892] at e_170_2
  have e_170_3 := arithEq_of_rows h (row := 170) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_3
  simp only [← c10893, ← c10894, ← c10895] at e_170_3
  have hr := e_170_3
  simp only [e_170_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 170 23) = bselect (a (.wire 160 43)) (a (.virt 17603)) (a (.wire 169 39)) := by
  have c10896 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10897 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10898 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10899 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10900 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10901 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_170_4 := arithEq_of_rows h (row := 170) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_4
  simp only [← c10896, ← c10897, ← c10898] at e_170_4
  have e_170_5 := arithEq_of_rows h (row := 170) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_5
  simp only [← c10899, ← c10900, ← c10901] at e_170_5
  have hr := e_170_5
  simp only [e_170_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 170 31) = bselect (a (.wire 160 43)) (a (.virt 17604)) (a (.wire 169 47)) := by
  have c10902 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10903 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10904 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10905 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10906 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10907 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_170_6 := arithEq_of_rows h (row := 170) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_6
  simp only [← c10902, ← c10903, ← c10904] at e_170_6
  have e_170_7 := arithEq_of_rows h (row := 170) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_7
  simp only [← c10905, ← c10906, ← c10907] at e_170_7
  have hr := e_170_7
  simp only [e_170_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 79) = bor (a (.wire 153 75)) (a (.wire 169 51)) := by
  have c10908 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10909 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10910 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10911 := (publicBatchWrapper89_copies340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10912 := (publicBatchWrapper89_copies341 a h).1
  have c10913 := (publicBatchWrapper89_copies341 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_153_19 := arithEq_of_rows h (row := 153) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_19
  simp only [← c10911, ← c10912, k3, ← c10913] at e_153_19
  have e_164_7 := arithEq_of_rows h (row := 164) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_7
  simp only [← c10908, ← c10909, ← c10910] at e_164_7
  have hr := e_153_19
  simp only [e_164_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1)) (a (.wire 170 23)) (a (.virt 18519)) (a (.virt 18520)) := by
  have c10914 := (publicBatchWrapper89_copies341 a h).2.2.1
  have c10915 := (publicBatchWrapper89_copies341 a h).2.2.2.1
  have c10916 := (publicBatchWrapper89_copies341 a h).2.2.2.2.1
  have c10917 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.1
  have c10918 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.1
  have c10919 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.1
  have c10920 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.1
  have c10921 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.1
  have c10922 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10923 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10924 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10925 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10926 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10927 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10928 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10929 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10930 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_160_11 := arithEq_of_rows h (row := 160) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_11
  simp only [← c10920, ← c10921, ← c10922] at e_160_11
  have e_160_12 := arithEq_of_rows h (row := 160) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_12
  simp only [← c10923, ← c10924, ← c10925] at e_160_12
  have e_170_8 := arithEq_of_rows h (row := 170) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_8
  simp only [← c10914, k3, ← c10915, k3, ← c10916] at e_170_8
  have e_170_9 := arithEq_of_rows h (row := 170) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_9
  simp only [← c10917, ← c10918, k3, ← c10919] at e_170_9
  have e_170_10 := arithEq_of_rows h (row := 170) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_10
  simp only [← c10926, ← c10927, k3, ← c10928] at e_170_10
  refine ⟨?_, ?_⟩
  · have hc := e_160_11
    simp only [e_170_9] at hc
    linear_combination c10929.trans k2 - hc
  · have hc := e_170_10
    simp only [e_170_8, e_160_12, e_170_9] at hc
    linear_combination c10930.trans k2 - hc

theorem publicBatchWrapper89_f1692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 3) = bor (a (.wire 1 59)) (a (.virt 18519)) := by
  have c10931 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10932 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10933 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10934 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10935 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10936 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_8 := arithEq_of_rows h (row := 164) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_8
  simp only [← c10931, ← c10932, ← c10933] at e_164_8
  have e_171_0 := arithEq_of_rows h (row := 171) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_0
  simp only [← c10934, ← c10935, k3, ← c10936] at e_171_0
  have hr := e_171_0
  simp only [e_164_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 3) = a (.virt 201) := by
  have c10937 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c10937

theorem publicBatchWrapper89_f1694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2)) (a (.wire 170 31)) (a (.virt 18521)) (a (.virt 18522)) := by
  have c10938 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10939 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10940 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10941 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10942 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10943 := (publicBatchWrapper89_copies341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10944 := (publicBatchWrapper89_copies342 a h).1
  have c10945 := (publicBatchWrapper89_copies342 a h).2.1
  have c10946 := (publicBatchWrapper89_copies342 a h).2.2.1
  have c10947 := (publicBatchWrapper89_copies342 a h).2.2.2.1
  have c10948 := (publicBatchWrapper89_copies342 a h).2.2.2.2.1
  have c10949 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.1
  have c10950 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.1
  have c10951 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.1
  have c10952 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.1
  have c10953 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.1
  have c10954 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_160_13 := arithEq_of_rows h (row := 160) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_13
  simp only [← c10944, ← c10945, ← c10946] at e_160_13
  have e_160_14 := arithEq_of_rows h (row := 160) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_14
  simp only [← c10947, ← c10948, ← c10949] at e_160_14
  have e_170_11 := arithEq_of_rows h (row := 170) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_11
  simp only [← c10938, k3, ← c10939, k3, ← c10940] at e_170_11
  have e_170_12 := arithEq_of_rows h (row := 170) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_12
  simp only [← c10941, ← c10942, k3, ← c10943] at e_170_12
  have e_170_13 := arithEq_of_rows h (row := 170) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_13
  simp only [← c10950, ← c10951, k3, ← c10952] at e_170_13
  refine ⟨?_, ?_⟩
  · have hc := e_160_13
    simp only [e_170_12] at hc
    linear_combination c10953.trans k2 - hc
  · have hc := e_170_13
    simp only [e_170_11, e_160_14, e_170_12] at hc
    linear_combination c10954.trans k2 - hc

theorem publicBatchWrapper89_f1695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 7) = bor (a (.wire 1 59)) (a (.virt 18521)) := by
  have c10955 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10956 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10957 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10958 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10959 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10960 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_9 := arithEq_of_rows h (row := 164) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_9
  simp only [← c10955, ← c10956, ← c10957] at e_164_9
  have e_171_1 := arithEq_of_rows h (row := 171) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_1
  simp only [← c10958, ← c10959, k3, ← c10960] at e_171_1
  have hr := e_171_1
  simp only [e_164_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 7) = a (.virt 201) := by
  have c10961 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c10961

theorem publicBatchWrapper89_f1697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3)) (a (.wire 169 63)) (a (.virt 18523)) (a (.virt 18524)) := by
  have c10962 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10963 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10964 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10965 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10966 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10967 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10968 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10969 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10970 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10971 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10972 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10973 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10974 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10975 := (publicBatchWrapper89_copies342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10976 := (publicBatchWrapper89_copies343 a h).1
  have c10977 := (publicBatchWrapper89_copies343 a h).2.1
  have c10978 := (publicBatchWrapper89_copies343 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_160_15 := arithEq_of_rows h (row := 160) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_15
  simp only [← c10968, ← c10969, ← c10970] at e_160_15
  have e_160_16 := arithEq_of_rows h (row := 160) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_16
  simp only [← c10971, ← c10972, ← c10973] at e_160_16
  have e_170_14 := arithEq_of_rows h (row := 170) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_14
  simp only [← c10962, k3, ← c10963, k3, ← c10964] at e_170_14
  have e_170_15 := arithEq_of_rows h (row := 170) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_15
  simp only [← c10965, ← c10966, k3, ← c10967] at e_170_15
  have e_170_16 := arithEq_of_rows h (row := 170) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_16
  simp only [← c10974, ← c10975, k3, ← c10976] at e_170_16
  refine ⟨?_, ?_⟩
  · have hc := e_160_15
    simp only [e_170_15] at hc
    linear_combination c10977.trans k2 - hc
  · have hc := e_170_16
    simp only [e_170_14, e_160_16, e_170_15] at hc
    linear_combination c10978.trans k2 - hc

theorem publicBatchWrapper89_f1698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4)) (a (.wire 169 71)) (a (.virt 18525)) (a (.virt 18526)) := by
  have c10979 := (publicBatchWrapper89_copies343 a h).2.2.2.1
  have c10980 := (publicBatchWrapper89_copies343 a h).2.2.2.2.1
  have c10981 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.1
  have c10982 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.1
  have c10983 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.1
  have c10984 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.1
  have c10985 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.1
  have c10986 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10987 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10988 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10989 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10990 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10991 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10992 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10993 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10994 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10995 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_160_17 := arithEq_of_rows h (row := 160) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_17
  simp only [← c10985, ← c10986, ← c10987] at e_160_17
  have e_160_18 := arithEq_of_rows h (row := 160) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_18
  simp only [← c10988, ← c10989, ← c10990] at e_160_18
  have e_170_17 := arithEq_of_rows h (row := 170) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_17
  simp only [← c10979, k3, ← c10980, k3, ← c10981] at e_170_17
  have e_170_18 := arithEq_of_rows h (row := 170) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_18
  simp only [← c10982, ← c10983, k3, ← c10984] at e_170_18
  have e_170_19 := arithEq_of_rows h (row := 170) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_170_19
  simp only [← c10991, ← c10992, k3, ← c10993] at e_170_19
  refine ⟨?_, ?_⟩
  · have hc := e_160_17
    simp only [e_170_18] at hc
    linear_combination c10994.trans k2 - hc
  · have hc := e_170_19
    simp only [e_170_17, e_160_18, e_170_18] at hc
    linear_combination c10995.trans k2 - hc

theorem publicBatchWrapper89_f1699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5)) (a (.wire 169 79)) (a (.virt 18527)) (a (.virt 18528)) := by
  have c10996 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10997 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10998 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10999 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11000 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11001 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11002 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11003 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11004 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11005 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11006 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11007 := (publicBatchWrapper89_copies343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11008 := (publicBatchWrapper89_copies344 a h).1
  have c11009 := (publicBatchWrapper89_copies344 a h).2.1
  have c11010 := (publicBatchWrapper89_copies344 a h).2.2.1
  have c11011 := (publicBatchWrapper89_copies344 a h).2.2.2.1
  have c11012 := (publicBatchWrapper89_copies344 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_160_19 := arithEq_of_rows h (row := 160) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_19
  simp only [← c11002, ← c11003, ← c11004] at e_160_19
  have e_172_0 := arithEq_of_rows h (row := 172) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_0
  simp only [← c10996, k3, ← c10997, k3, ← c10998] at e_172_0
  have e_172_1 := arithEq_of_rows h (row := 172) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_1
  simp only [← c10999, ← c11000, k3, ← c11001] at e_172_1
  have e_172_2 := arithEq_of_rows h (row := 172) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_2
  simp only [← c11008, ← c11009, k3, ← c11010] at e_172_2
  have e_173_0 := arithEq_of_rows h (row := 173) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_0
  simp only [← c11005, ← c11006, ← c11007] at e_173_0
  refine ⟨?_, ?_⟩
  · have hc := e_160_19
    simp only [e_172_1] at hc
    linear_combination c11011.trans k2 - hc
  · have hc := e_172_2
    simp only [e_172_0, e_173_0, e_172_1] at hc
    linear_combination c11012.trans k2 - hc

theorem publicBatchWrapper89_f1700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6)) (a (.wire 170 7)) (a (.virt 18529)) (a (.virt 18530)) := by
  have c11013 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.1
  have c11014 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.1
  have c11015 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.1
  have c11016 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.1
  have c11017 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.1
  have c11018 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11019 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11020 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11021 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11022 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11023 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11024 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11025 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11026 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11027 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11028 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11029 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_172_3 := arithEq_of_rows h (row := 172) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_3
  simp only [← c11013, k3, ← c11014, k3, ← c11015] at e_172_3
  have e_172_4 := arithEq_of_rows h (row := 172) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_4
  simp only [← c11016, ← c11017, k3, ← c11018] at e_172_4
  have e_172_5 := arithEq_of_rows h (row := 172) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_5
  simp only [← c11025, ← c11026, k3, ← c11027] at e_172_5
  have e_173_1 := arithEq_of_rows h (row := 173) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_1
  simp only [← c11019, ← c11020, ← c11021] at e_173_1
  have e_173_2 := arithEq_of_rows h (row := 173) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_2
  simp only [← c11022, ← c11023, ← c11024] at e_173_2
  refine ⟨?_, ?_⟩
  · have hc := e_173_1
    simp only [e_172_4] at hc
    linear_combination c11028.trans k2 - hc
  · have hc := e_172_5
    simp only [e_172_3, e_173_2, e_172_4] at hc
    linear_combination c11029.trans k2 - hc

theorem publicBatchWrapper89_f1701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 173 15) = band (a (.virt 18523)) (a (.virt 18525)) := by
  have c11030 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11031 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11032 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_3 := arithEq_of_rows h (row := 173) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_3
  simp only [← c11030, ← c11031, ← c11032] at e_173_3
  have hr := e_173_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 173 19) = band (a (.virt 18527)) (a (.virt 18529)) := by
  have c11033 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11034 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11035 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_4 := arithEq_of_rows h (row := 173) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_4
  simp only [← c11033, ← c11034, ← c11035] at e_173_4
  have hr := e_173_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 173 23) = band (a (.wire 173 15)) (a (.wire 173 19)) := by
  have c11036 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11037 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11038 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_5 := arithEq_of_rows h (row := 173) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_5
  simp only [← c11036, ← c11037, ← c11038] at e_173_5
  have hr := e_173_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 11) = bor (a (.wire 1 59)) (a (.wire 173 23)) := by
  have c11039 := (publicBatchWrapper89_copies344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11040 := (publicBatchWrapper89_copies345 a h).1
  have c11041 := (publicBatchWrapper89_copies345 a h).2.1
  have c11042 := (publicBatchWrapper89_copies345 a h).2.2.1
  have c11043 := (publicBatchWrapper89_copies345 a h).2.2.2.1
  have c11044 := (publicBatchWrapper89_copies345 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_10 := arithEq_of_rows h (row := 164) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_10
  simp only [← c11039, ← c11040, ← c11041] at e_164_10
  have e_171_2 := arithEq_of_rows h (row := 171) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_2
  simp only [← c11042, ← c11043, k3, ← c11044] at e_171_2
  have hr := e_171_2
  simp only [e_164_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 11) = a (.virt 201) := by
  have c11045 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.1
  exact c11045

theorem publicBatchWrapper89_f1706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 203)) (a (.wire 170 23)) (a (.virt 18531)) (a (.virt 18532)) := by
  have c11046 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.1
  have c11047 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.1
  have c11048 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.1
  have c11049 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.1
  have c11050 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11051 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11052 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11053 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11054 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11055 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11056 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11057 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11058 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11059 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11060 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11061 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11062 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_172_6 := arithEq_of_rows h (row := 172) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_6
  simp only [← c11046, k3, ← c11047, k3, ← c11048] at e_172_6
  have e_172_7 := arithEq_of_rows h (row := 172) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_7
  simp only [← c11049, ← c11050, k3, ← c11051] at e_172_7
  have e_172_8 := arithEq_of_rows h (row := 172) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_8
  simp only [← c11058, ← c11059, k3, ← c11060] at e_172_8
  have e_173_6 := arithEq_of_rows h (row := 173) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_6
  simp only [← c11052, ← c11053, ← c11054] at e_173_6
  have e_173_7 := arithEq_of_rows h (row := 173) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_7
  simp only [← c11055, ← c11056, ← c11057] at e_173_7
  refine ⟨?_, ?_⟩
  · have hc := e_173_6
    simp only [e_172_7] at hc
    linear_combination c11061.trans k2 - hc
  · have hc := e_172_8
    simp only [e_172_6, e_173_7, e_172_7] at hc
    linear_combination c11062.trans k2 - hc

theorem publicBatchWrapper89_f1707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 15) = bor (a (.wire 3 23)) (a (.virt 18531)) := by
  have c11063 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11064 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11065 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11066 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11067 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11068 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_11 := arithEq_of_rows h (row := 164) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_11
  simp only [← c11063, ← c11064, ← c11065] at e_164_11
  have e_171_3 := arithEq_of_rows h (row := 171) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_3
  simp only [← c11066, ← c11067, k3, ← c11068] at e_171_3
  have hr := e_171_3
  simp only [e_164_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 15) = a (.virt 201) := by
  have c11069 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11069

theorem publicBatchWrapper89_f1709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 204)) (a (.wire 170 31)) (a (.virt 18533)) (a (.virt 18534)) := by
  have c11070 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11071 := (publicBatchWrapper89_copies345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11072 := (publicBatchWrapper89_copies346 a h).1
  have c11073 := (publicBatchWrapper89_copies346 a h).2.1
  have c11074 := (publicBatchWrapper89_copies346 a h).2.2.1
  have c11075 := (publicBatchWrapper89_copies346 a h).2.2.2.1
  have c11076 := (publicBatchWrapper89_copies346 a h).2.2.2.2.1
  have c11077 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.1
  have c11078 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.1
  have c11079 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.1
  have c11080 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.1
  have c11081 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.1
  have c11082 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11083 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11084 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11085 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11086 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_172_9 := arithEq_of_rows h (row := 172) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_9
  simp only [← c11070, k3, ← c11071, k3, ← c11072] at e_172_9
  have e_172_10 := arithEq_of_rows h (row := 172) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_10
  simp only [← c11073, ← c11074, k3, ← c11075] at e_172_10
  have e_172_11 := arithEq_of_rows h (row := 172) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_11
  simp only [← c11082, ← c11083, k3, ← c11084] at e_172_11
  have e_173_8 := arithEq_of_rows h (row := 173) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_8
  simp only [← c11076, ← c11077, ← c11078] at e_173_8
  have e_173_9 := arithEq_of_rows h (row := 173) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_9
  simp only [← c11079, ← c11080, ← c11081] at e_173_9
  refine ⟨?_, ?_⟩
  · have hc := e_173_8
    simp only [e_172_10] at hc
    linear_combination c11085.trans k2 - hc
  · have hc := e_172_11
    simp only [e_172_9, e_173_9, e_172_10] at hc
    linear_combination c11086.trans k2 - hc

theorem publicBatchWrapper89_f1710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 19) = bor (a (.wire 3 23)) (a (.virt 18533)) := by
  have c11087 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11088 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11089 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11090 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11091 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11092 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_12 := arithEq_of_rows h (row := 164) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_12
  simp only [← c11087, ← c11088, ← c11089] at e_164_12
  have e_171_4 := arithEq_of_rows h (row := 171) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_4
  simp only [← c11090, ← c11091, k3, ← c11092] at e_171_4
  have hr := e_171_4
  simp only [e_164_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 19) = a (.virt 201) := by
  have c11093 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11093

theorem publicBatchWrapper89_f1712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 205)) (a (.wire 169 63)) (a (.virt 18535)) (a (.virt 18536)) := by
  have c11094 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11095 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11096 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11097 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11098 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11099 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11100 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11101 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11102 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11103 := (publicBatchWrapper89_copies346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11104 := (publicBatchWrapper89_copies347 a h).1
  have c11105 := (publicBatchWrapper89_copies347 a h).2.1
  have c11106 := (publicBatchWrapper89_copies347 a h).2.2.1
  have c11107 := (publicBatchWrapper89_copies347 a h).2.2.2.1
  have c11108 := (publicBatchWrapper89_copies347 a h).2.2.2.2.1
  have c11109 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.1
  have c11110 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_172_12 := arithEq_of_rows h (row := 172) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_12
  simp only [← c11094, k3, ← c11095, k3, ← c11096] at e_172_12
  have e_172_13 := arithEq_of_rows h (row := 172) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_13
  simp only [← c11097, ← c11098, k3, ← c11099] at e_172_13
  have e_172_14 := arithEq_of_rows h (row := 172) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_14
  simp only [← c11106, ← c11107, k3, ← c11108] at e_172_14
  have e_173_10 := arithEq_of_rows h (row := 173) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_10
  simp only [← c11100, ← c11101, ← c11102] at e_173_10
  have e_173_11 := arithEq_of_rows h (row := 173) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_11
  simp only [← c11103, ← c11104, ← c11105] at e_173_11
  refine ⟨?_, ?_⟩
  · have hc := e_173_10
    simp only [e_172_13] at hc
    linear_combination c11109.trans k2 - hc
  · have hc := e_172_14
    simp only [e_172_12, e_173_11, e_172_13] at hc
    linear_combination c11110.trans k2 - hc

theorem publicBatchWrapper89_f1713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 206)) (a (.wire 169 71)) (a (.virt 18537)) (a (.virt 18538)) := by
  have c11111 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.1
  have c11112 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.1
  have c11113 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.1
  have c11114 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11115 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11116 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11117 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11118 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11119 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11120 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11121 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11122 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11123 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11124 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11125 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11126 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11127 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_172_15 := arithEq_of_rows h (row := 172) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_15
  simp only [← c11111, k3, ← c11112, k3, ← c11113] at e_172_15
  have e_172_16 := arithEq_of_rows h (row := 172) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_16
  simp only [← c11114, ← c11115, k3, ← c11116] at e_172_16
  have e_172_17 := arithEq_of_rows h (row := 172) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_17
  simp only [← c11123, ← c11124, k3, ← c11125] at e_172_17
  have e_173_12 := arithEq_of_rows h (row := 173) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_12
  simp only [← c11117, ← c11118, ← c11119] at e_173_12
  have e_173_13 := arithEq_of_rows h (row := 173) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_13
  simp only [← c11120, ← c11121, ← c11122] at e_173_13
  refine ⟨?_, ?_⟩
  · have hc := e_173_12
    simp only [e_172_16] at hc
    linear_combination c11126.trans k2 - hc
  · have hc := e_172_17
    simp only [e_172_15, e_173_13, e_172_16] at hc
    linear_combination c11127.trans k2 - hc

theorem publicBatchWrapper89_f1714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 207)) (a (.wire 169 79)) (a (.virt 18539)) (a (.virt 18540)) := by
  have c11128 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11129 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11130 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11131 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11132 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11133 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11134 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11135 := (publicBatchWrapper89_copies347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11136 := (publicBatchWrapper89_copies348 a h).1
  have c11137 := (publicBatchWrapper89_copies348 a h).2.1
  have c11138 := (publicBatchWrapper89_copies348 a h).2.2.1
  have c11139 := (publicBatchWrapper89_copies348 a h).2.2.2.1
  have c11140 := (publicBatchWrapper89_copies348 a h).2.2.2.2.1
  have c11141 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.1
  have c11142 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.1
  have c11143 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.1
  have c11144 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_172_18 := arithEq_of_rows h (row := 172) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_18
  simp only [← c11128, k3, ← c11129, k3, ← c11130] at e_172_18
  have e_172_19 := arithEq_of_rows h (row := 172) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_172_19
  simp only [← c11131, ← c11132, k3, ← c11133] at e_172_19
  have e_173_14 := arithEq_of_rows h (row := 173) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_14
  simp only [← c11134, ← c11135, ← c11136] at e_173_14
  have e_173_15 := arithEq_of_rows h (row := 173) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_15
  simp only [← c11137, ← c11138, ← c11139] at e_173_15
  have e_174_0 := arithEq_of_rows h (row := 174) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_0
  simp only [← c11140, ← c11141, k3, ← c11142] at e_174_0
  refine ⟨?_, ?_⟩
  · have hc := e_173_14
    simp only [e_172_19] at hc
    linear_combination c11143.trans k2 - hc
  · have hc := e_174_0
    simp only [e_172_18, e_173_15, e_172_19] at hc
    linear_combination c11144.trans k2 - hc

theorem publicBatchWrapper89_f1715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 208)) (a (.wire 170 7)) (a (.virt 18541)) (a (.virt 18542)) := by
  have c11145 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.1
  have c11146 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11147 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11148 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11149 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11150 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11151 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11152 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11153 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11154 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11155 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11156 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11157 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11158 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11159 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11160 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11161 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_173_16 := arithEq_of_rows h (row := 173) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_16
  simp only [← c11151, ← c11152, ← c11153] at e_173_16
  have e_173_17 := arithEq_of_rows h (row := 173) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_17
  simp only [← c11154, ← c11155, ← c11156] at e_173_17
  have e_174_1 := arithEq_of_rows h (row := 174) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_1
  simp only [← c11145, k3, ← c11146, k3, ← c11147] at e_174_1
  have e_174_2 := arithEq_of_rows h (row := 174) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_2
  simp only [← c11148, ← c11149, k3, ← c11150] at e_174_2
  have e_174_3 := arithEq_of_rows h (row := 174) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_3
  simp only [← c11157, ← c11158, k3, ← c11159] at e_174_3
  refine ⟨?_, ?_⟩
  · have hc := e_173_16
    simp only [e_174_2] at hc
    linear_combination c11160.trans k2 - hc
  · have hc := e_174_3
    simp only [e_174_1, e_173_17, e_174_2] at hc
    linear_combination c11161.trans k2 - hc

theorem publicBatchWrapper89_f1716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 173 75) = band (a (.virt 18535)) (a (.virt 18537)) := by
  have c11162 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11163 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11164 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_173_18 := arithEq_of_rows h (row := 173) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_18
  simp only [← c11162, ← c11163, ← c11164] at e_173_18
  have hr := e_173_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 173 79) = band (a (.virt 18539)) (a (.virt 18541)) := by
  have c11165 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11166 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11167 := (publicBatchWrapper89_copies348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_173_19 := arithEq_of_rows h (row := 173) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_173_19
  simp only [← c11165, ← c11166, ← c11167] at e_173_19
  have hr := e_173_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 175 3) = band (a (.wire 173 75)) (a (.wire 173 79)) := by
  have c11168 := (publicBatchWrapper89_copies349 a h).1
  have c11169 := (publicBatchWrapper89_copies349 a h).2.1
  have c11170 := (publicBatchWrapper89_copies349 a h).2.2.1
  have e_175_0 := arithEq_of_rows h (row := 175) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_0
  simp only [← c11168, ← c11169, ← c11170] at e_175_0
  have hr := e_175_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 23) = bor (a (.wire 3 23)) (a (.wire 175 3)) := by
  have c11171 := (publicBatchWrapper89_copies349 a h).2.2.2.1
  have c11172 := (publicBatchWrapper89_copies349 a h).2.2.2.2.1
  have c11173 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.1
  have c11174 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.1
  have c11175 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.1
  have c11176 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_13 := arithEq_of_rows h (row := 164) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_13
  simp only [← c11171, ← c11172, ← c11173] at e_164_13
  have e_171_5 := arithEq_of_rows h (row := 171) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_5
  simp only [← c11174, ← c11175, k3, ← c11176] at e_171_5
  have hr := e_171_5
  simp only [e_164_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 23) = a (.virt 201) := by
  have c11177 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.1
  exact c11177

theorem publicBatchWrapper89_f1721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 403)) (a (.wire 170 23)) (a (.virt 18543)) (a (.virt 18544)) := by
  have c11178 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11179 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11180 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11181 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11182 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11183 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11184 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11185 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11186 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11187 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11188 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11189 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11190 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11191 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11192 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11193 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11194 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_174_4 := arithEq_of_rows h (row := 174) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_4
  simp only [← c11178, k3, ← c11179, k3, ← c11180] at e_174_4
  have e_174_5 := arithEq_of_rows h (row := 174) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_5
  simp only [← c11181, ← c11182, k3, ← c11183] at e_174_5
  have e_174_6 := arithEq_of_rows h (row := 174) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_6
  simp only [← c11190, ← c11191, k3, ← c11192] at e_174_6
  have e_175_1 := arithEq_of_rows h (row := 175) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_1
  simp only [← c11184, ← c11185, ← c11186] at e_175_1
  have e_175_2 := arithEq_of_rows h (row := 175) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_2
  simp only [← c11187, ← c11188, ← c11189] at e_175_2
  refine ⟨?_, ?_⟩
  · have hc := e_175_1
    simp only [e_174_5] at hc
    linear_combination c11193.trans k2 - hc
  · have hc := e_174_6
    simp only [e_174_4, e_175_2, e_174_5] at hc
    linear_combination c11194.trans k2 - hc

theorem publicBatchWrapper89_f1722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 27) = bor (a (.wire 3 67)) (a (.virt 18543)) := by
  have c11195 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11196 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11197 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11198 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11199 := (publicBatchWrapper89_copies349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11200 := (publicBatchWrapper89_copies350 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_14 := arithEq_of_rows h (row := 164) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_14
  simp only [← c11195, ← c11196, ← c11197] at e_164_14
  have e_171_6 := arithEq_of_rows h (row := 171) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_6
  simp only [← c11198, ← c11199, k3, ← c11200] at e_171_6
  have hr := e_171_6
  simp only [e_164_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 27) = a (.virt 201) := by
  have c11201 := (publicBatchWrapper89_copies350 a h).2.1
  exact c11201

theorem publicBatchWrapper89_f1724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 404)) (a (.wire 170 31)) (a (.virt 18545)) (a (.virt 18546)) := by
  have c11202 := (publicBatchWrapper89_copies350 a h).2.2.1
  have c11203 := (publicBatchWrapper89_copies350 a h).2.2.2.1
  have c11204 := (publicBatchWrapper89_copies350 a h).2.2.2.2.1
  have c11205 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.1
  have c11206 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.1
  have c11207 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.1
  have c11208 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.1
  have c11209 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.1
  have c11210 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11211 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11212 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11213 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11214 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11215 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11216 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11217 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11218 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_174_7 := arithEq_of_rows h (row := 174) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_7
  simp only [← c11202, k3, ← c11203, k3, ← c11204] at e_174_7
  have e_174_8 := arithEq_of_rows h (row := 174) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_8
  simp only [← c11205, ← c11206, k3, ← c11207] at e_174_8
  have e_174_9 := arithEq_of_rows h (row := 174) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_9
  simp only [← c11214, ← c11215, k3, ← c11216] at e_174_9
  have e_175_3 := arithEq_of_rows h (row := 175) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_3
  simp only [← c11208, ← c11209, ← c11210] at e_175_3
  have e_175_4 := arithEq_of_rows h (row := 175) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_4
  simp only [← c11211, ← c11212, ← c11213] at e_175_4
  refine ⟨?_, ?_⟩
  · have hc := e_175_3
    simp only [e_174_8] at hc
    linear_combination c11217.trans k2 - hc
  · have hc := e_174_9
    simp only [e_174_7, e_175_4, e_174_8] at hc
    linear_combination c11218.trans k2 - hc

theorem publicBatchWrapper89_f1725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 31) = bor (a (.wire 3 67)) (a (.virt 18545)) := by
  have c11219 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11220 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11221 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11222 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11223 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11224 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_15 := arithEq_of_rows h (row := 164) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_15
  simp only [← c11219, ← c11220, ← c11221] at e_164_15
  have e_171_7 := arithEq_of_rows h (row := 171) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_7
  simp only [← c11222, ← c11223, k3, ← c11224] at e_171_7
  have hr := e_171_7
  simp only [e_164_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 31) = a (.virt 201) := by
  have c11225 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11225

theorem publicBatchWrapper89_f1727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 405)) (a (.wire 169 63)) (a (.virt 18547)) (a (.virt 18548)) := by
  have c11226 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11227 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11228 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11229 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11230 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11231 := (publicBatchWrapper89_copies350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11232 := (publicBatchWrapper89_copies351 a h).1
  have c11233 := (publicBatchWrapper89_copies351 a h).2.1
  have c11234 := (publicBatchWrapper89_copies351 a h).2.2.1
  have c11235 := (publicBatchWrapper89_copies351 a h).2.2.2.1
  have c11236 := (publicBatchWrapper89_copies351 a h).2.2.2.2.1
  have c11237 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.1
  have c11238 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.1
  have c11239 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.1
  have c11240 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.1
  have c11241 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.1
  have c11242 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_174_10 := arithEq_of_rows h (row := 174) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_10
  simp only [← c11226, k3, ← c11227, k3, ← c11228] at e_174_10
  have e_174_11 := arithEq_of_rows h (row := 174) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_11
  simp only [← c11229, ← c11230, k3, ← c11231] at e_174_11
  have e_174_12 := arithEq_of_rows h (row := 174) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_12
  simp only [← c11238, ← c11239, k3, ← c11240] at e_174_12
  have e_175_5 := arithEq_of_rows h (row := 175) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_5
  simp only [← c11232, ← c11233, ← c11234] at e_175_5
  have e_175_6 := arithEq_of_rows h (row := 175) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_6
  simp only [← c11235, ← c11236, ← c11237] at e_175_6
  refine ⟨?_, ?_⟩
  · have hc := e_175_5
    simp only [e_174_11] at hc
    linear_combination c11241.trans k2 - hc
  · have hc := e_174_12
    simp only [e_174_10, e_175_6, e_174_11] at hc
    linear_combination c11242.trans k2 - hc

theorem publicBatchWrapper89_f1728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 406)) (a (.wire 169 71)) (a (.virt 18549)) (a (.virt 18550)) := by
  have c11243 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11244 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11245 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11246 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11247 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11248 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11249 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11250 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11251 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11252 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11253 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11254 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11255 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11256 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11257 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11258 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11259 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_174_13 := arithEq_of_rows h (row := 174) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_13
  simp only [← c11243, k3, ← c11244, k3, ← c11245] at e_174_13
  have e_174_14 := arithEq_of_rows h (row := 174) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_14
  simp only [← c11246, ← c11247, k3, ← c11248] at e_174_14
  have e_174_15 := arithEq_of_rows h (row := 174) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_15
  simp only [← c11255, ← c11256, k3, ← c11257] at e_174_15
  have e_175_7 := arithEq_of_rows h (row := 175) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_7
  simp only [← c11249, ← c11250, ← c11251] at e_175_7
  have e_175_8 := arithEq_of_rows h (row := 175) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_8
  simp only [← c11252, ← c11253, ← c11254] at e_175_8
  refine ⟨?_, ?_⟩
  · have hc := e_175_7
    simp only [e_174_14] at hc
    linear_combination c11258.trans k2 - hc
  · have hc := e_174_15
    simp only [e_174_13, e_175_8, e_174_14] at hc
    linear_combination c11259.trans k2 - hc

theorem publicBatchWrapper89_f1729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 407)) (a (.wire 169 79)) (a (.virt 18551)) (a (.virt 18552)) := by
  have c11260 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11261 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11262 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11263 := (publicBatchWrapper89_copies351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11264 := (publicBatchWrapper89_copies352 a h).1
  have c11265 := (publicBatchWrapper89_copies352 a h).2.1
  have c11266 := (publicBatchWrapper89_copies352 a h).2.2.1
  have c11267 := (publicBatchWrapper89_copies352 a h).2.2.2.1
  have c11268 := (publicBatchWrapper89_copies352 a h).2.2.2.2.1
  have c11269 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.1
  have c11270 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.1
  have c11271 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.1
  have c11272 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.1
  have c11273 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.1
  have c11274 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11275 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11276 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_174_16 := arithEq_of_rows h (row := 174) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_16
  simp only [← c11260, k3, ← c11261, k3, ← c11262] at e_174_16
  have e_174_17 := arithEq_of_rows h (row := 174) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_17
  simp only [← c11263, ← c11264, k3, ← c11265] at e_174_17
  have e_174_18 := arithEq_of_rows h (row := 174) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_18
  simp only [← c11272, ← c11273, k3, ← c11274] at e_174_18
  have e_175_9 := arithEq_of_rows h (row := 175) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_9
  simp only [← c11266, ← c11267, ← c11268] at e_175_9
  have e_175_10 := arithEq_of_rows h (row := 175) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_10
  simp only [← c11269, ← c11270, ← c11271] at e_175_10
  refine ⟨?_, ?_⟩
  · have hc := e_175_9
    simp only [e_174_17] at hc
    linear_combination c11275.trans k2 - hc
  · have hc := e_174_18
    simp only [e_174_16, e_175_10, e_174_17] at hc
    linear_combination c11276.trans k2 - hc

theorem publicBatchWrapper89_f1730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 408)) (a (.wire 170 7)) (a (.virt 18553)) (a (.virt 18554)) := by
  have c11277 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11278 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11279 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11280 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11281 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11282 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11283 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11284 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11285 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11286 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11287 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11288 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11289 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11290 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11291 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11292 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11293 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_174_19 := arithEq_of_rows h (row := 174) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_174_19
  simp only [← c11277, k3, ← c11278, k3, ← c11279] at e_174_19
  have e_175_11 := arithEq_of_rows h (row := 175) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_11
  simp only [← c11283, ← c11284, ← c11285] at e_175_11
  have e_175_12 := arithEq_of_rows h (row := 175) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_12
  simp only [← c11286, ← c11287, ← c11288] at e_175_12
  have e_176_0 := arithEq_of_rows h (row := 176) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_0
  simp only [← c11280, ← c11281, k3, ← c11282] at e_176_0
  have e_176_1 := arithEq_of_rows h (row := 176) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_1
  simp only [← c11289, ← c11290, k3, ← c11291] at e_176_1
  refine ⟨?_, ?_⟩
  · have hc := e_175_11
    simp only [e_176_0] at hc
    linear_combination c11292.trans k2 - hc
  · have hc := e_176_1
    simp only [e_174_19, e_175_12, e_176_0] at hc
    linear_combination c11293.trans k2 - hc

theorem publicBatchWrapper89_f1731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 175 55) = band (a (.virt 18547)) (a (.virt 18549)) := by
  have c11294 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11295 := (publicBatchWrapper89_copies352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11296 := (publicBatchWrapper89_copies353 a h).1
  have e_175_13 := arithEq_of_rows h (row := 175) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_13
  simp only [← c11294, ← c11295, ← c11296] at e_175_13
  have hr := e_175_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 175 59) = band (a (.virt 18551)) (a (.virt 18553)) := by
  have c11297 := (publicBatchWrapper89_copies353 a h).2.1
  have c11298 := (publicBatchWrapper89_copies353 a h).2.2.1
  have c11299 := (publicBatchWrapper89_copies353 a h).2.2.2.1
  have e_175_14 := arithEq_of_rows h (row := 175) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_14
  simp only [← c11297, ← c11298, ← c11299] at e_175_14
  have hr := e_175_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 175 63) = band (a (.wire 175 55)) (a (.wire 175 59)) := by
  have c11300 := (publicBatchWrapper89_copies353 a h).2.2.2.2.1
  have c11301 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.1
  have c11302 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.1
  have e_175_15 := arithEq_of_rows h (row := 175) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_15
  simp only [← c11300, ← c11301, ← c11302] at e_175_15
  have hr := e_175_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 35) = bor (a (.wire 3 67)) (a (.wire 175 63)) := by
  have c11303 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.1
  have c11304 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.1
  have c11305 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.1
  have c11306 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11307 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11308 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_16 := arithEq_of_rows h (row := 164) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_16
  simp only [← c11303, ← c11304, ← c11305] at e_164_16
  have e_171_8 := arithEq_of_rows h (row := 171) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_8
  simp only [← c11306, ← c11307, k3, ← c11308] at e_171_8
  have hr := e_171_8
  simp only [e_164_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 35) = a (.virt 201) := by
  have c11309 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11309

theorem publicBatchWrapper89_f1736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 603)) (a (.wire 170 23)) (a (.virt 18555)) (a (.virt 18556)) := by
  have c11310 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11311 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11312 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11313 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11314 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11315 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11316 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11317 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11318 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11319 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11320 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11321 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11322 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11323 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11324 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11325 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11326 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_175_16 := arithEq_of_rows h (row := 175) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_16
  simp only [← c11316, ← c11317, ← c11318] at e_175_16
  have e_175_17 := arithEq_of_rows h (row := 175) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_17
  simp only [← c11319, ← c11320, ← c11321] at e_175_17
  have e_176_2 := arithEq_of_rows h (row := 176) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_2
  simp only [← c11310, k3, ← c11311, k3, ← c11312] at e_176_2
  have e_176_3 := arithEq_of_rows h (row := 176) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_3
  simp only [← c11313, ← c11314, k3, ← c11315] at e_176_3
  have e_176_4 := arithEq_of_rows h (row := 176) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_4
  simp only [← c11322, ← c11323, k3, ← c11324] at e_176_4
  refine ⟨?_, ?_⟩
  · have hc := e_175_16
    simp only [e_176_3] at hc
    linear_combination c11325.trans k2 - hc
  · have hc := e_176_4
    simp only [e_176_2, e_175_17, e_176_3] at hc
    linear_combination c11326.trans k2 - hc

theorem publicBatchWrapper89_f1737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 39) = bor (a (.wire 5 31)) (a (.virt 18555)) := by
  have c11327 := (publicBatchWrapper89_copies353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11328 := (publicBatchWrapper89_copies354 a h).1
  have c11329 := (publicBatchWrapper89_copies354 a h).2.1
  have c11330 := (publicBatchWrapper89_copies354 a h).2.2.1
  have c11331 := (publicBatchWrapper89_copies354 a h).2.2.2.1
  have c11332 := (publicBatchWrapper89_copies354 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_17 := arithEq_of_rows h (row := 164) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_17
  simp only [← c11327, ← c11328, ← c11329] at e_164_17
  have e_171_9 := arithEq_of_rows h (row := 171) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_9
  simp only [← c11330, ← c11331, k3, ← c11332] at e_171_9
  have hr := e_171_9
  simp only [e_164_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 39) = a (.virt 201) := by
  have c11333 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.1
  exact c11333

theorem publicBatchWrapper89_f1739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 604)) (a (.wire 170 31)) (a (.virt 18557)) (a (.virt 18558)) := by
  have c11334 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.1
  have c11335 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.1
  have c11336 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.1
  have c11337 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.1
  have c11338 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11339 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11340 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11341 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11342 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11343 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11344 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11345 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11346 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11347 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11348 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11349 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11350 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_175_18 := arithEq_of_rows h (row := 175) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_18
  simp only [← c11340, ← c11341, ← c11342] at e_175_18
  have e_175_19 := arithEq_of_rows h (row := 175) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_175_19
  simp only [← c11343, ← c11344, ← c11345] at e_175_19
  have e_176_5 := arithEq_of_rows h (row := 176) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_5
  simp only [← c11334, k3, ← c11335, k3, ← c11336] at e_176_5
  have e_176_6 := arithEq_of_rows h (row := 176) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_6
  simp only [← c11337, ← c11338, k3, ← c11339] at e_176_6
  have e_176_7 := arithEq_of_rows h (row := 176) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_7
  simp only [← c11346, ← c11347, k3, ← c11348] at e_176_7
  refine ⟨?_, ?_⟩
  · have hc := e_175_18
    simp only [e_176_6] at hc
    linear_combination c11349.trans k2 - hc
  · have hc := e_176_7
    simp only [e_176_5, e_175_19, e_176_6] at hc
    linear_combination c11350.trans k2 - hc

theorem publicBatchWrapper89_f1740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 43) = bor (a (.wire 5 31)) (a (.virt 18557)) := by
  have c11351 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11352 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11353 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11354 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11355 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11356 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_18 := arithEq_of_rows h (row := 164) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_18
  simp only [← c11351, ← c11352, ← c11353] at e_164_18
  have e_171_10 := arithEq_of_rows h (row := 171) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_10
  simp only [← c11354, ← c11355, k3, ← c11356] at e_171_10
  have hr := e_171_10
  simp only [e_164_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 43) = a (.virt 201) := by
  have c11357 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11357

theorem publicBatchWrapper89_f1742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 605)) (a (.wire 169 63)) (a (.virt 18559)) (a (.virt 18560)) := by
  have c11358 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11359 := (publicBatchWrapper89_copies354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11360 := (publicBatchWrapper89_copies355 a h).1
  have c11361 := (publicBatchWrapper89_copies355 a h).2.1
  have c11362 := (publicBatchWrapper89_copies355 a h).2.2.1
  have c11363 := (publicBatchWrapper89_copies355 a h).2.2.2.1
  have c11364 := (publicBatchWrapper89_copies355 a h).2.2.2.2.1
  have c11365 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.1
  have c11366 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.1
  have c11367 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.1
  have c11368 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.1
  have c11369 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.1
  have c11370 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11371 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11372 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11373 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11374 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_176_8 := arithEq_of_rows h (row := 176) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_8
  simp only [← c11358, k3, ← c11359, k3, ← c11360] at e_176_8
  have e_176_9 := arithEq_of_rows h (row := 176) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_9
  simp only [← c11361, ← c11362, k3, ← c11363] at e_176_9
  have e_176_10 := arithEq_of_rows h (row := 176) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_10
  simp only [← c11370, ← c11371, k3, ← c11372] at e_176_10
  have e_177_0 := arithEq_of_rows h (row := 177) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_0
  simp only [← c11364, ← c11365, ← c11366] at e_177_0
  have e_177_1 := arithEq_of_rows h (row := 177) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_1
  simp only [← c11367, ← c11368, ← c11369] at e_177_1
  refine ⟨?_, ?_⟩
  · have hc := e_177_0
    simp only [e_176_9] at hc
    linear_combination c11373.trans k2 - hc
  · have hc := e_176_10
    simp only [e_176_8, e_177_1, e_176_9] at hc
    linear_combination c11374.trans k2 - hc

theorem publicBatchWrapper89_f1743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 606)) (a (.wire 169 71)) (a (.virt 18561)) (a (.virt 18562)) := by
  have c11375 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11376 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11377 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11378 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11379 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11380 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11381 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11382 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11383 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11384 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11385 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11386 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11387 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11388 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11389 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11390 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11391 := (publicBatchWrapper89_copies355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_176_11 := arithEq_of_rows h (row := 176) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_11
  simp only [← c11375, k3, ← c11376, k3, ← c11377] at e_176_11
  have e_176_12 := arithEq_of_rows h (row := 176) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_12
  simp only [← c11378, ← c11379, k3, ← c11380] at e_176_12
  have e_176_13 := arithEq_of_rows h (row := 176) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_13
  simp only [← c11387, ← c11388, k3, ← c11389] at e_176_13
  have e_177_2 := arithEq_of_rows h (row := 177) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_2
  simp only [← c11381, ← c11382, ← c11383] at e_177_2
  have e_177_3 := arithEq_of_rows h (row := 177) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_3
  simp only [← c11384, ← c11385, ← c11386] at e_177_3
  refine ⟨?_, ?_⟩
  · have hc := e_177_2
    simp only [e_176_12] at hc
    linear_combination c11390.trans k2 - hc
  · have hc := e_176_13
    simp only [e_176_11, e_177_3, e_176_12] at hc
    linear_combination c11391.trans k2 - hc

theorem publicBatchWrapper89_f1744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 607)) (a (.wire 169 79)) (a (.virt 18563)) (a (.virt 18564)) := by
  have c11392 := (publicBatchWrapper89_copies356 a h).1
  have c11393 := (publicBatchWrapper89_copies356 a h).2.1
  have c11394 := (publicBatchWrapper89_copies356 a h).2.2.1
  have c11395 := (publicBatchWrapper89_copies356 a h).2.2.2.1
  have c11396 := (publicBatchWrapper89_copies356 a h).2.2.2.2.1
  have c11397 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.1
  have c11398 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.1
  have c11399 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.1
  have c11400 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.1
  have c11401 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.1
  have c11402 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11403 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11404 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11405 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11406 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11407 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11408 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_176_14 := arithEq_of_rows h (row := 176) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_14
  simp only [← c11392, k3, ← c11393, k3, ← c11394] at e_176_14
  have e_176_15 := arithEq_of_rows h (row := 176) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_15
  simp only [← c11395, ← c11396, k3, ← c11397] at e_176_15
  have e_176_16 := arithEq_of_rows h (row := 176) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_16
  simp only [← c11404, ← c11405, k3, ← c11406] at e_176_16
  have e_177_4 := arithEq_of_rows h (row := 177) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_4
  simp only [← c11398, ← c11399, ← c11400] at e_177_4
  have e_177_5 := arithEq_of_rows h (row := 177) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_5
  simp only [← c11401, ← c11402, ← c11403] at e_177_5
  refine ⟨?_, ?_⟩
  · have hc := e_177_4
    simp only [e_176_15] at hc
    linear_combination c11407.trans k2 - hc
  · have hc := e_176_16
    simp only [e_176_14, e_177_5, e_176_15] at hc
    linear_combination c11408.trans k2 - hc

theorem publicBatchWrapper89_f1745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 608)) (a (.wire 170 7)) (a (.virt 18565)) (a (.virt 18566)) := by
  have c11409 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11410 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11411 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11412 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11413 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11414 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11415 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11416 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11417 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11418 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11419 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11420 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11421 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11422 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11423 := (publicBatchWrapper89_copies356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11424 := (publicBatchWrapper89_copies357 a h).1
  have c11425 := (publicBatchWrapper89_copies357 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_176_17 := arithEq_of_rows h (row := 176) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_17
  simp only [← c11409, k3, ← c11410, k3, ← c11411] at e_176_17
  have e_176_18 := arithEq_of_rows h (row := 176) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_18
  simp only [← c11412, ← c11413, k3, ← c11414] at e_176_18
  have e_176_19 := arithEq_of_rows h (row := 176) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_176_19
  simp only [← c11421, ← c11422, k3, ← c11423] at e_176_19
  have e_177_6 := arithEq_of_rows h (row := 177) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_6
  simp only [← c11415, ← c11416, ← c11417] at e_177_6
  have e_177_7 := arithEq_of_rows h (row := 177) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_7
  simp only [← c11418, ← c11419, ← c11420] at e_177_7
  refine ⟨?_, ?_⟩
  · have hc := e_177_6
    simp only [e_176_18] at hc
    linear_combination c11424.trans k2 - hc
  · have hc := e_176_19
    simp only [e_176_17, e_177_7, e_176_18] at hc
    linear_combination c11425.trans k2 - hc

theorem publicBatchWrapper89_f1746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 177 35) = band (a (.virt 18559)) (a (.virt 18561)) := by
  have c11426 := (publicBatchWrapper89_copies357 a h).2.2.1
  have c11427 := (publicBatchWrapper89_copies357 a h).2.2.2.1
  have c11428 := (publicBatchWrapper89_copies357 a h).2.2.2.2.1
  have e_177_8 := arithEq_of_rows h (row := 177) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_8
  simp only [← c11426, ← c11427, ← c11428] at e_177_8
  have hr := e_177_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 177 39) = band (a (.virt 18563)) (a (.virt 18565)) := by
  have c11429 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.1
  have c11430 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.1
  have c11431 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.1
  have e_177_9 := arithEq_of_rows h (row := 177) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_9
  simp only [← c11429, ← c11430, ← c11431] at e_177_9
  have hr := e_177_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 177 43) = band (a (.wire 177 35)) (a (.wire 177 39)) := by
  have c11432 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.1
  have c11433 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.1
  have c11434 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_177_10 := arithEq_of_rows h (row := 177) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_10
  simp only [← c11432, ← c11433, ← c11434] at e_177_10
  have hr := e_177_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 47) = bor (a (.wire 5 31)) (a (.wire 177 43)) := by
  have c11435 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11436 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11437 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11438 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11439 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11440 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_164_19 := arithEq_of_rows h (row := 164) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_19
  simp only [← c11435, ← c11436, ← c11437] at e_164_19
  have e_171_11 := arithEq_of_rows h (row := 171) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_11
  simp only [← c11438, ← c11439, k3, ← c11440] at e_171_11
  have hr := e_171_11
  simp only [e_164_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 47) = a (.virt 201) := by
  have c11441 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11441

theorem publicBatchWrapper89_f1751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 803)) (a (.wire 170 23)) (a (.virt 18567)) (a (.virt 18568)) := by
  have c11442 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11443 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11444 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11445 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11446 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11447 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11448 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11449 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11450 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11451 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11452 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11453 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11454 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11455 := (publicBatchWrapper89_copies357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11456 := (publicBatchWrapper89_copies358 a h).1
  have c11457 := (publicBatchWrapper89_copies358 a h).2.1
  have c11458 := (publicBatchWrapper89_copies358 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_177_11 := arithEq_of_rows h (row := 177) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_11
  simp only [← c11448, ← c11449, ← c11450] at e_177_11
  have e_177_12 := arithEq_of_rows h (row := 177) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_12
  simp only [← c11451, ← c11452, ← c11453] at e_177_12
  have e_178_0 := arithEq_of_rows h (row := 178) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_0
  simp only [← c11442, k3, ← c11443, k3, ← c11444] at e_178_0
  have e_178_1 := arithEq_of_rows h (row := 178) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_1
  simp only [← c11445, ← c11446, k3, ← c11447] at e_178_1
  have e_178_2 := arithEq_of_rows h (row := 178) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_2
  simp only [← c11454, ← c11455, k3, ← c11456] at e_178_2
  refine ⟨?_, ?_⟩
  · have hc := e_177_11
    simp only [e_178_1] at hc
    linear_combination c11457.trans k2 - hc
  · have hc := e_178_2
    simp only [e_178_0, e_177_12, e_178_1] at hc
    linear_combination c11458.trans k2 - hc

theorem publicBatchWrapper89_f1752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 51) = bor (a (.wire 5 75)) (a (.virt 18567)) := by
  have c11459 := (publicBatchWrapper89_copies358 a h).2.2.2.1
  have c11460 := (publicBatchWrapper89_copies358 a h).2.2.2.2.1
  have c11461 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.1
  have c11462 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.1
  have c11463 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.1
  have c11464 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_171_12 := arithEq_of_rows h (row := 171) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_12
  simp only [← c11462, ← c11463, k3, ← c11464] at e_171_12
  have e_179_0 := arithEq_of_rows h (row := 179) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_0
  simp only [← c11459, ← c11460, ← c11461] at e_179_0
  have hr := e_171_12
  simp only [e_179_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 51) = a (.virt 201) := by
  have c11465 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.1
  exact c11465

theorem publicBatchWrapper89_f1754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 804)) (a (.wire 170 31)) (a (.virt 18569)) (a (.virt 18570)) := by
  have c11466 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11467 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11468 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11469 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11470 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11471 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11472 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11473 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11474 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11475 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11476 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11477 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11478 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11479 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11480 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11481 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11482 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_177_13 := arithEq_of_rows h (row := 177) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_13
  simp only [← c11472, ← c11473, ← c11474] at e_177_13
  have e_177_14 := arithEq_of_rows h (row := 177) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_14
  simp only [← c11475, ← c11476, ← c11477] at e_177_14
  have e_178_3 := arithEq_of_rows h (row := 178) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_3
  simp only [← c11466, k3, ← c11467, k3, ← c11468] at e_178_3
  have e_178_4 := arithEq_of_rows h (row := 178) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_4
  simp only [← c11469, ← c11470, k3, ← c11471] at e_178_4
  have e_178_5 := arithEq_of_rows h (row := 178) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_5
  simp only [← c11478, ← c11479, k3, ← c11480] at e_178_5
  refine ⟨?_, ?_⟩
  · have hc := e_177_13
    simp only [e_178_4] at hc
    linear_combination c11481.trans k2 - hc
  · have hc := e_178_5
    simp only [e_178_3, e_177_14, e_178_4] at hc
    linear_combination c11482.trans k2 - hc

theorem publicBatchWrapper89_f1755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 55) = bor (a (.wire 5 75)) (a (.virt 18569)) := by
  have c11483 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11484 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11485 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11486 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11487 := (publicBatchWrapper89_copies358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11488 := (publicBatchWrapper89_copies359 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_171_13 := arithEq_of_rows h (row := 171) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_13
  simp only [← c11486, ← c11487, k3, ← c11488] at e_171_13
  have e_179_1 := arithEq_of_rows h (row := 179) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_1
  simp only [← c11483, ← c11484, ← c11485] at e_179_1
  have hr := e_171_13
  simp only [e_179_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 55) = a (.virt 201) := by
  have c11489 := (publicBatchWrapper89_copies359 a h).2.1
  exact c11489

theorem publicBatchWrapper89_f1757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 805)) (a (.wire 169 63)) (a (.virt 18571)) (a (.virt 18572)) := by
  have c11490 := (publicBatchWrapper89_copies359 a h).2.2.1
  have c11491 := (publicBatchWrapper89_copies359 a h).2.2.2.1
  have c11492 := (publicBatchWrapper89_copies359 a h).2.2.2.2.1
  have c11493 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.1
  have c11494 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.1
  have c11495 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.1
  have c11496 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.1
  have c11497 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.1
  have c11498 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11499 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11500 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11501 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11502 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11503 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11504 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11505 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11506 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_177_15 := arithEq_of_rows h (row := 177) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_15
  simp only [← c11496, ← c11497, ← c11498] at e_177_15
  have e_177_16 := arithEq_of_rows h (row := 177) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_16
  simp only [← c11499, ← c11500, ← c11501] at e_177_16
  have e_178_6 := arithEq_of_rows h (row := 178) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_6
  simp only [← c11490, k3, ← c11491, k3, ← c11492] at e_178_6
  have e_178_7 := arithEq_of_rows h (row := 178) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_7
  simp only [← c11493, ← c11494, k3, ← c11495] at e_178_7
  have e_178_8 := arithEq_of_rows h (row := 178) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_8
  simp only [← c11502, ← c11503, k3, ← c11504] at e_178_8
  refine ⟨?_, ?_⟩
  · have hc := e_177_15
    simp only [e_178_7] at hc
    linear_combination c11505.trans k2 - hc
  · have hc := e_178_8
    simp only [e_178_6, e_177_16, e_178_7] at hc
    linear_combination c11506.trans k2 - hc

theorem publicBatchWrapper89_f1758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 806)) (a (.wire 169 71)) (a (.virt 18573)) (a (.virt 18574)) := by
  have c11507 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11508 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11509 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11510 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11511 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11512 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11513 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11514 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11515 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11516 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11517 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11518 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11519 := (publicBatchWrapper89_copies359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11520 := (publicBatchWrapper89_copies360 a h).1
  have c11521 := (publicBatchWrapper89_copies360 a h).2.1
  have c11522 := (publicBatchWrapper89_copies360 a h).2.2.1
  have c11523 := (publicBatchWrapper89_copies360 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_177_17 := arithEq_of_rows h (row := 177) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_17
  simp only [← c11513, ← c11514, ← c11515] at e_177_17
  have e_177_18 := arithEq_of_rows h (row := 177) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_18
  simp only [← c11516, ← c11517, ← c11518] at e_177_18
  have e_178_9 := arithEq_of_rows h (row := 178) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_9
  simp only [← c11507, k3, ← c11508, k3, ← c11509] at e_178_9
  have e_178_10 := arithEq_of_rows h (row := 178) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_10
  simp only [← c11510, ← c11511, k3, ← c11512] at e_178_10
  have e_178_11 := arithEq_of_rows h (row := 178) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_11
  simp only [← c11519, ← c11520, k3, ← c11521] at e_178_11
  refine ⟨?_, ?_⟩
  · have hc := e_177_17
    simp only [e_178_10] at hc
    linear_combination c11522.trans k2 - hc
  · have hc := e_178_11
    simp only [e_178_9, e_177_18, e_178_10] at hc
    linear_combination c11523.trans k2 - hc

theorem publicBatchWrapper89_f1759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 807)) (a (.wire 169 79)) (a (.virt 18575)) (a (.virt 18576)) := by
  have c11524 := (publicBatchWrapper89_copies360 a h).2.2.2.2.1
  have c11525 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.1
  have c11526 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.1
  have c11527 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.1
  have c11528 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.1
  have c11529 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.1
  have c11530 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11531 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11532 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11533 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11534 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11535 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11536 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11537 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11538 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11539 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11540 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_177_19 := arithEq_of_rows h (row := 177) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_177_19
  simp only [← c11530, ← c11531, ← c11532] at e_177_19
  have e_178_12 := arithEq_of_rows h (row := 178) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_12
  simp only [← c11524, k3, ← c11525, k3, ← c11526] at e_178_12
  have e_178_13 := arithEq_of_rows h (row := 178) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_13
  simp only [← c11527, ← c11528, k3, ← c11529] at e_178_13
  have e_178_14 := arithEq_of_rows h (row := 178) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_14
  simp only [← c11536, ← c11537, k3, ← c11538] at e_178_14
  have e_180_0 := arithEq_of_rows h (row := 180) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_0
  simp only [← c11533, ← c11534, ← c11535] at e_180_0
  refine ⟨?_, ?_⟩
  · have hc := e_177_19
    simp only [e_178_13] at hc
    linear_combination c11539.trans k2 - hc
  · have hc := e_178_14
    simp only [e_178_12, e_180_0, e_178_13] at hc
    linear_combination c11540.trans k2 - hc

theorem publicBatchWrapper89_f1760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 808)) (a (.wire 170 7)) (a (.virt 18577)) (a (.virt 18578)) := by
  have c11541 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11542 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11543 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11544 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11545 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11546 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11547 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11548 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11549 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11550 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11551 := (publicBatchWrapper89_copies360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11552 := (publicBatchWrapper89_copies361 a h).1
  have c11553 := (publicBatchWrapper89_copies361 a h).2.1
  have c11554 := (publicBatchWrapper89_copies361 a h).2.2.1
  have c11555 := (publicBatchWrapper89_copies361 a h).2.2.2.1
  have c11556 := (publicBatchWrapper89_copies361 a h).2.2.2.2.1
  have c11557 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_178_15 := arithEq_of_rows h (row := 178) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_15
  simp only [← c11541, k3, ← c11542, k3, ← c11543] at e_178_15
  have e_178_16 := arithEq_of_rows h (row := 178) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_16
  simp only [← c11544, ← c11545, k3, ← c11546] at e_178_16
  have e_178_17 := arithEq_of_rows h (row := 178) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_17
  simp only [← c11553, ← c11554, k3, ← c11555] at e_178_17
  have e_180_1 := arithEq_of_rows h (row := 180) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_1
  simp only [← c11547, ← c11548, ← c11549] at e_180_1
  have e_180_2 := arithEq_of_rows h (row := 180) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_2
  simp only [← c11550, ← c11551, ← c11552] at e_180_2
  refine ⟨?_, ?_⟩
  · have hc := e_180_1
    simp only [e_178_16] at hc
    linear_combination c11556.trans k2 - hc
  · have hc := e_178_17
    simp only [e_178_15, e_180_2, e_178_16] at hc
    linear_combination c11557.trans k2 - hc

theorem publicBatchWrapper89_f1761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 180 15) = band (a (.virt 18571)) (a (.virt 18573)) := by
  have c11558 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.1
  have c11559 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.1
  have c11560 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.1
  have e_180_3 := arithEq_of_rows h (row := 180) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_3
  simp only [← c11558, ← c11559, ← c11560] at e_180_3
  have hr := e_180_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 180 19) = band (a (.virt 18575)) (a (.virt 18577)) := by
  have c11561 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.1
  have c11562 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11563 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_180_4 := arithEq_of_rows h (row := 180) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_4
  simp only [← c11561, ← c11562, ← c11563] at e_180_4
  have hr := e_180_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 180 23) = band (a (.wire 180 15)) (a (.wire 180 19)) := by
  have c11564 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11565 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11566 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_180_5 := arithEq_of_rows h (row := 180) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_5
  simp only [← c11564, ← c11565, ← c11566] at e_180_5
  have hr := e_180_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 59) = bor (a (.wire 5 75)) (a (.wire 180 23)) := by
  have c11567 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11568 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11569 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11570 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11571 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11572 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_171_14 := arithEq_of_rows h (row := 171) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_14
  simp only [← c11570, ← c11571, k3, ← c11572] at e_171_14
  have e_179_2 := arithEq_of_rows h (row := 179) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_2
  simp only [← c11567, ← c11568, ← c11569] at e_179_2
  have hr := e_171_14
  simp only [e_179_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 59) = a (.virt 201) := by
  have c11573 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11573

theorem publicBatchWrapper89_f1766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1003)) (a (.wire 170 23)) (a (.virt 18579)) (a (.virt 18580)) := by
  have c11574 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11575 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11576 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11577 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11578 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11579 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11580 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11581 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11582 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11583 := (publicBatchWrapper89_copies361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11584 := (publicBatchWrapper89_copies362 a h).1
  have c11585 := (publicBatchWrapper89_copies362 a h).2.1
  have c11586 := (publicBatchWrapper89_copies362 a h).2.2.1
  have c11587 := (publicBatchWrapper89_copies362 a h).2.2.2.1
  have c11588 := (publicBatchWrapper89_copies362 a h).2.2.2.2.1
  have c11589 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.1
  have c11590 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_178_18 := arithEq_of_rows h (row := 178) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_18
  simp only [← c11574, k3, ← c11575, k3, ← c11576] at e_178_18
  have e_178_19 := arithEq_of_rows h (row := 178) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_178_19
  simp only [← c11577, ← c11578, k3, ← c11579] at e_178_19
  have e_180_6 := arithEq_of_rows h (row := 180) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_6
  simp only [← c11580, ← c11581, ← c11582] at e_180_6
  have e_180_7 := arithEq_of_rows h (row := 180) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_7
  simp only [← c11583, ← c11584, ← c11585] at e_180_7
  have e_181_0 := arithEq_of_rows h (row := 181) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_0
  simp only [← c11586, ← c11587, k3, ← c11588] at e_181_0
  refine ⟨?_, ?_⟩
  · have hc := e_180_6
    simp only [e_178_19] at hc
    linear_combination c11589.trans k2 - hc
  · have hc := e_181_0
    simp only [e_178_18, e_180_7, e_178_19] at hc
    linear_combination c11590.trans k2 - hc

theorem publicBatchWrapper89_f1767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 63) = bor (a (.wire 7 39)) (a (.virt 18579)) := by
  have c11591 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.1
  have c11592 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.1
  have c11593 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.1
  have c11594 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11595 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11596 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_171_15 := arithEq_of_rows h (row := 171) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_15
  simp only [← c11594, ← c11595, k3, ← c11596] at e_171_15
  have e_179_3 := arithEq_of_rows h (row := 179) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_3
  simp only [← c11591, ← c11592, ← c11593] at e_179_3
  have hr := e_171_15
  simp only [e_179_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 63) = a (.virt 201) := by
  have c11597 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11597

theorem publicBatchWrapper89_f1769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1004)) (a (.wire 170 31)) (a (.virt 18581)) (a (.virt 18582)) := by
  have c11598 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11599 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11600 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11601 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11602 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11603 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11604 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11605 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11606 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11607 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11608 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11609 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11610 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11611 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11612 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11613 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11614 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_180_8 := arithEq_of_rows h (row := 180) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_8
  simp only [← c11604, ← c11605, ← c11606] at e_180_8
  have e_180_9 := arithEq_of_rows h (row := 180) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_9
  simp only [← c11607, ← c11608, ← c11609] at e_180_9
  have e_181_1 := arithEq_of_rows h (row := 181) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_1
  simp only [← c11598, k3, ← c11599, k3, ← c11600] at e_181_1
  have e_181_2 := arithEq_of_rows h (row := 181) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_2
  simp only [← c11601, ← c11602, k3, ← c11603] at e_181_2
  have e_181_3 := arithEq_of_rows h (row := 181) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_3
  simp only [← c11610, ← c11611, k3, ← c11612] at e_181_3
  refine ⟨?_, ?_⟩
  · have hc := e_180_8
    simp only [e_181_2] at hc
    linear_combination c11613.trans k2 - hc
  · have hc := e_181_3
    simp only [e_181_1, e_180_9, e_181_2] at hc
    linear_combination c11614.trans k2 - hc

theorem publicBatchWrapper89_f1770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 67) = bor (a (.wire 7 39)) (a (.virt 18581)) := by
  have c11615 := (publicBatchWrapper89_copies362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11616 := (publicBatchWrapper89_copies363 a h).1
  have c11617 := (publicBatchWrapper89_copies363 a h).2.1
  have c11618 := (publicBatchWrapper89_copies363 a h).2.2.1
  have c11619 := (publicBatchWrapper89_copies363 a h).2.2.2.1
  have c11620 := (publicBatchWrapper89_copies363 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_171_16 := arithEq_of_rows h (row := 171) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_16
  simp only [← c11618, ← c11619, k3, ← c11620] at e_171_16
  have e_179_4 := arithEq_of_rows h (row := 179) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_4
  simp only [← c11615, ← c11616, ← c11617] at e_179_4
  have hr := e_171_16
  simp only [e_179_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 67) = a (.virt 201) := by
  have c11621 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.1
  exact c11621

theorem publicBatchWrapper89_f1772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1005)) (a (.wire 169 63)) (a (.virt 18583)) (a (.virt 18584)) := by
  have c11622 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.1
  have c11623 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.1
  have c11624 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.1
  have c11625 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.1
  have c11626 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11627 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11628 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11629 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11630 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11631 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11632 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11633 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11634 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11635 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11636 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11637 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11638 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_180_10 := arithEq_of_rows h (row := 180) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_10
  simp only [← c11628, ← c11629, ← c11630] at e_180_10
  have e_180_11 := arithEq_of_rows h (row := 180) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_11
  simp only [← c11631, ← c11632, ← c11633] at e_180_11
  have e_181_4 := arithEq_of_rows h (row := 181) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_4
  simp only [← c11622, k3, ← c11623, k3, ← c11624] at e_181_4
  have e_181_5 := arithEq_of_rows h (row := 181) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_5
  simp only [← c11625, ← c11626, k3, ← c11627] at e_181_5
  have e_181_6 := arithEq_of_rows h (row := 181) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_6
  simp only [← c11634, ← c11635, k3, ← c11636] at e_181_6
  refine ⟨?_, ?_⟩
  · have hc := e_180_10
    simp only [e_181_5] at hc
    linear_combination c11637.trans k2 - hc
  · have hc := e_181_6
    simp only [e_181_4, e_180_11, e_181_5] at hc
    linear_combination c11638.trans k2 - hc

theorem publicBatchWrapper89_f1773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1006)) (a (.wire 169 71)) (a (.virt 18585)) (a (.virt 18586)) := by
  have c11639 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11640 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11641 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11642 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11643 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11644 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11645 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11646 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11647 := (publicBatchWrapper89_copies363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11648 := (publicBatchWrapper89_copies364 a h).1
  have c11649 := (publicBatchWrapper89_copies364 a h).2.1
  have c11650 := (publicBatchWrapper89_copies364 a h).2.2.1
  have c11651 := (publicBatchWrapper89_copies364 a h).2.2.2.1
  have c11652 := (publicBatchWrapper89_copies364 a h).2.2.2.2.1
  have c11653 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.1
  have c11654 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.1
  have c11655 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_180_12 := arithEq_of_rows h (row := 180) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_12
  simp only [← c11645, ← c11646, ← c11647] at e_180_12
  have e_180_13 := arithEq_of_rows h (row := 180) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_13
  simp only [← c11648, ← c11649, ← c11650] at e_180_13
  have e_181_7 := arithEq_of_rows h (row := 181) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_7
  simp only [← c11639, k3, ← c11640, k3, ← c11641] at e_181_7
  have e_181_8 := arithEq_of_rows h (row := 181) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_8
  simp only [← c11642, ← c11643, k3, ← c11644] at e_181_8
  have e_181_9 := arithEq_of_rows h (row := 181) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_9
  simp only [← c11651, ← c11652, k3, ← c11653] at e_181_9
  refine ⟨?_, ?_⟩
  · have hc := e_180_12
    simp only [e_181_8] at hc
    linear_combination c11654.trans k2 - hc
  · have hc := e_181_9
    simp only [e_181_7, e_180_13, e_181_8] at hc
    linear_combination c11655.trans k2 - hc

theorem publicBatchWrapper89_f1774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1007)) (a (.wire 169 79)) (a (.virt 18587)) (a (.virt 18588)) := by
  have c11656 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.1
  have c11657 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.1
  have c11658 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11659 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11660 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11661 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11662 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11663 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11664 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11665 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11666 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11667 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11668 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11669 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11670 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11671 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11672 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_180_14 := arithEq_of_rows h (row := 180) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_14
  simp only [← c11662, ← c11663, ← c11664] at e_180_14
  have e_180_15 := arithEq_of_rows h (row := 180) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_15
  simp only [← c11665, ← c11666, ← c11667] at e_180_15
  have e_181_10 := arithEq_of_rows h (row := 181) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_10
  simp only [← c11656, k3, ← c11657, k3, ← c11658] at e_181_10
  have e_181_11 := arithEq_of_rows h (row := 181) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_11
  simp only [← c11659, ← c11660, k3, ← c11661] at e_181_11
  have e_181_12 := arithEq_of_rows h (row := 181) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_12
  simp only [← c11668, ← c11669, k3, ← c11670] at e_181_12
  refine ⟨?_, ?_⟩
  · have hc := e_180_14
    simp only [e_181_11] at hc
    linear_combination c11671.trans k2 - hc
  · have hc := e_181_12
    simp only [e_181_10, e_180_15, e_181_11] at hc
    linear_combination c11672.trans k2 - hc

theorem publicBatchWrapper89_f1775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1008)) (a (.wire 170 7)) (a (.virt 18589)) (a (.virt 18590)) := by
  have c11673 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11674 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11675 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11676 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11677 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11678 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11679 := (publicBatchWrapper89_copies364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11680 := (publicBatchWrapper89_copies365 a h).1
  have c11681 := (publicBatchWrapper89_copies365 a h).2.1
  have c11682 := (publicBatchWrapper89_copies365 a h).2.2.1
  have c11683 := (publicBatchWrapper89_copies365 a h).2.2.2.1
  have c11684 := (publicBatchWrapper89_copies365 a h).2.2.2.2.1
  have c11685 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.1
  have c11686 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.1
  have c11687 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.1
  have c11688 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.1
  have c11689 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_180_16 := arithEq_of_rows h (row := 180) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_16
  simp only [← c11679, ← c11680, ← c11681] at e_180_16
  have e_180_17 := arithEq_of_rows h (row := 180) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_17
  simp only [← c11682, ← c11683, ← c11684] at e_180_17
  have e_181_13 := arithEq_of_rows h (row := 181) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_13
  simp only [← c11673, k3, ← c11674, k3, ← c11675] at e_181_13
  have e_181_14 := arithEq_of_rows h (row := 181) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_14
  simp only [← c11676, ← c11677, k3, ← c11678] at e_181_14
  have e_181_15 := arithEq_of_rows h (row := 181) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_15
  simp only [← c11685, ← c11686, k3, ← c11687] at e_181_15
  refine ⟨?_, ?_⟩
  · have hc := e_180_16
    simp only [e_181_14] at hc
    linear_combination c11688.trans k2 - hc
  · have hc := e_181_15
    simp only [e_181_13, e_180_17, e_181_14] at hc
    linear_combination c11689.trans k2 - hc

theorem publicBatchWrapper89_f1776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 180 75) = band (a (.virt 18583)) (a (.virt 18585)) := by
  have c11690 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11691 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11692 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_180_18 := arithEq_of_rows h (row := 180) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_18
  simp only [← c11690, ← c11691, ← c11692] at e_180_18
  have hr := e_180_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 180 79) = band (a (.virt 18587)) (a (.virt 18589)) := by
  have c11693 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11694 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11695 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_180_19 := arithEq_of_rows h (row := 180) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_180_19
  simp only [← c11693, ← c11694, ← c11695] at e_180_19
  have hr := e_180_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 182 3) = band (a (.wire 180 75)) (a (.wire 180 79)) := by
  have c11696 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11697 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11698 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_182_0 := arithEq_of_rows h (row := 182) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_0
  simp only [← c11696, ← c11697, ← c11698] at e_182_0
  have hr := e_182_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 71) = bor (a (.wire 7 39)) (a (.wire 182 3)) := by
  have c11699 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11700 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11701 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11702 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11703 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11704 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_171_17 := arithEq_of_rows h (row := 171) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_17
  simp only [← c11702, ← c11703, k3, ← c11704] at e_171_17
  have e_179_5 := arithEq_of_rows h (row := 179) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_5
  simp only [← c11699, ← c11700, ← c11701] at e_179_5
  have hr := e_171_17
  simp only [e_179_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 71) = a (.virt 201) := by
  have c11705 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11705

theorem publicBatchWrapper89_f1781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1203)) (a (.wire 170 23)) (a (.virt 18591)) (a (.virt 18592)) := by
  have c11706 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11707 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11708 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11709 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11710 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11711 := (publicBatchWrapper89_copies365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11712 := (publicBatchWrapper89_copies366 a h).1
  have c11713 := (publicBatchWrapper89_copies366 a h).2.1
  have c11714 := (publicBatchWrapper89_copies366 a h).2.2.1
  have c11715 := (publicBatchWrapper89_copies366 a h).2.2.2.1
  have c11716 := (publicBatchWrapper89_copies366 a h).2.2.2.2.1
  have c11717 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.1
  have c11718 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.1
  have c11719 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.1
  have c11720 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.1
  have c11721 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.1
  have c11722 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_181_16 := arithEq_of_rows h (row := 181) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_16
  simp only [← c11706, k3, ← c11707, k3, ← c11708] at e_181_16
  have e_181_17 := arithEq_of_rows h (row := 181) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_17
  simp only [← c11709, ← c11710, k3, ← c11711] at e_181_17
  have e_181_18 := arithEq_of_rows h (row := 181) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_18
  simp only [← c11718, ← c11719, k3, ← c11720] at e_181_18
  have e_182_1 := arithEq_of_rows h (row := 182) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_1
  simp only [← c11712, ← c11713, ← c11714] at e_182_1
  have e_182_2 := arithEq_of_rows h (row := 182) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_2
  simp only [← c11715, ← c11716, ← c11717] at e_182_2
  refine ⟨?_, ?_⟩
  · have hc := e_182_1
    simp only [e_181_17] at hc
    linear_combination c11721.trans k2 - hc
  · have hc := e_181_18
    simp only [e_181_16, e_182_2, e_181_17] at hc
    linear_combination c11722.trans k2 - hc

theorem publicBatchWrapper89_f1782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 75) = bor (a (.wire 8 3)) (a (.virt 18591)) := by
  have c11723 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11724 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11725 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11726 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11727 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11728 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_171_18 := arithEq_of_rows h (row := 171) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_18
  simp only [← c11726, ← c11727, k3, ← c11728] at e_171_18
  have e_179_6 := arithEq_of_rows h (row := 179) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_6
  simp only [← c11723, ← c11724, ← c11725] at e_179_6
  have hr := e_171_18
  simp only [e_179_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 75) = a (.virt 201) := by
  have c11729 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11729

theorem publicBatchWrapper89_f1784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1204)) (a (.wire 170 31)) (a (.virt 18593)) (a (.virt 18594)) := by
  have c11730 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11731 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11732 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11733 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11734 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11735 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11736 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11737 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11738 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11739 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11740 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11741 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11742 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11743 := (publicBatchWrapper89_copies366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11744 := (publicBatchWrapper89_copies367 a h).1
  have c11745 := (publicBatchWrapper89_copies367 a h).2.1
  have c11746 := (publicBatchWrapper89_copies367 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_181_19 := arithEq_of_rows h (row := 181) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_181_19
  simp only [← c11730, k3, ← c11731, k3, ← c11732] at e_181_19
  have e_182_3 := arithEq_of_rows h (row := 182) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_3
  simp only [← c11736, ← c11737, ← c11738] at e_182_3
  have e_182_4 := arithEq_of_rows h (row := 182) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_4
  simp only [← c11739, ← c11740, ← c11741] at e_182_4
  have e_183_0 := arithEq_of_rows h (row := 183) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_0
  simp only [← c11733, ← c11734, k3, ← c11735] at e_183_0
  have e_183_1 := arithEq_of_rows h (row := 183) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_1
  simp only [← c11742, ← c11743, k3, ← c11744] at e_183_1
  refine ⟨?_, ?_⟩
  · have hc := e_182_3
    simp only [e_183_0] at hc
    linear_combination c11745.trans k2 - hc
  · have hc := e_183_1
    simp only [e_181_19, e_182_4, e_183_0] at hc
    linear_combination c11746.trans k2 - hc

theorem publicBatchWrapper89_f1785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 79) = bor (a (.wire 8 3)) (a (.virt 18593)) := by
  have c11747 := (publicBatchWrapper89_copies367 a h).2.2.2.1
  have c11748 := (publicBatchWrapper89_copies367 a h).2.2.2.2.1
  have c11749 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.1
  have c11750 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.1
  have c11751 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.1
  have c11752 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_171_19 := arithEq_of_rows h (row := 171) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_171_19
  simp only [← c11750, ← c11751, k3, ← c11752] at e_171_19
  have e_179_7 := arithEq_of_rows h (row := 179) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_7
  simp only [← c11747, ← c11748, ← c11749] at e_179_7
  have hr := e_171_19
  simp only [e_179_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 171 79) = a (.virt 201) := by
  have c11753 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.1
  exact c11753

theorem publicBatchWrapper89_f1787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1205)) (a (.wire 169 63)) (a (.virt 18595)) (a (.virt 18596)) := by
  have c11754 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11755 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11756 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11757 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11758 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11759 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11760 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11761 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11762 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11763 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11764 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11765 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11766 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11767 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11768 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11769 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11770 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_182_5 := arithEq_of_rows h (row := 182) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_5
  simp only [← c11760, ← c11761, ← c11762] at e_182_5
  have e_182_6 := arithEq_of_rows h (row := 182) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_6
  simp only [← c11763, ← c11764, ← c11765] at e_182_6
  have e_183_2 := arithEq_of_rows h (row := 183) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_2
  simp only [← c11754, k3, ← c11755, k3, ← c11756] at e_183_2
  have e_183_3 := arithEq_of_rows h (row := 183) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_3
  simp only [← c11757, ← c11758, k3, ← c11759] at e_183_3
  have e_183_4 := arithEq_of_rows h (row := 183) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_4
  simp only [← c11766, ← c11767, k3, ← c11768] at e_183_4
  refine ⟨?_, ?_⟩
  · have hc := e_182_5
    simp only [e_183_3] at hc
    linear_combination c11769.trans k2 - hc
  · have hc := e_183_4
    simp only [e_183_2, e_182_6, e_183_3] at hc
    linear_combination c11770.trans k2 - hc

theorem publicBatchWrapper89_f1788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1206)) (a (.wire 169 71)) (a (.virt 18597)) (a (.virt 18598)) := by
  have c11771 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11772 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11773 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11774 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11775 := (publicBatchWrapper89_copies367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11776 := (publicBatchWrapper89_copies368 a h).1
  have c11777 := (publicBatchWrapper89_copies368 a h).2.1
  have c11778 := (publicBatchWrapper89_copies368 a h).2.2.1
  have c11779 := (publicBatchWrapper89_copies368 a h).2.2.2.1
  have c11780 := (publicBatchWrapper89_copies368 a h).2.2.2.2.1
  have c11781 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.1
  have c11782 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.1
  have c11783 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.1
  have c11784 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.1
  have c11785 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.1
  have c11786 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11787 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_182_7 := arithEq_of_rows h (row := 182) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_7
  simp only [← c11777, ← c11778, ← c11779] at e_182_7
  have e_182_8 := arithEq_of_rows h (row := 182) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_8
  simp only [← c11780, ← c11781, ← c11782] at e_182_8
  have e_183_5 := arithEq_of_rows h (row := 183) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_5
  simp only [← c11771, k3, ← c11772, k3, ← c11773] at e_183_5
  have e_183_6 := arithEq_of_rows h (row := 183) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_6
  simp only [← c11774, ← c11775, k3, ← c11776] at e_183_6
  have e_183_7 := arithEq_of_rows h (row := 183) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_7
  simp only [← c11783, ← c11784, k3, ← c11785] at e_183_7
  refine ⟨?_, ?_⟩
  · have hc := e_182_7
    simp only [e_183_6] at hc
    linear_combination c11786.trans k2 - hc
  · have hc := e_183_7
    simp only [e_183_5, e_182_8, e_183_6] at hc
    linear_combination c11787.trans k2 - hc

theorem publicBatchWrapper89_f1789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1207)) (a (.wire 169 79)) (a (.virt 18599)) (a (.virt 18600)) := by
  have c11788 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11789 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11790 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11791 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11792 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11793 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11794 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11795 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11796 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11797 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11798 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11799 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11800 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11801 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11802 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11803 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11804 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_182_9 := arithEq_of_rows h (row := 182) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_9
  simp only [← c11794, ← c11795, ← c11796] at e_182_9
  have e_182_10 := arithEq_of_rows h (row := 182) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_10
  simp only [← c11797, ← c11798, ← c11799] at e_182_10
  have e_183_8 := arithEq_of_rows h (row := 183) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_8
  simp only [← c11788, k3, ← c11789, k3, ← c11790] at e_183_8
  have e_183_9 := arithEq_of_rows h (row := 183) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_9
  simp only [← c11791, ← c11792, k3, ← c11793] at e_183_9
  have e_183_10 := arithEq_of_rows h (row := 183) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_10
  simp only [← c11800, ← c11801, k3, ← c11802] at e_183_10
  refine ⟨?_, ?_⟩
  · have hc := e_182_9
    simp only [e_183_9] at hc
    linear_combination c11803.trans k2 - hc
  · have hc := e_183_10
    simp only [e_183_8, e_182_10, e_183_9] at hc
    linear_combination c11804.trans k2 - hc

theorem publicBatchWrapper89_f1790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1208)) (a (.wire 170 7)) (a (.virt 18601)) (a (.virt 18602)) := by
  have c11805 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11806 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11807 := (publicBatchWrapper89_copies368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11808 := (publicBatchWrapper89_copies369 a h).1
  have c11809 := (publicBatchWrapper89_copies369 a h).2.1
  have c11810 := (publicBatchWrapper89_copies369 a h).2.2.1
  have c11811 := (publicBatchWrapper89_copies369 a h).2.2.2.1
  have c11812 := (publicBatchWrapper89_copies369 a h).2.2.2.2.1
  have c11813 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.1
  have c11814 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.1
  have c11815 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.1
  have c11816 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.1
  have c11817 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.1
  have c11818 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11819 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11820 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11821 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_182_11 := arithEq_of_rows h (row := 182) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_11
  simp only [← c11811, ← c11812, ← c11813] at e_182_11
  have e_182_12 := arithEq_of_rows h (row := 182) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_12
  simp only [← c11814, ← c11815, ← c11816] at e_182_12
  have e_183_11 := arithEq_of_rows h (row := 183) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_11
  simp only [← c11805, k3, ← c11806, k3, ← c11807] at e_183_11
  have e_183_12 := arithEq_of_rows h (row := 183) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_12
  simp only [← c11808, ← c11809, k3, ← c11810] at e_183_12
  have e_183_13 := arithEq_of_rows h (row := 183) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_13
  simp only [← c11817, ← c11818, k3, ← c11819] at e_183_13
  refine ⟨?_, ?_⟩
  · have hc := e_182_11
    simp only [e_183_12] at hc
    linear_combination c11820.trans k2 - hc
  · have hc := e_183_13
    simp only [e_183_11, e_182_12, e_183_12] at hc
    linear_combination c11821.trans k2 - hc

theorem publicBatchWrapper89_f1791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 182 55) = band (a (.virt 18595)) (a (.virt 18597)) := by
  have c11822 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11823 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11824 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_182_13 := arithEq_of_rows h (row := 182) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_13
  simp only [← c11822, ← c11823, ← c11824] at e_182_13
  have hr := e_182_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 182 59) = band (a (.virt 18599)) (a (.virt 18601)) := by
  have c11825 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11826 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11827 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_182_14 := arithEq_of_rows h (row := 182) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_14
  simp only [← c11825, ← c11826, ← c11827] at e_182_14
  have hr := e_182_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 182 63) = band (a (.wire 182 55)) (a (.wire 182 59)) := by
  have c11828 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11829 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11830 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_182_15 := arithEq_of_rows h (row := 182) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_15
  simp only [← c11828, ← c11829, ← c11830] at e_182_15
  have hr := e_182_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 3) = bor (a (.wire 8 3)) (a (.wire 182 63)) := by
  have c11831 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11832 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11833 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11834 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11835 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11836 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_8 := arithEq_of_rows h (row := 179) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_8
  simp only [← c11831, ← c11832, ← c11833] at e_179_8
  have e_184_0 := arithEq_of_rows h (row := 184) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_0
  simp only [← c11834, ← c11835, k3, ← c11836] at e_184_0
  have hr := e_184_0
  simp only [e_179_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 3) = a (.virt 201) := by
  have c11837 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11837

theorem publicBatchWrapper89_f1796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1403)) (a (.wire 170 23)) (a (.virt 18603)) (a (.virt 18604)) := by
  have c11838 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11839 := (publicBatchWrapper89_copies369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11840 := (publicBatchWrapper89_copies370 a h).1
  have c11841 := (publicBatchWrapper89_copies370 a h).2.1
  have c11842 := (publicBatchWrapper89_copies370 a h).2.2.1
  have c11843 := (publicBatchWrapper89_copies370 a h).2.2.2.1
  have c11844 := (publicBatchWrapper89_copies370 a h).2.2.2.2.1
  have c11845 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.1
  have c11846 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.1
  have c11847 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.1
  have c11848 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.1
  have c11849 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.1
  have c11850 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11851 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11852 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11853 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11854 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_182_16 := arithEq_of_rows h (row := 182) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_16
  simp only [← c11844, ← c11845, ← c11846] at e_182_16
  have e_182_17 := arithEq_of_rows h (row := 182) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_17
  simp only [← c11847, ← c11848, ← c11849] at e_182_17
  have e_183_14 := arithEq_of_rows h (row := 183) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_14
  simp only [← c11838, k3, ← c11839, k3, ← c11840] at e_183_14
  have e_183_15 := arithEq_of_rows h (row := 183) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_15
  simp only [← c11841, ← c11842, k3, ← c11843] at e_183_15
  have e_183_16 := arithEq_of_rows h (row := 183) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_16
  simp only [← c11850, ← c11851, k3, ← c11852] at e_183_16
  refine ⟨?_, ?_⟩
  · have hc := e_182_16
    simp only [e_183_15] at hc
    linear_combination c11853.trans k2 - hc
  · have hc := e_183_16
    simp only [e_183_14, e_182_17, e_183_15] at hc
    linear_combination c11854.trans k2 - hc

theorem publicBatchWrapper89_f1797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 7) = bor (a (.wire 8 47)) (a (.virt 18603)) := by
  have c11855 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11856 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11857 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11858 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11859 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11860 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_9 := arithEq_of_rows h (row := 179) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_9
  simp only [← c11855, ← c11856, ← c11857] at e_179_9
  have e_184_1 := arithEq_of_rows h (row := 184) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_1
  simp only [← c11858, ← c11859, k3, ← c11860] at e_184_1
  have hr := e_184_1
  simp only [e_179_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 7) = a (.virt 201) := by
  have c11861 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11861

theorem publicBatchWrapper89_f1799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1404)) (a (.wire 170 31)) (a (.virt 18605)) (a (.virt 18606)) := by
  have c11862 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11863 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11864 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11865 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11866 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11867 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11868 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11869 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11870 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11871 := (publicBatchWrapper89_copies370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11872 := (publicBatchWrapper89_copies371 a h).1
  have c11873 := (publicBatchWrapper89_copies371 a h).2.1
  have c11874 := (publicBatchWrapper89_copies371 a h).2.2.1
  have c11875 := (publicBatchWrapper89_copies371 a h).2.2.2.1
  have c11876 := (publicBatchWrapper89_copies371 a h).2.2.2.2.1
  have c11877 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.1
  have c11878 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_182_18 := arithEq_of_rows h (row := 182) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_18
  simp only [← c11868, ← c11869, ← c11870] at e_182_18
  have e_182_19 := arithEq_of_rows h (row := 182) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_182_19
  simp only [← c11871, ← c11872, ← c11873] at e_182_19
  have e_183_17 := arithEq_of_rows h (row := 183) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_17
  simp only [← c11862, k3, ← c11863, k3, ← c11864] at e_183_17
  have e_183_18 := arithEq_of_rows h (row := 183) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_18
  simp only [← c11865, ← c11866, k3, ← c11867] at e_183_18
  have e_183_19 := arithEq_of_rows h (row := 183) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_183_19
  simp only [← c11874, ← c11875, k3, ← c11876] at e_183_19
  refine ⟨?_, ?_⟩
  · have hc := e_182_18
    simp only [e_183_18] at hc
    linear_combination c11877.trans k2 - hc
  · have hc := e_183_19
    simp only [e_183_17, e_182_19, e_183_18] at hc
    linear_combination c11878.trans k2 - hc

theorem publicBatchWrapper89_f1800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 11) = bor (a (.wire 8 47)) (a (.virt 18605)) := by
  have c11879 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.1
  have c11880 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.1
  have c11881 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.1
  have c11882 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11883 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11884 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_10 := arithEq_of_rows h (row := 179) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_10
  simp only [← c11879, ← c11880, ← c11881] at e_179_10
  have e_184_2 := arithEq_of_rows h (row := 184) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_2
  simp only [← c11882, ← c11883, k3, ← c11884] at e_184_2
  have hr := e_184_2
  simp only [e_179_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 11) = a (.virt 201) := by
  have c11885 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11885

theorem publicBatchWrapper89_f1802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1405)) (a (.wire 169 63)) (a (.virt 18607)) (a (.virt 18608)) := by
  have c11886 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11887 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11888 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11889 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11890 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11891 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11892 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11893 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11894 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11895 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11896 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11897 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11898 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11899 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11900 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11901 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11902 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_185_0 := arithEq_of_rows h (row := 185) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_0
  simp only [← c11886, k3, ← c11887, k3, ← c11888] at e_185_0
  have e_185_1 := arithEq_of_rows h (row := 185) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_1
  simp only [← c11889, ← c11890, k3, ← c11891] at e_185_1
  have e_185_2 := arithEq_of_rows h (row := 185) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_2
  simp only [← c11898, ← c11899, k3, ← c11900] at e_185_2
  have e_186_0 := arithEq_of_rows h (row := 186) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_0
  simp only [← c11892, ← c11893, ← c11894] at e_186_0
  have e_186_1 := arithEq_of_rows h (row := 186) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_1
  simp only [← c11895, ← c11896, ← c11897] at e_186_1
  refine ⟨?_, ?_⟩
  · have hc := e_186_0
    simp only [e_185_1] at hc
    linear_combination c11901.trans k2 - hc
  · have hc := e_185_2
    simp only [e_185_0, e_186_1, e_185_1] at hc
    linear_combination c11902.trans k2 - hc

theorem publicBatchWrapper89_f1803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1406)) (a (.wire 169 71)) (a (.virt 18609)) (a (.virt 18610)) := by
  have c11903 := (publicBatchWrapper89_copies371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11904 := (publicBatchWrapper89_copies372 a h).1
  have c11905 := (publicBatchWrapper89_copies372 a h).2.1
  have c11906 := (publicBatchWrapper89_copies372 a h).2.2.1
  have c11907 := (publicBatchWrapper89_copies372 a h).2.2.2.1
  have c11908 := (publicBatchWrapper89_copies372 a h).2.2.2.2.1
  have c11909 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.1
  have c11910 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.1
  have c11911 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.1
  have c11912 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.1
  have c11913 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.1
  have c11914 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11915 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11916 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11917 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11918 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11919 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_185_3 := arithEq_of_rows h (row := 185) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_3
  simp only [← c11903, k3, ← c11904, k3, ← c11905] at e_185_3
  have e_185_4 := arithEq_of_rows h (row := 185) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_4
  simp only [← c11906, ← c11907, k3, ← c11908] at e_185_4
  have e_185_5 := arithEq_of_rows h (row := 185) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_5
  simp only [← c11915, ← c11916, k3, ← c11917] at e_185_5
  have e_186_2 := arithEq_of_rows h (row := 186) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_2
  simp only [← c11909, ← c11910, ← c11911] at e_186_2
  have e_186_3 := arithEq_of_rows h (row := 186) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_3
  simp only [← c11912, ← c11913, ← c11914] at e_186_3
  refine ⟨?_, ?_⟩
  · have hc := e_186_2
    simp only [e_185_4] at hc
    linear_combination c11918.trans k2 - hc
  · have hc := e_185_5
    simp only [e_185_3, e_186_3, e_185_4] at hc
    linear_combination c11919.trans k2 - hc

theorem publicBatchWrapper89_f1804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1407)) (a (.wire 169 79)) (a (.virt 18611)) (a (.virt 18612)) := by
  have c11920 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11921 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11922 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11923 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11924 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11925 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11926 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11927 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11928 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11929 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11930 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11931 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11932 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11933 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11934 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11935 := (publicBatchWrapper89_copies372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11936 := (publicBatchWrapper89_copies373 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_185_6 := arithEq_of_rows h (row := 185) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_6
  simp only [← c11920, k3, ← c11921, k3, ← c11922] at e_185_6
  have e_185_7 := arithEq_of_rows h (row := 185) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_7
  simp only [← c11923, ← c11924, k3, ← c11925] at e_185_7
  have e_185_8 := arithEq_of_rows h (row := 185) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_8
  simp only [← c11932, ← c11933, k3, ← c11934] at e_185_8
  have e_186_4 := arithEq_of_rows h (row := 186) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_4
  simp only [← c11926, ← c11927, ← c11928] at e_186_4
  have e_186_5 := arithEq_of_rows h (row := 186) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_5
  simp only [← c11929, ← c11930, ← c11931] at e_186_5
  refine ⟨?_, ?_⟩
  · have hc := e_186_4
    simp only [e_185_7] at hc
    linear_combination c11935.trans k2 - hc
  · have hc := e_185_8
    simp only [e_185_6, e_186_5, e_185_7] at hc
    linear_combination c11936.trans k2 - hc

theorem publicBatchWrapper89_f1805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1408)) (a (.wire 170 7)) (a (.virt 18613)) (a (.virt 18614)) := by
  have c11937 := (publicBatchWrapper89_copies373 a h).2.1
  have c11938 := (publicBatchWrapper89_copies373 a h).2.2.1
  have c11939 := (publicBatchWrapper89_copies373 a h).2.2.2.1
  have c11940 := (publicBatchWrapper89_copies373 a h).2.2.2.2.1
  have c11941 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.1
  have c11942 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.1
  have c11943 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.1
  have c11944 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.1
  have c11945 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.1
  have c11946 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11947 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11948 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11949 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11950 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11951 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11952 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11953 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_185_9 := arithEq_of_rows h (row := 185) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_9
  simp only [← c11937, k3, ← c11938, k3, ← c11939] at e_185_9
  have e_185_10 := arithEq_of_rows h (row := 185) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_10
  simp only [← c11940, ← c11941, k3, ← c11942] at e_185_10
  have e_185_11 := arithEq_of_rows h (row := 185) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_11
  simp only [← c11949, ← c11950, k3, ← c11951] at e_185_11
  have e_186_6 := arithEq_of_rows h (row := 186) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_6
  simp only [← c11943, ← c11944, ← c11945] at e_186_6
  have e_186_7 := arithEq_of_rows h (row := 186) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_7
  simp only [← c11946, ← c11947, ← c11948] at e_186_7
  refine ⟨?_, ?_⟩
  · have hc := e_186_6
    simp only [e_185_10] at hc
    linear_combination c11952.trans k2 - hc
  · have hc := e_185_11
    simp only [e_185_9, e_186_7, e_185_10] at hc
    linear_combination c11953.trans k2 - hc

theorem publicBatchWrapper89_f1806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 186 35) = band (a (.virt 18607)) (a (.virt 18609)) := by
  have c11954 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11955 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11956 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_186_8 := arithEq_of_rows h (row := 186) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_8
  simp only [← c11954, ← c11955, ← c11956] at e_186_8
  have hr := e_186_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 186 39) = band (a (.virt 18611)) (a (.virt 18613)) := by
  have c11957 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11958 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11959 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_186_9 := arithEq_of_rows h (row := 186) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_9
  simp only [← c11957, ← c11958, ← c11959] at e_186_9
  have hr := e_186_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 186 43) = band (a (.wire 186 35)) (a (.wire 186 39)) := by
  have c11960 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11961 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11962 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_186_10 := arithEq_of_rows h (row := 186) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_10
  simp only [← c11960, ← c11961, ← c11962] at e_186_10
  have hr := e_186_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 15) = bor (a (.wire 8 47)) (a (.wire 186 43)) := by
  have c11963 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11964 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11965 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11966 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11967 := (publicBatchWrapper89_copies373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c11968 := (publicBatchWrapper89_copies374 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_11 := arithEq_of_rows h (row := 179) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_11
  simp only [← c11963, ← c11964, ← c11965] at e_179_11
  have e_184_3 := arithEq_of_rows h (row := 184) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_3
  simp only [← c11966, ← c11967, k3, ← c11968] at e_184_3
  have hr := e_184_3
  simp only [e_179_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 15) = a (.virt 201) := by
  have c11969 := (publicBatchWrapper89_copies374 a h).2.1
  exact c11969

theorem publicBatchWrapper89_f1811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1603)) (a (.wire 170 23)) (a (.virt 18615)) (a (.virt 18616)) := by
  have c11970 := (publicBatchWrapper89_copies374 a h).2.2.1
  have c11971 := (publicBatchWrapper89_copies374 a h).2.2.2.1
  have c11972 := (publicBatchWrapper89_copies374 a h).2.2.2.2.1
  have c11973 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.1
  have c11974 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.1
  have c11975 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.1
  have c11976 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.1
  have c11977 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.1
  have c11978 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11979 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c11980 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11981 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11982 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11983 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11984 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11985 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11986 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_185_12 := arithEq_of_rows h (row := 185) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_12
  simp only [← c11970, k3, ← c11971, k3, ← c11972] at e_185_12
  have e_185_13 := arithEq_of_rows h (row := 185) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_13
  simp only [← c11973, ← c11974, k3, ← c11975] at e_185_13
  have e_185_14 := arithEq_of_rows h (row := 185) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_14
  simp only [← c11982, ← c11983, k3, ← c11984] at e_185_14
  have e_186_11 := arithEq_of_rows h (row := 186) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_11
  simp only [← c11976, ← c11977, ← c11978] at e_186_11
  have e_186_12 := arithEq_of_rows h (row := 186) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_12
  simp only [← c11979, ← c11980, ← c11981] at e_186_12
  refine ⟨?_, ?_⟩
  · have hc := e_186_11
    simp only [e_185_13] at hc
    linear_combination c11985.trans k2 - hc
  · have hc := e_185_14
    simp only [e_185_12, e_186_12, e_185_13] at hc
    linear_combination c11986.trans k2 - hc

theorem publicBatchWrapper89_f1812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 19) = bor (a (.wire 10 11)) (a (.virt 18615)) := by
  have c11987 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11988 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11989 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11990 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11991 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11992 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_12 := arithEq_of_rows h (row := 179) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_12
  simp only [← c11987, ← c11988, ← c11989] at e_179_12
  have e_184_4 := arithEq_of_rows h (row := 184) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_4
  simp only [← c11990, ← c11991, k3, ← c11992] at e_184_4
  have hr := e_184_4
  simp only [e_179_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 19) = a (.virt 201) := by
  have c11993 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c11993

theorem publicBatchWrapper89_f1814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1604)) (a (.wire 170 31)) (a (.virt 18617)) (a (.virt 18618)) := by
  have c11994 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11995 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11996 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11997 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11998 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c11999 := (publicBatchWrapper89_copies374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12000 := (publicBatchWrapper89_copies375 a h).1
  have c12001 := (publicBatchWrapper89_copies375 a h).2.1
  have c12002 := (publicBatchWrapper89_copies375 a h).2.2.1
  have c12003 := (publicBatchWrapper89_copies375 a h).2.2.2.1
  have c12004 := (publicBatchWrapper89_copies375 a h).2.2.2.2.1
  have c12005 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.1
  have c12006 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.1
  have c12007 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.1
  have c12008 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.1
  have c12009 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.1
  have c12010 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_185_15 := arithEq_of_rows h (row := 185) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_15
  simp only [← c11994, k3, ← c11995, k3, ← c11996] at e_185_15
  have e_185_16 := arithEq_of_rows h (row := 185) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_16
  simp only [← c11997, ← c11998, k3, ← c11999] at e_185_16
  have e_185_17 := arithEq_of_rows h (row := 185) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_17
  simp only [← c12006, ← c12007, k3, ← c12008] at e_185_17
  have e_186_13 := arithEq_of_rows h (row := 186) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_13
  simp only [← c12000, ← c12001, ← c12002] at e_186_13
  have e_186_14 := arithEq_of_rows h (row := 186) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_14
  simp only [← c12003, ← c12004, ← c12005] at e_186_14
  refine ⟨?_, ?_⟩
  · have hc := e_186_13
    simp only [e_185_16] at hc
    linear_combination c12009.trans k2 - hc
  · have hc := e_185_17
    simp only [e_185_15, e_186_14, e_185_16] at hc
    linear_combination c12010.trans k2 - hc

theorem publicBatchWrapper89_f1815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 23) = bor (a (.wire 10 11)) (a (.virt 18617)) := by
  have c12011 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12012 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12013 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12014 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12015 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12016 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_13 := arithEq_of_rows h (row := 179) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_13
  simp only [← c12011, ← c12012, ← c12013] at e_179_13
  have e_184_5 := arithEq_of_rows h (row := 184) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_5
  simp only [← c12014, ← c12015, k3, ← c12016] at e_184_5
  have hr := e_184_5
  simp only [e_179_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 23) = a (.virt 201) := by
  have c12017 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12017

theorem publicBatchWrapper89_f1817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1605)) (a (.wire 169 63)) (a (.virt 18619)) (a (.virt 18620)) := by
  have c12018 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12019 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12020 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12021 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12022 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12023 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12024 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12025 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12026 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12027 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12028 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12029 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12030 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12031 := (publicBatchWrapper89_copies375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12032 := (publicBatchWrapper89_copies376 a h).1
  have c12033 := (publicBatchWrapper89_copies376 a h).2.1
  have c12034 := (publicBatchWrapper89_copies376 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_185_18 := arithEq_of_rows h (row := 185) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_18
  simp only [← c12018, k3, ← c12019, k3, ← c12020] at e_185_18
  have e_185_19 := arithEq_of_rows h (row := 185) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_185_19
  simp only [← c12021, ← c12022, k3, ← c12023] at e_185_19
  have e_186_15 := arithEq_of_rows h (row := 186) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_15
  simp only [← c12024, ← c12025, ← c12026] at e_186_15
  have e_186_16 := arithEq_of_rows h (row := 186) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_16
  simp only [← c12027, ← c12028, ← c12029] at e_186_16
  have e_187_0 := arithEq_of_rows h (row := 187) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_0
  simp only [← c12030, ← c12031, k3, ← c12032] at e_187_0
  refine ⟨?_, ?_⟩
  · have hc := e_186_15
    simp only [e_185_19] at hc
    linear_combination c12033.trans k2 - hc
  · have hc := e_187_0
    simp only [e_185_18, e_186_16, e_185_19] at hc
    linear_combination c12034.trans k2 - hc

theorem publicBatchWrapper89_f1818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1606)) (a (.wire 169 71)) (a (.virt 18621)) (a (.virt 18622)) := by
  have c12035 := (publicBatchWrapper89_copies376 a h).2.2.2.1
  have c12036 := (publicBatchWrapper89_copies376 a h).2.2.2.2.1
  have c12037 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.1
  have c12038 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.1
  have c12039 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.1
  have c12040 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.1
  have c12041 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.1
  have c12042 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12043 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12044 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12045 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12046 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12047 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12048 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12049 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12050 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12051 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_186_17 := arithEq_of_rows h (row := 186) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_17
  simp only [← c12041, ← c12042, ← c12043] at e_186_17
  have e_186_18 := arithEq_of_rows h (row := 186) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_18
  simp only [← c12044, ← c12045, ← c12046] at e_186_18
  have e_187_1 := arithEq_of_rows h (row := 187) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_1
  simp only [← c12035, k3, ← c12036, k3, ← c12037] at e_187_1
  have e_187_2 := arithEq_of_rows h (row := 187) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_2
  simp only [← c12038, ← c12039, k3, ← c12040] at e_187_2
  have e_187_3 := arithEq_of_rows h (row := 187) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_3
  simp only [← c12047, ← c12048, k3, ← c12049] at e_187_3
  refine ⟨?_, ?_⟩
  · have hc := e_186_17
    simp only [e_187_2] at hc
    linear_combination c12050.trans k2 - hc
  · have hc := e_187_3
    simp only [e_187_1, e_186_18, e_187_2] at hc
    linear_combination c12051.trans k2 - hc

theorem publicBatchWrapper89_f1819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1607)) (a (.wire 169 79)) (a (.virt 18623)) (a (.virt 18624)) := by
  have c12052 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12053 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12054 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12055 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12056 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12057 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12058 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12059 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12060 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12061 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12062 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12063 := (publicBatchWrapper89_copies376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12064 := (publicBatchWrapper89_copies377 a h).1
  have c12065 := (publicBatchWrapper89_copies377 a h).2.1
  have c12066 := (publicBatchWrapper89_copies377 a h).2.2.1
  have c12067 := (publicBatchWrapper89_copies377 a h).2.2.2.1
  have c12068 := (publicBatchWrapper89_copies377 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_186_19 := arithEq_of_rows h (row := 186) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_186_19
  simp only [← c12058, ← c12059, ← c12060] at e_186_19
  have e_187_4 := arithEq_of_rows h (row := 187) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_4
  simp only [← c12052, k3, ← c12053, k3, ← c12054] at e_187_4
  have e_187_5 := arithEq_of_rows h (row := 187) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_5
  simp only [← c12055, ← c12056, k3, ← c12057] at e_187_5
  have e_187_6 := arithEq_of_rows h (row := 187) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_6
  simp only [← c12064, ← c12065, k3, ← c12066] at e_187_6
  have e_188_0 := arithEq_of_rows h (row := 188) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_0
  simp only [← c12061, ← c12062, ← c12063] at e_188_0
  refine ⟨?_, ?_⟩
  · have hc := e_186_19
    simp only [e_187_5] at hc
    linear_combination c12067.trans k2 - hc
  · have hc := e_187_6
    simp only [e_187_4, e_188_0, e_187_5] at hc
    linear_combination c12068.trans k2 - hc

theorem publicBatchWrapper89_f1820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1608)) (a (.wire 170 7)) (a (.virt 18625)) (a (.virt 18626)) := by
  have c12069 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.1
  have c12070 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.1
  have c12071 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.1
  have c12072 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.1
  have c12073 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.1
  have c12074 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12075 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12076 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12077 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12078 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12079 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12080 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12081 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12082 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12083 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12084 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12085 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_187_7 := arithEq_of_rows h (row := 187) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_7
  simp only [← c12069, k3, ← c12070, k3, ← c12071] at e_187_7
  have e_187_8 := arithEq_of_rows h (row := 187) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_8
  simp only [← c12072, ← c12073, k3, ← c12074] at e_187_8
  have e_187_9 := arithEq_of_rows h (row := 187) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_9
  simp only [← c12081, ← c12082, k3, ← c12083] at e_187_9
  have e_188_1 := arithEq_of_rows h (row := 188) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_1
  simp only [← c12075, ← c12076, ← c12077] at e_188_1
  have e_188_2 := arithEq_of_rows h (row := 188) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_2
  simp only [← c12078, ← c12079, ← c12080] at e_188_2
  refine ⟨?_, ?_⟩
  · have hc := e_188_1
    simp only [e_187_8] at hc
    linear_combination c12084.trans k2 - hc
  · have hc := e_187_9
    simp only [e_187_7, e_188_2, e_187_8] at hc
    linear_combination c12085.trans k2 - hc

theorem publicBatchWrapper89_f1821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 188 15) = band (a (.virt 18619)) (a (.virt 18621)) := by
  have c12086 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12087 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12088 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_3 := arithEq_of_rows h (row := 188) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_3
  simp only [← c12086, ← c12087, ← c12088] at e_188_3
  have hr := e_188_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 188 19) = band (a (.virt 18623)) (a (.virt 18625)) := by
  have c12089 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12090 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12091 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_4 := arithEq_of_rows h (row := 188) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_4
  simp only [← c12089, ← c12090, ← c12091] at e_188_4
  have hr := e_188_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 188 23) = band (a (.wire 188 15)) (a (.wire 188 19)) := by
  have c12092 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12093 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12094 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_5 := arithEq_of_rows h (row := 188) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_5
  simp only [← c12092, ← c12093, ← c12094] at e_188_5
  have hr := e_188_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 27) = bor (a (.wire 10 11)) (a (.wire 188 23)) := by
  have c12095 := (publicBatchWrapper89_copies377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12096 := (publicBatchWrapper89_copies378 a h).1
  have c12097 := (publicBatchWrapper89_copies378 a h).2.1
  have c12098 := (publicBatchWrapper89_copies378 a h).2.2.1
  have c12099 := (publicBatchWrapper89_copies378 a h).2.2.2.1
  have c12100 := (publicBatchWrapper89_copies378 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_14 := arithEq_of_rows h (row := 179) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_14
  simp only [← c12095, ← c12096, ← c12097] at e_179_14
  have e_184_6 := arithEq_of_rows h (row := 184) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_6
  simp only [← c12098, ← c12099, k3, ← c12100] at e_184_6
  have hr := e_184_6
  simp only [e_179_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 27) = a (.virt 201) := by
  have c12101 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.1
  exact c12101

theorem publicBatchWrapper89_f1826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1803)) (a (.wire 170 23)) (a (.virt 18627)) (a (.virt 18628)) := by
  have c12102 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.1
  have c12103 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.1
  have c12104 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.1
  have c12105 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.1
  have c12106 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12107 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12108 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12109 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12110 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12111 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12112 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12113 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12114 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12115 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12116 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12117 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12118 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_187_10 := arithEq_of_rows h (row := 187) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_10
  simp only [← c12102, k3, ← c12103, k3, ← c12104] at e_187_10
  have e_187_11 := arithEq_of_rows h (row := 187) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_11
  simp only [← c12105, ← c12106, k3, ← c12107] at e_187_11
  have e_187_12 := arithEq_of_rows h (row := 187) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_12
  simp only [← c12114, ← c12115, k3, ← c12116] at e_187_12
  have e_188_6 := arithEq_of_rows h (row := 188) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_6
  simp only [← c12108, ← c12109, ← c12110] at e_188_6
  have e_188_7 := arithEq_of_rows h (row := 188) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_7
  simp only [← c12111, ← c12112, ← c12113] at e_188_7
  refine ⟨?_, ?_⟩
  · have hc := e_188_6
    simp only [e_187_11] at hc
    linear_combination c12117.trans k2 - hc
  · have hc := e_187_12
    simp only [e_187_10, e_188_7, e_187_11] at hc
    linear_combination c12118.trans k2 - hc

theorem publicBatchWrapper89_f1827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 31) = bor (a (.wire 10 55)) (a (.virt 18627)) := by
  have c12119 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12120 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12121 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12122 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12123 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12124 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_15 := arithEq_of_rows h (row := 179) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_15
  simp only [← c12119, ← c12120, ← c12121] at e_179_15
  have e_184_7 := arithEq_of_rows h (row := 184) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_7
  simp only [← c12122, ← c12123, k3, ← c12124] at e_184_7
  have hr := e_184_7
  simp only [e_179_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 31) = a (.virt 201) := by
  have c12125 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12125

theorem publicBatchWrapper89_f1829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1804)) (a (.wire 170 31)) (a (.virt 18629)) (a (.virt 18630)) := by
  have c12126 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12127 := (publicBatchWrapper89_copies378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12128 := (publicBatchWrapper89_copies379 a h).1
  have c12129 := (publicBatchWrapper89_copies379 a h).2.1
  have c12130 := (publicBatchWrapper89_copies379 a h).2.2.1
  have c12131 := (publicBatchWrapper89_copies379 a h).2.2.2.1
  have c12132 := (publicBatchWrapper89_copies379 a h).2.2.2.2.1
  have c12133 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.1
  have c12134 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.1
  have c12135 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.1
  have c12136 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.1
  have c12137 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.1
  have c12138 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12139 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12140 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12141 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12142 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_187_13 := arithEq_of_rows h (row := 187) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_13
  simp only [← c12126, k3, ← c12127, k3, ← c12128] at e_187_13
  have e_187_14 := arithEq_of_rows h (row := 187) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_14
  simp only [← c12129, ← c12130, k3, ← c12131] at e_187_14
  have e_187_15 := arithEq_of_rows h (row := 187) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_15
  simp only [← c12138, ← c12139, k3, ← c12140] at e_187_15
  have e_188_8 := arithEq_of_rows h (row := 188) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_8
  simp only [← c12132, ← c12133, ← c12134] at e_188_8
  have e_188_9 := arithEq_of_rows h (row := 188) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_9
  simp only [← c12135, ← c12136, ← c12137] at e_188_9
  refine ⟨?_, ?_⟩
  · have hc := e_188_8
    simp only [e_187_14] at hc
    linear_combination c12141.trans k2 - hc
  · have hc := e_187_15
    simp only [e_187_13, e_188_9, e_187_14] at hc
    linear_combination c12142.trans k2 - hc

theorem publicBatchWrapper89_f1830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 35) = bor (a (.wire 10 55)) (a (.virt 18629)) := by
  have c12143 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12144 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12145 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12146 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12147 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12148 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_16 := arithEq_of_rows h (row := 179) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_16
  simp only [← c12143, ← c12144, ← c12145] at e_179_16
  have e_184_8 := arithEq_of_rows h (row := 184) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_8
  simp only [← c12146, ← c12147, k3, ← c12148] at e_184_8
  have hr := e_184_8
  simp only [e_179_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 35) = a (.virt 201) := by
  have c12149 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12149

theorem publicBatchWrapper89_f1832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1805)) (a (.wire 169 63)) (a (.virt 18631)) (a (.virt 18632)) := by
  have c12150 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12151 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12152 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12153 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12154 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12155 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12156 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12157 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12158 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12159 := (publicBatchWrapper89_copies379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12160 := (publicBatchWrapper89_copies380 a h).1
  have c12161 := (publicBatchWrapper89_copies380 a h).2.1
  have c12162 := (publicBatchWrapper89_copies380 a h).2.2.1
  have c12163 := (publicBatchWrapper89_copies380 a h).2.2.2.1
  have c12164 := (publicBatchWrapper89_copies380 a h).2.2.2.2.1
  have c12165 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.1
  have c12166 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_187_16 := arithEq_of_rows h (row := 187) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_16
  simp only [← c12150, k3, ← c12151, k3, ← c12152] at e_187_16
  have e_187_17 := arithEq_of_rows h (row := 187) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_17
  simp only [← c12153, ← c12154, k3, ← c12155] at e_187_17
  have e_187_18 := arithEq_of_rows h (row := 187) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_18
  simp only [← c12162, ← c12163, k3, ← c12164] at e_187_18
  have e_188_10 := arithEq_of_rows h (row := 188) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_10
  simp only [← c12156, ← c12157, ← c12158] at e_188_10
  have e_188_11 := arithEq_of_rows h (row := 188) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_11
  simp only [← c12159, ← c12160, ← c12161] at e_188_11
  refine ⟨?_, ?_⟩
  · have hc := e_188_10
    simp only [e_187_17] at hc
    linear_combination c12165.trans k2 - hc
  · have hc := e_187_18
    simp only [e_187_16, e_188_11, e_187_17] at hc
    linear_combination c12166.trans k2 - hc

theorem publicBatchWrapper89_f1833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1806)) (a (.wire 169 71)) (a (.virt 18633)) (a (.virt 18634)) := by
  have c12167 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.1
  have c12168 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.1
  have c12169 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.1
  have c12170 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12171 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12172 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12173 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12174 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12175 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12176 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12177 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12178 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12179 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12180 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12181 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12182 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12183 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_187_19 := arithEq_of_rows h (row := 187) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_187_19
  simp only [← c12167, k3, ← c12168, k3, ← c12169] at e_187_19
  have e_188_12 := arithEq_of_rows h (row := 188) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_12
  simp only [← c12173, ← c12174, ← c12175] at e_188_12
  have e_188_13 := arithEq_of_rows h (row := 188) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_13
  simp only [← c12176, ← c12177, ← c12178] at e_188_13
  have e_189_0 := arithEq_of_rows h (row := 189) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_0
  simp only [← c12170, ← c12171, k3, ← c12172] at e_189_0
  have e_189_1 := arithEq_of_rows h (row := 189) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_1
  simp only [← c12179, ← c12180, k3, ← c12181] at e_189_1
  refine ⟨?_, ?_⟩
  · have hc := e_188_12
    simp only [e_189_0] at hc
    linear_combination c12182.trans k2 - hc
  · have hc := e_189_1
    simp only [e_187_19, e_188_13, e_189_0] at hc
    linear_combination c12183.trans k2 - hc

theorem publicBatchWrapper89_f1834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1807)) (a (.wire 169 79)) (a (.virt 18635)) (a (.virt 18636)) := by
  have c12184 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12185 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12186 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12187 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12188 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12189 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12190 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12191 := (publicBatchWrapper89_copies380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12192 := (publicBatchWrapper89_copies381 a h).1
  have c12193 := (publicBatchWrapper89_copies381 a h).2.1
  have c12194 := (publicBatchWrapper89_copies381 a h).2.2.1
  have c12195 := (publicBatchWrapper89_copies381 a h).2.2.2.1
  have c12196 := (publicBatchWrapper89_copies381 a h).2.2.2.2.1
  have c12197 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.1
  have c12198 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.1
  have c12199 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.1
  have c12200 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_188_14 := arithEq_of_rows h (row := 188) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_14
  simp only [← c12190, ← c12191, ← c12192] at e_188_14
  have e_188_15 := arithEq_of_rows h (row := 188) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_15
  simp only [← c12193, ← c12194, ← c12195] at e_188_15
  have e_189_2 := arithEq_of_rows h (row := 189) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_2
  simp only [← c12184, k3, ← c12185, k3, ← c12186] at e_189_2
  have e_189_3 := arithEq_of_rows h (row := 189) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_3
  simp only [← c12187, ← c12188, k3, ← c12189] at e_189_3
  have e_189_4 := arithEq_of_rows h (row := 189) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_4
  simp only [← c12196, ← c12197, k3, ← c12198] at e_189_4
  refine ⟨?_, ?_⟩
  · have hc := e_188_14
    simp only [e_189_3] at hc
    linear_combination c12199.trans k2 - hc
  · have hc := e_189_4
    simp only [e_189_2, e_188_15, e_189_3] at hc
    linear_combination c12200.trans k2 - hc

theorem publicBatchWrapper89_f1835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1808)) (a (.wire 170 7)) (a (.virt 18637)) (a (.virt 18638)) := by
  have c12201 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.1
  have c12202 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12203 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12204 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12205 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12206 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12207 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12208 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12209 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12210 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12211 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12212 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12213 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12214 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12215 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12216 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12217 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_188_16 := arithEq_of_rows h (row := 188) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_16
  simp only [← c12207, ← c12208, ← c12209] at e_188_16
  have e_188_17 := arithEq_of_rows h (row := 188) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_17
  simp only [← c12210, ← c12211, ← c12212] at e_188_17
  have e_189_5 := arithEq_of_rows h (row := 189) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_5
  simp only [← c12201, k3, ← c12202, k3, ← c12203] at e_189_5
  have e_189_6 := arithEq_of_rows h (row := 189) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_6
  simp only [← c12204, ← c12205, k3, ← c12206] at e_189_6
  have e_189_7 := arithEq_of_rows h (row := 189) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_7
  simp only [← c12213, ← c12214, k3, ← c12215] at e_189_7
  refine ⟨?_, ?_⟩
  · have hc := e_188_16
    simp only [e_189_6] at hc
    linear_combination c12216.trans k2 - hc
  · have hc := e_189_7
    simp only [e_189_5, e_188_17, e_189_6] at hc
    linear_combination c12217.trans k2 - hc

theorem publicBatchWrapper89_f1836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 188 75) = band (a (.virt 18631)) (a (.virt 18633)) := by
  have c12218 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12219 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12220 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_188_18 := arithEq_of_rows h (row := 188) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_18
  simp only [← c12218, ← c12219, ← c12220] at e_188_18
  have hr := e_188_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 188 79) = band (a (.virt 18635)) (a (.virt 18637)) := by
  have c12221 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12222 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12223 := (publicBatchWrapper89_copies381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_188_19 := arithEq_of_rows h (row := 188) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_188_19
  simp only [← c12221, ← c12222, ← c12223] at e_188_19
  have hr := e_188_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 190 3) = band (a (.wire 188 75)) (a (.wire 188 79)) := by
  have c12224 := (publicBatchWrapper89_copies382 a h).1
  have c12225 := (publicBatchWrapper89_copies382 a h).2.1
  have c12226 := (publicBatchWrapper89_copies382 a h).2.2.1
  have e_190_0 := arithEq_of_rows h (row := 190) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_0
  simp only [← c12224, ← c12225, ← c12226] at e_190_0
  have hr := e_190_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 39) = bor (a (.wire 10 55)) (a (.wire 190 3)) := by
  have c12227 := (publicBatchWrapper89_copies382 a h).2.2.2.1
  have c12228 := (publicBatchWrapper89_copies382 a h).2.2.2.2.1
  have c12229 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.1
  have c12230 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.1
  have c12231 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.1
  have c12232 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_17 := arithEq_of_rows h (row := 179) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_17
  simp only [← c12227, ← c12228, ← c12229] at e_179_17
  have e_184_9 := arithEq_of_rows h (row := 184) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_9
  simp only [← c12230, ← c12231, k3, ← c12232] at e_184_9
  have hr := e_184_9
  simp only [e_179_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 39) = a (.virt 201) := by
  have c12233 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.1
  exact c12233

theorem publicBatchWrapper89_f1841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2003)) (a (.wire 170 23)) (a (.virt 18639)) (a (.virt 18640)) := by
  have c12234 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12235 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12236 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12237 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12238 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12239 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12240 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12241 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12242 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12243 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12244 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12245 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12246 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12247 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12248 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12249 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12250 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_189_8 := arithEq_of_rows h (row := 189) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_8
  simp only [← c12234, k3, ← c12235, k3, ← c12236] at e_189_8
  have e_189_9 := arithEq_of_rows h (row := 189) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_9
  simp only [← c12237, ← c12238, k3, ← c12239] at e_189_9
  have e_189_10 := arithEq_of_rows h (row := 189) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_10
  simp only [← c12246, ← c12247, k3, ← c12248] at e_189_10
  have e_190_1 := arithEq_of_rows h (row := 190) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_1
  simp only [← c12240, ← c12241, ← c12242] at e_190_1
  have e_190_2 := arithEq_of_rows h (row := 190) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_2
  simp only [← c12243, ← c12244, ← c12245] at e_190_2
  refine ⟨?_, ?_⟩
  · have hc := e_190_1
    simp only [e_189_9] at hc
    linear_combination c12249.trans k2 - hc
  · have hc := e_189_10
    simp only [e_189_8, e_190_2, e_189_9] at hc
    linear_combination c12250.trans k2 - hc

theorem publicBatchWrapper89_f1842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 43) = bor (a (.wire 12 19)) (a (.virt 18639)) := by
  have c12251 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12252 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12253 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12254 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12255 := (publicBatchWrapper89_copies382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12256 := (publicBatchWrapper89_copies383 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_18 := arithEq_of_rows h (row := 179) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_18
  simp only [← c12251, ← c12252, ← c12253] at e_179_18
  have e_184_10 := arithEq_of_rows h (row := 184) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_10
  simp only [← c12254, ← c12255, k3, ← c12256] at e_184_10
  have hr := e_184_10
  simp only [e_179_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 43) = a (.virt 201) := by
  have c12257 := (publicBatchWrapper89_copies383 a h).2.1
  exact c12257

theorem publicBatchWrapper89_f1844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2004)) (a (.wire 170 31)) (a (.virt 18641)) (a (.virt 18642)) := by
  have c12258 := (publicBatchWrapper89_copies383 a h).2.2.1
  have c12259 := (publicBatchWrapper89_copies383 a h).2.2.2.1
  have c12260 := (publicBatchWrapper89_copies383 a h).2.2.2.2.1
  have c12261 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.1
  have c12262 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.1
  have c12263 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.1
  have c12264 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.1
  have c12265 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.1
  have c12266 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12267 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12268 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12269 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12270 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12271 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12272 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12273 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12274 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_189_11 := arithEq_of_rows h (row := 189) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_11
  simp only [← c12258, k3, ← c12259, k3, ← c12260] at e_189_11
  have e_189_12 := arithEq_of_rows h (row := 189) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_12
  simp only [← c12261, ← c12262, k3, ← c12263] at e_189_12
  have e_189_13 := arithEq_of_rows h (row := 189) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_13
  simp only [← c12270, ← c12271, k3, ← c12272] at e_189_13
  have e_190_3 := arithEq_of_rows h (row := 190) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_3
  simp only [← c12264, ← c12265, ← c12266] at e_190_3
  have e_190_4 := arithEq_of_rows h (row := 190) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_4
  simp only [← c12267, ← c12268, ← c12269] at e_190_4
  refine ⟨?_, ?_⟩
  · have hc := e_190_3
    simp only [e_189_12] at hc
    linear_combination c12273.trans k2 - hc
  · have hc := e_189_13
    simp only [e_189_11, e_190_4, e_189_12] at hc
    linear_combination c12274.trans k2 - hc

theorem publicBatchWrapper89_f1845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 47) = bor (a (.wire 12 19)) (a (.virt 18641)) := by
  have c12275 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12276 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12277 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12278 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12279 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12280 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_179_19 := arithEq_of_rows h (row := 179) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_179_19
  simp only [← c12275, ← c12276, ← c12277] at e_179_19
  have e_184_11 := arithEq_of_rows h (row := 184) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_11
  simp only [← c12278, ← c12279, k3, ← c12280] at e_184_11
  have hr := e_184_11
  simp only [e_179_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 47) = a (.virt 201) := by
  have c12281 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12281

theorem publicBatchWrapper89_f1847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2005)) (a (.wire 169 63)) (a (.virt 18643)) (a (.virt 18644)) := by
  have c12282 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12283 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12284 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12285 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12286 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12287 := (publicBatchWrapper89_copies383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12288 := (publicBatchWrapper89_copies384 a h).1
  have c12289 := (publicBatchWrapper89_copies384 a h).2.1
  have c12290 := (publicBatchWrapper89_copies384 a h).2.2.1
  have c12291 := (publicBatchWrapper89_copies384 a h).2.2.2.1
  have c12292 := (publicBatchWrapper89_copies384 a h).2.2.2.2.1
  have c12293 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.1
  have c12294 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.1
  have c12295 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.1
  have c12296 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.1
  have c12297 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.1
  have c12298 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_189_14 := arithEq_of_rows h (row := 189) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_14
  simp only [← c12282, k3, ← c12283, k3, ← c12284] at e_189_14
  have e_189_15 := arithEq_of_rows h (row := 189) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_15
  simp only [← c12285, ← c12286, k3, ← c12287] at e_189_15
  have e_189_16 := arithEq_of_rows h (row := 189) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_16
  simp only [← c12294, ← c12295, k3, ← c12296] at e_189_16
  have e_190_5 := arithEq_of_rows h (row := 190) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_5
  simp only [← c12288, ← c12289, ← c12290] at e_190_5
  have e_190_6 := arithEq_of_rows h (row := 190) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_6
  simp only [← c12291, ← c12292, ← c12293] at e_190_6
  refine ⟨?_, ?_⟩
  · have hc := e_190_5
    simp only [e_189_15] at hc
    linear_combination c12297.trans k2 - hc
  · have hc := e_189_16
    simp only [e_189_14, e_190_6, e_189_15] at hc
    linear_combination c12298.trans k2 - hc

theorem publicBatchWrapper89_f1848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2006)) (a (.wire 169 71)) (a (.virt 18645)) (a (.virt 18646)) := by
  have c12299 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12300 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12301 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12302 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12303 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12304 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12305 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12306 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12307 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12308 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12309 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12310 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12311 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12312 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12313 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12314 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12315 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_189_17 := arithEq_of_rows h (row := 189) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_17
  simp only [← c12299, k3, ← c12300, k3, ← c12301] at e_189_17
  have e_189_18 := arithEq_of_rows h (row := 189) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_18
  simp only [← c12302, ← c12303, k3, ← c12304] at e_189_18
  have e_189_19 := arithEq_of_rows h (row := 189) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_189_19
  simp only [← c12311, ← c12312, k3, ← c12313] at e_189_19
  have e_190_7 := arithEq_of_rows h (row := 190) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_7
  simp only [← c12305, ← c12306, ← c12307] at e_190_7
  have e_190_8 := arithEq_of_rows h (row := 190) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_8
  simp only [← c12308, ← c12309, ← c12310] at e_190_8
  refine ⟨?_, ?_⟩
  · have hc := e_190_7
    simp only [e_189_18] at hc
    linear_combination c12314.trans k2 - hc
  · have hc := e_189_19
    simp only [e_189_17, e_190_8, e_189_18] at hc
    linear_combination c12315.trans k2 - hc

theorem publicBatchWrapper89_f1849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2007)) (a (.wire 169 79)) (a (.virt 18647)) (a (.virt 18648)) := by
  have c12316 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12317 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12318 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12319 := (publicBatchWrapper89_copies384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12320 := (publicBatchWrapper89_copies385 a h).1
  have c12321 := (publicBatchWrapper89_copies385 a h).2.1
  have c12322 := (publicBatchWrapper89_copies385 a h).2.2.1
  have c12323 := (publicBatchWrapper89_copies385 a h).2.2.2.1
  have c12324 := (publicBatchWrapper89_copies385 a h).2.2.2.2.1
  have c12325 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.1
  have c12326 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.1
  have c12327 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.1
  have c12328 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.1
  have c12329 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.1
  have c12330 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12331 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12332 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_190_9 := arithEq_of_rows h (row := 190) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_9
  simp only [← c12322, ← c12323, ← c12324] at e_190_9
  have e_190_10 := arithEq_of_rows h (row := 190) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_10
  simp only [← c12325, ← c12326, ← c12327] at e_190_10
  have e_191_0 := arithEq_of_rows h (row := 191) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_0
  simp only [← c12316, k3, ← c12317, k3, ← c12318] at e_191_0
  have e_191_1 := arithEq_of_rows h (row := 191) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_1
  simp only [← c12319, ← c12320, k3, ← c12321] at e_191_1
  have e_191_2 := arithEq_of_rows h (row := 191) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_2
  simp only [← c12328, ← c12329, k3, ← c12330] at e_191_2
  refine ⟨?_, ?_⟩
  · have hc := e_190_9
    simp only [e_191_1] at hc
    linear_combination c12331.trans k2 - hc
  · have hc := e_191_2
    simp only [e_191_0, e_190_10, e_191_1] at hc
    linear_combination c12332.trans k2 - hc

theorem publicBatchWrapper89_f1850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2008)) (a (.wire 170 7)) (a (.virt 18649)) (a (.virt 18650)) := by
  have c12333 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12334 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12335 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12336 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12337 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12338 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12339 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12340 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12341 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12342 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12343 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12344 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12345 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12346 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12347 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12348 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12349 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_190_11 := arithEq_of_rows h (row := 190) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_11
  simp only [← c12339, ← c12340, ← c12341] at e_190_11
  have e_190_12 := arithEq_of_rows h (row := 190) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_12
  simp only [← c12342, ← c12343, ← c12344] at e_190_12
  have e_191_3 := arithEq_of_rows h (row := 191) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_3
  simp only [← c12333, k3, ← c12334, k3, ← c12335] at e_191_3
  have e_191_4 := arithEq_of_rows h (row := 191) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_4
  simp only [← c12336, ← c12337, k3, ← c12338] at e_191_4
  have e_191_5 := arithEq_of_rows h (row := 191) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_5
  simp only [← c12345, ← c12346, k3, ← c12347] at e_191_5
  refine ⟨?_, ?_⟩
  · have hc := e_190_11
    simp only [e_191_4] at hc
    linear_combination c12348.trans k2 - hc
  · have hc := e_191_5
    simp only [e_191_3, e_190_12, e_191_4] at hc
    linear_combination c12349.trans k2 - hc

theorem publicBatchWrapper89_f1851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 190 55) = band (a (.virt 18643)) (a (.virt 18645)) := by
  have c12350 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12351 := (publicBatchWrapper89_copies385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12352 := (publicBatchWrapper89_copies386 a h).1
  have e_190_13 := arithEq_of_rows h (row := 190) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_13
  simp only [← c12350, ← c12351, ← c12352] at e_190_13
  have hr := e_190_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 190 59) = band (a (.virt 18647)) (a (.virt 18649)) := by
  have c12353 := (publicBatchWrapper89_copies386 a h).2.1
  have c12354 := (publicBatchWrapper89_copies386 a h).2.2.1
  have c12355 := (publicBatchWrapper89_copies386 a h).2.2.2.1
  have e_190_14 := arithEq_of_rows h (row := 190) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_14
  simp only [← c12353, ← c12354, ← c12355] at e_190_14
  have hr := e_190_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 190 63) = band (a (.wire 190 55)) (a (.wire 190 59)) := by
  have c12356 := (publicBatchWrapper89_copies386 a h).2.2.2.2.1
  have c12357 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.1
  have c12358 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.1
  have e_190_15 := arithEq_of_rows h (row := 190) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_15
  simp only [← c12356, ← c12357, ← c12358] at e_190_15
  have hr := e_190_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 51) = bor (a (.wire 12 19)) (a (.wire 190 63)) := by
  have c12359 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.1
  have c12360 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.1
  have c12361 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.1
  have c12362 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12363 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12364 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_184_12 := arithEq_of_rows h (row := 184) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_12
  simp only [← c12362, ← c12363, k3, ← c12364] at e_184_12
  have e_192_0 := arithEq_of_rows h (row := 192) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_0
  simp only [← c12359, ← c12360, ← c12361] at e_192_0
  have hr := e_184_12
  simp only [e_192_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 51) = a (.virt 201) := by
  have c12365 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12365

theorem publicBatchWrapper89_f1856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2203)) (a (.wire 170 23)) (a (.virt 18651)) (a (.virt 18652)) := by
  have c12366 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12367 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12368 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12369 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12370 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12371 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12372 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12373 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12374 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12375 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12376 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12377 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12378 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12379 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12380 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12381 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12382 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_190_16 := arithEq_of_rows h (row := 190) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_16
  simp only [← c12372, ← c12373, ← c12374] at e_190_16
  have e_190_17 := arithEq_of_rows h (row := 190) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_17
  simp only [← c12375, ← c12376, ← c12377] at e_190_17
  have e_191_6 := arithEq_of_rows h (row := 191) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_6
  simp only [← c12366, k3, ← c12367, k3, ← c12368] at e_191_6
  have e_191_7 := arithEq_of_rows h (row := 191) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_7
  simp only [← c12369, ← c12370, k3, ← c12371] at e_191_7
  have e_191_8 := arithEq_of_rows h (row := 191) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_8
  simp only [← c12378, ← c12379, k3, ← c12380] at e_191_8
  refine ⟨?_, ?_⟩
  · have hc := e_190_16
    simp only [e_191_7] at hc
    linear_combination c12381.trans k2 - hc
  · have hc := e_191_8
    simp only [e_191_6, e_190_17, e_191_7] at hc
    linear_combination c12382.trans k2 - hc

theorem publicBatchWrapper89_f1857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 55) = bor (a (.wire 12 63)) (a (.virt 18651)) := by
  have c12383 := (publicBatchWrapper89_copies386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12384 := (publicBatchWrapper89_copies387 a h).1
  have c12385 := (publicBatchWrapper89_copies387 a h).2.1
  have c12386 := (publicBatchWrapper89_copies387 a h).2.2.1
  have c12387 := (publicBatchWrapper89_copies387 a h).2.2.2.1
  have c12388 := (publicBatchWrapper89_copies387 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_184_13 := arithEq_of_rows h (row := 184) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_13
  simp only [← c12386, ← c12387, k3, ← c12388] at e_184_13
  have e_192_1 := arithEq_of_rows h (row := 192) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_1
  simp only [← c12383, ← c12384, ← c12385] at e_192_1
  have hr := e_184_13
  simp only [e_192_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 55) = a (.virt 201) := by
  have c12389 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.1
  exact c12389

theorem publicBatchWrapper89_f1859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2204)) (a (.wire 170 31)) (a (.virt 18653)) (a (.virt 18654)) := by
  have c12390 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.1
  have c12391 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.1
  have c12392 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.1
  have c12393 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.1
  have c12394 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12395 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12396 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12397 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12398 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12399 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12400 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12401 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12402 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12403 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12404 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12405 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12406 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_190_18 := arithEq_of_rows h (row := 190) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_18
  simp only [← c12396, ← c12397, ← c12398] at e_190_18
  have e_190_19 := arithEq_of_rows h (row := 190) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_190_19
  simp only [← c12399, ← c12400, ← c12401] at e_190_19
  have e_191_9 := arithEq_of_rows h (row := 191) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_9
  simp only [← c12390, k3, ← c12391, k3, ← c12392] at e_191_9
  have e_191_10 := arithEq_of_rows h (row := 191) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_10
  simp only [← c12393, ← c12394, k3, ← c12395] at e_191_10
  have e_191_11 := arithEq_of_rows h (row := 191) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_11
  simp only [← c12402, ← c12403, k3, ← c12404] at e_191_11
  refine ⟨?_, ?_⟩
  · have hc := e_190_18
    simp only [e_191_10] at hc
    linear_combination c12405.trans k2 - hc
  · have hc := e_191_11
    simp only [e_191_9, e_190_19, e_191_10] at hc
    linear_combination c12406.trans k2 - hc

theorem publicBatchWrapper89_f1860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 59) = bor (a (.wire 12 63)) (a (.virt 18653)) := by
  have c12407 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12408 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12409 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12410 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12411 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12412 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_184_14 := arithEq_of_rows h (row := 184) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_14
  simp only [← c12410, ← c12411, k3, ← c12412] at e_184_14
  have e_192_2 := arithEq_of_rows h (row := 192) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_2
  simp only [← c12407, ← c12408, ← c12409] at e_192_2
  have hr := e_184_14
  simp only [e_192_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 59) = a (.virt 201) := by
  have c12413 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12413

theorem publicBatchWrapper89_f1862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2205)) (a (.wire 169 63)) (a (.virt 18655)) (a (.virt 18656)) := by
  have c12414 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12415 := (publicBatchWrapper89_copies387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12416 := (publicBatchWrapper89_copies388 a h).1
  have c12417 := (publicBatchWrapper89_copies388 a h).2.1
  have c12418 := (publicBatchWrapper89_copies388 a h).2.2.1
  have c12419 := (publicBatchWrapper89_copies388 a h).2.2.2.1
  have c12420 := (publicBatchWrapper89_copies388 a h).2.2.2.2.1
  have c12421 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.1
  have c12422 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.1
  have c12423 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.1
  have c12424 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.1
  have c12425 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.1
  have c12426 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12427 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12428 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12429 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12430 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_191_12 := arithEq_of_rows h (row := 191) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_12
  simp only [← c12414, k3, ← c12415, k3, ← c12416] at e_191_12
  have e_191_13 := arithEq_of_rows h (row := 191) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_13
  simp only [← c12417, ← c12418, k3, ← c12419] at e_191_13
  have e_191_14 := arithEq_of_rows h (row := 191) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_14
  simp only [← c12426, ← c12427, k3, ← c12428] at e_191_14
  have e_193_0 := arithEq_of_rows h (row := 193) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_0
  simp only [← c12420, ← c12421, ← c12422] at e_193_0
  have e_193_1 := arithEq_of_rows h (row := 193) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_1
  simp only [← c12423, ← c12424, ← c12425] at e_193_1
  refine ⟨?_, ?_⟩
  · have hc := e_193_0
    simp only [e_191_13] at hc
    linear_combination c12429.trans k2 - hc
  · have hc := e_191_14
    simp only [e_191_12, e_193_1, e_191_13] at hc
    linear_combination c12430.trans k2 - hc

theorem publicBatchWrapper89_f1863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2206)) (a (.wire 169 71)) (a (.virt 18657)) (a (.virt 18658)) := by
  have c12431 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12432 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12433 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12434 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12435 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12436 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12437 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12438 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12439 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12440 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12441 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12442 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12443 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12444 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12445 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12446 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12447 := (publicBatchWrapper89_copies388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_191_15 := arithEq_of_rows h (row := 191) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_15
  simp only [← c12431, k3, ← c12432, k3, ← c12433] at e_191_15
  have e_191_16 := arithEq_of_rows h (row := 191) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_16
  simp only [← c12434, ← c12435, k3, ← c12436] at e_191_16
  have e_191_17 := arithEq_of_rows h (row := 191) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_17
  simp only [← c12443, ← c12444, k3, ← c12445] at e_191_17
  have e_193_2 := arithEq_of_rows h (row := 193) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_2
  simp only [← c12437, ← c12438, ← c12439] at e_193_2
  have e_193_3 := arithEq_of_rows h (row := 193) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_3
  simp only [← c12440, ← c12441, ← c12442] at e_193_3
  refine ⟨?_, ?_⟩
  · have hc := e_193_2
    simp only [e_191_16] at hc
    linear_combination c12446.trans k2 - hc
  · have hc := e_191_17
    simp only [e_191_15, e_193_3, e_191_16] at hc
    linear_combination c12447.trans k2 - hc

theorem publicBatchWrapper89_f1864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2207)) (a (.wire 169 79)) (a (.virt 18659)) (a (.virt 18660)) := by
  have c12448 := (publicBatchWrapper89_copies389 a h).1
  have c12449 := (publicBatchWrapper89_copies389 a h).2.1
  have c12450 := (publicBatchWrapper89_copies389 a h).2.2.1
  have c12451 := (publicBatchWrapper89_copies389 a h).2.2.2.1
  have c12452 := (publicBatchWrapper89_copies389 a h).2.2.2.2.1
  have c12453 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.1
  have c12454 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.1
  have c12455 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.1
  have c12456 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.1
  have c12457 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.1
  have c12458 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12459 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12460 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12461 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12462 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12463 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12464 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_191_18 := arithEq_of_rows h (row := 191) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_18
  simp only [← c12448, k3, ← c12449, k3, ← c12450] at e_191_18
  have e_191_19 := arithEq_of_rows h (row := 191) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_191_19
  simp only [← c12451, ← c12452, k3, ← c12453] at e_191_19
  have e_193_4 := arithEq_of_rows h (row := 193) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_4
  simp only [← c12454, ← c12455, ← c12456] at e_193_4
  have e_193_5 := arithEq_of_rows h (row := 193) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_5
  simp only [← c12457, ← c12458, ← c12459] at e_193_5
  have e_194_0 := arithEq_of_rows h (row := 194) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_0
  simp only [← c12460, ← c12461, k3, ← c12462] at e_194_0
  refine ⟨?_, ?_⟩
  · have hc := e_193_4
    simp only [e_191_19] at hc
    linear_combination c12463.trans k2 - hc
  · have hc := e_194_0
    simp only [e_191_18, e_193_5, e_191_19] at hc
    linear_combination c12464.trans k2 - hc

theorem publicBatchWrapper89_f1865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2208)) (a (.wire 170 7)) (a (.virt 18661)) (a (.virt 18662)) := by
  have c12465 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12466 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12467 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12468 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12469 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12470 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12471 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12472 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12473 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12474 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12475 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12476 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12477 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12478 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12479 := (publicBatchWrapper89_copies389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12480 := (publicBatchWrapper89_copies390 a h).1
  have c12481 := (publicBatchWrapper89_copies390 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_193_6 := arithEq_of_rows h (row := 193) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_6
  simp only [← c12471, ← c12472, ← c12473] at e_193_6
  have e_193_7 := arithEq_of_rows h (row := 193) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_7
  simp only [← c12474, ← c12475, ← c12476] at e_193_7
  have e_194_1 := arithEq_of_rows h (row := 194) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_1
  simp only [← c12465, k3, ← c12466, k3, ← c12467] at e_194_1
  have e_194_2 := arithEq_of_rows h (row := 194) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_2
  simp only [← c12468, ← c12469, k3, ← c12470] at e_194_2
  have e_194_3 := arithEq_of_rows h (row := 194) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_3
  simp only [← c12477, ← c12478, k3, ← c12479] at e_194_3
  refine ⟨?_, ?_⟩
  · have hc := e_193_6
    simp only [e_194_2] at hc
    linear_combination c12480.trans k2 - hc
  · have hc := e_194_3
    simp only [e_194_1, e_193_7, e_194_2] at hc
    linear_combination c12481.trans k2 - hc

theorem publicBatchWrapper89_f1866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 193 35) = band (a (.virt 18655)) (a (.virt 18657)) := by
  have c12482 := (publicBatchWrapper89_copies390 a h).2.2.1
  have c12483 := (publicBatchWrapper89_copies390 a h).2.2.2.1
  have c12484 := (publicBatchWrapper89_copies390 a h).2.2.2.2.1
  have e_193_8 := arithEq_of_rows h (row := 193) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_8
  simp only [← c12482, ← c12483, ← c12484] at e_193_8
  have hr := e_193_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 193 39) = band (a (.virt 18659)) (a (.virt 18661)) := by
  have c12485 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.1
  have c12486 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.1
  have c12487 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.1
  have e_193_9 := arithEq_of_rows h (row := 193) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_9
  simp only [← c12485, ← c12486, ← c12487] at e_193_9
  have hr := e_193_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 193 43) = band (a (.wire 193 35)) (a (.wire 193 39)) := by
  have c12488 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.1
  have c12489 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.1
  have c12490 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_193_10 := arithEq_of_rows h (row := 193) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_10
  simp only [← c12488, ← c12489, ← c12490] at e_193_10
  have hr := e_193_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 63) = bor (a (.wire 12 63)) (a (.wire 193 43)) := by
  have c12491 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12492 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12493 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12494 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12495 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12496 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_184_15 := arithEq_of_rows h (row := 184) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_15
  simp only [← c12494, ← c12495, k3, ← c12496] at e_184_15
  have e_192_3 := arithEq_of_rows h (row := 192) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_3
  simp only [← c12491, ← c12492, ← c12493] at e_192_3
  have hr := e_184_15
  simp only [e_192_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 63) = a (.virt 201) := by
  have c12497 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12497

theorem publicBatchWrapper89_f1871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2403)) (a (.wire 170 23)) (a (.virt 18663)) (a (.virt 18664)) := by
  have c12498 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12499 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12500 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12501 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12502 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12503 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12504 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12505 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12506 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12507 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12508 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12509 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12510 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12511 := (publicBatchWrapper89_copies390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12512 := (publicBatchWrapper89_copies391 a h).1
  have c12513 := (publicBatchWrapper89_copies391 a h).2.1
  have c12514 := (publicBatchWrapper89_copies391 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_193_11 := arithEq_of_rows h (row := 193) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_11
  simp only [← c12504, ← c12505, ← c12506] at e_193_11
  have e_193_12 := arithEq_of_rows h (row := 193) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_12
  simp only [← c12507, ← c12508, ← c12509] at e_193_12
  have e_194_4 := arithEq_of_rows h (row := 194) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_4
  simp only [← c12498, k3, ← c12499, k3, ← c12500] at e_194_4
  have e_194_5 := arithEq_of_rows h (row := 194) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_5
  simp only [← c12501, ← c12502, k3, ← c12503] at e_194_5
  have e_194_6 := arithEq_of_rows h (row := 194) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_6
  simp only [← c12510, ← c12511, k3, ← c12512] at e_194_6
  refine ⟨?_, ?_⟩
  · have hc := e_193_11
    simp only [e_194_5] at hc
    linear_combination c12513.trans k2 - hc
  · have hc := e_194_6
    simp only [e_194_4, e_193_12, e_194_5] at hc
    linear_combination c12514.trans k2 - hc

theorem publicBatchWrapper89_f1872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 67) = bor (a (.wire 14 27)) (a (.virt 18663)) := by
  have c12515 := (publicBatchWrapper89_copies391 a h).2.2.2.1
  have c12516 := (publicBatchWrapper89_copies391 a h).2.2.2.2.1
  have c12517 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.1
  have c12518 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.1
  have c12519 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.1
  have c12520 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_184_16 := arithEq_of_rows h (row := 184) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_16
  simp only [← c12518, ← c12519, k3, ← c12520] at e_184_16
  have e_192_4 := arithEq_of_rows h (row := 192) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_4
  simp only [← c12515, ← c12516, ← c12517] at e_192_4
  have hr := e_184_16
  simp only [e_192_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 67) = a (.virt 201) := by
  have c12521 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.1
  exact c12521

theorem publicBatchWrapper89_f1874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2404)) (a (.wire 170 31)) (a (.virt 18665)) (a (.virt 18666)) := by
  have c12522 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12523 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12524 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12525 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12526 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12527 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12528 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12529 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12530 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12531 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12532 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12533 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12534 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12535 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12536 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12537 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12538 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_193_13 := arithEq_of_rows h (row := 193) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_13
  simp only [← c12528, ← c12529, ← c12530] at e_193_13
  have e_193_14 := arithEq_of_rows h (row := 193) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_14
  simp only [← c12531, ← c12532, ← c12533] at e_193_14
  have e_194_7 := arithEq_of_rows h (row := 194) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_7
  simp only [← c12522, k3, ← c12523, k3, ← c12524] at e_194_7
  have e_194_8 := arithEq_of_rows h (row := 194) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_8
  simp only [← c12525, ← c12526, k3, ← c12527] at e_194_8
  have e_194_9 := arithEq_of_rows h (row := 194) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_9
  simp only [← c12534, ← c12535, k3, ← c12536] at e_194_9
  refine ⟨?_, ?_⟩
  · have hc := e_193_13
    simp only [e_194_8] at hc
    linear_combination c12537.trans k2 - hc
  · have hc := e_194_9
    simp only [e_194_7, e_193_14, e_194_8] at hc
    linear_combination c12538.trans k2 - hc

theorem publicBatchWrapper89_f1875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 71) = bor (a (.wire 14 27)) (a (.virt 18665)) := by
  have c12539 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12540 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12541 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12542 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12543 := (publicBatchWrapper89_copies391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12544 := (publicBatchWrapper89_copies392 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_184_17 := arithEq_of_rows h (row := 184) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_17
  simp only [← c12542, ← c12543, k3, ← c12544] at e_184_17
  have e_192_5 := arithEq_of_rows h (row := 192) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_5
  simp only [← c12539, ← c12540, ← c12541] at e_192_5
  have hr := e_184_17
  simp only [e_192_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 71) = a (.virt 201) := by
  have c12545 := (publicBatchWrapper89_copies392 a h).2.1
  exact c12545

theorem publicBatchWrapper89_f1877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2405)) (a (.wire 169 63)) (a (.virt 18667)) (a (.virt 18668)) := by
  have c12546 := (publicBatchWrapper89_copies392 a h).2.2.1
  have c12547 := (publicBatchWrapper89_copies392 a h).2.2.2.1
  have c12548 := (publicBatchWrapper89_copies392 a h).2.2.2.2.1
  have c12549 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.1
  have c12550 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.1
  have c12551 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.1
  have c12552 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.1
  have c12553 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.1
  have c12554 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12555 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12556 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12557 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12558 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12559 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12560 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12561 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12562 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_193_15 := arithEq_of_rows h (row := 193) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_15
  simp only [← c12552, ← c12553, ← c12554] at e_193_15
  have e_193_16 := arithEq_of_rows h (row := 193) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_16
  simp only [← c12555, ← c12556, ← c12557] at e_193_16
  have e_194_10 := arithEq_of_rows h (row := 194) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_10
  simp only [← c12546, k3, ← c12547, k3, ← c12548] at e_194_10
  have e_194_11 := arithEq_of_rows h (row := 194) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_11
  simp only [← c12549, ← c12550, k3, ← c12551] at e_194_11
  have e_194_12 := arithEq_of_rows h (row := 194) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_12
  simp only [← c12558, ← c12559, k3, ← c12560] at e_194_12
  refine ⟨?_, ?_⟩
  · have hc := e_193_15
    simp only [e_194_11] at hc
    linear_combination c12561.trans k2 - hc
  · have hc := e_194_12
    simp only [e_194_10, e_193_16, e_194_11] at hc
    linear_combination c12562.trans k2 - hc

theorem publicBatchWrapper89_f1878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2406)) (a (.wire 169 71)) (a (.virt 18669)) (a (.virt 18670)) := by
  have c12563 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12564 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12565 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12566 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12567 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12568 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12569 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12570 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12571 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12572 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12573 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12574 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12575 := (publicBatchWrapper89_copies392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12576 := (publicBatchWrapper89_copies393 a h).1
  have c12577 := (publicBatchWrapper89_copies393 a h).2.1
  have c12578 := (publicBatchWrapper89_copies393 a h).2.2.1
  have c12579 := (publicBatchWrapper89_copies393 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_193_17 := arithEq_of_rows h (row := 193) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_17
  simp only [← c12569, ← c12570, ← c12571] at e_193_17
  have e_193_18 := arithEq_of_rows h (row := 193) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_18
  simp only [← c12572, ← c12573, ← c12574] at e_193_18
  have e_194_13 := arithEq_of_rows h (row := 194) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_13
  simp only [← c12563, k3, ← c12564, k3, ← c12565] at e_194_13
  have e_194_14 := arithEq_of_rows h (row := 194) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_14
  simp only [← c12566, ← c12567, k3, ← c12568] at e_194_14
  have e_194_15 := arithEq_of_rows h (row := 194) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_15
  simp only [← c12575, ← c12576, k3, ← c12577] at e_194_15
  refine ⟨?_, ?_⟩
  · have hc := e_193_17
    simp only [e_194_14] at hc
    linear_combination c12578.trans k2 - hc
  · have hc := e_194_15
    simp only [e_194_13, e_193_18, e_194_14] at hc
    linear_combination c12579.trans k2 - hc

theorem publicBatchWrapper89_f1879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2407)) (a (.wire 169 79)) (a (.virt 18671)) (a (.virt 18672)) := by
  have c12580 := (publicBatchWrapper89_copies393 a h).2.2.2.2.1
  have c12581 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.1
  have c12582 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.1
  have c12583 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.1
  have c12584 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.1
  have c12585 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.1
  have c12586 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12587 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12588 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12589 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12590 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12591 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12592 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12593 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12594 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12595 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12596 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_193_19 := arithEq_of_rows h (row := 193) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_193_19
  simp only [← c12586, ← c12587, ← c12588] at e_193_19
  have e_194_16 := arithEq_of_rows h (row := 194) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_16
  simp only [← c12580, k3, ← c12581, k3, ← c12582] at e_194_16
  have e_194_17 := arithEq_of_rows h (row := 194) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_17
  simp only [← c12583, ← c12584, k3, ← c12585] at e_194_17
  have e_194_18 := arithEq_of_rows h (row := 194) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_18
  simp only [← c12592, ← c12593, k3, ← c12594] at e_194_18
  have e_195_0 := arithEq_of_rows h (row := 195) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_0
  simp only [← c12589, ← c12590, ← c12591] at e_195_0
  refine ⟨?_, ?_⟩
  · have hc := e_193_19
    simp only [e_194_17] at hc
    linear_combination c12595.trans k2 - hc
  · have hc := e_194_18
    simp only [e_194_16, e_195_0, e_194_17] at hc
    linear_combination c12596.trans k2 - hc

theorem publicBatchWrapper89_f1880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2408)) (a (.wire 170 7)) (a (.virt 18673)) (a (.virt 18674)) := by
  have c12597 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12598 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12599 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12600 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12601 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12602 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12603 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12604 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12605 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12606 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12607 := (publicBatchWrapper89_copies393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12608 := (publicBatchWrapper89_copies394 a h).1
  have c12609 := (publicBatchWrapper89_copies394 a h).2.1
  have c12610 := (publicBatchWrapper89_copies394 a h).2.2.1
  have c12611 := (publicBatchWrapper89_copies394 a h).2.2.2.1
  have c12612 := (publicBatchWrapper89_copies394 a h).2.2.2.2.1
  have c12613 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_194_19 := arithEq_of_rows h (row := 194) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_194_19
  simp only [← c12597, k3, ← c12598, k3, ← c12599] at e_194_19
  have e_195_1 := arithEq_of_rows h (row := 195) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_1
  simp only [← c12603, ← c12604, ← c12605] at e_195_1
  have e_195_2 := arithEq_of_rows h (row := 195) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_2
  simp only [← c12606, ← c12607, ← c12608] at e_195_2
  have e_196_0 := arithEq_of_rows h (row := 196) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_0
  simp only [← c12600, ← c12601, k3, ← c12602] at e_196_0
  have e_196_1 := arithEq_of_rows h (row := 196) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_1
  simp only [← c12609, ← c12610, k3, ← c12611] at e_196_1
  refine ⟨?_, ?_⟩
  · have hc := e_195_1
    simp only [e_196_0] at hc
    linear_combination c12612.trans k2 - hc
  · have hc := e_196_1
    simp only [e_194_19, e_195_2, e_196_0] at hc
    linear_combination c12613.trans k2 - hc

theorem publicBatchWrapper89_f1881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 195 15) = band (a (.virt 18667)) (a (.virt 18669)) := by
  have c12614 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.1
  have c12615 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.1
  have c12616 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.1
  have e_195_3 := arithEq_of_rows h (row := 195) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_3
  simp only [← c12614, ← c12615, ← c12616] at e_195_3
  have hr := e_195_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 195 19) = band (a (.virt 18671)) (a (.virt 18673)) := by
  have c12617 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.1
  have c12618 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12619 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_195_4 := arithEq_of_rows h (row := 195) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_4
  simp only [← c12617, ← c12618, ← c12619] at e_195_4
  have hr := e_195_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 195 23) = band (a (.wire 195 15)) (a (.wire 195 19)) := by
  have c12620 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12621 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12622 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_195_5 := arithEq_of_rows h (row := 195) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_5
  simp only [← c12620, ← c12621, ← c12622] at e_195_5
  have hr := e_195_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 75) = bor (a (.wire 14 27)) (a (.wire 195 23)) := by
  have c12623 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12624 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12625 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12626 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12627 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12628 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_184_18 := arithEq_of_rows h (row := 184) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_18
  simp only [← c12626, ← c12627, k3, ← c12628] at e_184_18
  have e_192_6 := arithEq_of_rows h (row := 192) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_6
  simp only [← c12623, ← c12624, ← c12625] at e_192_6
  have hr := e_184_18
  simp only [e_192_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 75) = a (.virt 201) := by
  have c12629 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12629

theorem publicBatchWrapper89_f1886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2603)) (a (.wire 170 23)) (a (.virt 18675)) (a (.virt 18676)) := by
  have c12630 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12631 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12632 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12633 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12634 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12635 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12636 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12637 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12638 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12639 := (publicBatchWrapper89_copies394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12640 := (publicBatchWrapper89_copies395 a h).1
  have c12641 := (publicBatchWrapper89_copies395 a h).2.1
  have c12642 := (publicBatchWrapper89_copies395 a h).2.2.1
  have c12643 := (publicBatchWrapper89_copies395 a h).2.2.2.1
  have c12644 := (publicBatchWrapper89_copies395 a h).2.2.2.2.1
  have c12645 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.1
  have c12646 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_195_6 := arithEq_of_rows h (row := 195) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_6
  simp only [← c12636, ← c12637, ← c12638] at e_195_6
  have e_195_7 := arithEq_of_rows h (row := 195) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_7
  simp only [← c12639, ← c12640, ← c12641] at e_195_7
  have e_196_2 := arithEq_of_rows h (row := 196) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_2
  simp only [← c12630, k3, ← c12631, k3, ← c12632] at e_196_2
  have e_196_3 := arithEq_of_rows h (row := 196) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_3
  simp only [← c12633, ← c12634, k3, ← c12635] at e_196_3
  have e_196_4 := arithEq_of_rows h (row := 196) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_4
  simp only [← c12642, ← c12643, k3, ← c12644] at e_196_4
  refine ⟨?_, ?_⟩
  · have hc := e_195_6
    simp only [e_196_3] at hc
    linear_combination c12645.trans k2 - hc
  · have hc := e_196_4
    simp only [e_196_2, e_195_7, e_196_3] at hc
    linear_combination c12646.trans k2 - hc

theorem publicBatchWrapper89_f1887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 79) = bor (a (.wire 14 71)) (a (.virt 18675)) := by
  have c12647 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.1
  have c12648 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.1
  have c12649 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.1
  have c12650 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12651 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12652 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_184_19 := arithEq_of_rows h (row := 184) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_184_19
  simp only [← c12650, ← c12651, k3, ← c12652] at e_184_19
  have e_192_7 := arithEq_of_rows h (row := 192) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_7
  simp only [← c12647, ← c12648, ← c12649] at e_192_7
  have hr := e_184_19
  simp only [e_192_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 184 79) = a (.virt 201) := by
  have c12653 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12653

theorem publicBatchWrapper89_f1889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2604)) (a (.wire 170 31)) (a (.virt 18677)) (a (.virt 18678)) := by
  have c12654 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12655 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12656 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12657 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12658 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12659 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12660 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12661 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12662 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12663 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12664 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12665 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12666 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12667 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12668 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12669 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12670 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_195_8 := arithEq_of_rows h (row := 195) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_8
  simp only [← c12660, ← c12661, ← c12662] at e_195_8
  have e_195_9 := arithEq_of_rows h (row := 195) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_9
  simp only [← c12663, ← c12664, ← c12665] at e_195_9
  have e_196_5 := arithEq_of_rows h (row := 196) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_5
  simp only [← c12654, k3, ← c12655, k3, ← c12656] at e_196_5
  have e_196_6 := arithEq_of_rows h (row := 196) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_6
  simp only [← c12657, ← c12658, k3, ← c12659] at e_196_6
  have e_196_7 := arithEq_of_rows h (row := 196) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_7
  simp only [← c12666, ← c12667, k3, ← c12668] at e_196_7
  refine ⟨?_, ?_⟩
  · have hc := e_195_8
    simp only [e_196_6] at hc
    linear_combination c12669.trans k2 - hc
  · have hc := e_196_7
    simp only [e_196_5, e_195_9, e_196_6] at hc
    linear_combination c12670.trans k2 - hc

theorem publicBatchWrapper89_f1890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 3) = bor (a (.wire 14 71)) (a (.virt 18677)) := by
  have c12671 := (publicBatchWrapper89_copies395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12672 := (publicBatchWrapper89_copies396 a h).1
  have c12673 := (publicBatchWrapper89_copies396 a h).2.1
  have c12674 := (publicBatchWrapper89_copies396 a h).2.2.1
  have c12675 := (publicBatchWrapper89_copies396 a h).2.2.2.1
  have c12676 := (publicBatchWrapper89_copies396 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_8 := arithEq_of_rows h (row := 192) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_8
  simp only [← c12671, ← c12672, ← c12673] at e_192_8
  have e_197_0 := arithEq_of_rows h (row := 197) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_0
  simp only [← c12674, ← c12675, k3, ← c12676] at e_197_0
  have hr := e_197_0
  simp only [e_192_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 3) = a (.virt 201) := by
  have c12677 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.1
  exact c12677

theorem publicBatchWrapper89_f1892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2605)) (a (.wire 169 63)) (a (.virt 18679)) (a (.virt 18680)) := by
  have c12678 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.1
  have c12679 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.1
  have c12680 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.1
  have c12681 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.1
  have c12682 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12683 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12684 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12685 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12686 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12687 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12688 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12689 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12690 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12691 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12692 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12693 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12694 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_195_10 := arithEq_of_rows h (row := 195) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_10
  simp only [← c12684, ← c12685, ← c12686] at e_195_10
  have e_195_11 := arithEq_of_rows h (row := 195) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_11
  simp only [← c12687, ← c12688, ← c12689] at e_195_11
  have e_196_8 := arithEq_of_rows h (row := 196) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_8
  simp only [← c12678, k3, ← c12679, k3, ← c12680] at e_196_8
  have e_196_9 := arithEq_of_rows h (row := 196) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_9
  simp only [← c12681, ← c12682, k3, ← c12683] at e_196_9
  have e_196_10 := arithEq_of_rows h (row := 196) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_10
  simp only [← c12690, ← c12691, k3, ← c12692] at e_196_10
  refine ⟨?_, ?_⟩
  · have hc := e_195_10
    simp only [e_196_9] at hc
    linear_combination c12693.trans k2 - hc
  · have hc := e_196_10
    simp only [e_196_8, e_195_11, e_196_9] at hc
    linear_combination c12694.trans k2 - hc

theorem publicBatchWrapper89_f1893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2606)) (a (.wire 169 71)) (a (.virt 18681)) (a (.virt 18682)) := by
  have c12695 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12696 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12697 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12698 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12699 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12700 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12701 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12702 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12703 := (publicBatchWrapper89_copies396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12704 := (publicBatchWrapper89_copies397 a h).1
  have c12705 := (publicBatchWrapper89_copies397 a h).2.1
  have c12706 := (publicBatchWrapper89_copies397 a h).2.2.1
  have c12707 := (publicBatchWrapper89_copies397 a h).2.2.2.1
  have c12708 := (publicBatchWrapper89_copies397 a h).2.2.2.2.1
  have c12709 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.1
  have c12710 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.1
  have c12711 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_195_12 := arithEq_of_rows h (row := 195) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_12
  simp only [← c12701, ← c12702, ← c12703] at e_195_12
  have e_195_13 := arithEq_of_rows h (row := 195) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_13
  simp only [← c12704, ← c12705, ← c12706] at e_195_13
  have e_196_11 := arithEq_of_rows h (row := 196) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_11
  simp only [← c12695, k3, ← c12696, k3, ← c12697] at e_196_11
  have e_196_12 := arithEq_of_rows h (row := 196) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_12
  simp only [← c12698, ← c12699, k3, ← c12700] at e_196_12
  have e_196_13 := arithEq_of_rows h (row := 196) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_13
  simp only [← c12707, ← c12708, k3, ← c12709] at e_196_13
  refine ⟨?_, ?_⟩
  · have hc := e_195_12
    simp only [e_196_12] at hc
    linear_combination c12710.trans k2 - hc
  · have hc := e_196_13
    simp only [e_196_11, e_195_13, e_196_12] at hc
    linear_combination c12711.trans k2 - hc

theorem publicBatchWrapper89_f1894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2607)) (a (.wire 169 79)) (a (.virt 18683)) (a (.virt 18684)) := by
  have c12712 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.1
  have c12713 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.1
  have c12714 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12715 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12716 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12717 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12718 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12719 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12720 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12721 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12722 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12723 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12724 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12725 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12726 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12727 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12728 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_195_14 := arithEq_of_rows h (row := 195) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_14
  simp only [← c12718, ← c12719, ← c12720] at e_195_14
  have e_195_15 := arithEq_of_rows h (row := 195) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_15
  simp only [← c12721, ← c12722, ← c12723] at e_195_15
  have e_196_14 := arithEq_of_rows h (row := 196) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_14
  simp only [← c12712, k3, ← c12713, k3, ← c12714] at e_196_14
  have e_196_15 := arithEq_of_rows h (row := 196) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_15
  simp only [← c12715, ← c12716, k3, ← c12717] at e_196_15
  have e_196_16 := arithEq_of_rows h (row := 196) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_16
  simp only [← c12724, ← c12725, k3, ← c12726] at e_196_16
  refine ⟨?_, ?_⟩
  · have hc := e_195_14
    simp only [e_196_15] at hc
    linear_combination c12727.trans k2 - hc
  · have hc := e_196_16
    simp only [e_196_14, e_195_15, e_196_15] at hc
    linear_combination c12728.trans k2 - hc

theorem publicBatchWrapper89_f1895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2608)) (a (.wire 170 7)) (a (.virt 18685)) (a (.virt 18686)) := by
  have c12729 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12730 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12731 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12732 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12733 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12734 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12735 := (publicBatchWrapper89_copies397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12736 := (publicBatchWrapper89_copies398 a h).1
  have c12737 := (publicBatchWrapper89_copies398 a h).2.1
  have c12738 := (publicBatchWrapper89_copies398 a h).2.2.1
  have c12739 := (publicBatchWrapper89_copies398 a h).2.2.2.1
  have c12740 := (publicBatchWrapper89_copies398 a h).2.2.2.2.1
  have c12741 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.1
  have c12742 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.1
  have c12743 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.1
  have c12744 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.1
  have c12745 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_195_16 := arithEq_of_rows h (row := 195) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_16
  simp only [← c12735, ← c12736, ← c12737] at e_195_16
  have e_195_17 := arithEq_of_rows h (row := 195) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_17
  simp only [← c12738, ← c12739, ← c12740] at e_195_17
  have e_196_17 := arithEq_of_rows h (row := 196) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_17
  simp only [← c12729, k3, ← c12730, k3, ← c12731] at e_196_17
  have e_196_18 := arithEq_of_rows h (row := 196) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_18
  simp only [← c12732, ← c12733, k3, ← c12734] at e_196_18
  have e_196_19 := arithEq_of_rows h (row := 196) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_196_19
  simp only [← c12741, ← c12742, k3, ← c12743] at e_196_19
  refine ⟨?_, ?_⟩
  · have hc := e_195_16
    simp only [e_196_18] at hc
    linear_combination c12744.trans k2 - hc
  · have hc := e_196_19
    simp only [e_196_17, e_195_17, e_196_18] at hc
    linear_combination c12745.trans k2 - hc

theorem publicBatchWrapper89_f1896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 195 75) = band (a (.virt 18679)) (a (.virt 18681)) := by
  have c12746 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12747 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12748 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_195_18 := arithEq_of_rows h (row := 195) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_18
  simp only [← c12746, ← c12747, ← c12748] at e_195_18
  have hr := e_195_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 195 79) = band (a (.virt 18683)) (a (.virt 18685)) := by
  have c12749 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12750 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12751 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_195_19 := arithEq_of_rows h (row := 195) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_195_19
  simp only [← c12749, ← c12750, ← c12751] at e_195_19
  have hr := e_195_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 198 3) = band (a (.wire 195 75)) (a (.wire 195 79)) := by
  have c12752 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12753 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12754 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_198_0 := arithEq_of_rows h (row := 198) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_0
  simp only [← c12752, ← c12753, ← c12754] at e_198_0
  have hr := e_198_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 7) = bor (a (.wire 14 71)) (a (.wire 198 3)) := by
  have c12755 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12756 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12757 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12758 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12759 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12760 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_9 := arithEq_of_rows h (row := 192) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_9
  simp only [← c12755, ← c12756, ← c12757] at e_192_9
  have e_197_1 := arithEq_of_rows h (row := 197) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_1
  simp only [← c12758, ← c12759, k3, ← c12760] at e_197_1
  have hr := e_197_1
  simp only [e_192_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 7) = a (.virt 201) := by
  have c12761 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12761

theorem publicBatchWrapper89_f1901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2803)) (a (.wire 170 23)) (a (.virt 18687)) (a (.virt 18688)) := by
  have c12762 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12763 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12764 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12765 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12766 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12767 := (publicBatchWrapper89_copies398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12768 := (publicBatchWrapper89_copies399 a h).1
  have c12769 := (publicBatchWrapper89_copies399 a h).2.1
  have c12770 := (publicBatchWrapper89_copies399 a h).2.2.1
  have c12771 := (publicBatchWrapper89_copies399 a h).2.2.2.1
  have c12772 := (publicBatchWrapper89_copies399 a h).2.2.2.2.1
  have c12773 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.1
  have c12774 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.1
  have c12775 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.1
  have c12776 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.1
  have c12777 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.1
  have c12778 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_198_1 := arithEq_of_rows h (row := 198) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_1
  simp only [← c12768, ← c12769, ← c12770] at e_198_1
  have e_198_2 := arithEq_of_rows h (row := 198) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_2
  simp only [← c12771, ← c12772, ← c12773] at e_198_2
  have e_199_0 := arithEq_of_rows h (row := 199) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_0
  simp only [← c12762, k3, ← c12763, k3, ← c12764] at e_199_0
  have e_199_1 := arithEq_of_rows h (row := 199) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_1
  simp only [← c12765, ← c12766, k3, ← c12767] at e_199_1
  have e_199_2 := arithEq_of_rows h (row := 199) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_2
  simp only [← c12774, ← c12775, k3, ← c12776] at e_199_2
  refine ⟨?_, ?_⟩
  · have hc := e_198_1
    simp only [e_199_1] at hc
    linear_combination c12777.trans k2 - hc
  · have hc := e_199_2
    simp only [e_199_0, e_198_2, e_199_1] at hc
    linear_combination c12778.trans k2 - hc

theorem publicBatchWrapper89_f1902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 11) = bor (a (.wire 15 35)) (a (.virt 18687)) := by
  have c12779 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12780 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12781 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12782 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12783 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12784 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_10 := arithEq_of_rows h (row := 192) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_10
  simp only [← c12779, ← c12780, ← c12781] at e_192_10
  have e_197_2 := arithEq_of_rows h (row := 197) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_2
  simp only [← c12782, ← c12783, k3, ← c12784] at e_197_2
  have hr := e_197_2
  simp only [e_192_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 11) = a (.virt 201) := by
  have c12785 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12785

theorem publicBatchWrapper89_f1904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2804)) (a (.wire 170 31)) (a (.virt 18689)) (a (.virt 18690)) := by
  have c12786 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12787 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12788 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12789 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12790 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12791 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12792 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12793 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12794 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12795 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12796 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12797 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12798 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12799 := (publicBatchWrapper89_copies399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12800 := (publicBatchWrapper89_copies400 a h).1
  have c12801 := (publicBatchWrapper89_copies400 a h).2.1
  have c12802 := (publicBatchWrapper89_copies400 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_198_3 := arithEq_of_rows h (row := 198) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_3
  simp only [← c12792, ← c12793, ← c12794] at e_198_3
  have e_198_4 := arithEq_of_rows h (row := 198) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_4
  simp only [← c12795, ← c12796, ← c12797] at e_198_4
  have e_199_3 := arithEq_of_rows h (row := 199) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_3
  simp only [← c12786, k3, ← c12787, k3, ← c12788] at e_199_3
  have e_199_4 := arithEq_of_rows h (row := 199) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_4
  simp only [← c12789, ← c12790, k3, ← c12791] at e_199_4
  have e_199_5 := arithEq_of_rows h (row := 199) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_5
  simp only [← c12798, ← c12799, k3, ← c12800] at e_199_5
  refine ⟨?_, ?_⟩
  · have hc := e_198_3
    simp only [e_199_4] at hc
    linear_combination c12801.trans k2 - hc
  · have hc := e_199_5
    simp only [e_199_3, e_198_4, e_199_4] at hc
    linear_combination c12802.trans k2 - hc

theorem publicBatchWrapper89_f1905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 15) = bor (a (.wire 15 35)) (a (.virt 18689)) := by
  have c12803 := (publicBatchWrapper89_copies400 a h).2.2.2.1
  have c12804 := (publicBatchWrapper89_copies400 a h).2.2.2.2.1
  have c12805 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.1
  have c12806 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.1
  have c12807 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.1
  have c12808 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_11 := arithEq_of_rows h (row := 192) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_11
  simp only [← c12803, ← c12804, ← c12805] at e_192_11
  have e_197_3 := arithEq_of_rows h (row := 197) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_3
  simp only [← c12806, ← c12807, k3, ← c12808] at e_197_3
  have hr := e_197_3
  simp only [e_192_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 15) = a (.virt 201) := by
  have c12809 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.1
  exact c12809

theorem publicBatchWrapper89_f1907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2805)) (a (.wire 169 63)) (a (.virt 18691)) (a (.virt 18692)) := by
  have c12810 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12811 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12812 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12813 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12814 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12815 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12816 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12817 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12818 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12819 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12820 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12821 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12822 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12823 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12824 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12825 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12826 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_198_5 := arithEq_of_rows h (row := 198) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_5
  simp only [← c12816, ← c12817, ← c12818] at e_198_5
  have e_198_6 := arithEq_of_rows h (row := 198) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_6
  simp only [← c12819, ← c12820, ← c12821] at e_198_6
  have e_199_6 := arithEq_of_rows h (row := 199) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_6
  simp only [← c12810, k3, ← c12811, k3, ← c12812] at e_199_6
  have e_199_7 := arithEq_of_rows h (row := 199) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_7
  simp only [← c12813, ← c12814, k3, ← c12815] at e_199_7
  have e_199_8 := arithEq_of_rows h (row := 199) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_8
  simp only [← c12822, ← c12823, k3, ← c12824] at e_199_8
  refine ⟨?_, ?_⟩
  · have hc := e_198_5
    simp only [e_199_7] at hc
    linear_combination c12825.trans k2 - hc
  · have hc := e_199_8
    simp only [e_199_6, e_198_6, e_199_7] at hc
    linear_combination c12826.trans k2 - hc

theorem publicBatchWrapper89_f1908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2806)) (a (.wire 169 71)) (a (.virt 18693)) (a (.virt 18694)) := by
  have c12827 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12828 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12829 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12830 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12831 := (publicBatchWrapper89_copies400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12832 := (publicBatchWrapper89_copies401 a h).1
  have c12833 := (publicBatchWrapper89_copies401 a h).2.1
  have c12834 := (publicBatchWrapper89_copies401 a h).2.2.1
  have c12835 := (publicBatchWrapper89_copies401 a h).2.2.2.1
  have c12836 := (publicBatchWrapper89_copies401 a h).2.2.2.2.1
  have c12837 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.1
  have c12838 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.1
  have c12839 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.1
  have c12840 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.1
  have c12841 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.1
  have c12842 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12843 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_198_7 := arithEq_of_rows h (row := 198) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_7
  simp only [← c12833, ← c12834, ← c12835] at e_198_7
  have e_198_8 := arithEq_of_rows h (row := 198) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_8
  simp only [← c12836, ← c12837, ← c12838] at e_198_8
  have e_199_9 := arithEq_of_rows h (row := 199) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_9
  simp only [← c12827, k3, ← c12828, k3, ← c12829] at e_199_9
  have e_199_10 := arithEq_of_rows h (row := 199) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_10
  simp only [← c12830, ← c12831, k3, ← c12832] at e_199_10
  have e_199_11 := arithEq_of_rows h (row := 199) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_11
  simp only [← c12839, ← c12840, k3, ← c12841] at e_199_11
  refine ⟨?_, ?_⟩
  · have hc := e_198_7
    simp only [e_199_10] at hc
    linear_combination c12842.trans k2 - hc
  · have hc := e_199_11
    simp only [e_199_9, e_198_8, e_199_10] at hc
    linear_combination c12843.trans k2 - hc

theorem publicBatchWrapper89_f1909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2807)) (a (.wire 169 79)) (a (.virt 18695)) (a (.virt 18696)) := by
  have c12844 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12845 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12846 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12847 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12848 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12849 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12850 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12851 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12852 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12853 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12854 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12855 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12856 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12857 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12858 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12859 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12860 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_198_9 := arithEq_of_rows h (row := 198) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_9
  simp only [← c12850, ← c12851, ← c12852] at e_198_9
  have e_198_10 := arithEq_of_rows h (row := 198) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_10
  simp only [← c12853, ← c12854, ← c12855] at e_198_10
  have e_199_12 := arithEq_of_rows h (row := 199) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_12
  simp only [← c12844, k3, ← c12845, k3, ← c12846] at e_199_12
  have e_199_13 := arithEq_of_rows h (row := 199) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_13
  simp only [← c12847, ← c12848, k3, ← c12849] at e_199_13
  have e_199_14 := arithEq_of_rows h (row := 199) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_14
  simp only [← c12856, ← c12857, k3, ← c12858] at e_199_14
  refine ⟨?_, ?_⟩
  · have hc := e_198_9
    simp only [e_199_13] at hc
    linear_combination c12859.trans k2 - hc
  · have hc := e_199_14
    simp only [e_199_12, e_198_10, e_199_13] at hc
    linear_combination c12860.trans k2 - hc

theorem publicBatchWrapper89_f1910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2808)) (a (.wire 170 7)) (a (.virt 18697)) (a (.virt 18698)) := by
  have c12861 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12862 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12863 := (publicBatchWrapper89_copies401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12864 := (publicBatchWrapper89_copies402 a h).1
  have c12865 := (publicBatchWrapper89_copies402 a h).2.1
  have c12866 := (publicBatchWrapper89_copies402 a h).2.2.1
  have c12867 := (publicBatchWrapper89_copies402 a h).2.2.2.1
  have c12868 := (publicBatchWrapper89_copies402 a h).2.2.2.2.1
  have c12869 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.1
  have c12870 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.1
  have c12871 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.1
  have c12872 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.1
  have c12873 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.1
  have c12874 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12875 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12876 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12877 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_198_11 := arithEq_of_rows h (row := 198) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_11
  simp only [← c12867, ← c12868, ← c12869] at e_198_11
  have e_198_12 := arithEq_of_rows h (row := 198) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_12
  simp only [← c12870, ← c12871, ← c12872] at e_198_12
  have e_199_15 := arithEq_of_rows h (row := 199) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_15
  simp only [← c12861, k3, ← c12862, k3, ← c12863] at e_199_15
  have e_199_16 := arithEq_of_rows h (row := 199) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_16
  simp only [← c12864, ← c12865, k3, ← c12866] at e_199_16
  have e_199_17 := arithEq_of_rows h (row := 199) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_17
  simp only [← c12873, ← c12874, k3, ← c12875] at e_199_17
  refine ⟨?_, ?_⟩
  · have hc := e_198_11
    simp only [e_199_16] at hc
    linear_combination c12876.trans k2 - hc
  · have hc := e_199_17
    simp only [e_199_15, e_198_12, e_199_16] at hc
    linear_combination c12877.trans k2 - hc

theorem publicBatchWrapper89_f1911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 198 55) = band (a (.virt 18691)) (a (.virt 18693)) := by
  have c12878 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12879 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12880 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_198_13 := arithEq_of_rows h (row := 198) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_13
  simp only [← c12878, ← c12879, ← c12880] at e_198_13
  have hr := e_198_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 198 59) = band (a (.virt 18695)) (a (.virt 18697)) := by
  have c12881 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12882 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12883 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_198_14 := arithEq_of_rows h (row := 198) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_14
  simp only [← c12881, ← c12882, ← c12883] at e_198_14
  have hr := e_198_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 198 63) = band (a (.wire 198 55)) (a (.wire 198 59)) := by
  have c12884 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12885 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12886 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_198_15 := arithEq_of_rows h (row := 198) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_15
  simp only [← c12884, ← c12885, ← c12886] at e_198_15
  have hr := e_198_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 19) = bor (a (.wire 15 35)) (a (.wire 198 63)) := by
  have c12887 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12888 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12889 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12890 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12891 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12892 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_12 := arithEq_of_rows h (row := 192) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_12
  simp only [← c12887, ← c12888, ← c12889] at e_192_12
  have e_197_4 := arithEq_of_rows h (row := 197) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_4
  simp only [← c12890, ← c12891, k3, ← c12892] at e_197_4
  have hr := e_197_4
  simp only [e_192_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 19) = a (.virt 201) := by
  have c12893 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12893

theorem publicBatchWrapper89_f1916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3003)) (a (.wire 170 23)) (a (.virt 18699)) (a (.virt 18700)) := by
  have c12894 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12895 := (publicBatchWrapper89_copies402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12896 := (publicBatchWrapper89_copies403 a h).1
  have c12897 := (publicBatchWrapper89_copies403 a h).2.1
  have c12898 := (publicBatchWrapper89_copies403 a h).2.2.1
  have c12899 := (publicBatchWrapper89_copies403 a h).2.2.2.1
  have c12900 := (publicBatchWrapper89_copies403 a h).2.2.2.2.1
  have c12901 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.1
  have c12902 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.1
  have c12903 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.1
  have c12904 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.1
  have c12905 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.1
  have c12906 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12907 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12908 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12909 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12910 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_198_16 := arithEq_of_rows h (row := 198) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_16
  simp only [← c12900, ← c12901, ← c12902] at e_198_16
  have e_198_17 := arithEq_of_rows h (row := 198) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_17
  simp only [← c12903, ← c12904, ← c12905] at e_198_17
  have e_199_18 := arithEq_of_rows h (row := 199) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_18
  simp only [← c12894, k3, ← c12895, k3, ← c12896] at e_199_18
  have e_199_19 := arithEq_of_rows h (row := 199) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_199_19
  simp only [← c12897, ← c12898, k3, ← c12899] at e_199_19
  have e_200_0 := arithEq_of_rows h (row := 200) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_0
  simp only [← c12906, ← c12907, k3, ← c12908] at e_200_0
  refine ⟨?_, ?_⟩
  · have hc := e_198_16
    simp only [e_199_19] at hc
    linear_combination c12909.trans k2 - hc
  · have hc := e_200_0
    simp only [e_199_18, e_198_17, e_199_19] at hc
    linear_combination c12910.trans k2 - hc

theorem publicBatchWrapper89_f1917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 23) = bor (a (.wire 15 79)) (a (.virt 18699)) := by
  have c12911 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12912 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12913 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12914 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12915 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12916 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_13 := arithEq_of_rows h (row := 192) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_13
  simp only [← c12911, ← c12912, ← c12913] at e_192_13
  have e_197_5 := arithEq_of_rows h (row := 197) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_5
  simp only [← c12914, ← c12915, k3, ← c12916] at e_197_5
  have hr := e_197_5
  simp only [e_192_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 23) = a (.virt 201) := by
  have c12917 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12917

theorem publicBatchWrapper89_f1919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3004)) (a (.wire 170 31)) (a (.virt 18701)) (a (.virt 18702)) := by
  have c12918 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12919 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12920 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12921 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12922 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12923 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12924 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12925 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12926 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12927 := (publicBatchWrapper89_copies403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12928 := (publicBatchWrapper89_copies404 a h).1
  have c12929 := (publicBatchWrapper89_copies404 a h).2.1
  have c12930 := (publicBatchWrapper89_copies404 a h).2.2.1
  have c12931 := (publicBatchWrapper89_copies404 a h).2.2.2.1
  have c12932 := (publicBatchWrapper89_copies404 a h).2.2.2.2.1
  have c12933 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.1
  have c12934 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_198_18 := arithEq_of_rows h (row := 198) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_18
  simp only [← c12924, ← c12925, ← c12926] at e_198_18
  have e_198_19 := arithEq_of_rows h (row := 198) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_198_19
  simp only [← c12927, ← c12928, ← c12929] at e_198_19
  have e_200_1 := arithEq_of_rows h (row := 200) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_1
  simp only [← c12918, k3, ← c12919, k3, ← c12920] at e_200_1
  have e_200_2 := arithEq_of_rows h (row := 200) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_2
  simp only [← c12921, ← c12922, k3, ← c12923] at e_200_2
  have e_200_3 := arithEq_of_rows h (row := 200) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_3
  simp only [← c12930, ← c12931, k3, ← c12932] at e_200_3
  refine ⟨?_, ?_⟩
  · have hc := e_198_18
    simp only [e_200_2] at hc
    linear_combination c12933.trans k2 - hc
  · have hc := e_200_3
    simp only [e_200_1, e_198_19, e_200_2] at hc
    linear_combination c12934.trans k2 - hc

theorem publicBatchWrapper89_f1920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 27) = bor (a (.wire 15 79)) (a (.virt 18701)) := by
  have c12935 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.1
  have c12936 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.1
  have c12937 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.1
  have c12938 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12939 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12940 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_14 := arithEq_of_rows h (row := 192) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_14
  simp only [← c12935, ← c12936, ← c12937] at e_192_14
  have e_197_6 := arithEq_of_rows h (row := 197) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_6
  simp only [← c12938, ← c12939, k3, ← c12940] at e_197_6
  have hr := e_197_6
  simp only [e_192_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 27) = a (.virt 201) := by
  have c12941 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12941

theorem publicBatchWrapper89_f1922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3005)) (a (.wire 169 63)) (a (.virt 18703)) (a (.virt 18704)) := by
  have c12942 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12943 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12944 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12945 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12946 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12947 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12948 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12949 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12950 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12951 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12952 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12953 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12954 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12955 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12956 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12957 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12958 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_200_4 := arithEq_of_rows h (row := 200) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_4
  simp only [← c12942, k3, ← c12943, k3, ← c12944] at e_200_4
  have e_200_5 := arithEq_of_rows h (row := 200) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_5
  simp only [← c12945, ← c12946, k3, ← c12947] at e_200_5
  have e_200_6 := arithEq_of_rows h (row := 200) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_6
  simp only [← c12954, ← c12955, k3, ← c12956] at e_200_6
  have e_201_0 := arithEq_of_rows h (row := 201) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_0
  simp only [← c12948, ← c12949, ← c12950] at e_201_0
  have e_201_1 := arithEq_of_rows h (row := 201) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_1
  simp only [← c12951, ← c12952, ← c12953] at e_201_1
  refine ⟨?_, ?_⟩
  · have hc := e_201_0
    simp only [e_200_5] at hc
    linear_combination c12957.trans k2 - hc
  · have hc := e_200_6
    simp only [e_200_4, e_201_1, e_200_5] at hc
    linear_combination c12958.trans k2 - hc

theorem publicBatchWrapper89_f1923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3006)) (a (.wire 169 71)) (a (.virt 18705)) (a (.virt 18706)) := by
  have c12959 := (publicBatchWrapper89_copies404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12960 := (publicBatchWrapper89_copies405 a h).1
  have c12961 := (publicBatchWrapper89_copies405 a h).2.1
  have c12962 := (publicBatchWrapper89_copies405 a h).2.2.1
  have c12963 := (publicBatchWrapper89_copies405 a h).2.2.2.1
  have c12964 := (publicBatchWrapper89_copies405 a h).2.2.2.2.1
  have c12965 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.1
  have c12966 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.1
  have c12967 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.1
  have c12968 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.1
  have c12969 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.1
  have c12970 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.1
  have c12971 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12972 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12973 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12974 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12975 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_200_7 := arithEq_of_rows h (row := 200) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_7
  simp only [← c12959, k3, ← c12960, k3, ← c12961] at e_200_7
  have e_200_8 := arithEq_of_rows h (row := 200) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_8
  simp only [← c12962, ← c12963, k3, ← c12964] at e_200_8
  have e_200_9 := arithEq_of_rows h (row := 200) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_9
  simp only [← c12971, ← c12972, k3, ← c12973] at e_200_9
  have e_201_2 := arithEq_of_rows h (row := 201) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_2
  simp only [← c12965, ← c12966, ← c12967] at e_201_2
  have e_201_3 := arithEq_of_rows h (row := 201) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_3
  simp only [← c12968, ← c12969, ← c12970] at e_201_3
  refine ⟨?_, ?_⟩
  · have hc := e_201_2
    simp only [e_200_8] at hc
    linear_combination c12974.trans k2 - hc
  · have hc := e_200_9
    simp only [e_200_7, e_201_3, e_200_8] at hc
    linear_combination c12975.trans k2 - hc

theorem publicBatchWrapper89_f1924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3007)) (a (.wire 169 79)) (a (.virt 18707)) (a (.virt 18708)) := by
  have c12976 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12977 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12978 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12979 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12980 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12981 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12982 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12983 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12984 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12985 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12986 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12987 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12988 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12989 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12990 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c12991 := (publicBatchWrapper89_copies405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c12992 := (publicBatchWrapper89_copies406 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_200_10 := arithEq_of_rows h (row := 200) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_10
  simp only [← c12976, k3, ← c12977, k3, ← c12978] at e_200_10
  have e_200_11 := arithEq_of_rows h (row := 200) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_11
  simp only [← c12979, ← c12980, k3, ← c12981] at e_200_11
  have e_200_12 := arithEq_of_rows h (row := 200) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_12
  simp only [← c12988, ← c12989, k3, ← c12990] at e_200_12
  have e_201_4 := arithEq_of_rows h (row := 201) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_4
  simp only [← c12982, ← c12983, ← c12984] at e_201_4
  have e_201_5 := arithEq_of_rows h (row := 201) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_5
  simp only [← c12985, ← c12986, ← c12987] at e_201_5
  refine ⟨?_, ?_⟩
  · have hc := e_201_4
    simp only [e_200_11] at hc
    linear_combination c12991.trans k2 - hc
  · have hc := e_200_12
    simp only [e_200_10, e_201_5, e_200_11] at hc
    linear_combination c12992.trans k2 - hc

theorem publicBatchWrapper89_f1925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3008)) (a (.wire 170 7)) (a (.virt 18709)) (a (.virt 18710)) := by
  have c12993 := (publicBatchWrapper89_copies406 a h).2.1
  have c12994 := (publicBatchWrapper89_copies406 a h).2.2.1
  have c12995 := (publicBatchWrapper89_copies406 a h).2.2.2.1
  have c12996 := (publicBatchWrapper89_copies406 a h).2.2.2.2.1
  have c12997 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.1
  have c12998 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.1
  have c12999 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.1
  have c13000 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.1
  have c13001 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.1
  have c13002 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13003 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13004 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13005 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13006 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13007 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13008 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13009 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_200_13 := arithEq_of_rows h (row := 200) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_13
  simp only [← c12993, k3, ← c12994, k3, ← c12995] at e_200_13
  have e_200_14 := arithEq_of_rows h (row := 200) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_14
  simp only [← c12996, ← c12997, k3, ← c12998] at e_200_14
  have e_200_15 := arithEq_of_rows h (row := 200) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_15
  simp only [← c13005, ← c13006, k3, ← c13007] at e_200_15
  have e_201_6 := arithEq_of_rows h (row := 201) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_6
  simp only [← c12999, ← c13000, ← c13001] at e_201_6
  have e_201_7 := arithEq_of_rows h (row := 201) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_7
  simp only [← c13002, ← c13003, ← c13004] at e_201_7
  refine ⟨?_, ?_⟩
  · have hc := e_201_6
    simp only [e_200_14] at hc
    linear_combination c13008.trans k2 - hc
  · have hc := e_200_15
    simp only [e_200_13, e_201_7, e_200_14] at hc
    linear_combination c13009.trans k2 - hc

theorem publicBatchWrapper89_f1926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 201 35) = band (a (.virt 18703)) (a (.virt 18705)) := by
  have c13010 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13011 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13012 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_8 := arithEq_of_rows h (row := 201) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_8
  simp only [← c13010, ← c13011, ← c13012] at e_201_8
  have hr := e_201_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 201 39) = band (a (.virt 18707)) (a (.virt 18709)) := by
  have c13013 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13014 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13015 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_9 := arithEq_of_rows h (row := 201) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_9
  simp only [← c13013, ← c13014, ← c13015] at e_201_9
  have hr := e_201_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 201 43) = band (a (.wire 201 35)) (a (.wire 201 39)) := by
  have c13016 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13017 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13018 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_201_10 := arithEq_of_rows h (row := 201) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_10
  simp only [← c13016, ← c13017, ← c13018] at e_201_10
  have hr := e_201_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 31) = bor (a (.wire 15 79)) (a (.wire 201 43)) := by
  have c13019 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13020 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13021 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13022 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13023 := (publicBatchWrapper89_copies406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13024 := (publicBatchWrapper89_copies407 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_15 := arithEq_of_rows h (row := 192) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_15
  simp only [← c13019, ← c13020, ← c13021] at e_192_15
  have e_197_7 := arithEq_of_rows h (row := 197) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_7
  simp only [← c13022, ← c13023, k3, ← c13024] at e_197_7
  have hr := e_197_7
  simp only [e_192_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 31) = a (.virt 201) := by
  have c13025 := (publicBatchWrapper89_copies407 a h).2.1
  exact c13025

theorem publicBatchWrapper89_f1931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3203)) (a (.wire 170 23)) (a (.virt 18711)) (a (.virt 18712)) := by
  have c13026 := (publicBatchWrapper89_copies407 a h).2.2.1
  have c13027 := (publicBatchWrapper89_copies407 a h).2.2.2.1
  have c13028 := (publicBatchWrapper89_copies407 a h).2.2.2.2.1
  have c13029 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.1
  have c13030 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.1
  have c13031 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.1
  have c13032 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.1
  have c13033 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.1
  have c13034 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13035 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13036 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13037 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13038 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13039 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13040 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13041 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13042 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_200_16 := arithEq_of_rows h (row := 200) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_16
  simp only [← c13026, k3, ← c13027, k3, ← c13028] at e_200_16
  have e_200_17 := arithEq_of_rows h (row := 200) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_17
  simp only [← c13029, ← c13030, k3, ← c13031] at e_200_17
  have e_200_18 := arithEq_of_rows h (row := 200) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_18
  simp only [← c13038, ← c13039, k3, ← c13040] at e_200_18
  have e_201_11 := arithEq_of_rows h (row := 201) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_11
  simp only [← c13032, ← c13033, ← c13034] at e_201_11
  have e_201_12 := arithEq_of_rows h (row := 201) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_12
  simp only [← c13035, ← c13036, ← c13037] at e_201_12
  refine ⟨?_, ?_⟩
  · have hc := e_201_11
    simp only [e_200_17] at hc
    linear_combination c13041.trans k2 - hc
  · have hc := e_200_18
    simp only [e_200_16, e_201_12, e_200_17] at hc
    linear_combination c13042.trans k2 - hc

theorem publicBatchWrapper89_f1932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 35) = bor (a (.wire 17 43)) (a (.virt 18711)) := by
  have c13043 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13044 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13045 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13046 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13047 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13048 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_16 := arithEq_of_rows h (row := 192) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_16
  simp only [← c13043, ← c13044, ← c13045] at e_192_16
  have e_197_8 := arithEq_of_rows h (row := 197) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_8
  simp only [← c13046, ← c13047, k3, ← c13048] at e_197_8
  have hr := e_197_8
  simp only [e_192_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 35) = a (.virt 201) := by
  have c13049 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13049

theorem publicBatchWrapper89_f1934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3204)) (a (.wire 170 31)) (a (.virt 18713)) (a (.virt 18714)) := by
  have c13050 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13051 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13052 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13053 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13054 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13055 := (publicBatchWrapper89_copies407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13056 := (publicBatchWrapper89_copies408 a h).1
  have c13057 := (publicBatchWrapper89_copies408 a h).2.1
  have c13058 := (publicBatchWrapper89_copies408 a h).2.2.1
  have c13059 := (publicBatchWrapper89_copies408 a h).2.2.2.1
  have c13060 := (publicBatchWrapper89_copies408 a h).2.2.2.2.1
  have c13061 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.1
  have c13062 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.1
  have c13063 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.1
  have c13064 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.1
  have c13065 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.1
  have c13066 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_200_19 := arithEq_of_rows h (row := 200) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_200_19
  simp only [← c13050, k3, ← c13051, k3, ← c13052] at e_200_19
  have e_201_13 := arithEq_of_rows h (row := 201) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_13
  simp only [← c13056, ← c13057, ← c13058] at e_201_13
  have e_201_14 := arithEq_of_rows h (row := 201) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_14
  simp only [← c13059, ← c13060, ← c13061] at e_201_14
  have e_202_0 := arithEq_of_rows h (row := 202) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_0
  simp only [← c13053, ← c13054, k3, ← c13055] at e_202_0
  have e_202_1 := arithEq_of_rows h (row := 202) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_1
  simp only [← c13062, ← c13063, k3, ← c13064] at e_202_1
  refine ⟨?_, ?_⟩
  · have hc := e_201_13
    simp only [e_202_0] at hc
    linear_combination c13065.trans k2 - hc
  · have hc := e_202_1
    simp only [e_200_19, e_201_14, e_202_0] at hc
    linear_combination c13066.trans k2 - hc

theorem publicBatchWrapper89_f1935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 39) = bor (a (.wire 17 43)) (a (.virt 18713)) := by
  have c13067 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13068 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13069 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13070 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13071 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13072 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_17 := arithEq_of_rows h (row := 192) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_17
  simp only [← c13067, ← c13068, ← c13069] at e_192_17
  have e_197_9 := arithEq_of_rows h (row := 197) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_9
  simp only [← c13070, ← c13071, k3, ← c13072] at e_197_9
  have hr := e_197_9
  simp only [e_192_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 39) = a (.virt 201) := by
  have c13073 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13073

theorem publicBatchWrapper89_f1937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3205)) (a (.wire 169 63)) (a (.virt 18715)) (a (.virt 18716)) := by
  have c13074 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13075 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13076 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13077 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13078 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13079 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13080 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13081 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13082 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13083 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13084 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13085 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13086 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13087 := (publicBatchWrapper89_copies408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13088 := (publicBatchWrapper89_copies409 a h).1
  have c13089 := (publicBatchWrapper89_copies409 a h).2.1
  have c13090 := (publicBatchWrapper89_copies409 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_201_15 := arithEq_of_rows h (row := 201) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_15
  simp only [← c13080, ← c13081, ← c13082] at e_201_15
  have e_201_16 := arithEq_of_rows h (row := 201) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_16
  simp only [← c13083, ← c13084, ← c13085] at e_201_16
  have e_202_2 := arithEq_of_rows h (row := 202) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_2
  simp only [← c13074, k3, ← c13075, k3, ← c13076] at e_202_2
  have e_202_3 := arithEq_of_rows h (row := 202) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_3
  simp only [← c13077, ← c13078, k3, ← c13079] at e_202_3
  have e_202_4 := arithEq_of_rows h (row := 202) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_4
  simp only [← c13086, ← c13087, k3, ← c13088] at e_202_4
  refine ⟨?_, ?_⟩
  · have hc := e_201_15
    simp only [e_202_3] at hc
    linear_combination c13089.trans k2 - hc
  · have hc := e_202_4
    simp only [e_202_2, e_201_16, e_202_3] at hc
    linear_combination c13090.trans k2 - hc

theorem publicBatchWrapper89_f1938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3206)) (a (.wire 169 71)) (a (.virt 18717)) (a (.virt 18718)) := by
  have c13091 := (publicBatchWrapper89_copies409 a h).2.2.2.1
  have c13092 := (publicBatchWrapper89_copies409 a h).2.2.2.2.1
  have c13093 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.1
  have c13094 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.1
  have c13095 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.1
  have c13096 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.1
  have c13097 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.1
  have c13098 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13099 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13100 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13101 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13102 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13103 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13104 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13105 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13106 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13107 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_201_17 := arithEq_of_rows h (row := 201) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_17
  simp only [← c13097, ← c13098, ← c13099] at e_201_17
  have e_201_18 := arithEq_of_rows h (row := 201) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_18
  simp only [← c13100, ← c13101, ← c13102] at e_201_18
  have e_202_5 := arithEq_of_rows h (row := 202) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_5
  simp only [← c13091, k3, ← c13092, k3, ← c13093] at e_202_5
  have e_202_6 := arithEq_of_rows h (row := 202) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_6
  simp only [← c13094, ← c13095, k3, ← c13096] at e_202_6
  have e_202_7 := arithEq_of_rows h (row := 202) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_7
  simp only [← c13103, ← c13104, k3, ← c13105] at e_202_7
  refine ⟨?_, ?_⟩
  · have hc := e_201_17
    simp only [e_202_6] at hc
    linear_combination c13106.trans k2 - hc
  · have hc := e_202_7
    simp only [e_202_5, e_201_18, e_202_6] at hc
    linear_combination c13107.trans k2 - hc

theorem publicBatchWrapper89_f1939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3207)) (a (.wire 169 79)) (a (.virt 18719)) (a (.virt 18720)) := by
  have c13108 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13109 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13110 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13111 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13112 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13113 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13114 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13115 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13116 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13117 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13118 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13119 := (publicBatchWrapper89_copies409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13120 := (publicBatchWrapper89_copies410 a h).1
  have c13121 := (publicBatchWrapper89_copies410 a h).2.1
  have c13122 := (publicBatchWrapper89_copies410 a h).2.2.1
  have c13123 := (publicBatchWrapper89_copies410 a h).2.2.2.1
  have c13124 := (publicBatchWrapper89_copies410 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_201_19 := arithEq_of_rows h (row := 201) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_201_19
  simp only [← c13114, ← c13115, ← c13116] at e_201_19
  have e_202_8 := arithEq_of_rows h (row := 202) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_8
  simp only [← c13108, k3, ← c13109, k3, ← c13110] at e_202_8
  have e_202_9 := arithEq_of_rows h (row := 202) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_9
  simp only [← c13111, ← c13112, k3, ← c13113] at e_202_9
  have e_202_10 := arithEq_of_rows h (row := 202) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_10
  simp only [← c13120, ← c13121, k3, ← c13122] at e_202_10
  have e_203_0 := arithEq_of_rows h (row := 203) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_0
  simp only [← c13117, ← c13118, ← c13119] at e_203_0
  refine ⟨?_, ?_⟩
  · have hc := e_201_19
    simp only [e_202_9] at hc
    linear_combination c13123.trans k2 - hc
  · have hc := e_202_10
    simp only [e_202_8, e_203_0, e_202_9] at hc
    linear_combination c13124.trans k2 - hc

theorem publicBatchWrapper89_f1940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3208)) (a (.wire 170 7)) (a (.virt 18721)) (a (.virt 18722)) := by
  have c13125 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.1
  have c13126 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.1
  have c13127 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.1
  have c13128 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.1
  have c13129 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.1
  have c13130 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13131 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13132 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13133 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13134 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13135 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13136 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13137 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13138 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13139 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13140 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13141 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_202_11 := arithEq_of_rows h (row := 202) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_11
  simp only [← c13125, k3, ← c13126, k3, ← c13127] at e_202_11
  have e_202_12 := arithEq_of_rows h (row := 202) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_12
  simp only [← c13128, ← c13129, k3, ← c13130] at e_202_12
  have e_202_13 := arithEq_of_rows h (row := 202) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_13
  simp only [← c13137, ← c13138, k3, ← c13139] at e_202_13
  have e_203_1 := arithEq_of_rows h (row := 203) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_1
  simp only [← c13131, ← c13132, ← c13133] at e_203_1
  have e_203_2 := arithEq_of_rows h (row := 203) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_2
  simp only [← c13134, ← c13135, ← c13136] at e_203_2
  refine ⟨?_, ?_⟩
  · have hc := e_203_1
    simp only [e_202_12] at hc
    linear_combination c13140.trans k2 - hc
  · have hc := e_202_13
    simp only [e_202_11, e_203_2, e_202_12] at hc
    linear_combination c13141.trans k2 - hc

theorem publicBatchWrapper89_f1941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 203 15) = band (a (.virt 18715)) (a (.virt 18717)) := by
  have c13142 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13143 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13144 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_203_3 := arithEq_of_rows h (row := 203) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_3
  simp only [← c13142, ← c13143, ← c13144] at e_203_3
  have hr := e_203_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 203 19) = band (a (.virt 18719)) (a (.virt 18721)) := by
  have c13145 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13146 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13147 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_203_4 := arithEq_of_rows h (row := 203) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_4
  simp only [← c13145, ← c13146, ← c13147] at e_203_4
  have hr := e_203_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 203 23) = band (a (.wire 203 15)) (a (.wire 203 19)) := by
  have c13148 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13149 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13150 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_203_5 := arithEq_of_rows h (row := 203) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_5
  simp only [← c13148, ← c13149, ← c13150] at e_203_5
  have hr := e_203_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 43) = bor (a (.wire 17 43)) (a (.wire 203 23)) := by
  have c13151 := (publicBatchWrapper89_copies410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13152 := (publicBatchWrapper89_copies411 a h).1
  have c13153 := (publicBatchWrapper89_copies411 a h).2.1
  have c13154 := (publicBatchWrapper89_copies411 a h).2.2.1
  have c13155 := (publicBatchWrapper89_copies411 a h).2.2.2.1
  have c13156 := (publicBatchWrapper89_copies411 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_18 := arithEq_of_rows h (row := 192) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_18
  simp only [← c13151, ← c13152, ← c13153] at e_192_18
  have e_197_10 := arithEq_of_rows h (row := 197) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_10
  simp only [← c13154, ← c13155, k3, ← c13156] at e_197_10
  have hr := e_197_10
  simp only [e_192_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 43) = a (.virt 201) := by
  have c13157 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.1
  exact c13157

theorem publicBatchWrapper89_f1946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3403)) (a (.wire 170 23)) (a (.virt 18723)) (a (.virt 18724)) := by
  have c13158 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.1
  have c13159 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.1
  have c13160 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.1
  have c13161 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.1
  have c13162 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13163 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13164 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13165 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13166 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13167 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13168 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13169 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13170 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13171 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13172 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13173 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13174 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_202_14 := arithEq_of_rows h (row := 202) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_14
  simp only [← c13158, k3, ← c13159, k3, ← c13160] at e_202_14
  have e_202_15 := arithEq_of_rows h (row := 202) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_15
  simp only [← c13161, ← c13162, k3, ← c13163] at e_202_15
  have e_202_16 := arithEq_of_rows h (row := 202) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_16
  simp only [← c13170, ← c13171, k3, ← c13172] at e_202_16
  have e_203_6 := arithEq_of_rows h (row := 203) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_6
  simp only [← c13164, ← c13165, ← c13166] at e_203_6
  have e_203_7 := arithEq_of_rows h (row := 203) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_7
  simp only [← c13167, ← c13168, ← c13169] at e_203_7
  refine ⟨?_, ?_⟩
  · have hc := e_203_6
    simp only [e_202_15] at hc
    linear_combination c13173.trans k2 - hc
  · have hc := e_202_16
    simp only [e_202_14, e_203_7, e_202_15] at hc
    linear_combination c13174.trans k2 - hc

theorem publicBatchWrapper89_f1947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 47) = bor (a (.wire 19 7)) (a (.virt 18723)) := by
  have c13175 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13176 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13177 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13178 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13179 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13180 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_192_19 := arithEq_of_rows h (row := 192) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_192_19
  simp only [← c13175, ← c13176, ← c13177] at e_192_19
  have e_197_11 := arithEq_of_rows h (row := 197) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_11
  simp only [← c13178, ← c13179, k3, ← c13180] at e_197_11
  have hr := e_197_11
  simp only [e_192_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 47) = a (.virt 201) := by
  have c13181 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13181

theorem publicBatchWrapper89_f1949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3404)) (a (.wire 170 31)) (a (.virt 18725)) (a (.virt 18726)) := by
  have c13182 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13183 := (publicBatchWrapper89_copies411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13184 := (publicBatchWrapper89_copies412 a h).1
  have c13185 := (publicBatchWrapper89_copies412 a h).2.1
  have c13186 := (publicBatchWrapper89_copies412 a h).2.2.1
  have c13187 := (publicBatchWrapper89_copies412 a h).2.2.2.1
  have c13188 := (publicBatchWrapper89_copies412 a h).2.2.2.2.1
  have c13189 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.1
  have c13190 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.1
  have c13191 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.1
  have c13192 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.1
  have c13193 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.1
  have c13194 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13195 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13196 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13197 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13198 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_202_17 := arithEq_of_rows h (row := 202) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_17
  simp only [← c13182, k3, ← c13183, k3, ← c13184] at e_202_17
  have e_202_18 := arithEq_of_rows h (row := 202) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_18
  simp only [← c13185, ← c13186, k3, ← c13187] at e_202_18
  have e_202_19 := arithEq_of_rows h (row := 202) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_202_19
  simp only [← c13194, ← c13195, k3, ← c13196] at e_202_19
  have e_203_8 := arithEq_of_rows h (row := 203) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_8
  simp only [← c13188, ← c13189, ← c13190] at e_203_8
  have e_203_9 := arithEq_of_rows h (row := 203) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_9
  simp only [← c13191, ← c13192, ← c13193] at e_203_9
  refine ⟨?_, ?_⟩
  · have hc := e_203_8
    simp only [e_202_18] at hc
    linear_combination c13197.trans k2 - hc
  · have hc := e_202_19
    simp only [e_202_17, e_203_9, e_202_18] at hc
    linear_combination c13198.trans k2 - hc

theorem publicBatchWrapper89_f1950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 51) = bor (a (.wire 19 7)) (a (.virt 18725)) := by
  have c13199 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13200 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13201 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13202 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13203 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13204 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_197_12 := arithEq_of_rows h (row := 197) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_12
  simp only [← c13202, ← c13203, k3, ← c13204] at e_197_12
  have e_204_0 := arithEq_of_rows h (row := 204) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_0
  simp only [← c13199, ← c13200, ← c13201] at e_204_0
  have hr := e_197_12
  simp only [e_204_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 51) = a (.virt 201) := by
  have c13205 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13205

theorem publicBatchWrapper89_f1952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3405)) (a (.wire 169 63)) (a (.virt 18727)) (a (.virt 18728)) := by
  have c13206 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13207 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13208 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13209 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13210 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13211 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13212 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13213 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13214 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13215 := (publicBatchWrapper89_copies412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13216 := (publicBatchWrapper89_copies413 a h).1
  have c13217 := (publicBatchWrapper89_copies413 a h).2.1
  have c13218 := (publicBatchWrapper89_copies413 a h).2.2.1
  have c13219 := (publicBatchWrapper89_copies413 a h).2.2.2.1
  have c13220 := (publicBatchWrapper89_copies413 a h).2.2.2.2.1
  have c13221 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.1
  have c13222 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_203_10 := arithEq_of_rows h (row := 203) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_10
  simp only [← c13212, ← c13213, ← c13214] at e_203_10
  have e_203_11 := arithEq_of_rows h (row := 203) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_11
  simp only [← c13215, ← c13216, ← c13217] at e_203_11
  have e_205_0 := arithEq_of_rows h (row := 205) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_0
  simp only [← c13206, k3, ← c13207, k3, ← c13208] at e_205_0
  have e_205_1 := arithEq_of_rows h (row := 205) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_1
  simp only [← c13209, ← c13210, k3, ← c13211] at e_205_1
  have e_205_2 := arithEq_of_rows h (row := 205) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_2
  simp only [← c13218, ← c13219, k3, ← c13220] at e_205_2
  refine ⟨?_, ?_⟩
  · have hc := e_203_10
    simp only [e_205_1] at hc
    linear_combination c13221.trans k2 - hc
  · have hc := e_205_2
    simp only [e_205_0, e_203_11, e_205_1] at hc
    linear_combination c13222.trans k2 - hc

theorem publicBatchWrapper89_f1953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3406)) (a (.wire 169 71)) (a (.virt 18729)) (a (.virt 18730)) := by
  have c13223 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.1
  have c13224 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.1
  have c13225 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.1
  have c13226 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13227 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13228 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13229 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13230 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13231 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13232 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13233 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13234 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13235 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13236 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13237 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13238 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13239 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_203_12 := arithEq_of_rows h (row := 203) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_12
  simp only [← c13229, ← c13230, ← c13231] at e_203_12
  have e_203_13 := arithEq_of_rows h (row := 203) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_13
  simp only [← c13232, ← c13233, ← c13234] at e_203_13
  have e_205_3 := arithEq_of_rows h (row := 205) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_3
  simp only [← c13223, k3, ← c13224, k3, ← c13225] at e_205_3
  have e_205_4 := arithEq_of_rows h (row := 205) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_4
  simp only [← c13226, ← c13227, k3, ← c13228] at e_205_4
  have e_205_5 := arithEq_of_rows h (row := 205) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_5
  simp only [← c13235, ← c13236, k3, ← c13237] at e_205_5
  refine ⟨?_, ?_⟩
  · have hc := e_203_12
    simp only [e_205_4] at hc
    linear_combination c13238.trans k2 - hc
  · have hc := e_205_5
    simp only [e_205_3, e_203_13, e_205_4] at hc
    linear_combination c13239.trans k2 - hc

theorem publicBatchWrapper89_f1954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3407)) (a (.wire 169 79)) (a (.virt 18731)) (a (.virt 18732)) := by
  have c13240 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13241 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13242 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13243 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13244 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13245 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13246 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13247 := (publicBatchWrapper89_copies413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13248 := (publicBatchWrapper89_copies414 a h).1
  have c13249 := (publicBatchWrapper89_copies414 a h).2.1
  have c13250 := (publicBatchWrapper89_copies414 a h).2.2.1
  have c13251 := (publicBatchWrapper89_copies414 a h).2.2.2.1
  have c13252 := (publicBatchWrapper89_copies414 a h).2.2.2.2.1
  have c13253 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.1
  have c13254 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.1
  have c13255 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.1
  have c13256 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_203_14 := arithEq_of_rows h (row := 203) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_14
  simp only [← c13246, ← c13247, ← c13248] at e_203_14
  have e_203_15 := arithEq_of_rows h (row := 203) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_15
  simp only [← c13249, ← c13250, ← c13251] at e_203_15
  have e_205_6 := arithEq_of_rows h (row := 205) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_6
  simp only [← c13240, k3, ← c13241, k3, ← c13242] at e_205_6
  have e_205_7 := arithEq_of_rows h (row := 205) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_7
  simp only [← c13243, ← c13244, k3, ← c13245] at e_205_7
  have e_205_8 := arithEq_of_rows h (row := 205) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_8
  simp only [← c13252, ← c13253, k3, ← c13254] at e_205_8
  refine ⟨?_, ?_⟩
  · have hc := e_203_14
    simp only [e_205_7] at hc
    linear_combination c13255.trans k2 - hc
  · have hc := e_205_8
    simp only [e_205_6, e_203_15, e_205_7] at hc
    linear_combination c13256.trans k2 - hc

theorem publicBatchWrapper89_f1955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3408)) (a (.wire 170 7)) (a (.virt 18733)) (a (.virt 18734)) := by
  have c13257 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.1
  have c13258 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13259 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13260 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13261 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13262 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13263 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13264 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13265 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13266 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13267 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13268 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13269 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13270 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13271 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13272 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13273 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_203_16 := arithEq_of_rows h (row := 203) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_16
  simp only [← c13263, ← c13264, ← c13265] at e_203_16
  have e_203_17 := arithEq_of_rows h (row := 203) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_17
  simp only [← c13266, ← c13267, ← c13268] at e_203_17
  have e_205_9 := arithEq_of_rows h (row := 205) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_9
  simp only [← c13257, k3, ← c13258, k3, ← c13259] at e_205_9
  have e_205_10 := arithEq_of_rows h (row := 205) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_10
  simp only [← c13260, ← c13261, k3, ← c13262] at e_205_10
  have e_205_11 := arithEq_of_rows h (row := 205) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_11
  simp only [← c13269, ← c13270, k3, ← c13271] at e_205_11
  refine ⟨?_, ?_⟩
  · have hc := e_203_16
    simp only [e_205_10] at hc
    linear_combination c13272.trans k2 - hc
  · have hc := e_205_11
    simp only [e_205_9, e_203_17, e_205_10] at hc
    linear_combination c13273.trans k2 - hc

theorem publicBatchWrapper89_f1956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 203 75) = band (a (.virt 18727)) (a (.virt 18729)) := by
  have c13274 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13275 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13276 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_203_18 := arithEq_of_rows h (row := 203) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_18
  simp only [← c13274, ← c13275, ← c13276] at e_203_18
  have hr := e_203_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 203 79) = band (a (.virt 18731)) (a (.virt 18733)) := by
  have c13277 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13278 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13279 := (publicBatchWrapper89_copies414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_203_19 := arithEq_of_rows h (row := 203) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_203_19
  simp only [← c13277, ← c13278, ← c13279] at e_203_19
  have hr := e_203_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 206 3) = band (a (.wire 203 75)) (a (.wire 203 79)) := by
  have c13280 := (publicBatchWrapper89_copies415 a h).1
  have c13281 := (publicBatchWrapper89_copies415 a h).2.1
  have c13282 := (publicBatchWrapper89_copies415 a h).2.2.1
  have e_206_0 := arithEq_of_rows h (row := 206) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_0
  simp only [← c13280, ← c13281, ← c13282] at e_206_0
  have hr := e_206_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 55) = bor (a (.wire 19 7)) (a (.wire 206 3)) := by
  have c13283 := (publicBatchWrapper89_copies415 a h).2.2.2.1
  have c13284 := (publicBatchWrapper89_copies415 a h).2.2.2.2.1
  have c13285 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.1
  have c13286 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.1
  have c13287 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.1
  have c13288 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_197_13 := arithEq_of_rows h (row := 197) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_13
  simp only [← c13286, ← c13287, k3, ← c13288] at e_197_13
  have e_204_1 := arithEq_of_rows h (row := 204) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_1
  simp only [← c13283, ← c13284, ← c13285] at e_204_1
  have hr := e_197_13
  simp only [e_204_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 55) = a (.virt 201) := by
  have c13289 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.1
  exact c13289

theorem publicBatchWrapper89_f1961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3603)) (a (.wire 170 23)) (a (.virt 18735)) (a (.virt 18736)) := by
  have c13290 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13291 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13292 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13293 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13294 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13295 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13296 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13297 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13298 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13299 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13300 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13301 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13302 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13303 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13304 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13305 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13306 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_205_12 := arithEq_of_rows h (row := 205) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_12
  simp only [← c13290, k3, ← c13291, k3, ← c13292] at e_205_12
  have e_205_13 := arithEq_of_rows h (row := 205) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_13
  simp only [← c13293, ← c13294, k3, ← c13295] at e_205_13
  have e_205_14 := arithEq_of_rows h (row := 205) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_14
  simp only [← c13302, ← c13303, k3, ← c13304] at e_205_14
  have e_206_1 := arithEq_of_rows h (row := 206) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_1
  simp only [← c13296, ← c13297, ← c13298] at e_206_1
  have e_206_2 := arithEq_of_rows h (row := 206) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_2
  simp only [← c13299, ← c13300, ← c13301] at e_206_2
  refine ⟨?_, ?_⟩
  · have hc := e_206_1
    simp only [e_205_13] at hc
    linear_combination c13305.trans k2 - hc
  · have hc := e_205_14
    simp only [e_205_12, e_206_2, e_205_13] at hc
    linear_combination c13306.trans k2 - hc

theorem publicBatchWrapper89_f1962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 59) = bor (a (.wire 19 51)) (a (.virt 18735)) := by
  have c13307 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13308 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13309 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13310 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13311 := (publicBatchWrapper89_copies415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13312 := (publicBatchWrapper89_copies416 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_197_14 := arithEq_of_rows h (row := 197) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_14
  simp only [← c13310, ← c13311, k3, ← c13312] at e_197_14
  have e_204_2 := arithEq_of_rows h (row := 204) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_2
  simp only [← c13307, ← c13308, ← c13309] at e_204_2
  have hr := e_197_14
  simp only [e_204_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 59) = a (.virt 201) := by
  have c13313 := (publicBatchWrapper89_copies416 a h).2.1
  exact c13313

theorem publicBatchWrapper89_f1964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3604)) (a (.wire 170 31)) (a (.virt 18737)) (a (.virt 18738)) := by
  have c13314 := (publicBatchWrapper89_copies416 a h).2.2.1
  have c13315 := (publicBatchWrapper89_copies416 a h).2.2.2.1
  have c13316 := (publicBatchWrapper89_copies416 a h).2.2.2.2.1
  have c13317 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.1
  have c13318 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.1
  have c13319 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.1
  have c13320 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.1
  have c13321 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.1
  have c13322 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13323 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13324 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13325 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13326 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13327 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13328 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13329 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13330 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_205_15 := arithEq_of_rows h (row := 205) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_15
  simp only [← c13314, k3, ← c13315, k3, ← c13316] at e_205_15
  have e_205_16 := arithEq_of_rows h (row := 205) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_16
  simp only [← c13317, ← c13318, k3, ← c13319] at e_205_16
  have e_205_17 := arithEq_of_rows h (row := 205) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_17
  simp only [← c13326, ← c13327, k3, ← c13328] at e_205_17
  have e_206_3 := arithEq_of_rows h (row := 206) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_3
  simp only [← c13320, ← c13321, ← c13322] at e_206_3
  have e_206_4 := arithEq_of_rows h (row := 206) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_4
  simp only [← c13323, ← c13324, ← c13325] at e_206_4
  refine ⟨?_, ?_⟩
  · have hc := e_206_3
    simp only [e_205_16] at hc
    linear_combination c13329.trans k2 - hc
  · have hc := e_205_17
    simp only [e_205_15, e_206_4, e_205_16] at hc
    linear_combination c13330.trans k2 - hc

theorem publicBatchWrapper89_f1965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 63) = bor (a (.wire 19 51)) (a (.virt 18737)) := by
  have c13331 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13332 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13333 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13334 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13335 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13336 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_197_15 := arithEq_of_rows h (row := 197) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_15
  simp only [← c13334, ← c13335, k3, ← c13336] at e_197_15
  have e_204_3 := arithEq_of_rows h (row := 204) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_3
  simp only [← c13331, ← c13332, ← c13333] at e_204_3
  have hr := e_197_15
  simp only [e_204_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 63) = a (.virt 201) := by
  have c13337 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13337

theorem publicBatchWrapper89_f1967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3605)) (a (.wire 169 63)) (a (.virt 18739)) (a (.virt 18740)) := by
  have c13338 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13339 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13340 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13341 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13342 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13343 := (publicBatchWrapper89_copies416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13344 := (publicBatchWrapper89_copies417 a h).1
  have c13345 := (publicBatchWrapper89_copies417 a h).2.1
  have c13346 := (publicBatchWrapper89_copies417 a h).2.2.1
  have c13347 := (publicBatchWrapper89_copies417 a h).2.2.2.1
  have c13348 := (publicBatchWrapper89_copies417 a h).2.2.2.2.1
  have c13349 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.1
  have c13350 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.1
  have c13351 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.1
  have c13352 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.1
  have c13353 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.1
  have c13354 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_205_18 := arithEq_of_rows h (row := 205) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_18
  simp only [← c13338, k3, ← c13339, k3, ← c13340] at e_205_18
  have e_205_19 := arithEq_of_rows h (row := 205) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_205_19
  simp only [← c13341, ← c13342, k3, ← c13343] at e_205_19
  have e_206_5 := arithEq_of_rows h (row := 206) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_5
  simp only [← c13344, ← c13345, ← c13346] at e_206_5
  have e_206_6 := arithEq_of_rows h (row := 206) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_6
  simp only [← c13347, ← c13348, ← c13349] at e_206_6
  have e_207_0 := arithEq_of_rows h (row := 207) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_0
  simp only [← c13350, ← c13351, k3, ← c13352] at e_207_0
  refine ⟨?_, ?_⟩
  · have hc := e_206_5
    simp only [e_205_19] at hc
    linear_combination c13353.trans k2 - hc
  · have hc := e_207_0
    simp only [e_205_18, e_206_6, e_205_19] at hc
    linear_combination c13354.trans k2 - hc

theorem publicBatchWrapper89_f1968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3606)) (a (.wire 169 71)) (a (.virt 18741)) (a (.virt 18742)) := by
  have c13355 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13356 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13357 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13358 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13359 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13360 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13361 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13362 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13363 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13364 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13365 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13366 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13367 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13368 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13369 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13370 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13371 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_206_7 := arithEq_of_rows h (row := 206) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_7
  simp only [← c13361, ← c13362, ← c13363] at e_206_7
  have e_206_8 := arithEq_of_rows h (row := 206) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_8
  simp only [← c13364, ← c13365, ← c13366] at e_206_8
  have e_207_1 := arithEq_of_rows h (row := 207) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_1
  simp only [← c13355, k3, ← c13356, k3, ← c13357] at e_207_1
  have e_207_2 := arithEq_of_rows h (row := 207) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_2
  simp only [← c13358, ← c13359, k3, ← c13360] at e_207_2
  have e_207_3 := arithEq_of_rows h (row := 207) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_3
  simp only [← c13367, ← c13368, k3, ← c13369] at e_207_3
  refine ⟨?_, ?_⟩
  · have hc := e_206_7
    simp only [e_207_2] at hc
    linear_combination c13370.trans k2 - hc
  · have hc := e_207_3
    simp only [e_207_1, e_206_8, e_207_2] at hc
    linear_combination c13371.trans k2 - hc

theorem publicBatchWrapper89_f1969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3607)) (a (.wire 169 79)) (a (.virt 18743)) (a (.virt 18744)) := by
  have c13372 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13373 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13374 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13375 := (publicBatchWrapper89_copies417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13376 := (publicBatchWrapper89_copies418 a h).1
  have c13377 := (publicBatchWrapper89_copies418 a h).2.1
  have c13378 := (publicBatchWrapper89_copies418 a h).2.2.1
  have c13379 := (publicBatchWrapper89_copies418 a h).2.2.2.1
  have c13380 := (publicBatchWrapper89_copies418 a h).2.2.2.2.1
  have c13381 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.1
  have c13382 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.1
  have c13383 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.1
  have c13384 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.1
  have c13385 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.1
  have c13386 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13387 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13388 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_206_9 := arithEq_of_rows h (row := 206) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_9
  simp only [← c13378, ← c13379, ← c13380] at e_206_9
  have e_206_10 := arithEq_of_rows h (row := 206) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_10
  simp only [← c13381, ← c13382, ← c13383] at e_206_10
  have e_207_4 := arithEq_of_rows h (row := 207) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_4
  simp only [← c13372, k3, ← c13373, k3, ← c13374] at e_207_4
  have e_207_5 := arithEq_of_rows h (row := 207) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_5
  simp only [← c13375, ← c13376, k3, ← c13377] at e_207_5
  have e_207_6 := arithEq_of_rows h (row := 207) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_6
  simp only [← c13384, ← c13385, k3, ← c13386] at e_207_6
  refine ⟨?_, ?_⟩
  · have hc := e_206_9
    simp only [e_207_5] at hc
    linear_combination c13387.trans k2 - hc
  · have hc := e_207_6
    simp only [e_207_4, e_206_10, e_207_5] at hc
    linear_combination c13388.trans k2 - hc

theorem publicBatchWrapper89_f1970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3608)) (a (.wire 170 7)) (a (.virt 18745)) (a (.virt 18746)) := by
  have c13389 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13390 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13391 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13392 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13393 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13394 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13395 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13396 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13397 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13398 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13399 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13400 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13401 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13402 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13403 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13404 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13405 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_206_11 := arithEq_of_rows h (row := 206) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_11
  simp only [← c13395, ← c13396, ← c13397] at e_206_11
  have e_206_12 := arithEq_of_rows h (row := 206) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_12
  simp only [← c13398, ← c13399, ← c13400] at e_206_12
  have e_207_7 := arithEq_of_rows h (row := 207) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_7
  simp only [← c13389, k3, ← c13390, k3, ← c13391] at e_207_7
  have e_207_8 := arithEq_of_rows h (row := 207) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_8
  simp only [← c13392, ← c13393, k3, ← c13394] at e_207_8
  have e_207_9 := arithEq_of_rows h (row := 207) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_9
  simp only [← c13401, ← c13402, k3, ← c13403] at e_207_9
  refine ⟨?_, ?_⟩
  · have hc := e_206_11
    simp only [e_207_8] at hc
    linear_combination c13404.trans k2 - hc
  · have hc := e_207_9
    simp only [e_207_7, e_206_12, e_207_8] at hc
    linear_combination c13405.trans k2 - hc

theorem publicBatchWrapper89_f1971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 206 55) = band (a (.virt 18739)) (a (.virt 18741)) := by
  have c13406 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13407 := (publicBatchWrapper89_copies418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13408 := (publicBatchWrapper89_copies419 a h).1
  have e_206_13 := arithEq_of_rows h (row := 206) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_13
  simp only [← c13406, ← c13407, ← c13408] at e_206_13
  have hr := e_206_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 206 59) = band (a (.virt 18743)) (a (.virt 18745)) := by
  have c13409 := (publicBatchWrapper89_copies419 a h).2.1
  have c13410 := (publicBatchWrapper89_copies419 a h).2.2.1
  have c13411 := (publicBatchWrapper89_copies419 a h).2.2.2.1
  have e_206_14 := arithEq_of_rows h (row := 206) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_14
  simp only [← c13409, ← c13410, ← c13411] at e_206_14
  have hr := e_206_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 206 63) = band (a (.wire 206 55)) (a (.wire 206 59)) := by
  have c13412 := (publicBatchWrapper89_copies419 a h).2.2.2.2.1
  have c13413 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.1
  have c13414 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.1
  have e_206_15 := arithEq_of_rows h (row := 206) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_15
  simp only [← c13412, ← c13413, ← c13414] at e_206_15
  have hr := e_206_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 67) = bor (a (.wire 19 51)) (a (.wire 206 63)) := by
  have c13415 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.1
  have c13416 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.1
  have c13417 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.1
  have c13418 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13419 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13420 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_197_16 := arithEq_of_rows h (row := 197) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_16
  simp only [← c13418, ← c13419, k3, ← c13420] at e_197_16
  have e_204_4 := arithEq_of_rows h (row := 204) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_4
  simp only [← c13415, ← c13416, ← c13417] at e_204_4
  have hr := e_197_16
  simp only [e_204_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 67) = a (.virt 201) := by
  have c13421 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13421

theorem publicBatchWrapper89_f1976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3803)) (a (.wire 170 23)) (a (.virt 18747)) (a (.virt 18748)) := by
  have c13422 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13423 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13424 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13425 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13426 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13427 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13428 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13429 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13430 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13431 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13432 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13433 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13434 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13435 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13436 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13437 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13438 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_206_16 := arithEq_of_rows h (row := 206) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_16
  simp only [← c13428, ← c13429, ← c13430] at e_206_16
  have e_206_17 := arithEq_of_rows h (row := 206) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_17
  simp only [← c13431, ← c13432, ← c13433] at e_206_17
  have e_207_10 := arithEq_of_rows h (row := 207) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_10
  simp only [← c13422, k3, ← c13423, k3, ← c13424] at e_207_10
  have e_207_11 := arithEq_of_rows h (row := 207) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_11
  simp only [← c13425, ← c13426, k3, ← c13427] at e_207_11
  have e_207_12 := arithEq_of_rows h (row := 207) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_12
  simp only [← c13434, ← c13435, k3, ← c13436] at e_207_12
  refine ⟨?_, ?_⟩
  · have hc := e_206_16
    simp only [e_207_11] at hc
    linear_combination c13437.trans k2 - hc
  · have hc := e_207_12
    simp only [e_207_10, e_206_17, e_207_11] at hc
    linear_combination c13438.trans k2 - hc

theorem publicBatchWrapper89_f1977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 71) = bor (a (.wire 20 15)) (a (.virt 18747)) := by
  have c13439 := (publicBatchWrapper89_copies419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13440 := (publicBatchWrapper89_copies420 a h).1
  have c13441 := (publicBatchWrapper89_copies420 a h).2.1
  have c13442 := (publicBatchWrapper89_copies420 a h).2.2.1
  have c13443 := (publicBatchWrapper89_copies420 a h).2.2.2.1
  have c13444 := (publicBatchWrapper89_copies420 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_197_17 := arithEq_of_rows h (row := 197) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_17
  simp only [← c13442, ← c13443, k3, ← c13444] at e_197_17
  have e_204_5 := arithEq_of_rows h (row := 204) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_5
  simp only [← c13439, ← c13440, ← c13441] at e_204_5
  have hr := e_197_17
  simp only [e_204_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 71) = a (.virt 201) := by
  have c13445 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.1
  exact c13445

theorem publicBatchWrapper89_f1979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3804)) (a (.wire 170 31)) (a (.virt 18749)) (a (.virt 18750)) := by
  have c13446 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.1
  have c13447 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.1
  have c13448 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.1
  have c13449 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.1
  have c13450 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13451 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13452 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13453 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13454 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13455 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13456 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13457 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13458 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13459 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13460 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13461 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13462 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_206_18 := arithEq_of_rows h (row := 206) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_18
  simp only [← c13452, ← c13453, ← c13454] at e_206_18
  have e_206_19 := arithEq_of_rows h (row := 206) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_206_19
  simp only [← c13455, ← c13456, ← c13457] at e_206_19
  have e_207_13 := arithEq_of_rows h (row := 207) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_13
  simp only [← c13446, k3, ← c13447, k3, ← c13448] at e_207_13
  have e_207_14 := arithEq_of_rows h (row := 207) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_14
  simp only [← c13449, ← c13450, k3, ← c13451] at e_207_14
  have e_207_15 := arithEq_of_rows h (row := 207) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_15
  simp only [← c13458, ← c13459, k3, ← c13460] at e_207_15
  refine ⟨?_, ?_⟩
  · have hc := e_206_18
    simp only [e_207_14] at hc
    linear_combination c13461.trans k2 - hc
  · have hc := e_207_15
    simp only [e_207_13, e_206_19, e_207_14] at hc
    linear_combination c13462.trans k2 - hc

theorem publicBatchWrapper89_f1980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 75) = bor (a (.wire 20 15)) (a (.virt 18749)) := by
  have c13463 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13464 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13465 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13466 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13467 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13468 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_197_18 := arithEq_of_rows h (row := 197) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_18
  simp only [← c13466, ← c13467, k3, ← c13468] at e_197_18
  have e_204_6 := arithEq_of_rows h (row := 204) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_6
  simp only [← c13463, ← c13464, ← c13465] at e_204_6
  have hr := e_197_18
  simp only [e_204_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 75) = a (.virt 201) := by
  have c13469 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13469

theorem publicBatchWrapper89_f1982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3805)) (a (.wire 169 63)) (a (.virt 18751)) (a (.virt 18752)) := by
  have c13470 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13471 := (publicBatchWrapper89_copies420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13472 := (publicBatchWrapper89_copies421 a h).1
  have c13473 := (publicBatchWrapper89_copies421 a h).2.1
  have c13474 := (publicBatchWrapper89_copies421 a h).2.2.1
  have c13475 := (publicBatchWrapper89_copies421 a h).2.2.2.1
  have c13476 := (publicBatchWrapper89_copies421 a h).2.2.2.2.1
  have c13477 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.1
  have c13478 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.1
  have c13479 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.1
  have c13480 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.1
  have c13481 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.1
  have c13482 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13483 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13484 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13485 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13486 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_207_16 := arithEq_of_rows h (row := 207) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_16
  simp only [← c13470, k3, ← c13471, k3, ← c13472] at e_207_16
  have e_207_17 := arithEq_of_rows h (row := 207) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_17
  simp only [← c13473, ← c13474, k3, ← c13475] at e_207_17
  have e_207_18 := arithEq_of_rows h (row := 207) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_18
  simp only [← c13482, ← c13483, k3, ← c13484] at e_207_18
  have e_208_0 := arithEq_of_rows h (row := 208) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_0
  simp only [← c13476, ← c13477, ← c13478] at e_208_0
  have e_208_1 := arithEq_of_rows h (row := 208) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_1
  simp only [← c13479, ← c13480, ← c13481] at e_208_1
  refine ⟨?_, ?_⟩
  · have hc := e_208_0
    simp only [e_207_17] at hc
    linear_combination c13485.trans k2 - hc
  · have hc := e_207_18
    simp only [e_207_16, e_208_1, e_207_17] at hc
    linear_combination c13486.trans k2 - hc

theorem publicBatchWrapper89_f1983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3806)) (a (.wire 169 71)) (a (.virt 18753)) (a (.virt 18754)) := by
  have c13487 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13488 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13489 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13490 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13491 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13492 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13493 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13494 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13495 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13496 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13497 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13498 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13499 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13500 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13501 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13502 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13503 := (publicBatchWrapper89_copies421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_207_19 := arithEq_of_rows h (row := 207) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_207_19
  simp only [← c13487, k3, ← c13488, k3, ← c13489] at e_207_19
  have e_208_2 := arithEq_of_rows h (row := 208) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_2
  simp only [← c13493, ← c13494, ← c13495] at e_208_2
  have e_208_3 := arithEq_of_rows h (row := 208) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_3
  simp only [← c13496, ← c13497, ← c13498] at e_208_3
  have e_209_0 := arithEq_of_rows h (row := 209) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_0
  simp only [← c13490, ← c13491, k3, ← c13492] at e_209_0
  have e_209_1 := arithEq_of_rows h (row := 209) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_1
  simp only [← c13499, ← c13500, k3, ← c13501] at e_209_1
  refine ⟨?_, ?_⟩
  · have hc := e_208_2
    simp only [e_209_0] at hc
    linear_combination c13502.trans k2 - hc
  · have hc := e_209_1
    simp only [e_207_19, e_208_3, e_209_0] at hc
    linear_combination c13503.trans k2 - hc

theorem publicBatchWrapper89_f1984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3807)) (a (.wire 169 79)) (a (.virt 18755)) (a (.virt 18756)) := by
  have c13504 := (publicBatchWrapper89_copies422 a h).1
  have c13505 := (publicBatchWrapper89_copies422 a h).2.1
  have c13506 := (publicBatchWrapper89_copies422 a h).2.2.1
  have c13507 := (publicBatchWrapper89_copies422 a h).2.2.2.1
  have c13508 := (publicBatchWrapper89_copies422 a h).2.2.2.2.1
  have c13509 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.1
  have c13510 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.1
  have c13511 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.1
  have c13512 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.1
  have c13513 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.1
  have c13514 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13515 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13516 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13517 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13518 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13519 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13520 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_208_4 := arithEq_of_rows h (row := 208) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_4
  simp only [← c13510, ← c13511, ← c13512] at e_208_4
  have e_208_5 := arithEq_of_rows h (row := 208) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_5
  simp only [← c13513, ← c13514, ← c13515] at e_208_5
  have e_209_2 := arithEq_of_rows h (row := 209) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_2
  simp only [← c13504, k3, ← c13505, k3, ← c13506] at e_209_2
  have e_209_3 := arithEq_of_rows h (row := 209) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_3
  simp only [← c13507, ← c13508, k3, ← c13509] at e_209_3
  have e_209_4 := arithEq_of_rows h (row := 209) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_4
  simp only [← c13516, ← c13517, k3, ← c13518] at e_209_4
  refine ⟨?_, ?_⟩
  · have hc := e_208_4
    simp only [e_209_3] at hc
    linear_combination c13519.trans k2 - hc
  · have hc := e_209_4
    simp only [e_209_2, e_208_5, e_209_3] at hc
    linear_combination c13520.trans k2 - hc

theorem publicBatchWrapper89_f1985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3808)) (a (.wire 170 7)) (a (.virt 18757)) (a (.virt 18758)) := by
  have c13521 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13522 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13523 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13524 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13525 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13526 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13527 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13528 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13529 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13530 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13531 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13532 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13533 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13534 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13535 := (publicBatchWrapper89_copies422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13536 := (publicBatchWrapper89_copies423 a h).1
  have c13537 := (publicBatchWrapper89_copies423 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_208_6 := arithEq_of_rows h (row := 208) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_6
  simp only [← c13527, ← c13528, ← c13529] at e_208_6
  have e_208_7 := arithEq_of_rows h (row := 208) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_7
  simp only [← c13530, ← c13531, ← c13532] at e_208_7
  have e_209_5 := arithEq_of_rows h (row := 209) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_5
  simp only [← c13521, k3, ← c13522, k3, ← c13523] at e_209_5
  have e_209_6 := arithEq_of_rows h (row := 209) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_6
  simp only [← c13524, ← c13525, k3, ← c13526] at e_209_6
  have e_209_7 := arithEq_of_rows h (row := 209) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_7
  simp only [← c13533, ← c13534, k3, ← c13535] at e_209_7
  refine ⟨?_, ?_⟩
  · have hc := e_208_6
    simp only [e_209_6] at hc
    linear_combination c13536.trans k2 - hc
  · have hc := e_209_7
    simp only [e_209_5, e_208_7, e_209_6] at hc
    linear_combination c13537.trans k2 - hc

theorem publicBatchWrapper89_f1986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 208 35) = band (a (.virt 18751)) (a (.virt 18753)) := by
  have c13538 := (publicBatchWrapper89_copies423 a h).2.2.1
  have c13539 := (publicBatchWrapper89_copies423 a h).2.2.2.1
  have c13540 := (publicBatchWrapper89_copies423 a h).2.2.2.2.1
  have e_208_8 := arithEq_of_rows h (row := 208) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_8
  simp only [← c13538, ← c13539, ← c13540] at e_208_8
  have hr := e_208_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 208 39) = band (a (.virt 18755)) (a (.virt 18757)) := by
  have c13541 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.1
  have c13542 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.1
  have c13543 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.1
  have e_208_9 := arithEq_of_rows h (row := 208) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_9
  simp only [← c13541, ← c13542, ← c13543] at e_208_9
  have hr := e_208_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 208 43) = band (a (.wire 208 35)) (a (.wire 208 39)) := by
  have c13544 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.1
  have c13545 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.1
  have c13546 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_208_10 := arithEq_of_rows h (row := 208) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_10
  simp only [← c13544, ← c13545, ← c13546] at e_208_10
  have hr := e_208_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 79) = bor (a (.wire 20 15)) (a (.wire 208 43)) := by
  have c13547 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13548 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13549 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13550 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13551 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13552 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_197_19 := arithEq_of_rows h (row := 197) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_197_19
  simp only [← c13550, ← c13551, k3, ← c13552] at e_197_19
  have e_204_7 := arithEq_of_rows h (row := 204) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_7
  simp only [← c13547, ← c13548, ← c13549] at e_204_7
  have hr := e_197_19
  simp only [e_204_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 197 79) = a (.virt 201) := by
  have c13553 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13553

theorem publicBatchWrapper89_f1991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4003)) (a (.wire 170 23)) (a (.virt 18759)) (a (.virt 18760)) := by
  have c13554 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13555 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13556 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13557 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13558 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13559 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13560 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13561 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13562 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13563 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13564 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13565 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13566 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13567 := (publicBatchWrapper89_copies423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13568 := (publicBatchWrapper89_copies424 a h).1
  have c13569 := (publicBatchWrapper89_copies424 a h).2.1
  have c13570 := (publicBatchWrapper89_copies424 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_208_11 := arithEq_of_rows h (row := 208) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_11
  simp only [← c13560, ← c13561, ← c13562] at e_208_11
  have e_208_12 := arithEq_of_rows h (row := 208) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_12
  simp only [← c13563, ← c13564, ← c13565] at e_208_12
  have e_209_8 := arithEq_of_rows h (row := 209) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_8
  simp only [← c13554, k3, ← c13555, k3, ← c13556] at e_209_8
  have e_209_9 := arithEq_of_rows h (row := 209) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_9
  simp only [← c13557, ← c13558, k3, ← c13559] at e_209_9
  have e_209_10 := arithEq_of_rows h (row := 209) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_10
  simp only [← c13566, ← c13567, k3, ← c13568] at e_209_10
  refine ⟨?_, ?_⟩
  · have hc := e_208_11
    simp only [e_209_9] at hc
    linear_combination c13569.trans k2 - hc
  · have hc := e_209_10
    simp only [e_209_8, e_208_12, e_209_9] at hc
    linear_combination c13570.trans k2 - hc

theorem publicBatchWrapper89_f1992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 3) = bor (a (.wire 20 59)) (a (.virt 18759)) := by
  have c13571 := (publicBatchWrapper89_copies424 a h).2.2.2.1
  have c13572 := (publicBatchWrapper89_copies424 a h).2.2.2.2.1
  have c13573 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.1
  have c13574 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.1
  have c13575 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.1
  have c13576 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_8 := arithEq_of_rows h (row := 204) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_8
  simp only [← c13571, ← c13572, ← c13573] at e_204_8
  have e_210_0 := arithEq_of_rows h (row := 210) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_0
  simp only [← c13574, ← c13575, k3, ← c13576] at e_210_0
  have hr := e_210_0
  simp only [e_204_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 3) = a (.virt 201) := by
  have c13577 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.1
  exact c13577

theorem publicBatchWrapper89_f1994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4004)) (a (.wire 170 31)) (a (.virt 18761)) (a (.virt 18762)) := by
  have c13578 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13579 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13580 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13581 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13582 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13583 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13584 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13585 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13586 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13587 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13588 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13589 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13590 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13591 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13592 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13593 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13594 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_208_13 := arithEq_of_rows h (row := 208) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_13
  simp only [← c13584, ← c13585, ← c13586] at e_208_13
  have e_208_14 := arithEq_of_rows h (row := 208) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_14
  simp only [← c13587, ← c13588, ← c13589] at e_208_14
  have e_209_11 := arithEq_of_rows h (row := 209) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_11
  simp only [← c13578, k3, ← c13579, k3, ← c13580] at e_209_11
  have e_209_12 := arithEq_of_rows h (row := 209) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_12
  simp only [← c13581, ← c13582, k3, ← c13583] at e_209_12
  have e_209_13 := arithEq_of_rows h (row := 209) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_13
  simp only [← c13590, ← c13591, k3, ← c13592] at e_209_13
  refine ⟨?_, ?_⟩
  · have hc := e_208_13
    simp only [e_209_12] at hc
    linear_combination c13593.trans k2 - hc
  · have hc := e_209_13
    simp only [e_209_11, e_208_14, e_209_12] at hc
    linear_combination c13594.trans k2 - hc

theorem publicBatchWrapper89_f1995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 7) = bor (a (.wire 20 59)) (a (.virt 18761)) := by
  have c13595 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13596 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13597 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13598 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13599 := (publicBatchWrapper89_copies424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13600 := (publicBatchWrapper89_copies425 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_9 := arithEq_of_rows h (row := 204) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_9
  simp only [← c13595, ← c13596, ← c13597] at e_204_9
  have e_210_1 := arithEq_of_rows h (row := 210) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_1
  simp only [← c13598, ← c13599, k3, ← c13600] at e_210_1
  have hr := e_210_1
  simp only [e_204_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 7) = a (.virt 201) := by
  have c13601 := (publicBatchWrapper89_copies425 a h).2.1
  exact c13601

theorem publicBatchWrapper89_f1997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4005)) (a (.wire 169 63)) (a (.virt 18763)) (a (.virt 18764)) := by
  have c13602 := (publicBatchWrapper89_copies425 a h).2.2.1
  have c13603 := (publicBatchWrapper89_copies425 a h).2.2.2.1
  have c13604 := (publicBatchWrapper89_copies425 a h).2.2.2.2.1
  have c13605 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.1
  have c13606 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.1
  have c13607 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.1
  have c13608 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.1
  have c13609 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.1
  have c13610 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13611 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13612 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13613 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13614 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13615 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13616 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13617 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13618 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_208_15 := arithEq_of_rows h (row := 208) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_15
  simp only [← c13608, ← c13609, ← c13610] at e_208_15
  have e_208_16 := arithEq_of_rows h (row := 208) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_16
  simp only [← c13611, ← c13612, ← c13613] at e_208_16
  have e_209_14 := arithEq_of_rows h (row := 209) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_14
  simp only [← c13602, k3, ← c13603, k3, ← c13604] at e_209_14
  have e_209_15 := arithEq_of_rows h (row := 209) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_15
  simp only [← c13605, ← c13606, k3, ← c13607] at e_209_15
  have e_209_16 := arithEq_of_rows h (row := 209) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_16
  simp only [← c13614, ← c13615, k3, ← c13616] at e_209_16
  refine ⟨?_, ?_⟩
  · have hc := e_208_15
    simp only [e_209_15] at hc
    linear_combination c13617.trans k2 - hc
  · have hc := e_209_16
    simp only [e_209_14, e_208_16, e_209_15] at hc
    linear_combination c13618.trans k2 - hc

theorem publicBatchWrapper89_f1998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4006)) (a (.wire 169 71)) (a (.virt 18765)) (a (.virt 18766)) := by
  have c13619 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13620 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13621 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13622 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13623 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13624 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13625 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13626 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13627 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13628 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13629 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13630 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13631 := (publicBatchWrapper89_copies425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13632 := (publicBatchWrapper89_copies426 a h).1
  have c13633 := (publicBatchWrapper89_copies426 a h).2.1
  have c13634 := (publicBatchWrapper89_copies426 a h).2.2.1
  have c13635 := (publicBatchWrapper89_copies426 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_208_17 := arithEq_of_rows h (row := 208) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_17
  simp only [← c13625, ← c13626, ← c13627] at e_208_17
  have e_208_18 := arithEq_of_rows h (row := 208) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_18
  simp only [← c13628, ← c13629, ← c13630] at e_208_18
  have e_209_17 := arithEq_of_rows h (row := 209) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_17
  simp only [← c13619, k3, ← c13620, k3, ← c13621] at e_209_17
  have e_209_18 := arithEq_of_rows h (row := 209) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_18
  simp only [← c13622, ← c13623, k3, ← c13624] at e_209_18
  have e_209_19 := arithEq_of_rows h (row := 209) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_209_19
  simp only [← c13631, ← c13632, k3, ← c13633] at e_209_19
  refine ⟨?_, ?_⟩
  · have hc := e_208_17
    simp only [e_209_18] at hc
    linear_combination c13634.trans k2 - hc
  · have hc := e_209_19
    simp only [e_209_17, e_208_18, e_209_18] at hc
    linear_combination c13635.trans k2 - hc

theorem publicBatchWrapper89_f1999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4007)) (a (.wire 169 79)) (a (.virt 18767)) (a (.virt 18768)) := by
  have c13636 := (publicBatchWrapper89_copies426 a h).2.2.2.2.1
  have c13637 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.1
  have c13638 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.1
  have c13639 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.1
  have c13640 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.1
  have c13641 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.1
  have c13642 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13643 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13644 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13645 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13646 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13647 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13648 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13649 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13650 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13651 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13652 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_208_19 := arithEq_of_rows h (row := 208) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_208_19
  simp only [← c13642, ← c13643, ← c13644] at e_208_19
  have e_211_0 := arithEq_of_rows h (row := 211) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_0
  simp only [← c13636, k3, ← c13637, k3, ← c13638] at e_211_0
  have e_211_1 := arithEq_of_rows h (row := 211) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_1
  simp only [← c13639, ← c13640, k3, ← c13641] at e_211_1
  have e_211_2 := arithEq_of_rows h (row := 211) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_2
  simp only [← c13648, ← c13649, k3, ← c13650] at e_211_2
  have e_212_0 := arithEq_of_rows h (row := 212) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_0
  simp only [← c13645, ← c13646, ← c13647] at e_212_0
  refine ⟨?_, ?_⟩
  · have hc := e_208_19
    simp only [e_211_1] at hc
    linear_combination c13651.trans k2 - hc
  · have hc := e_211_2
    simp only [e_211_0, e_212_0, e_211_1] at hc
    linear_combination c13652.trans k2 - hc

theorem publicBatchWrapper89_f2000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4008)) (a (.wire 170 7)) (a (.virt 18769)) (a (.virt 18770)) := by
  have c13653 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13654 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13655 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13656 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13657 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13658 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13659 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13660 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13661 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13662 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13663 := (publicBatchWrapper89_copies426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13664 := (publicBatchWrapper89_copies427 a h).1
  have c13665 := (publicBatchWrapper89_copies427 a h).2.1
  have c13666 := (publicBatchWrapper89_copies427 a h).2.2.1
  have c13667 := (publicBatchWrapper89_copies427 a h).2.2.2.1
  have c13668 := (publicBatchWrapper89_copies427 a h).2.2.2.2.1
  have c13669 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_211_3 := arithEq_of_rows h (row := 211) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_3
  simp only [← c13653, k3, ← c13654, k3, ← c13655] at e_211_3
  have e_211_4 := arithEq_of_rows h (row := 211) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_4
  simp only [← c13656, ← c13657, k3, ← c13658] at e_211_4
  have e_211_5 := arithEq_of_rows h (row := 211) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_5
  simp only [← c13665, ← c13666, k3, ← c13667] at e_211_5
  have e_212_1 := arithEq_of_rows h (row := 212) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_1
  simp only [← c13659, ← c13660, ← c13661] at e_212_1
  have e_212_2 := arithEq_of_rows h (row := 212) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_2
  simp only [← c13662, ← c13663, ← c13664] at e_212_2
  refine ⟨?_, ?_⟩
  · have hc := e_212_1
    simp only [e_211_4] at hc
    linear_combination c13668.trans k2 - hc
  · have hc := e_211_5
    simp only [e_211_3, e_212_2, e_211_4] at hc
    linear_combination c13669.trans k2 - hc

theorem publicBatchWrapper89_f2001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 212 15) = band (a (.virt 18763)) (a (.virt 18765)) := by
  have c13670 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.1
  have c13671 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.1
  have c13672 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.1
  have e_212_3 := arithEq_of_rows h (row := 212) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_3
  simp only [← c13670, ← c13671, ← c13672] at e_212_3
  have hr := e_212_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 212 19) = band (a (.virt 18767)) (a (.virt 18769)) := by
  have c13673 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.1
  have c13674 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13675 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_212_4 := arithEq_of_rows h (row := 212) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_4
  simp only [← c13673, ← c13674, ← c13675] at e_212_4
  have hr := e_212_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 212 23) = band (a (.wire 212 15)) (a (.wire 212 19)) := by
  have c13676 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13677 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13678 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_212_5 := arithEq_of_rows h (row := 212) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_5
  simp only [← c13676, ← c13677, ← c13678] at e_212_5
  have hr := e_212_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 11) = bor (a (.wire 20 59)) (a (.wire 212 23)) := by
  have c13679 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13680 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13681 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13682 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13683 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13684 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_10 := arithEq_of_rows h (row := 204) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_10
  simp only [← c13679, ← c13680, ← c13681] at e_204_10
  have e_210_2 := arithEq_of_rows h (row := 210) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_2
  simp only [← c13682, ← c13683, k3, ← c13684] at e_210_2
  have hr := e_210_2
  simp only [e_204_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 11) = a (.virt 201) := by
  have c13685 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13685

theorem publicBatchWrapper89_f2006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4203)) (a (.wire 170 23)) (a (.virt 18771)) (a (.virt 18772)) := by
  have c13686 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13687 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13688 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13689 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13690 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13691 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13692 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13693 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13694 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13695 := (publicBatchWrapper89_copies427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13696 := (publicBatchWrapper89_copies428 a h).1
  have c13697 := (publicBatchWrapper89_copies428 a h).2.1
  have c13698 := (publicBatchWrapper89_copies428 a h).2.2.1
  have c13699 := (publicBatchWrapper89_copies428 a h).2.2.2.1
  have c13700 := (publicBatchWrapper89_copies428 a h).2.2.2.2.1
  have c13701 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.1
  have c13702 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_211_6 := arithEq_of_rows h (row := 211) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_6
  simp only [← c13686, k3, ← c13687, k3, ← c13688] at e_211_6
  have e_211_7 := arithEq_of_rows h (row := 211) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_7
  simp only [← c13689, ← c13690, k3, ← c13691] at e_211_7
  have e_211_8 := arithEq_of_rows h (row := 211) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_8
  simp only [← c13698, ← c13699, k3, ← c13700] at e_211_8
  have e_212_6 := arithEq_of_rows h (row := 212) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_6
  simp only [← c13692, ← c13693, ← c13694] at e_212_6
  have e_212_7 := arithEq_of_rows h (row := 212) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_7
  simp only [← c13695, ← c13696, ← c13697] at e_212_7
  refine ⟨?_, ?_⟩
  · have hc := e_212_6
    simp only [e_211_7] at hc
    linear_combination c13701.trans k2 - hc
  · have hc := e_211_8
    simp only [e_211_6, e_212_7, e_211_7] at hc
    linear_combination c13702.trans k2 - hc

theorem publicBatchWrapper89_f2007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 15) = bor (a (.wire 22 23)) (a (.virt 18771)) := by
  have c13703 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.1
  have c13704 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.1
  have c13705 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.1
  have c13706 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13707 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13708 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_11 := arithEq_of_rows h (row := 204) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_11
  simp only [← c13703, ← c13704, ← c13705] at e_204_11
  have e_210_3 := arithEq_of_rows h (row := 210) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_3
  simp only [← c13706, ← c13707, k3, ← c13708] at e_210_3
  have hr := e_210_3
  simp only [e_204_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 15) = a (.virt 201) := by
  have c13709 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13709

theorem publicBatchWrapper89_f2009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4204)) (a (.wire 170 31)) (a (.virt 18773)) (a (.virt 18774)) := by
  have c13710 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13711 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13712 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13713 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13714 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13715 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13716 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13717 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13718 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13719 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13720 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13721 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13722 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13723 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13724 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13725 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13726 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_211_9 := arithEq_of_rows h (row := 211) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_9
  simp only [← c13710, k3, ← c13711, k3, ← c13712] at e_211_9
  have e_211_10 := arithEq_of_rows h (row := 211) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_10
  simp only [← c13713, ← c13714, k3, ← c13715] at e_211_10
  have e_211_11 := arithEq_of_rows h (row := 211) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_11
  simp only [← c13722, ← c13723, k3, ← c13724] at e_211_11
  have e_212_8 := arithEq_of_rows h (row := 212) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_8
  simp only [← c13716, ← c13717, ← c13718] at e_212_8
  have e_212_9 := arithEq_of_rows h (row := 212) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_9
  simp only [← c13719, ← c13720, ← c13721] at e_212_9
  refine ⟨?_, ?_⟩
  · have hc := e_212_8
    simp only [e_211_10] at hc
    linear_combination c13725.trans k2 - hc
  · have hc := e_211_11
    simp only [e_211_9, e_212_9, e_211_10] at hc
    linear_combination c13726.trans k2 - hc

theorem publicBatchWrapper89_f2010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 19) = bor (a (.wire 22 23)) (a (.virt 18773)) := by
  have c13727 := (publicBatchWrapper89_copies428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13728 := (publicBatchWrapper89_copies429 a h).1
  have c13729 := (publicBatchWrapper89_copies429 a h).2.1
  have c13730 := (publicBatchWrapper89_copies429 a h).2.2.1
  have c13731 := (publicBatchWrapper89_copies429 a h).2.2.2.1
  have c13732 := (publicBatchWrapper89_copies429 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_12 := arithEq_of_rows h (row := 204) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_12
  simp only [← c13727, ← c13728, ← c13729] at e_204_12
  have e_210_4 := arithEq_of_rows h (row := 210) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_4
  simp only [← c13730, ← c13731, k3, ← c13732] at e_210_4
  have hr := e_210_4
  simp only [e_204_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 19) = a (.virt 201) := by
  have c13733 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.1
  exact c13733

theorem publicBatchWrapper89_f2012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4205)) (a (.wire 169 63)) (a (.virt 18775)) (a (.virt 18776)) := by
  have c13734 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.1
  have c13735 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.1
  have c13736 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.1
  have c13737 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.1
  have c13738 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13739 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13740 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13741 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13742 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13743 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13744 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13745 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13746 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13747 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13748 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13749 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13750 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_211_12 := arithEq_of_rows h (row := 211) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_12
  simp only [← c13734, k3, ← c13735, k3, ← c13736] at e_211_12
  have e_211_13 := arithEq_of_rows h (row := 211) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_13
  simp only [← c13737, ← c13738, k3, ← c13739] at e_211_13
  have e_211_14 := arithEq_of_rows h (row := 211) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_14
  simp only [← c13746, ← c13747, k3, ← c13748] at e_211_14
  have e_212_10 := arithEq_of_rows h (row := 212) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_10
  simp only [← c13740, ← c13741, ← c13742] at e_212_10
  have e_212_11 := arithEq_of_rows h (row := 212) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_11
  simp only [← c13743, ← c13744, ← c13745] at e_212_11
  refine ⟨?_, ?_⟩
  · have hc := e_212_10
    simp only [e_211_13] at hc
    linear_combination c13749.trans k2 - hc
  · have hc := e_211_14
    simp only [e_211_12, e_212_11, e_211_13] at hc
    linear_combination c13750.trans k2 - hc

theorem publicBatchWrapper89_f2013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4206)) (a (.wire 169 71)) (a (.virt 18777)) (a (.virt 18778)) := by
  have c13751 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13752 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13753 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13754 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13755 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13756 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13757 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13758 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13759 := (publicBatchWrapper89_copies429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13760 := (publicBatchWrapper89_copies430 a h).1
  have c13761 := (publicBatchWrapper89_copies430 a h).2.1
  have c13762 := (publicBatchWrapper89_copies430 a h).2.2.1
  have c13763 := (publicBatchWrapper89_copies430 a h).2.2.2.1
  have c13764 := (publicBatchWrapper89_copies430 a h).2.2.2.2.1
  have c13765 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.1
  have c13766 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.1
  have c13767 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_211_15 := arithEq_of_rows h (row := 211) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_15
  simp only [← c13751, k3, ← c13752, k3, ← c13753] at e_211_15
  have e_211_16 := arithEq_of_rows h (row := 211) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_16
  simp only [← c13754, ← c13755, k3, ← c13756] at e_211_16
  have e_211_17 := arithEq_of_rows h (row := 211) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_17
  simp only [← c13763, ← c13764, k3, ← c13765] at e_211_17
  have e_212_12 := arithEq_of_rows h (row := 212) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_12
  simp only [← c13757, ← c13758, ← c13759] at e_212_12
  have e_212_13 := arithEq_of_rows h (row := 212) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_13
  simp only [← c13760, ← c13761, ← c13762] at e_212_13
  refine ⟨?_, ?_⟩
  · have hc := e_212_12
    simp only [e_211_16] at hc
    linear_combination c13766.trans k2 - hc
  · have hc := e_211_17
    simp only [e_211_15, e_212_13, e_211_16] at hc
    linear_combination c13767.trans k2 - hc

theorem publicBatchWrapper89_f2014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4207)) (a (.wire 169 79)) (a (.virt 18779)) (a (.virt 18780)) := by
  have c13768 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.1
  have c13769 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.1
  have c13770 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13771 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13772 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13773 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13774 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13775 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13776 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13777 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13778 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13779 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13780 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13781 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13782 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13783 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13784 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_211_18 := arithEq_of_rows h (row := 211) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_18
  simp only [← c13768, k3, ← c13769, k3, ← c13770] at e_211_18
  have e_211_19 := arithEq_of_rows h (row := 211) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_211_19
  simp only [← c13771, ← c13772, k3, ← c13773] at e_211_19
  have e_212_14 := arithEq_of_rows h (row := 212) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_14
  simp only [← c13774, ← c13775, ← c13776] at e_212_14
  have e_212_15 := arithEq_of_rows h (row := 212) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_15
  simp only [← c13777, ← c13778, ← c13779] at e_212_15
  have e_213_0 := arithEq_of_rows h (row := 213) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_0
  simp only [← c13780, ← c13781, k3, ← c13782] at e_213_0
  refine ⟨?_, ?_⟩
  · have hc := e_212_14
    simp only [e_211_19] at hc
    linear_combination c13783.trans k2 - hc
  · have hc := e_213_0
    simp only [e_211_18, e_212_15, e_211_19] at hc
    linear_combination c13784.trans k2 - hc

theorem publicBatchWrapper89_f2015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4208)) (a (.wire 170 7)) (a (.virt 18781)) (a (.virt 18782)) := by
  have c13785 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13786 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13787 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13788 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13789 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13790 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13791 := (publicBatchWrapper89_copies430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13792 := (publicBatchWrapper89_copies431 a h).1
  have c13793 := (publicBatchWrapper89_copies431 a h).2.1
  have c13794 := (publicBatchWrapper89_copies431 a h).2.2.1
  have c13795 := (publicBatchWrapper89_copies431 a h).2.2.2.1
  have c13796 := (publicBatchWrapper89_copies431 a h).2.2.2.2.1
  have c13797 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.1
  have c13798 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.1
  have c13799 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.1
  have c13800 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.1
  have c13801 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_212_16 := arithEq_of_rows h (row := 212) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_16
  simp only [← c13791, ← c13792, ← c13793] at e_212_16
  have e_212_17 := arithEq_of_rows h (row := 212) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_17
  simp only [← c13794, ← c13795, ← c13796] at e_212_17
  have e_213_1 := arithEq_of_rows h (row := 213) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_1
  simp only [← c13785, k3, ← c13786, k3, ← c13787] at e_213_1
  have e_213_2 := arithEq_of_rows h (row := 213) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_2
  simp only [← c13788, ← c13789, k3, ← c13790] at e_213_2
  have e_213_3 := arithEq_of_rows h (row := 213) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_3
  simp only [← c13797, ← c13798, k3, ← c13799] at e_213_3
  refine ⟨?_, ?_⟩
  · have hc := e_212_16
    simp only [e_213_2] at hc
    linear_combination c13800.trans k2 - hc
  · have hc := e_213_3
    simp only [e_213_1, e_212_17, e_213_2] at hc
    linear_combination c13801.trans k2 - hc

theorem publicBatchWrapper89_f2016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 212 75) = band (a (.virt 18775)) (a (.virt 18777)) := by
  have c13802 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13803 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13804 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_212_18 := arithEq_of_rows h (row := 212) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_18
  simp only [← c13802, ← c13803, ← c13804] at e_212_18
  have hr := e_212_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 212 79) = band (a (.virt 18779)) (a (.virt 18781)) := by
  have c13805 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13806 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13807 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_212_19 := arithEq_of_rows h (row := 212) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_212_19
  simp only [← c13805, ← c13806, ← c13807] at e_212_19
  have hr := e_212_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 214 3) = band (a (.wire 212 75)) (a (.wire 212 79)) := by
  have c13808 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13809 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13810 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_214_0 := arithEq_of_rows h (row := 214) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_0
  simp only [← c13808, ← c13809, ← c13810] at e_214_0
  have hr := e_214_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 23) = bor (a (.wire 22 23)) (a (.wire 214 3)) := by
  have c13811 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13812 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13813 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13814 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13815 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13816 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_13 := arithEq_of_rows h (row := 204) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_13
  simp only [← c13811, ← c13812, ← c13813] at e_204_13
  have e_210_5 := arithEq_of_rows h (row := 210) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_5
  simp only [← c13814, ← c13815, k3, ← c13816] at e_210_5
  have hr := e_210_5
  simp only [e_204_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 23) = a (.virt 201) := by
  have c13817 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13817

theorem publicBatchWrapper89_f2021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4403)) (a (.wire 170 23)) (a (.virt 18783)) (a (.virt 18784)) := by
  have c13818 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13819 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13820 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13821 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13822 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13823 := (publicBatchWrapper89_copies431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13824 := (publicBatchWrapper89_copies432 a h).1
  have c13825 := (publicBatchWrapper89_copies432 a h).2.1
  have c13826 := (publicBatchWrapper89_copies432 a h).2.2.1
  have c13827 := (publicBatchWrapper89_copies432 a h).2.2.2.1
  have c13828 := (publicBatchWrapper89_copies432 a h).2.2.2.2.1
  have c13829 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.1
  have c13830 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.1
  have c13831 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.1
  have c13832 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.1
  have c13833 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.1
  have c13834 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_213_4 := arithEq_of_rows h (row := 213) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_4
  simp only [← c13818, k3, ← c13819, k3, ← c13820] at e_213_4
  have e_213_5 := arithEq_of_rows h (row := 213) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_5
  simp only [← c13821, ← c13822, k3, ← c13823] at e_213_5
  have e_213_6 := arithEq_of_rows h (row := 213) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_6
  simp only [← c13830, ← c13831, k3, ← c13832] at e_213_6
  have e_214_1 := arithEq_of_rows h (row := 214) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_1
  simp only [← c13824, ← c13825, ← c13826] at e_214_1
  have e_214_2 := arithEq_of_rows h (row := 214) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_2
  simp only [← c13827, ← c13828, ← c13829] at e_214_2
  refine ⟨?_, ?_⟩
  · have hc := e_214_1
    simp only [e_213_5] at hc
    linear_combination c13833.trans k2 - hc
  · have hc := e_213_6
    simp only [e_213_4, e_214_2, e_213_5] at hc
    linear_combination c13834.trans k2 - hc

theorem publicBatchWrapper89_f2022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 27) = bor (a (.wire 22 67)) (a (.virt 18783)) := by
  have c13835 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13836 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13837 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13838 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13839 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13840 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_14 := arithEq_of_rows h (row := 204) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_14
  simp only [← c13835, ← c13836, ← c13837] at e_204_14
  have e_210_6 := arithEq_of_rows h (row := 210) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_6
  simp only [← c13838, ← c13839, k3, ← c13840] at e_210_6
  have hr := e_210_6
  simp only [e_204_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 27) = a (.virt 201) := by
  have c13841 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13841

theorem publicBatchWrapper89_f2024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4404)) (a (.wire 170 31)) (a (.virt 18785)) (a (.virt 18786)) := by
  have c13842 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13843 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13844 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13845 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13846 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13847 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13848 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13849 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13850 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13851 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13852 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13853 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13854 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13855 := (publicBatchWrapper89_copies432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13856 := (publicBatchWrapper89_copies433 a h).1
  have c13857 := (publicBatchWrapper89_copies433 a h).2.1
  have c13858 := (publicBatchWrapper89_copies433 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_213_7 := arithEq_of_rows h (row := 213) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_7
  simp only [← c13842, k3, ← c13843, k3, ← c13844] at e_213_7
  have e_213_8 := arithEq_of_rows h (row := 213) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_8
  simp only [← c13845, ← c13846, k3, ← c13847] at e_213_8
  have e_213_9 := arithEq_of_rows h (row := 213) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_9
  simp only [← c13854, ← c13855, k3, ← c13856] at e_213_9
  have e_214_3 := arithEq_of_rows h (row := 214) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_3
  simp only [← c13848, ← c13849, ← c13850] at e_214_3
  have e_214_4 := arithEq_of_rows h (row := 214) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_4
  simp only [← c13851, ← c13852, ← c13853] at e_214_4
  refine ⟨?_, ?_⟩
  · have hc := e_214_3
    simp only [e_213_8] at hc
    linear_combination c13857.trans k2 - hc
  · have hc := e_213_9
    simp only [e_213_7, e_214_4, e_213_8] at hc
    linear_combination c13858.trans k2 - hc

theorem publicBatchWrapper89_f2025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 31) = bor (a (.wire 22 67)) (a (.virt 18785)) := by
  have c13859 := (publicBatchWrapper89_copies433 a h).2.2.2.1
  have c13860 := (publicBatchWrapper89_copies433 a h).2.2.2.2.1
  have c13861 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.1
  have c13862 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.1
  have c13863 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.1
  have c13864 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_15 := arithEq_of_rows h (row := 204) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_15
  simp only [← c13859, ← c13860, ← c13861] at e_204_15
  have e_210_7 := arithEq_of_rows h (row := 210) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_7
  simp only [← c13862, ← c13863, k3, ← c13864] at e_210_7
  have hr := e_210_7
  simp only [e_204_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 31) = a (.virt 201) := by
  have c13865 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.1
  exact c13865

theorem publicBatchWrapper89_f2027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4405)) (a (.wire 169 63)) (a (.virt 18787)) (a (.virt 18788)) := by
  have c13866 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13867 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13868 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13869 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13870 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13871 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13872 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13873 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13874 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13875 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13876 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13877 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13878 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13879 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13880 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13881 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13882 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_213_10 := arithEq_of_rows h (row := 213) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_10
  simp only [← c13866, k3, ← c13867, k3, ← c13868] at e_213_10
  have e_213_11 := arithEq_of_rows h (row := 213) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_11
  simp only [← c13869, ← c13870, k3, ← c13871] at e_213_11
  have e_213_12 := arithEq_of_rows h (row := 213) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_12
  simp only [← c13878, ← c13879, k3, ← c13880] at e_213_12
  have e_214_5 := arithEq_of_rows h (row := 214) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_5
  simp only [← c13872, ← c13873, ← c13874] at e_214_5
  have e_214_6 := arithEq_of_rows h (row := 214) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_6
  simp only [← c13875, ← c13876, ← c13877] at e_214_6
  refine ⟨?_, ?_⟩
  · have hc := e_214_5
    simp only [e_213_11] at hc
    linear_combination c13881.trans k2 - hc
  · have hc := e_213_12
    simp only [e_213_10, e_214_6, e_213_11] at hc
    linear_combination c13882.trans k2 - hc

theorem publicBatchWrapper89_f2028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4406)) (a (.wire 169 71)) (a (.virt 18789)) (a (.virt 18790)) := by
  have c13883 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13884 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13885 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13886 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13887 := (publicBatchWrapper89_copies433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13888 := (publicBatchWrapper89_copies434 a h).1
  have c13889 := (publicBatchWrapper89_copies434 a h).2.1
  have c13890 := (publicBatchWrapper89_copies434 a h).2.2.1
  have c13891 := (publicBatchWrapper89_copies434 a h).2.2.2.1
  have c13892 := (publicBatchWrapper89_copies434 a h).2.2.2.2.1
  have c13893 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.1
  have c13894 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.1
  have c13895 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.1
  have c13896 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.1
  have c13897 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.1
  have c13898 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13899 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_213_13 := arithEq_of_rows h (row := 213) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_13
  simp only [← c13883, k3, ← c13884, k3, ← c13885] at e_213_13
  have e_213_14 := arithEq_of_rows h (row := 213) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_14
  simp only [← c13886, ← c13887, k3, ← c13888] at e_213_14
  have e_213_15 := arithEq_of_rows h (row := 213) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_15
  simp only [← c13895, ← c13896, k3, ← c13897] at e_213_15
  have e_214_7 := arithEq_of_rows h (row := 214) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_7
  simp only [← c13889, ← c13890, ← c13891] at e_214_7
  have e_214_8 := arithEq_of_rows h (row := 214) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_8
  simp only [← c13892, ← c13893, ← c13894] at e_214_8
  refine ⟨?_, ?_⟩
  · have hc := e_214_7
    simp only [e_213_14] at hc
    linear_combination c13898.trans k2 - hc
  · have hc := e_213_15
    simp only [e_213_13, e_214_8, e_213_14] at hc
    linear_combination c13899.trans k2 - hc

theorem publicBatchWrapper89_f2029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4407)) (a (.wire 169 79)) (a (.virt 18791)) (a (.virt 18792)) := by
  have c13900 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13901 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13902 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13903 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13904 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13905 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13906 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13907 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13908 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13909 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13910 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13911 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13912 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13913 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13914 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13915 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13916 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_213_16 := arithEq_of_rows h (row := 213) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_16
  simp only [← c13900, k3, ← c13901, k3, ← c13902] at e_213_16
  have e_213_17 := arithEq_of_rows h (row := 213) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_17
  simp only [← c13903, ← c13904, k3, ← c13905] at e_213_17
  have e_213_18 := arithEq_of_rows h (row := 213) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_18
  simp only [← c13912, ← c13913, k3, ← c13914] at e_213_18
  have e_214_9 := arithEq_of_rows h (row := 214) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_9
  simp only [← c13906, ← c13907, ← c13908] at e_214_9
  have e_214_10 := arithEq_of_rows h (row := 214) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_10
  simp only [← c13909, ← c13910, ← c13911] at e_214_10
  refine ⟨?_, ?_⟩
  · have hc := e_214_9
    simp only [e_213_17] at hc
    linear_combination c13915.trans k2 - hc
  · have hc := e_213_18
    simp only [e_213_16, e_214_10, e_213_17] at hc
    linear_combination c13916.trans k2 - hc

theorem publicBatchWrapper89_f2030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4408)) (a (.wire 170 7)) (a (.virt 18793)) (a (.virt 18794)) := by
  have c13917 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13918 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13919 := (publicBatchWrapper89_copies434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13920 := (publicBatchWrapper89_copies435 a h).1
  have c13921 := (publicBatchWrapper89_copies435 a h).2.1
  have c13922 := (publicBatchWrapper89_copies435 a h).2.2.1
  have c13923 := (publicBatchWrapper89_copies435 a h).2.2.2.1
  have c13924 := (publicBatchWrapper89_copies435 a h).2.2.2.2.1
  have c13925 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.1
  have c13926 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.1
  have c13927 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.1
  have c13928 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.1
  have c13929 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.1
  have c13930 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13931 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13932 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13933 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_213_19 := arithEq_of_rows h (row := 213) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_213_19
  simp only [← c13917, k3, ← c13918, k3, ← c13919] at e_213_19
  have e_214_11 := arithEq_of_rows h (row := 214) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_11
  simp only [← c13923, ← c13924, ← c13925] at e_214_11
  have e_214_12 := arithEq_of_rows h (row := 214) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_12
  simp only [← c13926, ← c13927, ← c13928] at e_214_12
  have e_215_0 := arithEq_of_rows h (row := 215) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_0
  simp only [← c13920, ← c13921, k3, ← c13922] at e_215_0
  have e_215_1 := arithEq_of_rows h (row := 215) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_1
  simp only [← c13929, ← c13930, k3, ← c13931] at e_215_1
  refine ⟨?_, ?_⟩
  · have hc := e_214_11
    simp only [e_215_0] at hc
    linear_combination c13932.trans k2 - hc
  · have hc := e_215_1
    simp only [e_213_19, e_214_12, e_215_0] at hc
    linear_combination c13933.trans k2 - hc

theorem publicBatchWrapper89_f2031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 214 55) = band (a (.virt 18787)) (a (.virt 18789)) := by
  have c13934 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13935 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13936 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_214_13 := arithEq_of_rows h (row := 214) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_13
  simp only [← c13934, ← c13935, ← c13936] at e_214_13
  have hr := e_214_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 214 59) = band (a (.virt 18791)) (a (.virt 18793)) := by
  have c13937 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13938 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13939 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_214_14 := arithEq_of_rows h (row := 214) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_14
  simp only [← c13937, ← c13938, ← c13939] at e_214_14
  have hr := e_214_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 214 63) = band (a (.wire 214 55)) (a (.wire 214 59)) := by
  have c13940 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13941 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13942 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_214_15 := arithEq_of_rows h (row := 214) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_15
  simp only [← c13940, ← c13941, ← c13942] at e_214_15
  have hr := e_214_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 35) = bor (a (.wire 22 67)) (a (.wire 214 63)) := by
  have c13943 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13944 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13945 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13946 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13947 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13948 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_16 := arithEq_of_rows h (row := 204) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_16
  simp only [← c13943, ← c13944, ← c13945] at e_204_16
  have e_210_8 := arithEq_of_rows h (row := 210) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_8
  simp only [← c13946, ← c13947, k3, ← c13948] at e_210_8
  have hr := e_210_8
  simp only [e_204_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 35) = a (.virt 201) := by
  have c13949 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13949

theorem publicBatchWrapper89_f2036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4603)) (a (.wire 170 23)) (a (.virt 18795)) (a (.virt 18796)) := by
  have c13950 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13951 := (publicBatchWrapper89_copies435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13952 := (publicBatchWrapper89_copies436 a h).1
  have c13953 := (publicBatchWrapper89_copies436 a h).2.1
  have c13954 := (publicBatchWrapper89_copies436 a h).2.2.1
  have c13955 := (publicBatchWrapper89_copies436 a h).2.2.2.1
  have c13956 := (publicBatchWrapper89_copies436 a h).2.2.2.2.1
  have c13957 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.1
  have c13958 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.1
  have c13959 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.1
  have c13960 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.1
  have c13961 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.1
  have c13962 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13963 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13964 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13965 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13966 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_214_16 := arithEq_of_rows h (row := 214) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_16
  simp only [← c13956, ← c13957, ← c13958] at e_214_16
  have e_214_17 := arithEq_of_rows h (row := 214) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_17
  simp only [← c13959, ← c13960, ← c13961] at e_214_17
  have e_215_2 := arithEq_of_rows h (row := 215) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_2
  simp only [← c13950, k3, ← c13951, k3, ← c13952] at e_215_2
  have e_215_3 := arithEq_of_rows h (row := 215) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_3
  simp only [← c13953, ← c13954, k3, ← c13955] at e_215_3
  have e_215_4 := arithEq_of_rows h (row := 215) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_4
  simp only [← c13962, ← c13963, k3, ← c13964] at e_215_4
  refine ⟨?_, ?_⟩
  · have hc := e_214_16
    simp only [e_215_3] at hc
    linear_combination c13965.trans k2 - hc
  · have hc := e_215_4
    simp only [e_215_2, e_214_17, e_215_3] at hc
    linear_combination c13966.trans k2 - hc

theorem publicBatchWrapper89_f2037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 39) = bor (a (.wire 24 31)) (a (.virt 18795)) := by
  have c13967 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13968 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13969 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13970 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13971 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13972 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_17 := arithEq_of_rows h (row := 204) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_17
  simp only [← c13967, ← c13968, ← c13969] at e_204_17
  have e_210_9 := arithEq_of_rows h (row := 210) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_9
  simp only [← c13970, ← c13971, k3, ← c13972] at e_210_9
  have hr := e_210_9
  simp only [e_204_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 39) = a (.virt 201) := by
  have c13973 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13973

theorem publicBatchWrapper89_f2039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4604)) (a (.wire 170 31)) (a (.virt 18797)) (a (.virt 18798)) := by
  have c13974 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13975 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13976 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13977 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13978 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13979 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13980 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13981 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13982 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13983 := (publicBatchWrapper89_copies436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c13984 := (publicBatchWrapper89_copies437 a h).1
  have c13985 := (publicBatchWrapper89_copies437 a h).2.1
  have c13986 := (publicBatchWrapper89_copies437 a h).2.2.1
  have c13987 := (publicBatchWrapper89_copies437 a h).2.2.2.1
  have c13988 := (publicBatchWrapper89_copies437 a h).2.2.2.2.1
  have c13989 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.1
  have c13990 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_214_18 := arithEq_of_rows h (row := 214) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_18
  simp only [← c13980, ← c13981, ← c13982] at e_214_18
  have e_214_19 := arithEq_of_rows h (row := 214) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_214_19
  simp only [← c13983, ← c13984, ← c13985] at e_214_19
  have e_215_5 := arithEq_of_rows h (row := 215) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_5
  simp only [← c13974, k3, ← c13975, k3, ← c13976] at e_215_5
  have e_215_6 := arithEq_of_rows h (row := 215) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_6
  simp only [← c13977, ← c13978, k3, ← c13979] at e_215_6
  have e_215_7 := arithEq_of_rows h (row := 215) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_7
  simp only [← c13986, ← c13987, k3, ← c13988] at e_215_7
  refine ⟨?_, ?_⟩
  · have hc := e_214_18
    simp only [e_215_6] at hc
    linear_combination c13989.trans k2 - hc
  · have hc := e_215_7
    simp only [e_215_5, e_214_19, e_215_6] at hc
    linear_combination c13990.trans k2 - hc

theorem publicBatchWrapper89_f2040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 43) = bor (a (.wire 24 31)) (a (.virt 18797)) := by
  have c13991 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.1
  have c13992 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.1
  have c13993 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.1
  have c13994 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.1
  have c13995 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c13996 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_18 := arithEq_of_rows h (row := 204) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_18
  simp only [← c13991, ← c13992, ← c13993] at e_204_18
  have e_210_10 := arithEq_of_rows h (row := 210) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_10
  simp only [← c13994, ← c13995, k3, ← c13996] at e_210_10
  have hr := e_210_10
  simp only [e_204_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 43) = a (.virt 201) := by
  have c13997 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13997

theorem publicBatchWrapper89_f2042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4605)) (a (.wire 169 63)) (a (.virt 18799)) (a (.virt 18800)) := by
  have c13998 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13999 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14000 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14001 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14002 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14003 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14004 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14005 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14006 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14007 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14008 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14009 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14010 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14011 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14012 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14013 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14014 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_215_8 := arithEq_of_rows h (row := 215) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_8
  simp only [← c13998, k3, ← c13999, k3, ← c14000] at e_215_8
  have e_215_9 := arithEq_of_rows h (row := 215) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_9
  simp only [← c14001, ← c14002, k3, ← c14003] at e_215_9
  have e_215_10 := arithEq_of_rows h (row := 215) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_10
  simp only [← c14010, ← c14011, k3, ← c14012] at e_215_10
  have e_216_0 := arithEq_of_rows h (row := 216) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_0
  simp only [← c14004, ← c14005, ← c14006] at e_216_0
  have e_216_1 := arithEq_of_rows h (row := 216) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_1
  simp only [← c14007, ← c14008, ← c14009] at e_216_1
  refine ⟨?_, ?_⟩
  · have hc := e_216_0
    simp only [e_215_9] at hc
    linear_combination c14013.trans k2 - hc
  · have hc := e_215_10
    simp only [e_215_8, e_216_1, e_215_9] at hc
    linear_combination c14014.trans k2 - hc

theorem publicBatchWrapper89_f2043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4606)) (a (.wire 169 71)) (a (.virt 18801)) (a (.virt 18802)) := by
  have c14015 := (publicBatchWrapper89_copies437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14016 := (publicBatchWrapper89_copies438 a h).1
  have c14017 := (publicBatchWrapper89_copies438 a h).2.1
  have c14018 := (publicBatchWrapper89_copies438 a h).2.2.1
  have c14019 := (publicBatchWrapper89_copies438 a h).2.2.2.1
  have c14020 := (publicBatchWrapper89_copies438 a h).2.2.2.2.1
  have c14021 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.1
  have c14022 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.1
  have c14023 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.1
  have c14024 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.1
  have c14025 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.1
  have c14026 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14027 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14028 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14029 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14030 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14031 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_215_11 := arithEq_of_rows h (row := 215) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_11
  simp only [← c14015, k3, ← c14016, k3, ← c14017] at e_215_11
  have e_215_12 := arithEq_of_rows h (row := 215) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_12
  simp only [← c14018, ← c14019, k3, ← c14020] at e_215_12
  have e_215_13 := arithEq_of_rows h (row := 215) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_13
  simp only [← c14027, ← c14028, k3, ← c14029] at e_215_13
  have e_216_2 := arithEq_of_rows h (row := 216) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_2
  simp only [← c14021, ← c14022, ← c14023] at e_216_2
  have e_216_3 := arithEq_of_rows h (row := 216) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_3
  simp only [← c14024, ← c14025, ← c14026] at e_216_3
  refine ⟨?_, ?_⟩
  · have hc := e_216_2
    simp only [e_215_12] at hc
    linear_combination c14030.trans k2 - hc
  · have hc := e_215_13
    simp only [e_215_11, e_216_3, e_215_12] at hc
    linear_combination c14031.trans k2 - hc

theorem publicBatchWrapper89_f2044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4607)) (a (.wire 169 79)) (a (.virt 18803)) (a (.virt 18804)) := by
  have c14032 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14033 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14034 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14035 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14036 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14037 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14038 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14039 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14040 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14041 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14042 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14043 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14044 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14045 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14046 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14047 := (publicBatchWrapper89_copies438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14048 := (publicBatchWrapper89_copies439 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_215_14 := arithEq_of_rows h (row := 215) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_14
  simp only [← c14032, k3, ← c14033, k3, ← c14034] at e_215_14
  have e_215_15 := arithEq_of_rows h (row := 215) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_15
  simp only [← c14035, ← c14036, k3, ← c14037] at e_215_15
  have e_215_16 := arithEq_of_rows h (row := 215) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_16
  simp only [← c14044, ← c14045, k3, ← c14046] at e_215_16
  have e_216_4 := arithEq_of_rows h (row := 216) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_4
  simp only [← c14038, ← c14039, ← c14040] at e_216_4
  have e_216_5 := arithEq_of_rows h (row := 216) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_5
  simp only [← c14041, ← c14042, ← c14043] at e_216_5
  refine ⟨?_, ?_⟩
  · have hc := e_216_4
    simp only [e_215_15] at hc
    linear_combination c14047.trans k2 - hc
  · have hc := e_215_16
    simp only [e_215_14, e_216_5, e_215_15] at hc
    linear_combination c14048.trans k2 - hc

theorem publicBatchWrapper89_f2045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4608)) (a (.wire 170 7)) (a (.virt 18805)) (a (.virt 18806)) := by
  have c14049 := (publicBatchWrapper89_copies439 a h).2.1
  have c14050 := (publicBatchWrapper89_copies439 a h).2.2.1
  have c14051 := (publicBatchWrapper89_copies439 a h).2.2.2.1
  have c14052 := (publicBatchWrapper89_copies439 a h).2.2.2.2.1
  have c14053 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.1
  have c14054 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.1
  have c14055 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.1
  have c14056 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.1
  have c14057 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.1
  have c14058 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14059 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14060 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14061 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14062 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14063 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14064 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14065 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_215_17 := arithEq_of_rows h (row := 215) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_17
  simp only [← c14049, k3, ← c14050, k3, ← c14051] at e_215_17
  have e_215_18 := arithEq_of_rows h (row := 215) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_18
  simp only [← c14052, ← c14053, k3, ← c14054] at e_215_18
  have e_215_19 := arithEq_of_rows h (row := 215) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_215_19
  simp only [← c14061, ← c14062, k3, ← c14063] at e_215_19
  have e_216_6 := arithEq_of_rows h (row := 216) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_6
  simp only [← c14055, ← c14056, ← c14057] at e_216_6
  have e_216_7 := arithEq_of_rows h (row := 216) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_7
  simp only [← c14058, ← c14059, ← c14060] at e_216_7
  refine ⟨?_, ?_⟩
  · have hc := e_216_6
    simp only [e_215_18] at hc
    linear_combination c14064.trans k2 - hc
  · have hc := e_215_19
    simp only [e_215_17, e_216_7, e_215_18] at hc
    linear_combination c14065.trans k2 - hc

theorem publicBatchWrapper89_f2046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 216 35) = band (a (.virt 18799)) (a (.virt 18801)) := by
  have c14066 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14067 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14068 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_216_8 := arithEq_of_rows h (row := 216) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_8
  simp only [← c14066, ← c14067, ← c14068] at e_216_8
  have hr := e_216_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 216 39) = band (a (.virt 18803)) (a (.virt 18805)) := by
  have c14069 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14070 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14071 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_216_9 := arithEq_of_rows h (row := 216) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_9
  simp only [← c14069, ← c14070, ← c14071] at e_216_9
  have hr := e_216_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 216 43) = band (a (.wire 216 35)) (a (.wire 216 39)) := by
  have c14072 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14073 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14074 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_216_10 := arithEq_of_rows h (row := 216) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_10
  simp only [← c14072, ← c14073, ← c14074] at e_216_10
  have hr := e_216_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 47) = bor (a (.wire 24 31)) (a (.wire 216 43)) := by
  have c14075 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14076 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14077 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14078 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14079 := (publicBatchWrapper89_copies439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14080 := (publicBatchWrapper89_copies440 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_204_19 := arithEq_of_rows h (row := 204) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_204_19
  simp only [← c14075, ← c14076, ← c14077] at e_204_19
  have e_210_11 := arithEq_of_rows h (row := 210) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_11
  simp only [← c14078, ← c14079, k3, ← c14080] at e_210_11
  have hr := e_210_11
  simp only [e_204_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 47) = a (.virt 201) := by
  have c14081 := (publicBatchWrapper89_copies440 a h).2.1
  exact c14081

theorem publicBatchWrapper89_f2051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4803)) (a (.wire 170 23)) (a (.virt 18807)) (a (.virt 18808)) := by
  have c14082 := (publicBatchWrapper89_copies440 a h).2.2.1
  have c14083 := (publicBatchWrapper89_copies440 a h).2.2.2.1
  have c14084 := (publicBatchWrapper89_copies440 a h).2.2.2.2.1
  have c14085 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.1
  have c14086 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.1
  have c14087 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.1
  have c14088 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.1
  have c14089 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.1
  have c14090 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14091 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14092 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14093 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14094 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14095 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14096 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14097 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14098 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_216_11 := arithEq_of_rows h (row := 216) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_11
  simp only [← c14088, ← c14089, ← c14090] at e_216_11
  have e_216_12 := arithEq_of_rows h (row := 216) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_12
  simp only [← c14091, ← c14092, ← c14093] at e_216_12
  have e_217_0 := arithEq_of_rows h (row := 217) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_0
  simp only [← c14082, k3, ← c14083, k3, ← c14084] at e_217_0
  have e_217_1 := arithEq_of_rows h (row := 217) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_1
  simp only [← c14085, ← c14086, k3, ← c14087] at e_217_1
  have e_217_2 := arithEq_of_rows h (row := 217) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_2
  simp only [← c14094, ← c14095, k3, ← c14096] at e_217_2
  refine ⟨?_, ?_⟩
  · have hc := e_216_11
    simp only [e_217_1] at hc
    linear_combination c14097.trans k2 - hc
  · have hc := e_217_2
    simp only [e_217_0, e_216_12, e_217_1] at hc
    linear_combination c14098.trans k2 - hc

theorem publicBatchWrapper89_f2052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 51) = bor (a (.wire 24 75)) (a (.virt 18807)) := by
  have c14099 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14100 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14101 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14102 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14103 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14104 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_210_12 := arithEq_of_rows h (row := 210) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_12
  simp only [← c14102, ← c14103, k3, ← c14104] at e_210_12
  have e_218_0 := arithEq_of_rows h (row := 218) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_0
  simp only [← c14099, ← c14100, ← c14101] at e_218_0
  have hr := e_210_12
  simp only [e_218_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 51) = a (.virt 201) := by
  have c14105 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14105

theorem publicBatchWrapper89_f2054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4804)) (a (.wire 170 31)) (a (.virt 18809)) (a (.virt 18810)) := by
  have c14106 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14107 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14108 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14109 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14110 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14111 := (publicBatchWrapper89_copies440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14112 := (publicBatchWrapper89_copies441 a h).1
  have c14113 := (publicBatchWrapper89_copies441 a h).2.1
  have c14114 := (publicBatchWrapper89_copies441 a h).2.2.1
  have c14115 := (publicBatchWrapper89_copies441 a h).2.2.2.1
  have c14116 := (publicBatchWrapper89_copies441 a h).2.2.2.2.1
  have c14117 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.1
  have c14118 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.1
  have c14119 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.1
  have c14120 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.1
  have c14121 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.1
  have c14122 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_216_13 := arithEq_of_rows h (row := 216) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_13
  simp only [← c14112, ← c14113, ← c14114] at e_216_13
  have e_216_14 := arithEq_of_rows h (row := 216) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_14
  simp only [← c14115, ← c14116, ← c14117] at e_216_14
  have e_217_3 := arithEq_of_rows h (row := 217) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_3
  simp only [← c14106, k3, ← c14107, k3, ← c14108] at e_217_3
  have e_217_4 := arithEq_of_rows h (row := 217) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_4
  simp only [← c14109, ← c14110, k3, ← c14111] at e_217_4
  have e_217_5 := arithEq_of_rows h (row := 217) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_5
  simp only [← c14118, ← c14119, k3, ← c14120] at e_217_5
  refine ⟨?_, ?_⟩
  · have hc := e_216_13
    simp only [e_217_4] at hc
    linear_combination c14121.trans k2 - hc
  · have hc := e_217_5
    simp only [e_217_3, e_216_14, e_217_4] at hc
    linear_combination c14122.trans k2 - hc

theorem publicBatchWrapper89_f2055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 55) = bor (a (.wire 24 75)) (a (.virt 18809)) := by
  have c14123 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14124 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14125 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14126 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14127 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14128 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_210_13 := arithEq_of_rows h (row := 210) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_13
  simp only [← c14126, ← c14127, k3, ← c14128] at e_210_13
  have e_218_1 := arithEq_of_rows h (row := 218) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_1
  simp only [← c14123, ← c14124, ← c14125] at e_218_1
  have hr := e_210_13
  simp only [e_218_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 55) = a (.virt 201) := by
  have c14129 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14129

theorem publicBatchWrapper89_f2057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4805)) (a (.wire 169 63)) (a (.virt 18811)) (a (.virt 18812)) := by
  have c14130 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14131 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14132 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14133 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14134 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14135 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14136 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14137 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14138 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14139 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14140 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14141 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14142 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14143 := (publicBatchWrapper89_copies441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14144 := (publicBatchWrapper89_copies442 a h).1
  have c14145 := (publicBatchWrapper89_copies442 a h).2.1
  have c14146 := (publicBatchWrapper89_copies442 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_216_15 := arithEq_of_rows h (row := 216) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_15
  simp only [← c14136, ← c14137, ← c14138] at e_216_15
  have e_216_16 := arithEq_of_rows h (row := 216) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_16
  simp only [← c14139, ← c14140, ← c14141] at e_216_16
  have e_217_6 := arithEq_of_rows h (row := 217) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_6
  simp only [← c14130, k3, ← c14131, k3, ← c14132] at e_217_6
  have e_217_7 := arithEq_of_rows h (row := 217) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_7
  simp only [← c14133, ← c14134, k3, ← c14135] at e_217_7
  have e_217_8 := arithEq_of_rows h (row := 217) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_8
  simp only [← c14142, ← c14143, k3, ← c14144] at e_217_8
  refine ⟨?_, ?_⟩
  · have hc := e_216_15
    simp only [e_217_7] at hc
    linear_combination c14145.trans k2 - hc
  · have hc := e_217_8
    simp only [e_217_6, e_216_16, e_217_7] at hc
    linear_combination c14146.trans k2 - hc

theorem publicBatchWrapper89_f2058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4806)) (a (.wire 169 71)) (a (.virt 18813)) (a (.virt 18814)) := by
  have c14147 := (publicBatchWrapper89_copies442 a h).2.2.2.1
  have c14148 := (publicBatchWrapper89_copies442 a h).2.2.2.2.1
  have c14149 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.1
  have c14150 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.1
  have c14151 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.1
  have c14152 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.1
  have c14153 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.1
  have c14154 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14155 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14156 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14157 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14158 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14159 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14160 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14161 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14162 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14163 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_216_17 := arithEq_of_rows h (row := 216) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_17
  simp only [← c14153, ← c14154, ← c14155] at e_216_17
  have e_216_18 := arithEq_of_rows h (row := 216) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_18
  simp only [← c14156, ← c14157, ← c14158] at e_216_18
  have e_217_9 := arithEq_of_rows h (row := 217) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_9
  simp only [← c14147, k3, ← c14148, k3, ← c14149] at e_217_9
  have e_217_10 := arithEq_of_rows h (row := 217) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_10
  simp only [← c14150, ← c14151, k3, ← c14152] at e_217_10
  have e_217_11 := arithEq_of_rows h (row := 217) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_11
  simp only [← c14159, ← c14160, k3, ← c14161] at e_217_11
  refine ⟨?_, ?_⟩
  · have hc := e_216_17
    simp only [e_217_10] at hc
    linear_combination c14162.trans k2 - hc
  · have hc := e_217_11
    simp only [e_217_9, e_216_18, e_217_10] at hc
    linear_combination c14163.trans k2 - hc

theorem publicBatchWrapper89_f2059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4807)) (a (.wire 169 79)) (a (.virt 18815)) (a (.virt 18816)) := by
  have c14164 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14165 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14166 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14167 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14168 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14169 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14170 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14171 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14172 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14173 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14174 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14175 := (publicBatchWrapper89_copies442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14176 := (publicBatchWrapper89_copies443 a h).1
  have c14177 := (publicBatchWrapper89_copies443 a h).2.1
  have c14178 := (publicBatchWrapper89_copies443 a h).2.2.1
  have c14179 := (publicBatchWrapper89_copies443 a h).2.2.2.1
  have c14180 := (publicBatchWrapper89_copies443 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_216_19 := arithEq_of_rows h (row := 216) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_216_19
  simp only [← c14170, ← c14171, ← c14172] at e_216_19
  have e_217_12 := arithEq_of_rows h (row := 217) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_12
  simp only [← c14164, k3, ← c14165, k3, ← c14166] at e_217_12
  have e_217_13 := arithEq_of_rows h (row := 217) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_13
  simp only [← c14167, ← c14168, k3, ← c14169] at e_217_13
  have e_217_14 := arithEq_of_rows h (row := 217) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_14
  simp only [← c14176, ← c14177, k3, ← c14178] at e_217_14
  have e_219_0 := arithEq_of_rows h (row := 219) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_0
  simp only [← c14173, ← c14174, ← c14175] at e_219_0
  refine ⟨?_, ?_⟩
  · have hc := e_216_19
    simp only [e_217_13] at hc
    linear_combination c14179.trans k2 - hc
  · have hc := e_217_14
    simp only [e_217_12, e_219_0, e_217_13] at hc
    linear_combination c14180.trans k2 - hc

theorem publicBatchWrapper89_f2060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4808)) (a (.wire 170 7)) (a (.virt 18817)) (a (.virt 18818)) := by
  have c14181 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.1
  have c14182 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.1
  have c14183 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.1
  have c14184 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.1
  have c14185 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.1
  have c14186 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14187 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14188 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14189 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14190 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14191 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14192 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14193 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14194 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14195 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14196 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14197 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_217_15 := arithEq_of_rows h (row := 217) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_15
  simp only [← c14181, k3, ← c14182, k3, ← c14183] at e_217_15
  have e_217_16 := arithEq_of_rows h (row := 217) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_16
  simp only [← c14184, ← c14185, k3, ← c14186] at e_217_16
  have e_217_17 := arithEq_of_rows h (row := 217) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_17
  simp only [← c14193, ← c14194, k3, ← c14195] at e_217_17
  have e_219_1 := arithEq_of_rows h (row := 219) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_1
  simp only [← c14187, ← c14188, ← c14189] at e_219_1
  have e_219_2 := arithEq_of_rows h (row := 219) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_2
  simp only [← c14190, ← c14191, ← c14192] at e_219_2
  refine ⟨?_, ?_⟩
  · have hc := e_219_1
    simp only [e_217_16] at hc
    linear_combination c14196.trans k2 - hc
  · have hc := e_217_17
    simp only [e_217_15, e_219_2, e_217_16] at hc
    linear_combination c14197.trans k2 - hc

theorem publicBatchWrapper89_f2061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 219 15) = band (a (.virt 18811)) (a (.virt 18813)) := by
  have c14198 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14199 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14200 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_219_3 := arithEq_of_rows h (row := 219) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_3
  simp only [← c14198, ← c14199, ← c14200] at e_219_3
  have hr := e_219_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 219 19) = band (a (.virt 18815)) (a (.virt 18817)) := by
  have c14201 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14202 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14203 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_219_4 := arithEq_of_rows h (row := 219) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_4
  simp only [← c14201, ← c14202, ← c14203] at e_219_4
  have hr := e_219_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 219 23) = band (a (.wire 219 15)) (a (.wire 219 19)) := by
  have c14204 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14205 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14206 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_219_5 := arithEq_of_rows h (row := 219) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_5
  simp only [← c14204, ← c14205, ← c14206] at e_219_5
  have hr := e_219_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 59) = bor (a (.wire 24 75)) (a (.wire 219 23)) := by
  have c14207 := (publicBatchWrapper89_copies443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14208 := (publicBatchWrapper89_copies444 a h).1
  have c14209 := (publicBatchWrapper89_copies444 a h).2.1
  have c14210 := (publicBatchWrapper89_copies444 a h).2.2.1
  have c14211 := (publicBatchWrapper89_copies444 a h).2.2.2.1
  have c14212 := (publicBatchWrapper89_copies444 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_210_14 := arithEq_of_rows h (row := 210) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_14
  simp only [← c14210, ← c14211, k3, ← c14212] at e_210_14
  have e_218_2 := arithEq_of_rows h (row := 218) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_2
  simp only [← c14207, ← c14208, ← c14209] at e_218_2
  have hr := e_210_14
  simp only [e_218_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 59) = a (.virt 201) := by
  have c14213 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.1
  exact c14213

theorem publicBatchWrapper89_f2066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5003)) (a (.wire 170 23)) (a (.virt 18819)) (a (.virt 18820)) := by
  have c14214 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.1
  have c14215 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.1
  have c14216 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.1
  have c14217 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.1
  have c14218 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14219 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14220 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14221 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14222 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14223 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14224 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14225 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14226 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14227 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14228 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14229 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14230 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_217_18 := arithEq_of_rows h (row := 217) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_18
  simp only [← c14214, k3, ← c14215, k3, ← c14216] at e_217_18
  have e_217_19 := arithEq_of_rows h (row := 217) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_217_19
  simp only [← c14217, ← c14218, k3, ← c14219] at e_217_19
  have e_219_6 := arithEq_of_rows h (row := 219) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_6
  simp only [← c14220, ← c14221, ← c14222] at e_219_6
  have e_219_7 := arithEq_of_rows h (row := 219) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_7
  simp only [← c14223, ← c14224, ← c14225] at e_219_7
  have e_220_0 := arithEq_of_rows h (row := 220) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_0
  simp only [← c14226, ← c14227, k3, ← c14228] at e_220_0
  refine ⟨?_, ?_⟩
  · have hc := e_219_6
    simp only [e_217_19] at hc
    linear_combination c14229.trans k2 - hc
  · have hc := e_220_0
    simp only [e_217_18, e_219_7, e_217_19] at hc
    linear_combination c14230.trans k2 - hc

theorem publicBatchWrapper89_f2067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 63) = bor (a (.wire 26 39)) (a (.virt 18819)) := by
  have c14231 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14232 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14233 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14234 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14235 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14236 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_210_15 := arithEq_of_rows h (row := 210) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_15
  simp only [← c14234, ← c14235, k3, ← c14236] at e_210_15
  have e_218_3 := arithEq_of_rows h (row := 218) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_3
  simp only [← c14231, ← c14232, ← c14233] at e_218_3
  have hr := e_210_15
  simp only [e_218_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 63) = a (.virt 201) := by
  have c14237 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14237

theorem publicBatchWrapper89_f2069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5004)) (a (.wire 170 31)) (a (.virt 18821)) (a (.virt 18822)) := by
  have c14238 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14239 := (publicBatchWrapper89_copies444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14240 := (publicBatchWrapper89_copies445 a h).1
  have c14241 := (publicBatchWrapper89_copies445 a h).2.1
  have c14242 := (publicBatchWrapper89_copies445 a h).2.2.1
  have c14243 := (publicBatchWrapper89_copies445 a h).2.2.2.1
  have c14244 := (publicBatchWrapper89_copies445 a h).2.2.2.2.1
  have c14245 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.1
  have c14246 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.1
  have c14247 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.1
  have c14248 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.1
  have c14249 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.1
  have c14250 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14251 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14252 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14253 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14254 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_219_8 := arithEq_of_rows h (row := 219) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_8
  simp only [← c14244, ← c14245, ← c14246] at e_219_8
  have e_219_9 := arithEq_of_rows h (row := 219) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_9
  simp only [← c14247, ← c14248, ← c14249] at e_219_9
  have e_220_1 := arithEq_of_rows h (row := 220) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_1
  simp only [← c14238, k3, ← c14239, k3, ← c14240] at e_220_1
  have e_220_2 := arithEq_of_rows h (row := 220) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_2
  simp only [← c14241, ← c14242, k3, ← c14243] at e_220_2
  have e_220_3 := arithEq_of_rows h (row := 220) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_3
  simp only [← c14250, ← c14251, k3, ← c14252] at e_220_3
  refine ⟨?_, ?_⟩
  · have hc := e_219_8
    simp only [e_220_2] at hc
    linear_combination c14253.trans k2 - hc
  · have hc := e_220_3
    simp only [e_220_1, e_219_9, e_220_2] at hc
    linear_combination c14254.trans k2 - hc

theorem publicBatchWrapper89_f2070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 67) = bor (a (.wire 26 39)) (a (.virt 18821)) := by
  have c14255 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14256 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14257 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14258 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14259 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14260 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_210_16 := arithEq_of_rows h (row := 210) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_16
  simp only [← c14258, ← c14259, k3, ← c14260] at e_210_16
  have e_218_4 := arithEq_of_rows h (row := 218) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_4
  simp only [← c14255, ← c14256, ← c14257] at e_218_4
  have hr := e_210_16
  simp only [e_218_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 67) = a (.virt 201) := by
  have c14261 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14261

theorem publicBatchWrapper89_f2072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5005)) (a (.wire 169 63)) (a (.virt 18823)) (a (.virt 18824)) := by
  have c14262 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14263 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14264 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14265 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14266 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14267 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14268 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14269 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14270 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14271 := (publicBatchWrapper89_copies445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14272 := (publicBatchWrapper89_copies446 a h).1
  have c14273 := (publicBatchWrapper89_copies446 a h).2.1
  have c14274 := (publicBatchWrapper89_copies446 a h).2.2.1
  have c14275 := (publicBatchWrapper89_copies446 a h).2.2.2.1
  have c14276 := (publicBatchWrapper89_copies446 a h).2.2.2.2.1
  have c14277 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.1
  have c14278 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_219_10 := arithEq_of_rows h (row := 219) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_10
  simp only [← c14268, ← c14269, ← c14270] at e_219_10
  have e_219_11 := arithEq_of_rows h (row := 219) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_11
  simp only [← c14271, ← c14272, ← c14273] at e_219_11
  have e_220_4 := arithEq_of_rows h (row := 220) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_4
  simp only [← c14262, k3, ← c14263, k3, ← c14264] at e_220_4
  have e_220_5 := arithEq_of_rows h (row := 220) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_5
  simp only [← c14265, ← c14266, k3, ← c14267] at e_220_5
  have e_220_6 := arithEq_of_rows h (row := 220) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_6
  simp only [← c14274, ← c14275, k3, ← c14276] at e_220_6
  refine ⟨?_, ?_⟩
  · have hc := e_219_10
    simp only [e_220_5] at hc
    linear_combination c14277.trans k2 - hc
  · have hc := e_220_6
    simp only [e_220_4, e_219_11, e_220_5] at hc
    linear_combination c14278.trans k2 - hc

theorem publicBatchWrapper89_f2073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5006)) (a (.wire 169 71)) (a (.virt 18825)) (a (.virt 18826)) := by
  have c14279 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.1
  have c14280 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.1
  have c14281 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.1
  have c14282 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14283 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14284 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14285 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14286 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14287 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14288 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14289 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14290 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14291 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14292 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14293 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14294 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14295 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_219_12 := arithEq_of_rows h (row := 219) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_12
  simp only [← c14285, ← c14286, ← c14287] at e_219_12
  have e_219_13 := arithEq_of_rows h (row := 219) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_13
  simp only [← c14288, ← c14289, ← c14290] at e_219_13
  have e_220_7 := arithEq_of_rows h (row := 220) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_7
  simp only [← c14279, k3, ← c14280, k3, ← c14281] at e_220_7
  have e_220_8 := arithEq_of_rows h (row := 220) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_8
  simp only [← c14282, ← c14283, k3, ← c14284] at e_220_8
  have e_220_9 := arithEq_of_rows h (row := 220) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_9
  simp only [← c14291, ← c14292, k3, ← c14293] at e_220_9
  refine ⟨?_, ?_⟩
  · have hc := e_219_12
    simp only [e_220_8] at hc
    linear_combination c14294.trans k2 - hc
  · have hc := e_220_9
    simp only [e_220_7, e_219_13, e_220_8] at hc
    linear_combination c14295.trans k2 - hc

theorem publicBatchWrapper89_f2074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5007)) (a (.wire 169 79)) (a (.virt 18827)) (a (.virt 18828)) := by
  have c14296 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14297 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14298 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14299 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14300 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14301 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14302 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14303 := (publicBatchWrapper89_copies446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14304 := (publicBatchWrapper89_copies447 a h).1
  have c14305 := (publicBatchWrapper89_copies447 a h).2.1
  have c14306 := (publicBatchWrapper89_copies447 a h).2.2.1
  have c14307 := (publicBatchWrapper89_copies447 a h).2.2.2.1
  have c14308 := (publicBatchWrapper89_copies447 a h).2.2.2.2.1
  have c14309 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.1
  have c14310 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.1
  have c14311 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.1
  have c14312 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_219_14 := arithEq_of_rows h (row := 219) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_14
  simp only [← c14302, ← c14303, ← c14304] at e_219_14
  have e_219_15 := arithEq_of_rows h (row := 219) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_15
  simp only [← c14305, ← c14306, ← c14307] at e_219_15
  have e_220_10 := arithEq_of_rows h (row := 220) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_10
  simp only [← c14296, k3, ← c14297, k3, ← c14298] at e_220_10
  have e_220_11 := arithEq_of_rows h (row := 220) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_11
  simp only [← c14299, ← c14300, k3, ← c14301] at e_220_11
  have e_220_12 := arithEq_of_rows h (row := 220) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_12
  simp only [← c14308, ← c14309, k3, ← c14310] at e_220_12
  refine ⟨?_, ?_⟩
  · have hc := e_219_14
    simp only [e_220_11] at hc
    linear_combination c14311.trans k2 - hc
  · have hc := e_220_12
    simp only [e_220_10, e_219_15, e_220_11] at hc
    linear_combination c14312.trans k2 - hc

theorem publicBatchWrapper89_f2075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5008)) (a (.wire 170 7)) (a (.virt 18829)) (a (.virt 18830)) := by
  have c14313 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.1
  have c14314 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14315 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14316 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14317 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14318 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14319 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14320 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14321 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14322 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14323 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14324 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14325 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14326 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14327 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14328 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14329 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_219_16 := arithEq_of_rows h (row := 219) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_16
  simp only [← c14319, ← c14320, ← c14321] at e_219_16
  have e_219_17 := arithEq_of_rows h (row := 219) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_17
  simp only [← c14322, ← c14323, ← c14324] at e_219_17
  have e_220_13 := arithEq_of_rows h (row := 220) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_13
  simp only [← c14313, k3, ← c14314, k3, ← c14315] at e_220_13
  have e_220_14 := arithEq_of_rows h (row := 220) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_14
  simp only [← c14316, ← c14317, k3, ← c14318] at e_220_14
  have e_220_15 := arithEq_of_rows h (row := 220) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_15
  simp only [← c14325, ← c14326, k3, ← c14327] at e_220_15
  refine ⟨?_, ?_⟩
  · have hc := e_219_16
    simp only [e_220_14] at hc
    linear_combination c14328.trans k2 - hc
  · have hc := e_220_15
    simp only [e_220_13, e_219_17, e_220_14] at hc
    linear_combination c14329.trans k2 - hc

theorem publicBatchWrapper89_f2076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 219 75) = band (a (.virt 18823)) (a (.virt 18825)) := by
  have c14330 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14331 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14332 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_219_18 := arithEq_of_rows h (row := 219) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_18
  simp only [← c14330, ← c14331, ← c14332] at e_219_18
  have hr := e_219_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 219 79) = band (a (.virt 18827)) (a (.virt 18829)) := by
  have c14333 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14334 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14335 := (publicBatchWrapper89_copies447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_219_19 := arithEq_of_rows h (row := 219) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_219_19
  simp only [← c14333, ← c14334, ← c14335] at e_219_19
  have hr := e_219_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 221 3) = band (a (.wire 219 75)) (a (.wire 219 79)) := by
  have c14336 := (publicBatchWrapper89_copies448 a h).1
  have c14337 := (publicBatchWrapper89_copies448 a h).2.1
  have c14338 := (publicBatchWrapper89_copies448 a h).2.2.1
  have e_221_0 := arithEq_of_rows h (row := 221) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_0
  simp only [← c14336, ← c14337, ← c14338] at e_221_0
  have hr := e_221_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 71) = bor (a (.wire 26 39)) (a (.wire 221 3)) := by
  have c14339 := (publicBatchWrapper89_copies448 a h).2.2.2.1
  have c14340 := (publicBatchWrapper89_copies448 a h).2.2.2.2.1
  have c14341 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.1
  have c14342 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.1
  have c14343 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.1
  have c14344 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_210_17 := arithEq_of_rows h (row := 210) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_17
  simp only [← c14342, ← c14343, k3, ← c14344] at e_210_17
  have e_218_5 := arithEq_of_rows h (row := 218) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_5
  simp only [← c14339, ← c14340, ← c14341] at e_218_5
  have hr := e_210_17
  simp only [e_218_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 71) = a (.virt 201) := by
  have c14345 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.1
  exact c14345

theorem publicBatchWrapper89_f2081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5203)) (a (.wire 170 23)) (a (.virt 18831)) (a (.virt 18832)) := by
  have c14346 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14347 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14348 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14349 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14350 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14351 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14352 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14353 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14354 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14355 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14356 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14357 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14358 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14359 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14360 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14361 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14362 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_220_16 := arithEq_of_rows h (row := 220) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_16
  simp only [← c14346, k3, ← c14347, k3, ← c14348] at e_220_16
  have e_220_17 := arithEq_of_rows h (row := 220) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_17
  simp only [← c14349, ← c14350, k3, ← c14351] at e_220_17
  have e_220_18 := arithEq_of_rows h (row := 220) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_18
  simp only [← c14358, ← c14359, k3, ← c14360] at e_220_18
  have e_221_1 := arithEq_of_rows h (row := 221) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_1
  simp only [← c14352, ← c14353, ← c14354] at e_221_1
  have e_221_2 := arithEq_of_rows h (row := 221) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_2
  simp only [← c14355, ← c14356, ← c14357] at e_221_2
  refine ⟨?_, ?_⟩
  · have hc := e_221_1
    simp only [e_220_17] at hc
    linear_combination c14361.trans k2 - hc
  · have hc := e_220_18
    simp only [e_220_16, e_221_2, e_220_17] at hc
    linear_combination c14362.trans k2 - hc

theorem publicBatchWrapper89_f2082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 75) = bor (a (.wire 27 3)) (a (.virt 18831)) := by
  have c14363 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14364 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14365 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14366 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14367 := (publicBatchWrapper89_copies448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14368 := (publicBatchWrapper89_copies449 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_210_18 := arithEq_of_rows h (row := 210) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_18
  simp only [← c14366, ← c14367, k3, ← c14368] at e_210_18
  have e_218_6 := arithEq_of_rows h (row := 218) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_6
  simp only [← c14363, ← c14364, ← c14365] at e_218_6
  have hr := e_210_18
  simp only [e_218_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 75) = a (.virt 201) := by
  have c14369 := (publicBatchWrapper89_copies449 a h).2.1
  exact c14369

theorem publicBatchWrapper89_f2084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5204)) (a (.wire 170 31)) (a (.virt 18833)) (a (.virt 18834)) := by
  have c14370 := (publicBatchWrapper89_copies449 a h).2.2.1
  have c14371 := (publicBatchWrapper89_copies449 a h).2.2.2.1
  have c14372 := (publicBatchWrapper89_copies449 a h).2.2.2.2.1
  have c14373 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.1
  have c14374 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.1
  have c14375 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.1
  have c14376 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.1
  have c14377 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.1
  have c14378 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14379 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14380 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14381 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14382 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14383 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14384 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14385 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14386 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_220_19 := arithEq_of_rows h (row := 220) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_220_19
  simp only [← c14370, k3, ← c14371, k3, ← c14372] at e_220_19
  have e_221_3 := arithEq_of_rows h (row := 221) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_3
  simp only [← c14376, ← c14377, ← c14378] at e_221_3
  have e_221_4 := arithEq_of_rows h (row := 221) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_4
  simp only [← c14379, ← c14380, ← c14381] at e_221_4
  have e_222_0 := arithEq_of_rows h (row := 222) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_0
  simp only [← c14373, ← c14374, k3, ← c14375] at e_222_0
  have e_222_1 := arithEq_of_rows h (row := 222) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_1
  simp only [← c14382, ← c14383, k3, ← c14384] at e_222_1
  refine ⟨?_, ?_⟩
  · have hc := e_221_3
    simp only [e_222_0] at hc
    linear_combination c14385.trans k2 - hc
  · have hc := e_222_1
    simp only [e_220_19, e_221_4, e_222_0] at hc
    linear_combination c14386.trans k2 - hc

theorem publicBatchWrapper89_f2085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 79) = bor (a (.wire 27 3)) (a (.virt 18833)) := by
  have c14387 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14388 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14389 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14390 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14391 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14392 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_210_19 := arithEq_of_rows h (row := 210) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_210_19
  simp only [← c14390, ← c14391, k3, ← c14392] at e_210_19
  have e_218_7 := arithEq_of_rows h (row := 218) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_7
  simp only [← c14387, ← c14388, ← c14389] at e_218_7
  have hr := e_210_19
  simp only [e_218_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 210 79) = a (.virt 201) := by
  have c14393 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14393

theorem publicBatchWrapper89_f2087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5205)) (a (.wire 169 63)) (a (.virt 18835)) (a (.virt 18836)) := by
  have c14394 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14395 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14396 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14397 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14398 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14399 := (publicBatchWrapper89_copies449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14400 := (publicBatchWrapper89_copies450 a h).1
  have c14401 := (publicBatchWrapper89_copies450 a h).2.1
  have c14402 := (publicBatchWrapper89_copies450 a h).2.2.1
  have c14403 := (publicBatchWrapper89_copies450 a h).2.2.2.1
  have c14404 := (publicBatchWrapper89_copies450 a h).2.2.2.2.1
  have c14405 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.1
  have c14406 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.1
  have c14407 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.1
  have c14408 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.1
  have c14409 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.1
  have c14410 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_221_5 := arithEq_of_rows h (row := 221) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_5
  simp only [← c14400, ← c14401, ← c14402] at e_221_5
  have e_221_6 := arithEq_of_rows h (row := 221) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_6
  simp only [← c14403, ← c14404, ← c14405] at e_221_6
  have e_222_2 := arithEq_of_rows h (row := 222) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_2
  simp only [← c14394, k3, ← c14395, k3, ← c14396] at e_222_2
  have e_222_3 := arithEq_of_rows h (row := 222) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_3
  simp only [← c14397, ← c14398, k3, ← c14399] at e_222_3
  have e_222_4 := arithEq_of_rows h (row := 222) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_4
  simp only [← c14406, ← c14407, k3, ← c14408] at e_222_4
  refine ⟨?_, ?_⟩
  · have hc := e_221_5
    simp only [e_222_3] at hc
    linear_combination c14409.trans k2 - hc
  · have hc := e_222_4
    simp only [e_222_2, e_221_6, e_222_3] at hc
    linear_combination c14410.trans k2 - hc

theorem publicBatchWrapper89_f2088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5206)) (a (.wire 169 71)) (a (.virt 18837)) (a (.virt 18838)) := by
  have c14411 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14412 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14413 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14414 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14415 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14416 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14417 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14418 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14419 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14420 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14421 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14422 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14423 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14424 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14425 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14426 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14427 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_221_7 := arithEq_of_rows h (row := 221) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_7
  simp only [← c14417, ← c14418, ← c14419] at e_221_7
  have e_221_8 := arithEq_of_rows h (row := 221) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_8
  simp only [← c14420, ← c14421, ← c14422] at e_221_8
  have e_222_5 := arithEq_of_rows h (row := 222) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_5
  simp only [← c14411, k3, ← c14412, k3, ← c14413] at e_222_5
  have e_222_6 := arithEq_of_rows h (row := 222) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_6
  simp only [← c14414, ← c14415, k3, ← c14416] at e_222_6
  have e_222_7 := arithEq_of_rows h (row := 222) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_7
  simp only [← c14423, ← c14424, k3, ← c14425] at e_222_7
  refine ⟨?_, ?_⟩
  · have hc := e_221_7
    simp only [e_222_6] at hc
    linear_combination c14426.trans k2 - hc
  · have hc := e_222_7
    simp only [e_222_5, e_221_8, e_222_6] at hc
    linear_combination c14427.trans k2 - hc

theorem publicBatchWrapper89_f2089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5207)) (a (.wire 169 79)) (a (.virt 18839)) (a (.virt 18840)) := by
  have c14428 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14429 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14430 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14431 := (publicBatchWrapper89_copies450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14432 := (publicBatchWrapper89_copies451 a h).1
  have c14433 := (publicBatchWrapper89_copies451 a h).2.1
  have c14434 := (publicBatchWrapper89_copies451 a h).2.2.1
  have c14435 := (publicBatchWrapper89_copies451 a h).2.2.2.1
  have c14436 := (publicBatchWrapper89_copies451 a h).2.2.2.2.1
  have c14437 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.1
  have c14438 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.1
  have c14439 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.1
  have c14440 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.1
  have c14441 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.1
  have c14442 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14443 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14444 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_221_9 := arithEq_of_rows h (row := 221) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_9
  simp only [← c14434, ← c14435, ← c14436] at e_221_9
  have e_221_10 := arithEq_of_rows h (row := 221) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_10
  simp only [← c14437, ← c14438, ← c14439] at e_221_10
  have e_222_8 := arithEq_of_rows h (row := 222) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_8
  simp only [← c14428, k3, ← c14429, k3, ← c14430] at e_222_8
  have e_222_9 := arithEq_of_rows h (row := 222) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_9
  simp only [← c14431, ← c14432, k3, ← c14433] at e_222_9
  have e_222_10 := arithEq_of_rows h (row := 222) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_10
  simp only [← c14440, ← c14441, k3, ← c14442] at e_222_10
  refine ⟨?_, ?_⟩
  · have hc := e_221_9
    simp only [e_222_9] at hc
    linear_combination c14443.trans k2 - hc
  · have hc := e_222_10
    simp only [e_222_8, e_221_10, e_222_9] at hc
    linear_combination c14444.trans k2 - hc

theorem publicBatchWrapper89_f2090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5208)) (a (.wire 170 7)) (a (.virt 18841)) (a (.virt 18842)) := by
  have c14445 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14446 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14447 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14448 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14449 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14450 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14451 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14452 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14453 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14454 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14455 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14456 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14457 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14458 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14459 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14460 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14461 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_221_11 := arithEq_of_rows h (row := 221) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_11
  simp only [← c14451, ← c14452, ← c14453] at e_221_11
  have e_221_12 := arithEq_of_rows h (row := 221) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_12
  simp only [← c14454, ← c14455, ← c14456] at e_221_12
  have e_222_11 := arithEq_of_rows h (row := 222) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_11
  simp only [← c14445, k3, ← c14446, k3, ← c14447] at e_222_11
  have e_222_12 := arithEq_of_rows h (row := 222) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_12
  simp only [← c14448, ← c14449, k3, ← c14450] at e_222_12
  have e_222_13 := arithEq_of_rows h (row := 222) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_13
  simp only [← c14457, ← c14458, k3, ← c14459] at e_222_13
  refine ⟨?_, ?_⟩
  · have hc := e_221_11
    simp only [e_222_12] at hc
    linear_combination c14460.trans k2 - hc
  · have hc := e_222_13
    simp only [e_222_11, e_221_12, e_222_12] at hc
    linear_combination c14461.trans k2 - hc

theorem publicBatchWrapper89_f2091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 221 55) = band (a (.virt 18835)) (a (.virt 18837)) := by
  have c14462 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14463 := (publicBatchWrapper89_copies451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14464 := (publicBatchWrapper89_copies452 a h).1
  have e_221_13 := arithEq_of_rows h (row := 221) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_13
  simp only [← c14462, ← c14463, ← c14464] at e_221_13
  have hr := e_221_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 221 59) = band (a (.virt 18839)) (a (.virt 18841)) := by
  have c14465 := (publicBatchWrapper89_copies452 a h).2.1
  have c14466 := (publicBatchWrapper89_copies452 a h).2.2.1
  have c14467 := (publicBatchWrapper89_copies452 a h).2.2.2.1
  have e_221_14 := arithEq_of_rows h (row := 221) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_14
  simp only [← c14465, ← c14466, ← c14467] at e_221_14
  have hr := e_221_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 221 63) = band (a (.wire 221 55)) (a (.wire 221 59)) := by
  have c14468 := (publicBatchWrapper89_copies452 a h).2.2.2.2.1
  have c14469 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.1
  have c14470 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.1
  have e_221_15 := arithEq_of_rows h (row := 221) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_15
  simp only [← c14468, ← c14469, ← c14470] at e_221_15
  have hr := e_221_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 3) = bor (a (.wire 27 3)) (a (.wire 221 63)) := by
  have c14471 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.1
  have c14472 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.1
  have c14473 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.1
  have c14474 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14475 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14476 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_8 := arithEq_of_rows h (row := 218) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_8
  simp only [← c14471, ← c14472, ← c14473] at e_218_8
  have e_223_0 := arithEq_of_rows h (row := 223) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_0
  simp only [← c14474, ← c14475, k3, ← c14476] at e_223_0
  have hr := e_223_0
  simp only [e_218_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 3) = a (.virt 201) := by
  have c14477 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14477

theorem publicBatchWrapper89_f2096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5403)) (a (.wire 170 23)) (a (.virt 18843)) (a (.virt 18844)) := by
  have c14478 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14479 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14480 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14481 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14482 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14483 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14484 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14485 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14486 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14487 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14488 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14489 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14490 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14491 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14492 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14493 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14494 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_221_16 := arithEq_of_rows h (row := 221) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_16
  simp only [← c14484, ← c14485, ← c14486] at e_221_16
  have e_221_17 := arithEq_of_rows h (row := 221) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_17
  simp only [← c14487, ← c14488, ← c14489] at e_221_17
  have e_222_14 := arithEq_of_rows h (row := 222) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_14
  simp only [← c14478, k3, ← c14479, k3, ← c14480] at e_222_14
  have e_222_15 := arithEq_of_rows h (row := 222) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_15
  simp only [← c14481, ← c14482, k3, ← c14483] at e_222_15
  have e_222_16 := arithEq_of_rows h (row := 222) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_16
  simp only [← c14490, ← c14491, k3, ← c14492] at e_222_16
  refine ⟨?_, ?_⟩
  · have hc := e_221_16
    simp only [e_222_15] at hc
    linear_combination c14493.trans k2 - hc
  · have hc := e_222_16
    simp only [e_222_14, e_221_17, e_222_15] at hc
    linear_combination c14494.trans k2 - hc

theorem publicBatchWrapper89_f2097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 7) = bor (a (.wire 27 47)) (a (.virt 18843)) := by
  have c14495 := (publicBatchWrapper89_copies452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14496 := (publicBatchWrapper89_copies453 a h).1
  have c14497 := (publicBatchWrapper89_copies453 a h).2.1
  have c14498 := (publicBatchWrapper89_copies453 a h).2.2.1
  have c14499 := (publicBatchWrapper89_copies453 a h).2.2.2.1
  have c14500 := (publicBatchWrapper89_copies453 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_9 := arithEq_of_rows h (row := 218) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_9
  simp only [← c14495, ← c14496, ← c14497] at e_218_9
  have e_223_1 := arithEq_of_rows h (row := 223) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_1
  simp only [← c14498, ← c14499, k3, ← c14500] at e_223_1
  have hr := e_223_1
  simp only [e_218_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 7) = a (.virt 201) := by
  have c14501 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.1
  exact c14501

theorem publicBatchWrapper89_f2099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5404)) (a (.wire 170 31)) (a (.virt 18845)) (a (.virt 18846)) := by
  have c14502 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.1
  have c14503 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.1
  have c14504 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.1
  have c14505 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.1
  have c14506 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14507 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14508 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14509 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14510 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14511 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14512 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14513 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14514 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14515 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14516 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14517 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14518 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_221_18 := arithEq_of_rows h (row := 221) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_18
  simp only [← c14508, ← c14509, ← c14510] at e_221_18
  have e_221_19 := arithEq_of_rows h (row := 221) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_221_19
  simp only [← c14511, ← c14512, ← c14513] at e_221_19
  have e_222_17 := arithEq_of_rows h (row := 222) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_17
  simp only [← c14502, k3, ← c14503, k3, ← c14504] at e_222_17
  have e_222_18 := arithEq_of_rows h (row := 222) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_18
  simp only [← c14505, ← c14506, k3, ← c14507] at e_222_18
  have e_222_19 := arithEq_of_rows h (row := 222) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_222_19
  simp only [← c14514, ← c14515, k3, ← c14516] at e_222_19
  refine ⟨?_, ?_⟩
  · have hc := e_221_18
    simp only [e_222_18] at hc
    linear_combination c14517.trans k2 - hc
  · have hc := e_222_19
    simp only [e_222_17, e_221_19, e_222_18] at hc
    linear_combination c14518.trans k2 - hc

theorem publicBatchWrapper89_f2100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 11) = bor (a (.wire 27 47)) (a (.virt 18845)) := by
  have c14519 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14520 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14521 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14522 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14523 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14524 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_10 := arithEq_of_rows h (row := 218) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_10
  simp only [← c14519, ← c14520, ← c14521] at e_218_10
  have e_223_2 := arithEq_of_rows h (row := 223) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_2
  simp only [← c14522, ← c14523, k3, ← c14524] at e_223_2
  have hr := e_223_2
  simp only [e_218_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 11) = a (.virt 201) := by
  have c14525 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14525

theorem publicBatchWrapper89_f2102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5405)) (a (.wire 169 63)) (a (.virt 18847)) (a (.virt 18848)) := by
  have c14526 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14527 := (publicBatchWrapper89_copies453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14528 := (publicBatchWrapper89_copies454 a h).1
  have c14529 := (publicBatchWrapper89_copies454 a h).2.1
  have c14530 := (publicBatchWrapper89_copies454 a h).2.2.1
  have c14531 := (publicBatchWrapper89_copies454 a h).2.2.2.1
  have c14532 := (publicBatchWrapper89_copies454 a h).2.2.2.2.1
  have c14533 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.1
  have c14534 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.1
  have c14535 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.1
  have c14536 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.1
  have c14537 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.1
  have c14538 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14539 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14540 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14541 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14542 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_224_0 := arithEq_of_rows h (row := 224) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_0
  simp only [← c14526, k3, ← c14527, k3, ← c14528] at e_224_0
  have e_224_1 := arithEq_of_rows h (row := 224) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_1
  simp only [← c14529, ← c14530, k3, ← c14531] at e_224_1
  have e_224_2 := arithEq_of_rows h (row := 224) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_2
  simp only [← c14538, ← c14539, k3, ← c14540] at e_224_2
  have e_225_0 := arithEq_of_rows h (row := 225) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_0
  simp only [← c14532, ← c14533, ← c14534] at e_225_0
  have e_225_1 := arithEq_of_rows h (row := 225) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_1
  simp only [← c14535, ← c14536, ← c14537] at e_225_1
  refine ⟨?_, ?_⟩
  · have hc := e_225_0
    simp only [e_224_1] at hc
    linear_combination c14541.trans k2 - hc
  · have hc := e_224_2
    simp only [e_224_0, e_225_1, e_224_1] at hc
    linear_combination c14542.trans k2 - hc

theorem publicBatchWrapper89_f2103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5406)) (a (.wire 169 71)) (a (.virt 18849)) (a (.virt 18850)) := by
  have c14543 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14544 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14545 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14546 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14547 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14548 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14549 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14550 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14551 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14552 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14553 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14554 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14555 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14556 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14557 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14558 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14559 := (publicBatchWrapper89_copies454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_224_3 := arithEq_of_rows h (row := 224) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_3
  simp only [← c14543, k3, ← c14544, k3, ← c14545] at e_224_3
  have e_224_4 := arithEq_of_rows h (row := 224) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_4
  simp only [← c14546, ← c14547, k3, ← c14548] at e_224_4
  have e_224_5 := arithEq_of_rows h (row := 224) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_5
  simp only [← c14555, ← c14556, k3, ← c14557] at e_224_5
  have e_225_2 := arithEq_of_rows h (row := 225) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_2
  simp only [← c14549, ← c14550, ← c14551] at e_225_2
  have e_225_3 := arithEq_of_rows h (row := 225) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_3
  simp only [← c14552, ← c14553, ← c14554] at e_225_3
  refine ⟨?_, ?_⟩
  · have hc := e_225_2
    simp only [e_224_4] at hc
    linear_combination c14558.trans k2 - hc
  · have hc := e_224_5
    simp only [e_224_3, e_225_3, e_224_4] at hc
    linear_combination c14559.trans k2 - hc

theorem publicBatchWrapper89_f2104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5407)) (a (.wire 169 79)) (a (.virt 18851)) (a (.virt 18852)) := by
  have c14560 := (publicBatchWrapper89_copies455 a h).1
  have c14561 := (publicBatchWrapper89_copies455 a h).2.1
  have c14562 := (publicBatchWrapper89_copies455 a h).2.2.1
  have c14563 := (publicBatchWrapper89_copies455 a h).2.2.2.1
  have c14564 := (publicBatchWrapper89_copies455 a h).2.2.2.2.1
  have c14565 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.1
  have c14566 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.1
  have c14567 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.1
  have c14568 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.1
  have c14569 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.1
  have c14570 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14571 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14572 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14573 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14574 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14575 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14576 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_224_6 := arithEq_of_rows h (row := 224) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_6
  simp only [← c14560, k3, ← c14561, k3, ← c14562] at e_224_6
  have e_224_7 := arithEq_of_rows h (row := 224) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_7
  simp only [← c14563, ← c14564, k3, ← c14565] at e_224_7
  have e_224_8 := arithEq_of_rows h (row := 224) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_8
  simp only [← c14572, ← c14573, k3, ← c14574] at e_224_8
  have e_225_4 := arithEq_of_rows h (row := 225) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_4
  simp only [← c14566, ← c14567, ← c14568] at e_225_4
  have e_225_5 := arithEq_of_rows h (row := 225) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_5
  simp only [← c14569, ← c14570, ← c14571] at e_225_5
  refine ⟨?_, ?_⟩
  · have hc := e_225_4
    simp only [e_224_7] at hc
    linear_combination c14575.trans k2 - hc
  · have hc := e_224_8
    simp only [e_224_6, e_225_5, e_224_7] at hc
    linear_combination c14576.trans k2 - hc

theorem publicBatchWrapper89_f2105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5408)) (a (.wire 170 7)) (a (.virt 18853)) (a (.virt 18854)) := by
  have c14577 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14578 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14579 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14580 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14581 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14582 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14583 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14584 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14585 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14586 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14587 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14588 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14589 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14590 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14591 := (publicBatchWrapper89_copies455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14592 := (publicBatchWrapper89_copies456 a h).1
  have c14593 := (publicBatchWrapper89_copies456 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_224_9 := arithEq_of_rows h (row := 224) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_9
  simp only [← c14577, k3, ← c14578, k3, ← c14579] at e_224_9
  have e_224_10 := arithEq_of_rows h (row := 224) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_10
  simp only [← c14580, ← c14581, k3, ← c14582] at e_224_10
  have e_224_11 := arithEq_of_rows h (row := 224) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_11
  simp only [← c14589, ← c14590, k3, ← c14591] at e_224_11
  have e_225_6 := arithEq_of_rows h (row := 225) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_6
  simp only [← c14583, ← c14584, ← c14585] at e_225_6
  have e_225_7 := arithEq_of_rows h (row := 225) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_7
  simp only [← c14586, ← c14587, ← c14588] at e_225_7
  refine ⟨?_, ?_⟩
  · have hc := e_225_6
    simp only [e_224_10] at hc
    linear_combination c14592.trans k2 - hc
  · have hc := e_224_11
    simp only [e_224_9, e_225_7, e_224_10] at hc
    linear_combination c14593.trans k2 - hc

theorem publicBatchWrapper89_f2106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 225 35) = band (a (.virt 18847)) (a (.virt 18849)) := by
  have c14594 := (publicBatchWrapper89_copies456 a h).2.2.1
  have c14595 := (publicBatchWrapper89_copies456 a h).2.2.2.1
  have c14596 := (publicBatchWrapper89_copies456 a h).2.2.2.2.1
  have e_225_8 := arithEq_of_rows h (row := 225) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_8
  simp only [← c14594, ← c14595, ← c14596] at e_225_8
  have hr := e_225_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 225 39) = band (a (.virt 18851)) (a (.virt 18853)) := by
  have c14597 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.1
  have c14598 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.1
  have c14599 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.1
  have e_225_9 := arithEq_of_rows h (row := 225) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_9
  simp only [← c14597, ← c14598, ← c14599] at e_225_9
  have hr := e_225_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 225 43) = band (a (.wire 225 35)) (a (.wire 225 39)) := by
  have c14600 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.1
  have c14601 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.1
  have c14602 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_225_10 := arithEq_of_rows h (row := 225) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_10
  simp only [← c14600, ← c14601, ← c14602] at e_225_10
  have hr := e_225_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 15) = bor (a (.wire 27 47)) (a (.wire 225 43)) := by
  have c14603 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14604 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14605 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14606 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14607 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14608 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_11 := arithEq_of_rows h (row := 218) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_11
  simp only [← c14603, ← c14604, ← c14605] at e_218_11
  have e_223_3 := arithEq_of_rows h (row := 223) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_3
  simp only [← c14606, ← c14607, k3, ← c14608] at e_223_3
  have hr := e_223_3
  simp only [e_218_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 15) = a (.virt 201) := by
  have c14609 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14609

theorem publicBatchWrapper89_f2111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5603)) (a (.wire 170 23)) (a (.virt 18855)) (a (.virt 18856)) := by
  have c14610 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14611 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14612 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14613 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14614 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14615 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14616 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14617 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14618 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14619 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14620 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14621 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14622 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14623 := (publicBatchWrapper89_copies456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14624 := (publicBatchWrapper89_copies457 a h).1
  have c14625 := (publicBatchWrapper89_copies457 a h).2.1
  have c14626 := (publicBatchWrapper89_copies457 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_224_12 := arithEq_of_rows h (row := 224) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_12
  simp only [← c14610, k3, ← c14611, k3, ← c14612] at e_224_12
  have e_224_13 := arithEq_of_rows h (row := 224) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_13
  simp only [← c14613, ← c14614, k3, ← c14615] at e_224_13
  have e_224_14 := arithEq_of_rows h (row := 224) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_14
  simp only [← c14622, ← c14623, k3, ← c14624] at e_224_14
  have e_225_11 := arithEq_of_rows h (row := 225) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_11
  simp only [← c14616, ← c14617, ← c14618] at e_225_11
  have e_225_12 := arithEq_of_rows h (row := 225) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_12
  simp only [← c14619, ← c14620, ← c14621] at e_225_12
  refine ⟨?_, ?_⟩
  · have hc := e_225_11
    simp only [e_224_13] at hc
    linear_combination c14625.trans k2 - hc
  · have hc := e_224_14
    simp only [e_224_12, e_225_12, e_224_13] at hc
    linear_combination c14626.trans k2 - hc

theorem publicBatchWrapper89_f2112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 19) = bor (a (.wire 29 11)) (a (.virt 18855)) := by
  have c14627 := (publicBatchWrapper89_copies457 a h).2.2.2.1
  have c14628 := (publicBatchWrapper89_copies457 a h).2.2.2.2.1
  have c14629 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.1
  have c14630 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.1
  have c14631 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.1
  have c14632 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_12 := arithEq_of_rows h (row := 218) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_12
  simp only [← c14627, ← c14628, ← c14629] at e_218_12
  have e_223_4 := arithEq_of_rows h (row := 223) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_4
  simp only [← c14630, ← c14631, k3, ← c14632] at e_223_4
  have hr := e_223_4
  simp only [e_218_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 19) = a (.virt 201) := by
  have c14633 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.1
  exact c14633

theorem publicBatchWrapper89_f2114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5604)) (a (.wire 170 31)) (a (.virt 18857)) (a (.virt 18858)) := by
  have c14634 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14635 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14636 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14637 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14638 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14639 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14640 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14641 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14642 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14643 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14644 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14645 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14646 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14647 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14648 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14649 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14650 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_224_15 := arithEq_of_rows h (row := 224) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_15
  simp only [← c14634, k3, ← c14635, k3, ← c14636] at e_224_15
  have e_224_16 := arithEq_of_rows h (row := 224) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_16
  simp only [← c14637, ← c14638, k3, ← c14639] at e_224_16
  have e_224_17 := arithEq_of_rows h (row := 224) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_17
  simp only [← c14646, ← c14647, k3, ← c14648] at e_224_17
  have e_225_13 := arithEq_of_rows h (row := 225) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_13
  simp only [← c14640, ← c14641, ← c14642] at e_225_13
  have e_225_14 := arithEq_of_rows h (row := 225) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_14
  simp only [← c14643, ← c14644, ← c14645] at e_225_14
  refine ⟨?_, ?_⟩
  · have hc := e_225_13
    simp only [e_224_16] at hc
    linear_combination c14649.trans k2 - hc
  · have hc := e_224_17
    simp only [e_224_15, e_225_14, e_224_16] at hc
    linear_combination c14650.trans k2 - hc

theorem publicBatchWrapper89_f2115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 23) = bor (a (.wire 29 11)) (a (.virt 18857)) := by
  have c14651 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14652 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14653 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14654 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14655 := (publicBatchWrapper89_copies457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14656 := (publicBatchWrapper89_copies458 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_13 := arithEq_of_rows h (row := 218) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_13
  simp only [← c14651, ← c14652, ← c14653] at e_218_13
  have e_223_5 := arithEq_of_rows h (row := 223) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_5
  simp only [← c14654, ← c14655, k3, ← c14656] at e_223_5
  have hr := e_223_5
  simp only [e_218_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 23) = a (.virt 201) := by
  have c14657 := (publicBatchWrapper89_copies458 a h).2.1
  exact c14657

theorem publicBatchWrapper89_f2117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5605)) (a (.wire 169 63)) (a (.virt 18859)) (a (.virt 18860)) := by
  have c14658 := (publicBatchWrapper89_copies458 a h).2.2.1
  have c14659 := (publicBatchWrapper89_copies458 a h).2.2.2.1
  have c14660 := (publicBatchWrapper89_copies458 a h).2.2.2.2.1
  have c14661 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.1
  have c14662 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.1
  have c14663 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.1
  have c14664 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.1
  have c14665 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.1
  have c14666 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14667 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14668 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14669 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14670 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14671 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14672 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14673 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14674 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_224_18 := arithEq_of_rows h (row := 224) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_18
  simp only [← c14658, k3, ← c14659, k3, ← c14660] at e_224_18
  have e_224_19 := arithEq_of_rows h (row := 224) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_224_19
  simp only [← c14661, ← c14662, k3, ← c14663] at e_224_19
  have e_225_15 := arithEq_of_rows h (row := 225) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_15
  simp only [← c14664, ← c14665, ← c14666] at e_225_15
  have e_225_16 := arithEq_of_rows h (row := 225) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_16
  simp only [← c14667, ← c14668, ← c14669] at e_225_16
  have e_226_0 := arithEq_of_rows h (row := 226) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_0
  simp only [← c14670, ← c14671, k3, ← c14672] at e_226_0
  refine ⟨?_, ?_⟩
  · have hc := e_225_15
    simp only [e_224_19] at hc
    linear_combination c14673.trans k2 - hc
  · have hc := e_226_0
    simp only [e_224_18, e_225_16, e_224_19] at hc
    linear_combination c14674.trans k2 - hc

theorem publicBatchWrapper89_f2118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5606)) (a (.wire 169 71)) (a (.virt 18861)) (a (.virt 18862)) := by
  have c14675 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14676 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14677 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14678 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14679 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14680 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14681 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14682 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14683 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14684 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14685 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14686 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14687 := (publicBatchWrapper89_copies458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14688 := (publicBatchWrapper89_copies459 a h).1
  have c14689 := (publicBatchWrapper89_copies459 a h).2.1
  have c14690 := (publicBatchWrapper89_copies459 a h).2.2.1
  have c14691 := (publicBatchWrapper89_copies459 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_225_17 := arithEq_of_rows h (row := 225) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_17
  simp only [← c14681, ← c14682, ← c14683] at e_225_17
  have e_225_18 := arithEq_of_rows h (row := 225) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_18
  simp only [← c14684, ← c14685, ← c14686] at e_225_18
  have e_226_1 := arithEq_of_rows h (row := 226) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_1
  simp only [← c14675, k3, ← c14676, k3, ← c14677] at e_226_1
  have e_226_2 := arithEq_of_rows h (row := 226) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_2
  simp only [← c14678, ← c14679, k3, ← c14680] at e_226_2
  have e_226_3 := arithEq_of_rows h (row := 226) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_3
  simp only [← c14687, ← c14688, k3, ← c14689] at e_226_3
  refine ⟨?_, ?_⟩
  · have hc := e_225_17
    simp only [e_226_2] at hc
    linear_combination c14690.trans k2 - hc
  · have hc := e_226_3
    simp only [e_226_1, e_225_18, e_226_2] at hc
    linear_combination c14691.trans k2 - hc

theorem publicBatchWrapper89_f2119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5607)) (a (.wire 169 79)) (a (.virt 18863)) (a (.virt 18864)) := by
  have c14692 := (publicBatchWrapper89_copies459 a h).2.2.2.2.1
  have c14693 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.1
  have c14694 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.1
  have c14695 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.1
  have c14696 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.1
  have c14697 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.1
  have c14698 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14699 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14700 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14701 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14702 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14703 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14704 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14705 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14706 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14707 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14708 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_225_19 := arithEq_of_rows h (row := 225) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_225_19
  simp only [← c14698, ← c14699, ← c14700] at e_225_19
  have e_226_4 := arithEq_of_rows h (row := 226) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_4
  simp only [← c14692, k3, ← c14693, k3, ← c14694] at e_226_4
  have e_226_5 := arithEq_of_rows h (row := 226) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_5
  simp only [← c14695, ← c14696, k3, ← c14697] at e_226_5
  have e_226_6 := arithEq_of_rows h (row := 226) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_6
  simp only [← c14704, ← c14705, k3, ← c14706] at e_226_6
  have e_227_0 := arithEq_of_rows h (row := 227) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_0
  simp only [← c14701, ← c14702, ← c14703] at e_227_0
  refine ⟨?_, ?_⟩
  · have hc := e_225_19
    simp only [e_226_5] at hc
    linear_combination c14707.trans k2 - hc
  · have hc := e_226_6
    simp only [e_226_4, e_227_0, e_226_5] at hc
    linear_combination c14708.trans k2 - hc

theorem publicBatchWrapper89_f2120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5608)) (a (.wire 170 7)) (a (.virt 18865)) (a (.virt 18866)) := by
  have c14709 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14710 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14711 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14712 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14713 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14714 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14715 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14716 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14717 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14718 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14719 := (publicBatchWrapper89_copies459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14720 := (publicBatchWrapper89_copies460 a h).1
  have c14721 := (publicBatchWrapper89_copies460 a h).2.1
  have c14722 := (publicBatchWrapper89_copies460 a h).2.2.1
  have c14723 := (publicBatchWrapper89_copies460 a h).2.2.2.1
  have c14724 := (publicBatchWrapper89_copies460 a h).2.2.2.2.1
  have c14725 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_226_7 := arithEq_of_rows h (row := 226) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_7
  simp only [← c14709, k3, ← c14710, k3, ← c14711] at e_226_7
  have e_226_8 := arithEq_of_rows h (row := 226) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_8
  simp only [← c14712, ← c14713, k3, ← c14714] at e_226_8
  have e_226_9 := arithEq_of_rows h (row := 226) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_9
  simp only [← c14721, ← c14722, k3, ← c14723] at e_226_9
  have e_227_1 := arithEq_of_rows h (row := 227) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_1
  simp only [← c14715, ← c14716, ← c14717] at e_227_1
  have e_227_2 := arithEq_of_rows h (row := 227) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_2
  simp only [← c14718, ← c14719, ← c14720] at e_227_2
  refine ⟨?_, ?_⟩
  · have hc := e_227_1
    simp only [e_226_8] at hc
    linear_combination c14724.trans k2 - hc
  · have hc := e_226_9
    simp only [e_226_7, e_227_2, e_226_8] at hc
    linear_combination c14725.trans k2 - hc

theorem publicBatchWrapper89_f2121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 227 15) = band (a (.virt 18859)) (a (.virt 18861)) := by
  have c14726 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.1
  have c14727 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.1
  have c14728 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.1
  have e_227_3 := arithEq_of_rows h (row := 227) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_3
  simp only [← c14726, ← c14727, ← c14728] at e_227_3
  have hr := e_227_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 227 19) = band (a (.virt 18863)) (a (.virt 18865)) := by
  have c14729 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.1
  have c14730 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14731 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_227_4 := arithEq_of_rows h (row := 227) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_4
  simp only [← c14729, ← c14730, ← c14731] at e_227_4
  have hr := e_227_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 227 23) = band (a (.wire 227 15)) (a (.wire 227 19)) := by
  have c14732 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14733 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14734 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_227_5 := arithEq_of_rows h (row := 227) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_5
  simp only [← c14732, ← c14733, ← c14734] at e_227_5
  have hr := e_227_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 27) = bor (a (.wire 29 11)) (a (.wire 227 23)) := by
  have c14735 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14736 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14737 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14738 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14739 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14740 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_14 := arithEq_of_rows h (row := 218) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_14
  simp only [← c14735, ← c14736, ← c14737] at e_218_14
  have e_223_6 := arithEq_of_rows h (row := 223) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_6
  simp only [← c14738, ← c14739, k3, ← c14740] at e_223_6
  have hr := e_223_6
  simp only [e_218_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 27) = a (.virt 201) := by
  have c14741 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14741

theorem publicBatchWrapper89_f2126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5803)) (a (.wire 170 23)) (a (.virt 18867)) (a (.virt 18868)) := by
  have c14742 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14743 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14744 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14745 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14746 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14747 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14748 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14749 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14750 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14751 := (publicBatchWrapper89_copies460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14752 := (publicBatchWrapper89_copies461 a h).1
  have c14753 := (publicBatchWrapper89_copies461 a h).2.1
  have c14754 := (publicBatchWrapper89_copies461 a h).2.2.1
  have c14755 := (publicBatchWrapper89_copies461 a h).2.2.2.1
  have c14756 := (publicBatchWrapper89_copies461 a h).2.2.2.2.1
  have c14757 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.1
  have c14758 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_226_10 := arithEq_of_rows h (row := 226) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_10
  simp only [← c14742, k3, ← c14743, k3, ← c14744] at e_226_10
  have e_226_11 := arithEq_of_rows h (row := 226) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_11
  simp only [← c14745, ← c14746, k3, ← c14747] at e_226_11
  have e_226_12 := arithEq_of_rows h (row := 226) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_12
  simp only [← c14754, ← c14755, k3, ← c14756] at e_226_12
  have e_227_6 := arithEq_of_rows h (row := 227) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_6
  simp only [← c14748, ← c14749, ← c14750] at e_227_6
  have e_227_7 := arithEq_of_rows h (row := 227) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_7
  simp only [← c14751, ← c14752, ← c14753] at e_227_7
  refine ⟨?_, ?_⟩
  · have hc := e_227_6
    simp only [e_226_11] at hc
    linear_combination c14757.trans k2 - hc
  · have hc := e_226_12
    simp only [e_226_10, e_227_7, e_226_11] at hc
    linear_combination c14758.trans k2 - hc

theorem publicBatchWrapper89_f2127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 31) = bor (a (.wire 29 55)) (a (.virt 18867)) := by
  have c14759 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.1
  have c14760 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.1
  have c14761 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.1
  have c14762 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14763 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14764 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_15 := arithEq_of_rows h (row := 218) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_15
  simp only [← c14759, ← c14760, ← c14761] at e_218_15
  have e_223_7 := arithEq_of_rows h (row := 223) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_7
  simp only [← c14762, ← c14763, k3, ← c14764] at e_223_7
  have hr := e_223_7
  simp only [e_218_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 31) = a (.virt 201) := by
  have c14765 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14765

theorem publicBatchWrapper89_f2129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5804)) (a (.wire 170 31)) (a (.virt 18869)) (a (.virt 18870)) := by
  have c14766 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14767 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14768 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14769 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14770 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14771 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14772 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14773 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14774 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14775 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14776 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14777 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14778 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14779 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14780 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14781 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14782 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_226_13 := arithEq_of_rows h (row := 226) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_13
  simp only [← c14766, k3, ← c14767, k3, ← c14768] at e_226_13
  have e_226_14 := arithEq_of_rows h (row := 226) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_14
  simp only [← c14769, ← c14770, k3, ← c14771] at e_226_14
  have e_226_15 := arithEq_of_rows h (row := 226) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_15
  simp only [← c14778, ← c14779, k3, ← c14780] at e_226_15
  have e_227_8 := arithEq_of_rows h (row := 227) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_8
  simp only [← c14772, ← c14773, ← c14774] at e_227_8
  have e_227_9 := arithEq_of_rows h (row := 227) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_9
  simp only [← c14775, ← c14776, ← c14777] at e_227_9
  refine ⟨?_, ?_⟩
  · have hc := e_227_8
    simp only [e_226_14] at hc
    linear_combination c14781.trans k2 - hc
  · have hc := e_226_15
    simp only [e_226_13, e_227_9, e_226_14] at hc
    linear_combination c14782.trans k2 - hc

theorem publicBatchWrapper89_f2130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 35) = bor (a (.wire 29 55)) (a (.virt 18869)) := by
  have c14783 := (publicBatchWrapper89_copies461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14784 := (publicBatchWrapper89_copies462 a h).1
  have c14785 := (publicBatchWrapper89_copies462 a h).2.1
  have c14786 := (publicBatchWrapper89_copies462 a h).2.2.1
  have c14787 := (publicBatchWrapper89_copies462 a h).2.2.2.1
  have c14788 := (publicBatchWrapper89_copies462 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_16 := arithEq_of_rows h (row := 218) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_16
  simp only [← c14783, ← c14784, ← c14785] at e_218_16
  have e_223_8 := arithEq_of_rows h (row := 223) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_8
  simp only [← c14786, ← c14787, k3, ← c14788] at e_223_8
  have hr := e_223_8
  simp only [e_218_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 35) = a (.virt 201) := by
  have c14789 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.1
  exact c14789

theorem publicBatchWrapper89_f2132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5805)) (a (.wire 169 63)) (a (.virt 18871)) (a (.virt 18872)) := by
  have c14790 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.1
  have c14791 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.1
  have c14792 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.1
  have c14793 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.1
  have c14794 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14795 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14796 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14797 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14798 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14799 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14800 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14801 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14802 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14803 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14804 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14805 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14806 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_226_16 := arithEq_of_rows h (row := 226) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_16
  simp only [← c14790, k3, ← c14791, k3, ← c14792] at e_226_16
  have e_226_17 := arithEq_of_rows h (row := 226) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_17
  simp only [← c14793, ← c14794, k3, ← c14795] at e_226_17
  have e_226_18 := arithEq_of_rows h (row := 226) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_18
  simp only [← c14802, ← c14803, k3, ← c14804] at e_226_18
  have e_227_10 := arithEq_of_rows h (row := 227) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_10
  simp only [← c14796, ← c14797, ← c14798] at e_227_10
  have e_227_11 := arithEq_of_rows h (row := 227) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_11
  simp only [← c14799, ← c14800, ← c14801] at e_227_11
  refine ⟨?_, ?_⟩
  · have hc := e_227_10
    simp only [e_226_17] at hc
    linear_combination c14805.trans k2 - hc
  · have hc := e_226_18
    simp only [e_226_16, e_227_11, e_226_17] at hc
    linear_combination c14806.trans k2 - hc

theorem publicBatchWrapper89_f2133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5806)) (a (.wire 169 71)) (a (.virt 18873)) (a (.virt 18874)) := by
  have c14807 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14808 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14809 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14810 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14811 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14812 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14813 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14814 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14815 := (publicBatchWrapper89_copies462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14816 := (publicBatchWrapper89_copies463 a h).1
  have c14817 := (publicBatchWrapper89_copies463 a h).2.1
  have c14818 := (publicBatchWrapper89_copies463 a h).2.2.1
  have c14819 := (publicBatchWrapper89_copies463 a h).2.2.2.1
  have c14820 := (publicBatchWrapper89_copies463 a h).2.2.2.2.1
  have c14821 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.1
  have c14822 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.1
  have c14823 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_226_19 := arithEq_of_rows h (row := 226) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_226_19
  simp only [← c14807, k3, ← c14808, k3, ← c14809] at e_226_19
  have e_227_12 := arithEq_of_rows h (row := 227) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_12
  simp only [← c14813, ← c14814, ← c14815] at e_227_12
  have e_227_13 := arithEq_of_rows h (row := 227) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_13
  simp only [← c14816, ← c14817, ← c14818] at e_227_13
  have e_228_0 := arithEq_of_rows h (row := 228) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_0
  simp only [← c14810, ← c14811, k3, ← c14812] at e_228_0
  have e_228_1 := arithEq_of_rows h (row := 228) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_1
  simp only [← c14819, ← c14820, k3, ← c14821] at e_228_1
  refine ⟨?_, ?_⟩
  · have hc := e_227_12
    simp only [e_228_0] at hc
    linear_combination c14822.trans k2 - hc
  · have hc := e_228_1
    simp only [e_226_19, e_227_13, e_228_0] at hc
    linear_combination c14823.trans k2 - hc

theorem publicBatchWrapper89_f2134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5807)) (a (.wire 169 79)) (a (.virt 18875)) (a (.virt 18876)) := by
  have c14824 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.1
  have c14825 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.1
  have c14826 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14827 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14828 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14829 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14830 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14831 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14832 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14833 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14834 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14835 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14836 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14837 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14838 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14839 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14840 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_227_14 := arithEq_of_rows h (row := 227) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_14
  simp only [← c14830, ← c14831, ← c14832] at e_227_14
  have e_227_15 := arithEq_of_rows h (row := 227) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_15
  simp only [← c14833, ← c14834, ← c14835] at e_227_15
  have e_228_2 := arithEq_of_rows h (row := 228) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_2
  simp only [← c14824, k3, ← c14825, k3, ← c14826] at e_228_2
  have e_228_3 := arithEq_of_rows h (row := 228) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_3
  simp only [← c14827, ← c14828, k3, ← c14829] at e_228_3
  have e_228_4 := arithEq_of_rows h (row := 228) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_4
  simp only [← c14836, ← c14837, k3, ← c14838] at e_228_4
  refine ⟨?_, ?_⟩
  · have hc := e_227_14
    simp only [e_228_3] at hc
    linear_combination c14839.trans k2 - hc
  · have hc := e_228_4
    simp only [e_228_2, e_227_15, e_228_3] at hc
    linear_combination c14840.trans k2 - hc

theorem publicBatchWrapper89_f2135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5808)) (a (.wire 170 7)) (a (.virt 18877)) (a (.virt 18878)) := by
  have c14841 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14842 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14843 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14844 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14845 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14846 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14847 := (publicBatchWrapper89_copies463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14848 := (publicBatchWrapper89_copies464 a h).1
  have c14849 := (publicBatchWrapper89_copies464 a h).2.1
  have c14850 := (publicBatchWrapper89_copies464 a h).2.2.1
  have c14851 := (publicBatchWrapper89_copies464 a h).2.2.2.1
  have c14852 := (publicBatchWrapper89_copies464 a h).2.2.2.2.1
  have c14853 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.1
  have c14854 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.1
  have c14855 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.1
  have c14856 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.1
  have c14857 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_227_16 := arithEq_of_rows h (row := 227) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_16
  simp only [← c14847, ← c14848, ← c14849] at e_227_16
  have e_227_17 := arithEq_of_rows h (row := 227) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_17
  simp only [← c14850, ← c14851, ← c14852] at e_227_17
  have e_228_5 := arithEq_of_rows h (row := 228) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_5
  simp only [← c14841, k3, ← c14842, k3, ← c14843] at e_228_5
  have e_228_6 := arithEq_of_rows h (row := 228) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_6
  simp only [← c14844, ← c14845, k3, ← c14846] at e_228_6
  have e_228_7 := arithEq_of_rows h (row := 228) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_7
  simp only [← c14853, ← c14854, k3, ← c14855] at e_228_7
  refine ⟨?_, ?_⟩
  · have hc := e_227_16
    simp only [e_228_6] at hc
    linear_combination c14856.trans k2 - hc
  · have hc := e_228_7
    simp only [e_228_5, e_227_17, e_228_6] at hc
    linear_combination c14857.trans k2 - hc

theorem publicBatchWrapper89_f2136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 227 75) = band (a (.virt 18871)) (a (.virt 18873)) := by
  have c14858 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14859 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14860 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_227_18 := arithEq_of_rows h (row := 227) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_18
  simp only [← c14858, ← c14859, ← c14860] at e_227_18
  have hr := e_227_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 227 79) = band (a (.virt 18875)) (a (.virt 18877)) := by
  have c14861 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14862 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14863 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_227_19 := arithEq_of_rows h (row := 227) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_227_19
  simp only [← c14861, ← c14862, ← c14863] at e_227_19
  have hr := e_227_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 229 3) = band (a (.wire 227 75)) (a (.wire 227 79)) := by
  have c14864 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14865 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14866 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_229_0 := arithEq_of_rows h (row := 229) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_0
  simp only [← c14864, ← c14865, ← c14866] at e_229_0
  have hr := e_229_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 39) = bor (a (.wire 29 55)) (a (.wire 229 3)) := by
  have c14867 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14868 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14869 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14870 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14871 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14872 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_17 := arithEq_of_rows h (row := 218) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_17
  simp only [← c14867, ← c14868, ← c14869] at e_218_17
  have e_223_9 := arithEq_of_rows h (row := 223) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_9
  simp only [← c14870, ← c14871, k3, ← c14872] at e_223_9
  have hr := e_223_9
  simp only [e_218_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 39) = a (.virt 201) := by
  have c14873 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14873

theorem publicBatchWrapper89_f2141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6003)) (a (.wire 170 23)) (a (.virt 18879)) (a (.virt 18880)) := by
  have c14874 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14875 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14876 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14877 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14878 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14879 := (publicBatchWrapper89_copies464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14880 := (publicBatchWrapper89_copies465 a h).1
  have c14881 := (publicBatchWrapper89_copies465 a h).2.1
  have c14882 := (publicBatchWrapper89_copies465 a h).2.2.1
  have c14883 := (publicBatchWrapper89_copies465 a h).2.2.2.1
  have c14884 := (publicBatchWrapper89_copies465 a h).2.2.2.2.1
  have c14885 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.1
  have c14886 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.1
  have c14887 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.1
  have c14888 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.1
  have c14889 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.1
  have c14890 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_228_8 := arithEq_of_rows h (row := 228) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_8
  simp only [← c14874, k3, ← c14875, k3, ← c14876] at e_228_8
  have e_228_9 := arithEq_of_rows h (row := 228) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_9
  simp only [← c14877, ← c14878, k3, ← c14879] at e_228_9
  have e_228_10 := arithEq_of_rows h (row := 228) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_10
  simp only [← c14886, ← c14887, k3, ← c14888] at e_228_10
  have e_229_1 := arithEq_of_rows h (row := 229) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_1
  simp only [← c14880, ← c14881, ← c14882] at e_229_1
  have e_229_2 := arithEq_of_rows h (row := 229) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_2
  simp only [← c14883, ← c14884, ← c14885] at e_229_2
  refine ⟨?_, ?_⟩
  · have hc := e_229_1
    simp only [e_228_9] at hc
    linear_combination c14889.trans k2 - hc
  · have hc := e_228_10
    simp only [e_228_8, e_229_2, e_228_9] at hc
    linear_combination c14890.trans k2 - hc

theorem publicBatchWrapper89_f2142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 43) = bor (a (.wire 31 19)) (a (.virt 18879)) := by
  have c14891 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14892 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14893 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14894 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14895 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14896 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_18 := arithEq_of_rows h (row := 218) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_18
  simp only [← c14891, ← c14892, ← c14893] at e_218_18
  have e_223_10 := arithEq_of_rows h (row := 223) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_10
  simp only [← c14894, ← c14895, k3, ← c14896] at e_223_10
  have hr := e_223_10
  simp only [e_218_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 43) = a (.virt 201) := by
  have c14897 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14897

theorem publicBatchWrapper89_f2144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6004)) (a (.wire 170 31)) (a (.virt 18881)) (a (.virt 18882)) := by
  have c14898 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14899 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14900 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14901 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14902 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14903 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14904 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14905 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14906 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14907 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14908 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14909 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14910 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14911 := (publicBatchWrapper89_copies465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14912 := (publicBatchWrapper89_copies466 a h).1
  have c14913 := (publicBatchWrapper89_copies466 a h).2.1
  have c14914 := (publicBatchWrapper89_copies466 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_228_11 := arithEq_of_rows h (row := 228) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_11
  simp only [← c14898, k3, ← c14899, k3, ← c14900] at e_228_11
  have e_228_12 := arithEq_of_rows h (row := 228) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_12
  simp only [← c14901, ← c14902, k3, ← c14903] at e_228_12
  have e_228_13 := arithEq_of_rows h (row := 228) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_13
  simp only [← c14910, ← c14911, k3, ← c14912] at e_228_13
  have e_229_3 := arithEq_of_rows h (row := 229) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_3
  simp only [← c14904, ← c14905, ← c14906] at e_229_3
  have e_229_4 := arithEq_of_rows h (row := 229) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_4
  simp only [← c14907, ← c14908, ← c14909] at e_229_4
  refine ⟨?_, ?_⟩
  · have hc := e_229_3
    simp only [e_228_12] at hc
    linear_combination c14913.trans k2 - hc
  · have hc := e_228_13
    simp only [e_228_11, e_229_4, e_228_12] at hc
    linear_combination c14914.trans k2 - hc

theorem publicBatchWrapper89_f2145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 47) = bor (a (.wire 31 19)) (a (.virt 18881)) := by
  have c14915 := (publicBatchWrapper89_copies466 a h).2.2.2.1
  have c14916 := (publicBatchWrapper89_copies466 a h).2.2.2.2.1
  have c14917 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.1
  have c14918 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.1
  have c14919 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.1
  have c14920 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_218_19 := arithEq_of_rows h (row := 218) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_218_19
  simp only [← c14915, ← c14916, ← c14917] at e_218_19
  have e_223_11 := arithEq_of_rows h (row := 223) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_11
  simp only [← c14918, ← c14919, k3, ← c14920] at e_223_11
  have hr := e_223_11
  simp only [e_218_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 47) = a (.virt 201) := by
  have c14921 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.1
  exact c14921

theorem publicBatchWrapper89_f2147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6005)) (a (.wire 169 63)) (a (.virt 18883)) (a (.virt 18884)) := by
  have c14922 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14923 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14924 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14925 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14926 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14927 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14928 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14929 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14930 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14931 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14932 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14933 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14934 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14935 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14936 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14937 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14938 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_228_14 := arithEq_of_rows h (row := 228) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_14
  simp only [← c14922, k3, ← c14923, k3, ← c14924] at e_228_14
  have e_228_15 := arithEq_of_rows h (row := 228) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_15
  simp only [← c14925, ← c14926, k3, ← c14927] at e_228_15
  have e_228_16 := arithEq_of_rows h (row := 228) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_16
  simp only [← c14934, ← c14935, k3, ← c14936] at e_228_16
  have e_229_5 := arithEq_of_rows h (row := 229) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_5
  simp only [← c14928, ← c14929, ← c14930] at e_229_5
  have e_229_6 := arithEq_of_rows h (row := 229) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_6
  simp only [← c14931, ← c14932, ← c14933] at e_229_6
  refine ⟨?_, ?_⟩
  · have hc := e_229_5
    simp only [e_228_15] at hc
    linear_combination c14937.trans k2 - hc
  · have hc := e_228_16
    simp only [e_228_14, e_229_6, e_228_15] at hc
    linear_combination c14938.trans k2 - hc

theorem publicBatchWrapper89_f2148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6006)) (a (.wire 169 71)) (a (.virt 18885)) (a (.virt 18886)) := by
  have c14939 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14940 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14941 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14942 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14943 := (publicBatchWrapper89_copies466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14944 := (publicBatchWrapper89_copies467 a h).1
  have c14945 := (publicBatchWrapper89_copies467 a h).2.1
  have c14946 := (publicBatchWrapper89_copies467 a h).2.2.1
  have c14947 := (publicBatchWrapper89_copies467 a h).2.2.2.1
  have c14948 := (publicBatchWrapper89_copies467 a h).2.2.2.2.1
  have c14949 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.1
  have c14950 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.1
  have c14951 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.1
  have c14952 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.1
  have c14953 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.1
  have c14954 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14955 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_228_17 := arithEq_of_rows h (row := 228) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_17
  simp only [← c14939, k3, ← c14940, k3, ← c14941] at e_228_17
  have e_228_18 := arithEq_of_rows h (row := 228) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_18
  simp only [← c14942, ← c14943, k3, ← c14944] at e_228_18
  have e_228_19 := arithEq_of_rows h (row := 228) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_228_19
  simp only [← c14951, ← c14952, k3, ← c14953] at e_228_19
  have e_229_7 := arithEq_of_rows h (row := 229) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_7
  simp only [← c14945, ← c14946, ← c14947] at e_229_7
  have e_229_8 := arithEq_of_rows h (row := 229) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_8
  simp only [← c14948, ← c14949, ← c14950] at e_229_8
  refine ⟨?_, ?_⟩
  · have hc := e_229_7
    simp only [e_228_18] at hc
    linear_combination c14954.trans k2 - hc
  · have hc := e_228_19
    simp only [e_228_17, e_229_8, e_228_18] at hc
    linear_combination c14955.trans k2 - hc

theorem publicBatchWrapper89_f2149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6007)) (a (.wire 169 79)) (a (.virt 18887)) (a (.virt 18888)) := by
  have c14956 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14957 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14958 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14959 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14960 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14961 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14962 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14963 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14964 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14965 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14966 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14967 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14968 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14969 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14970 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14971 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14972 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_229_9 := arithEq_of_rows h (row := 229) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_9
  simp only [← c14962, ← c14963, ← c14964] at e_229_9
  have e_229_10 := arithEq_of_rows h (row := 229) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_10
  simp only [← c14965, ← c14966, ← c14967] at e_229_10
  have e_230_0 := arithEq_of_rows h (row := 230) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_0
  simp only [← c14956, k3, ← c14957, k3, ← c14958] at e_230_0
  have e_230_1 := arithEq_of_rows h (row := 230) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_1
  simp only [← c14959, ← c14960, k3, ← c14961] at e_230_1
  have e_230_2 := arithEq_of_rows h (row := 230) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_2
  simp only [← c14968, ← c14969, k3, ← c14970] at e_230_2
  refine ⟨?_, ?_⟩
  · have hc := e_229_9
    simp only [e_230_1] at hc
    linear_combination c14971.trans k2 - hc
  · have hc := e_230_2
    simp only [e_230_0, e_229_10, e_230_1] at hc
    linear_combination c14972.trans k2 - hc

theorem publicBatchWrapper89_f2150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6008)) (a (.wire 170 7)) (a (.virt 18889)) (a (.virt 18890)) := by
  have c14973 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14974 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14975 := (publicBatchWrapper89_copies467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c14976 := (publicBatchWrapper89_copies468 a h).1
  have c14977 := (publicBatchWrapper89_copies468 a h).2.1
  have c14978 := (publicBatchWrapper89_copies468 a h).2.2.1
  have c14979 := (publicBatchWrapper89_copies468 a h).2.2.2.1
  have c14980 := (publicBatchWrapper89_copies468 a h).2.2.2.2.1
  have c14981 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.1
  have c14982 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.1
  have c14983 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.1
  have c14984 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.1
  have c14985 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.1
  have c14986 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.1
  have c14987 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c14988 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14989 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_229_11 := arithEq_of_rows h (row := 229) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_11
  simp only [← c14979, ← c14980, ← c14981] at e_229_11
  have e_229_12 := arithEq_of_rows h (row := 229) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_12
  simp only [← c14982, ← c14983, ← c14984] at e_229_12
  have e_230_3 := arithEq_of_rows h (row := 230) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_3
  simp only [← c14973, k3, ← c14974, k3, ← c14975] at e_230_3
  have e_230_4 := arithEq_of_rows h (row := 230) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_4
  simp only [← c14976, ← c14977, k3, ← c14978] at e_230_4
  have e_230_5 := arithEq_of_rows h (row := 230) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_5
  simp only [← c14985, ← c14986, k3, ← c14987] at e_230_5
  refine ⟨?_, ?_⟩
  · have hc := e_229_11
    simp only [e_230_4] at hc
    linear_combination c14988.trans k2 - hc
  · have hc := e_230_5
    simp only [e_230_3, e_229_12, e_230_4] at hc
    linear_combination c14989.trans k2 - hc

theorem publicBatchWrapper89_f2151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 229 55) = band (a (.virt 18883)) (a (.virt 18885)) := by
  have c14990 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14991 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14992 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_229_13 := arithEq_of_rows h (row := 229) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_13
  simp only [← c14990, ← c14991, ← c14992] at e_229_13
  have hr := e_229_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 229 59) = band (a (.virt 18887)) (a (.virt 18889)) := by
  have c14993 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14994 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14995 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_229_14 := arithEq_of_rows h (row := 229) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_14
  simp only [← c14993, ← c14994, ← c14995] at e_229_14
  have hr := e_229_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 229 63) = band (a (.wire 229 55)) (a (.wire 229 59)) := by
  have c14996 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14997 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14998 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_229_15 := arithEq_of_rows h (row := 229) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_15
  simp only [← c14996, ← c14997, ← c14998] at e_229_15
  have hr := e_229_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 51) = bor (a (.wire 31 19)) (a (.wire 229 63)) := by
  have c14999 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15000 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15001 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15002 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15003 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15004 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_223_12 := arithEq_of_rows h (row := 223) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_12
  simp only [← c15002, ← c15003, k3, ← c15004] at e_223_12
  have e_231_0 := arithEq_of_rows h (row := 231) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_0
  simp only [← c14999, ← c15000, ← c15001] at e_231_0
  have hr := e_223_12
  simp only [e_231_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 51) = a (.virt 201) := by
  have c15005 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15005

theorem publicBatchWrapper89_f2156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6203)) (a (.wire 170 23)) (a (.virt 18891)) (a (.virt 18892)) := by
  have c15006 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15007 := (publicBatchWrapper89_copies468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15008 := (publicBatchWrapper89_copies469 a h).1
  have c15009 := (publicBatchWrapper89_copies469 a h).2.1
  have c15010 := (publicBatchWrapper89_copies469 a h).2.2.1
  have c15011 := (publicBatchWrapper89_copies469 a h).2.2.2.1
  have c15012 := (publicBatchWrapper89_copies469 a h).2.2.2.2.1
  have c15013 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.1
  have c15014 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.1
  have c15015 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.1
  have c15016 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.1
  have c15017 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.1
  have c15018 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15019 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15020 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15021 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15022 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_229_16 := arithEq_of_rows h (row := 229) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_16
  simp only [← c15012, ← c15013, ← c15014] at e_229_16
  have e_229_17 := arithEq_of_rows h (row := 229) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_17
  simp only [← c15015, ← c15016, ← c15017] at e_229_17
  have e_230_6 := arithEq_of_rows h (row := 230) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_6
  simp only [← c15006, k3, ← c15007, k3, ← c15008] at e_230_6
  have e_230_7 := arithEq_of_rows h (row := 230) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_7
  simp only [← c15009, ← c15010, k3, ← c15011] at e_230_7
  have e_230_8 := arithEq_of_rows h (row := 230) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_8
  simp only [← c15018, ← c15019, k3, ← c15020] at e_230_8
  refine ⟨?_, ?_⟩
  · have hc := e_229_16
    simp only [e_230_7] at hc
    linear_combination c15021.trans k2 - hc
  · have hc := e_230_8
    simp only [e_230_6, e_229_17, e_230_7] at hc
    linear_combination c15022.trans k2 - hc

theorem publicBatchWrapper89_f2157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 55) = bor (a (.wire 31 63)) (a (.virt 18891)) := by
  have c15023 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15024 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15025 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15026 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15027 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15028 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_223_13 := arithEq_of_rows h (row := 223) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_13
  simp only [← c15026, ← c15027, k3, ← c15028] at e_223_13
  have e_231_1 := arithEq_of_rows h (row := 231) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_1
  simp only [← c15023, ← c15024, ← c15025] at e_231_1
  have hr := e_223_13
  simp only [e_231_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 55) = a (.virt 201) := by
  have c15029 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15029

theorem publicBatchWrapper89_f2159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6204)) (a (.wire 170 31)) (a (.virt 18893)) (a (.virt 18894)) := by
  have c15030 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15031 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15032 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15033 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15034 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15035 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15036 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15037 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15038 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15039 := (publicBatchWrapper89_copies469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15040 := (publicBatchWrapper89_copies470 a h).1
  have c15041 := (publicBatchWrapper89_copies470 a h).2.1
  have c15042 := (publicBatchWrapper89_copies470 a h).2.2.1
  have c15043 := (publicBatchWrapper89_copies470 a h).2.2.2.1
  have c15044 := (publicBatchWrapper89_copies470 a h).2.2.2.2.1
  have c15045 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.1
  have c15046 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_229_18 := arithEq_of_rows h (row := 229) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_18
  simp only [← c15036, ← c15037, ← c15038] at e_229_18
  have e_229_19 := arithEq_of_rows h (row := 229) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_229_19
  simp only [← c15039, ← c15040, ← c15041] at e_229_19
  have e_230_9 := arithEq_of_rows h (row := 230) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_9
  simp only [← c15030, k3, ← c15031, k3, ← c15032] at e_230_9
  have e_230_10 := arithEq_of_rows h (row := 230) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_10
  simp only [← c15033, ← c15034, k3, ← c15035] at e_230_10
  have e_230_11 := arithEq_of_rows h (row := 230) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_11
  simp only [← c15042, ← c15043, k3, ← c15044] at e_230_11
  refine ⟨?_, ?_⟩
  · have hc := e_229_18
    simp only [e_230_10] at hc
    linear_combination c15045.trans k2 - hc
  · have hc := e_230_11
    simp only [e_230_9, e_229_19, e_230_10] at hc
    linear_combination c15046.trans k2 - hc

theorem publicBatchWrapper89_f2160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 59) = bor (a (.wire 31 63)) (a (.virt 18893)) := by
  have c15047 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.1
  have c15048 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.1
  have c15049 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.1
  have c15050 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15051 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15052 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_223_14 := arithEq_of_rows h (row := 223) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_14
  simp only [← c15050, ← c15051, k3, ← c15052] at e_223_14
  have e_231_2 := arithEq_of_rows h (row := 231) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_2
  simp only [← c15047, ← c15048, ← c15049] at e_231_2
  have hr := e_223_14
  simp only [e_231_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 59) = a (.virt 201) := by
  have c15053 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15053

theorem publicBatchWrapper89_f2162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6205)) (a (.wire 169 63)) (a (.virt 18895)) (a (.virt 18896)) := by
  have c15054 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15055 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15056 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15057 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15058 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15059 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15060 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15061 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15062 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15063 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15064 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15065 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15066 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15067 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15068 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15069 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15070 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_230_12 := arithEq_of_rows h (row := 230) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_12
  simp only [← c15054, k3, ← c15055, k3, ← c15056] at e_230_12
  have e_230_13 := arithEq_of_rows h (row := 230) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_13
  simp only [← c15057, ← c15058, k3, ← c15059] at e_230_13
  have e_230_14 := arithEq_of_rows h (row := 230) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_14
  simp only [← c15066, ← c15067, k3, ← c15068] at e_230_14
  have e_232_0 := arithEq_of_rows h (row := 232) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_0
  simp only [← c15060, ← c15061, ← c15062] at e_232_0
  have e_232_1 := arithEq_of_rows h (row := 232) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_1
  simp only [← c15063, ← c15064, ← c15065] at e_232_1
  refine ⟨?_, ?_⟩
  · have hc := e_232_0
    simp only [e_230_13] at hc
    linear_combination c15069.trans k2 - hc
  · have hc := e_230_14
    simp only [e_230_12, e_232_1, e_230_13] at hc
    linear_combination c15070.trans k2 - hc

theorem publicBatchWrapper89_f2163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6206)) (a (.wire 169 71)) (a (.virt 18897)) (a (.virt 18898)) := by
  have c15071 := (publicBatchWrapper89_copies470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15072 := (publicBatchWrapper89_copies471 a h).1
  have c15073 := (publicBatchWrapper89_copies471 a h).2.1
  have c15074 := (publicBatchWrapper89_copies471 a h).2.2.1
  have c15075 := (publicBatchWrapper89_copies471 a h).2.2.2.1
  have c15076 := (publicBatchWrapper89_copies471 a h).2.2.2.2.1
  have c15077 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.1
  have c15078 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.1
  have c15079 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.1
  have c15080 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.1
  have c15081 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.1
  have c15082 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15083 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15084 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15085 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15086 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15087 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_230_15 := arithEq_of_rows h (row := 230) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_15
  simp only [← c15071, k3, ← c15072, k3, ← c15073] at e_230_15
  have e_230_16 := arithEq_of_rows h (row := 230) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_16
  simp only [← c15074, ← c15075, k3, ← c15076] at e_230_16
  have e_230_17 := arithEq_of_rows h (row := 230) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_17
  simp only [← c15083, ← c15084, k3, ← c15085] at e_230_17
  have e_232_2 := arithEq_of_rows h (row := 232) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_2
  simp only [← c15077, ← c15078, ← c15079] at e_232_2
  have e_232_3 := arithEq_of_rows h (row := 232) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_3
  simp only [← c15080, ← c15081, ← c15082] at e_232_3
  refine ⟨?_, ?_⟩
  · have hc := e_232_2
    simp only [e_230_16] at hc
    linear_combination c15086.trans k2 - hc
  · have hc := e_230_17
    simp only [e_230_15, e_232_3, e_230_16] at hc
    linear_combination c15087.trans k2 - hc

theorem publicBatchWrapper89_f2164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6207)) (a (.wire 169 79)) (a (.virt 18899)) (a (.virt 18900)) := by
  have c15088 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15089 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15090 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15091 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15092 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15093 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15094 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15095 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15096 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15097 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15098 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15099 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15100 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15101 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15102 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15103 := (publicBatchWrapper89_copies471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15104 := (publicBatchWrapper89_copies472 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_230_18 := arithEq_of_rows h (row := 230) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_18
  simp only [← c15088, k3, ← c15089, k3, ← c15090] at e_230_18
  have e_230_19 := arithEq_of_rows h (row := 230) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_230_19
  simp only [← c15091, ← c15092, k3, ← c15093] at e_230_19
  have e_232_4 := arithEq_of_rows h (row := 232) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_4
  simp only [← c15094, ← c15095, ← c15096] at e_232_4
  have e_232_5 := arithEq_of_rows h (row := 232) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_5
  simp only [← c15097, ← c15098, ← c15099] at e_232_5
  have e_233_0 := arithEq_of_rows h (row := 233) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_0
  simp only [← c15100, ← c15101, k3, ← c15102] at e_233_0
  refine ⟨?_, ?_⟩
  · have hc := e_232_4
    simp only [e_230_19] at hc
    linear_combination c15103.trans k2 - hc
  · have hc := e_233_0
    simp only [e_230_18, e_232_5, e_230_19] at hc
    linear_combination c15104.trans k2 - hc

theorem publicBatchWrapper89_f2165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6208)) (a (.wire 170 7)) (a (.virt 18901)) (a (.virt 18902)) := by
  have c15105 := (publicBatchWrapper89_copies472 a h).2.1
  have c15106 := (publicBatchWrapper89_copies472 a h).2.2.1
  have c15107 := (publicBatchWrapper89_copies472 a h).2.2.2.1
  have c15108 := (publicBatchWrapper89_copies472 a h).2.2.2.2.1
  have c15109 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.1
  have c15110 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.1
  have c15111 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.1
  have c15112 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.1
  have c15113 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.1
  have c15114 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15115 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15116 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15117 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15118 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15119 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15120 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15121 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_232_6 := arithEq_of_rows h (row := 232) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_6
  simp only [← c15111, ← c15112, ← c15113] at e_232_6
  have e_232_7 := arithEq_of_rows h (row := 232) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_7
  simp only [← c15114, ← c15115, ← c15116] at e_232_7
  have e_233_1 := arithEq_of_rows h (row := 233) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_1
  simp only [← c15105, k3, ← c15106, k3, ← c15107] at e_233_1
  have e_233_2 := arithEq_of_rows h (row := 233) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_2
  simp only [← c15108, ← c15109, k3, ← c15110] at e_233_2
  have e_233_3 := arithEq_of_rows h (row := 233) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_3
  simp only [← c15117, ← c15118, k3, ← c15119] at e_233_3
  refine ⟨?_, ?_⟩
  · have hc := e_232_6
    simp only [e_233_2] at hc
    linear_combination c15120.trans k2 - hc
  · have hc := e_233_3
    simp only [e_233_1, e_232_7, e_233_2] at hc
    linear_combination c15121.trans k2 - hc

theorem publicBatchWrapper89_f2166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 232 35) = band (a (.virt 18895)) (a (.virt 18897)) := by
  have c15122 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15123 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15124 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_232_8 := arithEq_of_rows h (row := 232) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_8
  simp only [← c15122, ← c15123, ← c15124] at e_232_8
  have hr := e_232_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 232 39) = band (a (.virt 18899)) (a (.virt 18901)) := by
  have c15125 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15126 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15127 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_232_9 := arithEq_of_rows h (row := 232) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_9
  simp only [← c15125, ← c15126, ← c15127] at e_232_9
  have hr := e_232_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 232 43) = band (a (.wire 232 35)) (a (.wire 232 39)) := by
  have c15128 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15129 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15130 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_232_10 := arithEq_of_rows h (row := 232) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_10
  simp only [← c15128, ← c15129, ← c15130] at e_232_10
  have hr := e_232_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 63) = bor (a (.wire 31 63)) (a (.wire 232 43)) := by
  have c15131 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15132 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15133 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15134 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15135 := (publicBatchWrapper89_copies472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15136 := (publicBatchWrapper89_copies473 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_223_15 := arithEq_of_rows h (row := 223) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_15
  simp only [← c15134, ← c15135, k3, ← c15136] at e_223_15
  have e_231_3 := arithEq_of_rows h (row := 231) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_3
  simp only [← c15131, ← c15132, ← c15133] at e_231_3
  have hr := e_223_15
  simp only [e_231_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 63) = a (.virt 201) := by
  have c15137 := (publicBatchWrapper89_copies473 a h).2.1
  exact c15137

theorem publicBatchWrapper89_f2171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6403)) (a (.wire 170 23)) (a (.virt 18903)) (a (.virt 18904)) := by
  have c15138 := (publicBatchWrapper89_copies473 a h).2.2.1
  have c15139 := (publicBatchWrapper89_copies473 a h).2.2.2.1
  have c15140 := (publicBatchWrapper89_copies473 a h).2.2.2.2.1
  have c15141 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.1
  have c15142 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.1
  have c15143 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.1
  have c15144 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.1
  have c15145 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.1
  have c15146 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15147 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15148 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15149 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15150 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15151 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15152 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15153 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15154 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_232_11 := arithEq_of_rows h (row := 232) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_11
  simp only [← c15144, ← c15145, ← c15146] at e_232_11
  have e_232_12 := arithEq_of_rows h (row := 232) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_12
  simp only [← c15147, ← c15148, ← c15149] at e_232_12
  have e_233_4 := arithEq_of_rows h (row := 233) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_4
  simp only [← c15138, k3, ← c15139, k3, ← c15140] at e_233_4
  have e_233_5 := arithEq_of_rows h (row := 233) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_5
  simp only [← c15141, ← c15142, k3, ← c15143] at e_233_5
  have e_233_6 := arithEq_of_rows h (row := 233) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_6
  simp only [← c15150, ← c15151, k3, ← c15152] at e_233_6
  refine ⟨?_, ?_⟩
  · have hc := e_232_11
    simp only [e_233_5] at hc
    linear_combination c15153.trans k2 - hc
  · have hc := e_233_6
    simp only [e_233_4, e_232_12, e_233_5] at hc
    linear_combination c15154.trans k2 - hc

theorem publicBatchWrapper89_f2172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 67) = bor (a (.wire 33 27)) (a (.virt 18903)) := by
  have c15155 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15156 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15157 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15158 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15159 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15160 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_223_16 := arithEq_of_rows h (row := 223) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_16
  simp only [← c15158, ← c15159, k3, ← c15160] at e_223_16
  have e_231_4 := arithEq_of_rows h (row := 231) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_4
  simp only [← c15155, ← c15156, ← c15157] at e_231_4
  have hr := e_223_16
  simp only [e_231_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 67) = a (.virt 201) := by
  have c15161 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15161

theorem publicBatchWrapper89_f2174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6404)) (a (.wire 170 31)) (a (.virt 18905)) (a (.virt 18906)) := by
  have c15162 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15163 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15164 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15165 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15166 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15167 := (publicBatchWrapper89_copies473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15168 := (publicBatchWrapper89_copies474 a h).1
  have c15169 := (publicBatchWrapper89_copies474 a h).2.1
  have c15170 := (publicBatchWrapper89_copies474 a h).2.2.1
  have c15171 := (publicBatchWrapper89_copies474 a h).2.2.2.1
  have c15172 := (publicBatchWrapper89_copies474 a h).2.2.2.2.1
  have c15173 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.1
  have c15174 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.1
  have c15175 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.1
  have c15176 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.1
  have c15177 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.1
  have c15178 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_232_13 := arithEq_of_rows h (row := 232) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_13
  simp only [← c15168, ← c15169, ← c15170] at e_232_13
  have e_232_14 := arithEq_of_rows h (row := 232) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_14
  simp only [← c15171, ← c15172, ← c15173] at e_232_14
  have e_233_7 := arithEq_of_rows h (row := 233) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_7
  simp only [← c15162, k3, ← c15163, k3, ← c15164] at e_233_7
  have e_233_8 := arithEq_of_rows h (row := 233) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_8
  simp only [← c15165, ← c15166, k3, ← c15167] at e_233_8
  have e_233_9 := arithEq_of_rows h (row := 233) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_9
  simp only [← c15174, ← c15175, k3, ← c15176] at e_233_9
  refine ⟨?_, ?_⟩
  · have hc := e_232_13
    simp only [e_233_8] at hc
    linear_combination c15177.trans k2 - hc
  · have hc := e_233_9
    simp only [e_233_7, e_232_14, e_233_8] at hc
    linear_combination c15178.trans k2 - hc

theorem publicBatchWrapper89_f2175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 71) = bor (a (.wire 33 27)) (a (.virt 18905)) := by
  have c15179 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15180 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15181 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15182 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15183 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15184 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_223_17 := arithEq_of_rows h (row := 223) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_17
  simp only [← c15182, ← c15183, k3, ← c15184] at e_223_17
  have e_231_5 := arithEq_of_rows h (row := 231) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_5
  simp only [← c15179, ← c15180, ← c15181] at e_231_5
  have hr := e_223_17
  simp only [e_231_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 71) = a (.virt 201) := by
  have c15185 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15185

theorem publicBatchWrapper89_f2177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6405)) (a (.wire 169 63)) (a (.virt 18907)) (a (.virt 18908)) := by
  have c15186 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15187 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15188 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15189 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15190 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15191 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15192 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15193 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15194 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15195 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15196 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15197 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15198 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15199 := (publicBatchWrapper89_copies474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15200 := (publicBatchWrapper89_copies475 a h).1
  have c15201 := (publicBatchWrapper89_copies475 a h).2.1
  have c15202 := (publicBatchWrapper89_copies475 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_232_15 := arithEq_of_rows h (row := 232) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_15
  simp only [← c15192, ← c15193, ← c15194] at e_232_15
  have e_232_16 := arithEq_of_rows h (row := 232) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_16
  simp only [← c15195, ← c15196, ← c15197] at e_232_16
  have e_233_10 := arithEq_of_rows h (row := 233) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_10
  simp only [← c15186, k3, ← c15187, k3, ← c15188] at e_233_10
  have e_233_11 := arithEq_of_rows h (row := 233) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_11
  simp only [← c15189, ← c15190, k3, ← c15191] at e_233_11
  have e_233_12 := arithEq_of_rows h (row := 233) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_12
  simp only [← c15198, ← c15199, k3, ← c15200] at e_233_12
  refine ⟨?_, ?_⟩
  · have hc := e_232_15
    simp only [e_233_11] at hc
    linear_combination c15201.trans k2 - hc
  · have hc := e_233_12
    simp only [e_233_10, e_232_16, e_233_11] at hc
    linear_combination c15202.trans k2 - hc

theorem publicBatchWrapper89_f2178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6406)) (a (.wire 169 71)) (a (.virt 18909)) (a (.virt 18910)) := by
  have c15203 := (publicBatchWrapper89_copies475 a h).2.2.2.1
  have c15204 := (publicBatchWrapper89_copies475 a h).2.2.2.2.1
  have c15205 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.1
  have c15206 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.1
  have c15207 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.1
  have c15208 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.1
  have c15209 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.1
  have c15210 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15211 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15212 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15213 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15214 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15215 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15216 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15217 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15218 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15219 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_232_17 := arithEq_of_rows h (row := 232) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_17
  simp only [← c15209, ← c15210, ← c15211] at e_232_17
  have e_232_18 := arithEq_of_rows h (row := 232) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_18
  simp only [← c15212, ← c15213, ← c15214] at e_232_18
  have e_233_13 := arithEq_of_rows h (row := 233) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_13
  simp only [← c15203, k3, ← c15204, k3, ← c15205] at e_233_13
  have e_233_14 := arithEq_of_rows h (row := 233) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_14
  simp only [← c15206, ← c15207, k3, ← c15208] at e_233_14
  have e_233_15 := arithEq_of_rows h (row := 233) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_15
  simp only [← c15215, ← c15216, k3, ← c15217] at e_233_15
  refine ⟨?_, ?_⟩
  · have hc := e_232_17
    simp only [e_233_14] at hc
    linear_combination c15218.trans k2 - hc
  · have hc := e_233_15
    simp only [e_233_13, e_232_18, e_233_14] at hc
    linear_combination c15219.trans k2 - hc

theorem publicBatchWrapper89_f2179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6407)) (a (.wire 169 79)) (a (.virt 18911)) (a (.virt 18912)) := by
  have c15220 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15221 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15222 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15223 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15224 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15225 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15226 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15227 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15228 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15229 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15230 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15231 := (publicBatchWrapper89_copies475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15232 := (publicBatchWrapper89_copies476 a h).1
  have c15233 := (publicBatchWrapper89_copies476 a h).2.1
  have c15234 := (publicBatchWrapper89_copies476 a h).2.2.1
  have c15235 := (publicBatchWrapper89_copies476 a h).2.2.2.1
  have c15236 := (publicBatchWrapper89_copies476 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_232_19 := arithEq_of_rows h (row := 232) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_232_19
  simp only [← c15226, ← c15227, ← c15228] at e_232_19
  have e_233_16 := arithEq_of_rows h (row := 233) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_16
  simp only [← c15220, k3, ← c15221, k3, ← c15222] at e_233_16
  have e_233_17 := arithEq_of_rows h (row := 233) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_17
  simp only [← c15223, ← c15224, k3, ← c15225] at e_233_17
  have e_233_18 := arithEq_of_rows h (row := 233) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_18
  simp only [← c15232, ← c15233, k3, ← c15234] at e_233_18
  have e_234_0 := arithEq_of_rows h (row := 234) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_0
  simp only [← c15229, ← c15230, ← c15231] at e_234_0
  refine ⟨?_, ?_⟩
  · have hc := e_232_19
    simp only [e_233_17] at hc
    linear_combination c15235.trans k2 - hc
  · have hc := e_233_18
    simp only [e_233_16, e_234_0, e_233_17] at hc
    linear_combination c15236.trans k2 - hc

theorem publicBatchWrapper89_f2180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6408)) (a (.wire 170 7)) (a (.virt 18913)) (a (.virt 18914)) := by
  have c15237 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.1
  have c15238 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.1
  have c15239 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.1
  have c15240 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.1
  have c15241 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.1
  have c15242 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15243 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15244 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15245 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15246 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15247 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15248 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15249 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15250 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15251 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15252 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15253 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_233_19 := arithEq_of_rows h (row := 233) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_233_19
  simp only [← c15237, k3, ← c15238, k3, ← c15239] at e_233_19
  have e_234_1 := arithEq_of_rows h (row := 234) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_1
  simp only [← c15243, ← c15244, ← c15245] at e_234_1
  have e_234_2 := arithEq_of_rows h (row := 234) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_2
  simp only [← c15246, ← c15247, ← c15248] at e_234_2
  have e_235_0 := arithEq_of_rows h (row := 235) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_0
  simp only [← c15240, ← c15241, k3, ← c15242] at e_235_0
  have e_235_1 := arithEq_of_rows h (row := 235) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_1
  simp only [← c15249, ← c15250, k3, ← c15251] at e_235_1
  refine ⟨?_, ?_⟩
  · have hc := e_234_1
    simp only [e_235_0] at hc
    linear_combination c15252.trans k2 - hc
  · have hc := e_235_1
    simp only [e_233_19, e_234_2, e_235_0] at hc
    linear_combination c15253.trans k2 - hc

theorem publicBatchWrapper89_f2181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 234 15) = band (a (.virt 18907)) (a (.virt 18909)) := by
  have c15254 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15255 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15256 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_234_3 := arithEq_of_rows h (row := 234) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_3
  simp only [← c15254, ← c15255, ← c15256] at e_234_3
  have hr := e_234_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 234 19) = band (a (.virt 18911)) (a (.virt 18913)) := by
  have c15257 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15258 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15259 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_234_4 := arithEq_of_rows h (row := 234) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_4
  simp only [← c15257, ← c15258, ← c15259] at e_234_4
  have hr := e_234_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 234 23) = band (a (.wire 234 15)) (a (.wire 234 19)) := by
  have c15260 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15261 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15262 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_234_5 := arithEq_of_rows h (row := 234) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_5
  simp only [← c15260, ← c15261, ← c15262] at e_234_5
  have hr := e_234_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 75) = bor (a (.wire 33 27)) (a (.wire 234 23)) := by
  have c15263 := (publicBatchWrapper89_copies476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15264 := (publicBatchWrapper89_copies477 a h).1
  have c15265 := (publicBatchWrapper89_copies477 a h).2.1
  have c15266 := (publicBatchWrapper89_copies477 a h).2.2.1
  have c15267 := (publicBatchWrapper89_copies477 a h).2.2.2.1
  have c15268 := (publicBatchWrapper89_copies477 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_223_18 := arithEq_of_rows h (row := 223) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_18
  simp only [← c15266, ← c15267, k3, ← c15268] at e_223_18
  have e_231_6 := arithEq_of_rows h (row := 231) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_6
  simp only [← c15263, ← c15264, ← c15265] at e_231_6
  have hr := e_223_18
  simp only [e_231_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 75) = a (.virt 201) := by
  have c15269 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.1
  exact c15269

theorem publicBatchWrapper89_f2186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6603)) (a (.wire 170 23)) (a (.virt 18915)) (a (.virt 18916)) := by
  have c15270 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.1
  have c15271 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.1
  have c15272 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.1
  have c15273 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.1
  have c15274 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15275 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15276 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15277 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15278 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15279 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15280 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15281 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15282 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15283 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15284 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15285 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15286 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_234_6 := arithEq_of_rows h (row := 234) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_6
  simp only [← c15276, ← c15277, ← c15278] at e_234_6
  have e_234_7 := arithEq_of_rows h (row := 234) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_7
  simp only [← c15279, ← c15280, ← c15281] at e_234_7
  have e_235_2 := arithEq_of_rows h (row := 235) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_2
  simp only [← c15270, k3, ← c15271, k3, ← c15272] at e_235_2
  have e_235_3 := arithEq_of_rows h (row := 235) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_3
  simp only [← c15273, ← c15274, k3, ← c15275] at e_235_3
  have e_235_4 := arithEq_of_rows h (row := 235) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_4
  simp only [← c15282, ← c15283, k3, ← c15284] at e_235_4
  refine ⟨?_, ?_⟩
  · have hc := e_234_6
    simp only [e_235_3] at hc
    linear_combination c15285.trans k2 - hc
  · have hc := e_235_4
    simp only [e_235_2, e_234_7, e_235_3] at hc
    linear_combination c15286.trans k2 - hc

theorem publicBatchWrapper89_f2187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 79) = bor (a (.wire 33 71)) (a (.virt 18915)) := by
  have c15287 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15288 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15289 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15290 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15291 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15292 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_223_19 := arithEq_of_rows h (row := 223) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_223_19
  simp only [← c15290, ← c15291, k3, ← c15292] at e_223_19
  have e_231_7 := arithEq_of_rows h (row := 231) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_7
  simp only [← c15287, ← c15288, ← c15289] at e_231_7
  have hr := e_223_19
  simp only [e_231_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 223 79) = a (.virt 201) := by
  have c15293 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15293

theorem publicBatchWrapper89_f2189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6604)) (a (.wire 170 31)) (a (.virt 18917)) (a (.virt 18918)) := by
  have c15294 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15295 := (publicBatchWrapper89_copies477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15296 := (publicBatchWrapper89_copies478 a h).1
  have c15297 := (publicBatchWrapper89_copies478 a h).2.1
  have c15298 := (publicBatchWrapper89_copies478 a h).2.2.1
  have c15299 := (publicBatchWrapper89_copies478 a h).2.2.2.1
  have c15300 := (publicBatchWrapper89_copies478 a h).2.2.2.2.1
  have c15301 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.1
  have c15302 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.1
  have c15303 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.1
  have c15304 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.1
  have c15305 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.1
  have c15306 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15307 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15308 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15309 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15310 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_234_8 := arithEq_of_rows h (row := 234) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_8
  simp only [← c15300, ← c15301, ← c15302] at e_234_8
  have e_234_9 := arithEq_of_rows h (row := 234) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_9
  simp only [← c15303, ← c15304, ← c15305] at e_234_9
  have e_235_5 := arithEq_of_rows h (row := 235) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_5
  simp only [← c15294, k3, ← c15295, k3, ← c15296] at e_235_5
  have e_235_6 := arithEq_of_rows h (row := 235) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_6
  simp only [← c15297, ← c15298, k3, ← c15299] at e_235_6
  have e_235_7 := arithEq_of_rows h (row := 235) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_7
  simp only [← c15306, ← c15307, k3, ← c15308] at e_235_7
  refine ⟨?_, ?_⟩
  · have hc := e_234_8
    simp only [e_235_6] at hc
    linear_combination c15309.trans k2 - hc
  · have hc := e_235_7
    simp only [e_235_5, e_234_9, e_235_6] at hc
    linear_combination c15310.trans k2 - hc

theorem publicBatchWrapper89_f2190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 3) = bor (a (.wire 33 71)) (a (.virt 18917)) := by
  have c15311 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15312 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15313 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15314 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15315 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15316 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_8 := arithEq_of_rows h (row := 231) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_8
  simp only [← c15311, ← c15312, ← c15313] at e_231_8
  have e_236_0 := arithEq_of_rows h (row := 236) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_0
  simp only [← c15314, ← c15315, k3, ← c15316] at e_236_0
  have hr := e_236_0
  simp only [e_231_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 3) = a (.virt 201) := by
  have c15317 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15317

theorem publicBatchWrapper89_f2192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6605)) (a (.wire 169 63)) (a (.virt 18919)) (a (.virt 18920)) := by
  have c15318 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15319 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15320 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15321 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15322 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15323 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15324 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15325 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15326 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15327 := (publicBatchWrapper89_copies478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15328 := (publicBatchWrapper89_copies479 a h).1
  have c15329 := (publicBatchWrapper89_copies479 a h).2.1
  have c15330 := (publicBatchWrapper89_copies479 a h).2.2.1
  have c15331 := (publicBatchWrapper89_copies479 a h).2.2.2.1
  have c15332 := (publicBatchWrapper89_copies479 a h).2.2.2.2.1
  have c15333 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.1
  have c15334 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_234_10 := arithEq_of_rows h (row := 234) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_10
  simp only [← c15324, ← c15325, ← c15326] at e_234_10
  have e_234_11 := arithEq_of_rows h (row := 234) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_11
  simp only [← c15327, ← c15328, ← c15329] at e_234_11
  have e_235_8 := arithEq_of_rows h (row := 235) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_8
  simp only [← c15318, k3, ← c15319, k3, ← c15320] at e_235_8
  have e_235_9 := arithEq_of_rows h (row := 235) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_9
  simp only [← c15321, ← c15322, k3, ← c15323] at e_235_9
  have e_235_10 := arithEq_of_rows h (row := 235) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_10
  simp only [← c15330, ← c15331, k3, ← c15332] at e_235_10
  refine ⟨?_, ?_⟩
  · have hc := e_234_10
    simp only [e_235_9] at hc
    linear_combination c15333.trans k2 - hc
  · have hc := e_235_10
    simp only [e_235_8, e_234_11, e_235_9] at hc
    linear_combination c15334.trans k2 - hc

theorem publicBatchWrapper89_f2193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6606)) (a (.wire 169 71)) (a (.virt 18921)) (a (.virt 18922)) := by
  have c15335 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.1
  have c15336 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.1
  have c15337 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.1
  have c15338 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15339 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15340 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15341 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15342 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15343 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15344 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15345 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15346 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15347 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15348 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15349 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15350 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15351 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_234_12 := arithEq_of_rows h (row := 234) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_12
  simp only [← c15341, ← c15342, ← c15343] at e_234_12
  have e_234_13 := arithEq_of_rows h (row := 234) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_13
  simp only [← c15344, ← c15345, ← c15346] at e_234_13
  have e_235_11 := arithEq_of_rows h (row := 235) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_11
  simp only [← c15335, k3, ← c15336, k3, ← c15337] at e_235_11
  have e_235_12 := arithEq_of_rows h (row := 235) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_12
  simp only [← c15338, ← c15339, k3, ← c15340] at e_235_12
  have e_235_13 := arithEq_of_rows h (row := 235) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_13
  simp only [← c15347, ← c15348, k3, ← c15349] at e_235_13
  refine ⟨?_, ?_⟩
  · have hc := e_234_12
    simp only [e_235_12] at hc
    linear_combination c15350.trans k2 - hc
  · have hc := e_235_13
    simp only [e_235_11, e_234_13, e_235_12] at hc
    linear_combination c15351.trans k2 - hc

theorem publicBatchWrapper89_f2194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6607)) (a (.wire 169 79)) (a (.virt 18923)) (a (.virt 18924)) := by
  have c15352 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15353 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15354 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15355 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15356 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15357 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15358 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15359 := (publicBatchWrapper89_copies479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15360 := (publicBatchWrapper89_copies480 a h).1
  have c15361 := (publicBatchWrapper89_copies480 a h).2.1
  have c15362 := (publicBatchWrapper89_copies480 a h).2.2.1
  have c15363 := (publicBatchWrapper89_copies480 a h).2.2.2.1
  have c15364 := (publicBatchWrapper89_copies480 a h).2.2.2.2.1
  have c15365 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.1
  have c15366 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.1
  have c15367 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.1
  have c15368 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_234_14 := arithEq_of_rows h (row := 234) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_14
  simp only [← c15358, ← c15359, ← c15360] at e_234_14
  have e_234_15 := arithEq_of_rows h (row := 234) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_15
  simp only [← c15361, ← c15362, ← c15363] at e_234_15
  have e_235_14 := arithEq_of_rows h (row := 235) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_14
  simp only [← c15352, k3, ← c15353, k3, ← c15354] at e_235_14
  have e_235_15 := arithEq_of_rows h (row := 235) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_15
  simp only [← c15355, ← c15356, k3, ← c15357] at e_235_15
  have e_235_16 := arithEq_of_rows h (row := 235) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_16
  simp only [← c15364, ← c15365, k3, ← c15366] at e_235_16
  refine ⟨?_, ?_⟩
  · have hc := e_234_14
    simp only [e_235_15] at hc
    linear_combination c15367.trans k2 - hc
  · have hc := e_235_16
    simp only [e_235_14, e_234_15, e_235_15] at hc
    linear_combination c15368.trans k2 - hc

theorem publicBatchWrapper89_f2195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6608)) (a (.wire 170 7)) (a (.virt 18925)) (a (.virt 18926)) := by
  have c15369 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.1
  have c15370 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15371 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15372 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15373 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15374 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15375 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15376 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15377 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15378 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15379 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15380 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15381 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15382 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15383 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15384 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15385 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_234_16 := arithEq_of_rows h (row := 234) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_16
  simp only [← c15375, ← c15376, ← c15377] at e_234_16
  have e_234_17 := arithEq_of_rows h (row := 234) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_17
  simp only [← c15378, ← c15379, ← c15380] at e_234_17
  have e_235_17 := arithEq_of_rows h (row := 235) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_17
  simp only [← c15369, k3, ← c15370, k3, ← c15371] at e_235_17
  have e_235_18 := arithEq_of_rows h (row := 235) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_18
  simp only [← c15372, ← c15373, k3, ← c15374] at e_235_18
  have e_235_19 := arithEq_of_rows h (row := 235) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_235_19
  simp only [← c15381, ← c15382, k3, ← c15383] at e_235_19
  refine ⟨?_, ?_⟩
  · have hc := e_234_16
    simp only [e_235_18] at hc
    linear_combination c15384.trans k2 - hc
  · have hc := e_235_19
    simp only [e_235_17, e_234_17, e_235_18] at hc
    linear_combination c15385.trans k2 - hc

theorem publicBatchWrapper89_f2196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 234 75) = band (a (.virt 18919)) (a (.virt 18921)) := by
  have c15386 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15387 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15388 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_234_18 := arithEq_of_rows h (row := 234) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_18
  simp only [← c15386, ← c15387, ← c15388] at e_234_18
  have hr := e_234_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 234 79) = band (a (.virt 18923)) (a (.virt 18925)) := by
  have c15389 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15390 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15391 := (publicBatchWrapper89_copies480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_234_19 := arithEq_of_rows h (row := 234) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_234_19
  simp only [← c15389, ← c15390, ← c15391] at e_234_19
  have hr := e_234_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 237 3) = band (a (.wire 234 75)) (a (.wire 234 79)) := by
  have c15392 := (publicBatchWrapper89_copies481 a h).1
  have c15393 := (publicBatchWrapper89_copies481 a h).2.1
  have c15394 := (publicBatchWrapper89_copies481 a h).2.2.1
  have e_237_0 := arithEq_of_rows h (row := 237) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_0
  simp only [← c15392, ← c15393, ← c15394] at e_237_0
  have hr := e_237_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 7) = bor (a (.wire 33 71)) (a (.wire 237 3)) := by
  have c15395 := (publicBatchWrapper89_copies481 a h).2.2.2.1
  have c15396 := (publicBatchWrapper89_copies481 a h).2.2.2.2.1
  have c15397 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.1
  have c15398 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.1
  have c15399 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.1
  have c15400 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_9 := arithEq_of_rows h (row := 231) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_9
  simp only [← c15395, ← c15396, ← c15397] at e_231_9
  have e_236_1 := arithEq_of_rows h (row := 236) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_1
  simp only [← c15398, ← c15399, k3, ← c15400] at e_236_1
  have hr := e_236_1
  simp only [e_231_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 7) = a (.virt 201) := by
  have c15401 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.1
  exact c15401

theorem publicBatchWrapper89_f2201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6803)) (a (.wire 170 23)) (a (.virt 18927)) (a (.virt 18928)) := by
  have c15402 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15403 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15404 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15405 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15406 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15407 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15408 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15409 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15410 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15411 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15412 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15413 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15414 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15415 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15416 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15417 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15418 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_237_1 := arithEq_of_rows h (row := 237) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_1
  simp only [← c15408, ← c15409, ← c15410] at e_237_1
  have e_237_2 := arithEq_of_rows h (row := 237) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_2
  simp only [← c15411, ← c15412, ← c15413] at e_237_2
  have e_238_0 := arithEq_of_rows h (row := 238) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_0
  simp only [← c15402, k3, ← c15403, k3, ← c15404] at e_238_0
  have e_238_1 := arithEq_of_rows h (row := 238) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_1
  simp only [← c15405, ← c15406, k3, ← c15407] at e_238_1
  have e_238_2 := arithEq_of_rows h (row := 238) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_2
  simp only [← c15414, ← c15415, k3, ← c15416] at e_238_2
  refine ⟨?_, ?_⟩
  · have hc := e_237_1
    simp only [e_238_1] at hc
    linear_combination c15417.trans k2 - hc
  · have hc := e_238_2
    simp only [e_238_0, e_237_2, e_238_1] at hc
    linear_combination c15418.trans k2 - hc

theorem publicBatchWrapper89_f2202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 11) = bor (a (.wire 34 35)) (a (.virt 18927)) := by
  have c15419 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15420 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15421 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15422 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15423 := (publicBatchWrapper89_copies481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15424 := (publicBatchWrapper89_copies482 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_10 := arithEq_of_rows h (row := 231) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_10
  simp only [← c15419, ← c15420, ← c15421] at e_231_10
  have e_236_2 := arithEq_of_rows h (row := 236) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_2
  simp only [← c15422, ← c15423, k3, ← c15424] at e_236_2
  have hr := e_236_2
  simp only [e_231_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 11) = a (.virt 201) := by
  have c15425 := (publicBatchWrapper89_copies482 a h).2.1
  exact c15425

theorem publicBatchWrapper89_f2204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6804)) (a (.wire 170 31)) (a (.virt 18929)) (a (.virt 18930)) := by
  have c15426 := (publicBatchWrapper89_copies482 a h).2.2.1
  have c15427 := (publicBatchWrapper89_copies482 a h).2.2.2.1
  have c15428 := (publicBatchWrapper89_copies482 a h).2.2.2.2.1
  have c15429 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.1
  have c15430 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.1
  have c15431 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.1
  have c15432 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.1
  have c15433 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.1
  have c15434 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15435 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15436 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15437 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15438 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15439 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15440 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15441 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15442 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_237_3 := arithEq_of_rows h (row := 237) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_3
  simp only [← c15432, ← c15433, ← c15434] at e_237_3
  have e_237_4 := arithEq_of_rows h (row := 237) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_4
  simp only [← c15435, ← c15436, ← c15437] at e_237_4
  have e_238_3 := arithEq_of_rows h (row := 238) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_3
  simp only [← c15426, k3, ← c15427, k3, ← c15428] at e_238_3
  have e_238_4 := arithEq_of_rows h (row := 238) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_4
  simp only [← c15429, ← c15430, k3, ← c15431] at e_238_4
  have e_238_5 := arithEq_of_rows h (row := 238) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_5
  simp only [← c15438, ← c15439, k3, ← c15440] at e_238_5
  refine ⟨?_, ?_⟩
  · have hc := e_237_3
    simp only [e_238_4] at hc
    linear_combination c15441.trans k2 - hc
  · have hc := e_238_5
    simp only [e_238_3, e_237_4, e_238_4] at hc
    linear_combination c15442.trans k2 - hc

theorem publicBatchWrapper89_f2205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 15) = bor (a (.wire 34 35)) (a (.virt 18929)) := by
  have c15443 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15444 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15445 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15446 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15447 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15448 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_11 := arithEq_of_rows h (row := 231) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_11
  simp only [← c15443, ← c15444, ← c15445] at e_231_11
  have e_236_3 := arithEq_of_rows h (row := 236) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_3
  simp only [← c15446, ← c15447, k3, ← c15448] at e_236_3
  have hr := e_236_3
  simp only [e_231_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 15) = a (.virt 201) := by
  have c15449 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15449

theorem publicBatchWrapper89_f2207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6805)) (a (.wire 169 63)) (a (.virt 18931)) (a (.virt 18932)) := by
  have c15450 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15451 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15452 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15453 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15454 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15455 := (publicBatchWrapper89_copies482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15456 := (publicBatchWrapper89_copies483 a h).1
  have c15457 := (publicBatchWrapper89_copies483 a h).2.1
  have c15458 := (publicBatchWrapper89_copies483 a h).2.2.1
  have c15459 := (publicBatchWrapper89_copies483 a h).2.2.2.1
  have c15460 := (publicBatchWrapper89_copies483 a h).2.2.2.2.1
  have c15461 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.1
  have c15462 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.1
  have c15463 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.1
  have c15464 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.1
  have c15465 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.1
  have c15466 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_237_5 := arithEq_of_rows h (row := 237) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_5
  simp only [← c15456, ← c15457, ← c15458] at e_237_5
  have e_237_6 := arithEq_of_rows h (row := 237) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_6
  simp only [← c15459, ← c15460, ← c15461] at e_237_6
  have e_238_6 := arithEq_of_rows h (row := 238) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_6
  simp only [← c15450, k3, ← c15451, k3, ← c15452] at e_238_6
  have e_238_7 := arithEq_of_rows h (row := 238) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_7
  simp only [← c15453, ← c15454, k3, ← c15455] at e_238_7
  have e_238_8 := arithEq_of_rows h (row := 238) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_8
  simp only [← c15462, ← c15463, k3, ← c15464] at e_238_8
  refine ⟨?_, ?_⟩
  · have hc := e_237_5
    simp only [e_238_7] at hc
    linear_combination c15465.trans k2 - hc
  · have hc := e_238_8
    simp only [e_238_6, e_237_6, e_238_7] at hc
    linear_combination c15466.trans k2 - hc

theorem publicBatchWrapper89_f2208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6806)) (a (.wire 169 71)) (a (.virt 18933)) (a (.virt 18934)) := by
  have c15467 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15468 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15469 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15470 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15471 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15472 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15473 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15474 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15475 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15476 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15477 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15478 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15479 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15480 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15481 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15482 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15483 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_237_7 := arithEq_of_rows h (row := 237) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_7
  simp only [← c15473, ← c15474, ← c15475] at e_237_7
  have e_237_8 := arithEq_of_rows h (row := 237) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_8
  simp only [← c15476, ← c15477, ← c15478] at e_237_8
  have e_238_9 := arithEq_of_rows h (row := 238) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_9
  simp only [← c15467, k3, ← c15468, k3, ← c15469] at e_238_9
  have e_238_10 := arithEq_of_rows h (row := 238) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_10
  simp only [← c15470, ← c15471, k3, ← c15472] at e_238_10
  have e_238_11 := arithEq_of_rows h (row := 238) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_11
  simp only [← c15479, ← c15480, k3, ← c15481] at e_238_11
  refine ⟨?_, ?_⟩
  · have hc := e_237_7
    simp only [e_238_10] at hc
    linear_combination c15482.trans k2 - hc
  · have hc := e_238_11
    simp only [e_238_9, e_237_8, e_238_10] at hc
    linear_combination c15483.trans k2 - hc

theorem publicBatchWrapper89_f2209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6807)) (a (.wire 169 79)) (a (.virt 18935)) (a (.virt 18936)) := by
  have c15484 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15485 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15486 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15487 := (publicBatchWrapper89_copies483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15488 := (publicBatchWrapper89_copies484 a h).1
  have c15489 := (publicBatchWrapper89_copies484 a h).2.1
  have c15490 := (publicBatchWrapper89_copies484 a h).2.2.1
  have c15491 := (publicBatchWrapper89_copies484 a h).2.2.2.1
  have c15492 := (publicBatchWrapper89_copies484 a h).2.2.2.2.1
  have c15493 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.1
  have c15494 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.1
  have c15495 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.1
  have c15496 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.1
  have c15497 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.1
  have c15498 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15499 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15500 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_237_9 := arithEq_of_rows h (row := 237) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_9
  simp only [← c15490, ← c15491, ← c15492] at e_237_9
  have e_237_10 := arithEq_of_rows h (row := 237) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_10
  simp only [← c15493, ← c15494, ← c15495] at e_237_10
  have e_238_12 := arithEq_of_rows h (row := 238) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_12
  simp only [← c15484, k3, ← c15485, k3, ← c15486] at e_238_12
  have e_238_13 := arithEq_of_rows h (row := 238) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_13
  simp only [← c15487, ← c15488, k3, ← c15489] at e_238_13
  have e_238_14 := arithEq_of_rows h (row := 238) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_14
  simp only [← c15496, ← c15497, k3, ← c15498] at e_238_14
  refine ⟨?_, ?_⟩
  · have hc := e_237_9
    simp only [e_238_13] at hc
    linear_combination c15499.trans k2 - hc
  · have hc := e_238_14
    simp only [e_238_12, e_237_10, e_238_13] at hc
    linear_combination c15500.trans k2 - hc

theorem publicBatchWrapper89_f2210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6808)) (a (.wire 170 7)) (a (.virt 18937)) (a (.virt 18938)) := by
  have c15501 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15502 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15503 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15504 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15505 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15506 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15507 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15508 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15509 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15510 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15511 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15512 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15513 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15514 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15515 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15516 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15517 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_237_11 := arithEq_of_rows h (row := 237) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_11
  simp only [← c15507, ← c15508, ← c15509] at e_237_11
  have e_237_12 := arithEq_of_rows h (row := 237) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_12
  simp only [← c15510, ← c15511, ← c15512] at e_237_12
  have e_238_15 := arithEq_of_rows h (row := 238) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_15
  simp only [← c15501, k3, ← c15502, k3, ← c15503] at e_238_15
  have e_238_16 := arithEq_of_rows h (row := 238) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_16
  simp only [← c15504, ← c15505, k3, ← c15506] at e_238_16
  have e_238_17 := arithEq_of_rows h (row := 238) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_17
  simp only [← c15513, ← c15514, k3, ← c15515] at e_238_17
  refine ⟨?_, ?_⟩
  · have hc := e_237_11
    simp only [e_238_16] at hc
    linear_combination c15516.trans k2 - hc
  · have hc := e_238_17
    simp only [e_238_15, e_237_12, e_238_16] at hc
    linear_combination c15517.trans k2 - hc

theorem publicBatchWrapper89_f2211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 237 55) = band (a (.virt 18931)) (a (.virt 18933)) := by
  have c15518 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15519 := (publicBatchWrapper89_copies484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15520 := (publicBatchWrapper89_copies485 a h).1
  have e_237_13 := arithEq_of_rows h (row := 237) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_13
  simp only [← c15518, ← c15519, ← c15520] at e_237_13
  have hr := e_237_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 237 59) = band (a (.virt 18935)) (a (.virt 18937)) := by
  have c15521 := (publicBatchWrapper89_copies485 a h).2.1
  have c15522 := (publicBatchWrapper89_copies485 a h).2.2.1
  have c15523 := (publicBatchWrapper89_copies485 a h).2.2.2.1
  have e_237_14 := arithEq_of_rows h (row := 237) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_14
  simp only [← c15521, ← c15522, ← c15523] at e_237_14
  have hr := e_237_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 237 63) = band (a (.wire 237 55)) (a (.wire 237 59)) := by
  have c15524 := (publicBatchWrapper89_copies485 a h).2.2.2.2.1
  have c15525 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.1
  have c15526 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.1
  have e_237_15 := arithEq_of_rows h (row := 237) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_15
  simp only [← c15524, ← c15525, ← c15526] at e_237_15
  have hr := e_237_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 19) = bor (a (.wire 34 35)) (a (.wire 237 63)) := by
  have c15527 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.1
  have c15528 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.1
  have c15529 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.1
  have c15530 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15531 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15532 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_12 := arithEq_of_rows h (row := 231) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_12
  simp only [← c15527, ← c15528, ← c15529] at e_231_12
  have e_236_4 := arithEq_of_rows h (row := 236) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_4
  simp only [← c15530, ← c15531, k3, ← c15532] at e_236_4
  have hr := e_236_4
  simp only [e_231_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 19) = a (.virt 201) := by
  have c15533 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15533

theorem publicBatchWrapper89_f2216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7003)) (a (.wire 170 23)) (a (.virt 18939)) (a (.virt 18940)) := by
  have c15534 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15535 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15536 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15537 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15538 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15539 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15540 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15541 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15542 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15543 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15544 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15545 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15546 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15547 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15548 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15549 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15550 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_237_16 := arithEq_of_rows h (row := 237) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_16
  simp only [← c15540, ← c15541, ← c15542] at e_237_16
  have e_237_17 := arithEq_of_rows h (row := 237) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_17
  simp only [← c15543, ← c15544, ← c15545] at e_237_17
  have e_238_18 := arithEq_of_rows h (row := 238) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_18
  simp only [← c15534, k3, ← c15535, k3, ← c15536] at e_238_18
  have e_238_19 := arithEq_of_rows h (row := 238) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_238_19
  simp only [← c15537, ← c15538, k3, ← c15539] at e_238_19
  have e_239_0 := arithEq_of_rows h (row := 239) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_0
  simp only [← c15546, ← c15547, k3, ← c15548] at e_239_0
  refine ⟨?_, ?_⟩
  · have hc := e_237_16
    simp only [e_238_19] at hc
    linear_combination c15549.trans k2 - hc
  · have hc := e_239_0
    simp only [e_238_18, e_237_17, e_238_19] at hc
    linear_combination c15550.trans k2 - hc

theorem publicBatchWrapper89_f2217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 23) = bor (a (.wire 34 79)) (a (.virt 18939)) := by
  have c15551 := (publicBatchWrapper89_copies485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15552 := (publicBatchWrapper89_copies486 a h).1
  have c15553 := (publicBatchWrapper89_copies486 a h).2.1
  have c15554 := (publicBatchWrapper89_copies486 a h).2.2.1
  have c15555 := (publicBatchWrapper89_copies486 a h).2.2.2.1
  have c15556 := (publicBatchWrapper89_copies486 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_13 := arithEq_of_rows h (row := 231) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_13
  simp only [← c15551, ← c15552, ← c15553] at e_231_13
  have e_236_5 := arithEq_of_rows h (row := 236) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_5
  simp only [← c15554, ← c15555, k3, ← c15556] at e_236_5
  have hr := e_236_5
  simp only [e_231_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 23) = a (.virt 201) := by
  have c15557 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.1
  exact c15557

theorem publicBatchWrapper89_f2219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7004)) (a (.wire 170 31)) (a (.virt 18941)) (a (.virt 18942)) := by
  have c15558 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.1
  have c15559 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.1
  have c15560 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.1
  have c15561 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.1
  have c15562 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15563 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15564 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15565 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15566 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15567 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15568 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15569 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15570 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15571 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15572 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15573 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15574 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_237_18 := arithEq_of_rows h (row := 237) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_18
  simp only [← c15564, ← c15565, ← c15566] at e_237_18
  have e_237_19 := arithEq_of_rows h (row := 237) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_237_19
  simp only [← c15567, ← c15568, ← c15569] at e_237_19
  have e_239_1 := arithEq_of_rows h (row := 239) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_1
  simp only [← c15558, k3, ← c15559, k3, ← c15560] at e_239_1
  have e_239_2 := arithEq_of_rows h (row := 239) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_2
  simp only [← c15561, ← c15562, k3, ← c15563] at e_239_2
  have e_239_3 := arithEq_of_rows h (row := 239) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_3
  simp only [← c15570, ← c15571, k3, ← c15572] at e_239_3
  refine ⟨?_, ?_⟩
  · have hc := e_237_18
    simp only [e_239_2] at hc
    linear_combination c15573.trans k2 - hc
  · have hc := e_239_3
    simp only [e_239_1, e_237_19, e_239_2] at hc
    linear_combination c15574.trans k2 - hc

theorem publicBatchWrapper89_f2220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 27) = bor (a (.wire 34 79)) (a (.virt 18941)) := by
  have c15575 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15576 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15577 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15578 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15579 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15580 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_14 := arithEq_of_rows h (row := 231) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_14
  simp only [← c15575, ← c15576, ← c15577] at e_231_14
  have e_236_6 := arithEq_of_rows h (row := 236) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_6
  simp only [← c15578, ← c15579, k3, ← c15580] at e_236_6
  have hr := e_236_6
  simp only [e_231_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 27) = a (.virt 201) := by
  have c15581 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15581

theorem publicBatchWrapper89_f2222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7005)) (a (.wire 169 63)) (a (.virt 18943)) (a (.virt 18944)) := by
  have c15582 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15583 := (publicBatchWrapper89_copies486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15584 := (publicBatchWrapper89_copies487 a h).1
  have c15585 := (publicBatchWrapper89_copies487 a h).2.1
  have c15586 := (publicBatchWrapper89_copies487 a h).2.2.1
  have c15587 := (publicBatchWrapper89_copies487 a h).2.2.2.1
  have c15588 := (publicBatchWrapper89_copies487 a h).2.2.2.2.1
  have c15589 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.1
  have c15590 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.1
  have c15591 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.1
  have c15592 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.1
  have c15593 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.1
  have c15594 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15595 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15596 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15597 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15598 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_239_4 := arithEq_of_rows h (row := 239) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_4
  simp only [← c15582, k3, ← c15583, k3, ← c15584] at e_239_4
  have e_239_5 := arithEq_of_rows h (row := 239) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_5
  simp only [← c15585, ← c15586, k3, ← c15587] at e_239_5
  have e_239_6 := arithEq_of_rows h (row := 239) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_6
  simp only [← c15594, ← c15595, k3, ← c15596] at e_239_6
  have e_240_0 := arithEq_of_rows h (row := 240) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_0
  simp only [← c15588, ← c15589, ← c15590] at e_240_0
  have e_240_1 := arithEq_of_rows h (row := 240) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_1
  simp only [← c15591, ← c15592, ← c15593] at e_240_1
  refine ⟨?_, ?_⟩
  · have hc := e_240_0
    simp only [e_239_5] at hc
    linear_combination c15597.trans k2 - hc
  · have hc := e_239_6
    simp only [e_239_4, e_240_1, e_239_5] at hc
    linear_combination c15598.trans k2 - hc

theorem publicBatchWrapper89_f2223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7006)) (a (.wire 169 71)) (a (.virt 18945)) (a (.virt 18946)) := by
  have c15599 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15600 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15601 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15602 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15603 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15604 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15605 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15606 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15607 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15608 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15609 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15610 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15611 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15612 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15613 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15614 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15615 := (publicBatchWrapper89_copies487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_239_7 := arithEq_of_rows h (row := 239) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_7
  simp only [← c15599, k3, ← c15600, k3, ← c15601] at e_239_7
  have e_239_8 := arithEq_of_rows h (row := 239) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_8
  simp only [← c15602, ← c15603, k3, ← c15604] at e_239_8
  have e_239_9 := arithEq_of_rows h (row := 239) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_9
  simp only [← c15611, ← c15612, k3, ← c15613] at e_239_9
  have e_240_2 := arithEq_of_rows h (row := 240) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_2
  simp only [← c15605, ← c15606, ← c15607] at e_240_2
  have e_240_3 := arithEq_of_rows h (row := 240) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_3
  simp only [← c15608, ← c15609, ← c15610] at e_240_3
  refine ⟨?_, ?_⟩
  · have hc := e_240_2
    simp only [e_239_8] at hc
    linear_combination c15614.trans k2 - hc
  · have hc := e_239_9
    simp only [e_239_7, e_240_3, e_239_8] at hc
    linear_combination c15615.trans k2 - hc

theorem publicBatchWrapper89_f2224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7007)) (a (.wire 169 79)) (a (.virt 18947)) (a (.virt 18948)) := by
  have c15616 := (publicBatchWrapper89_copies488 a h).1
  have c15617 := (publicBatchWrapper89_copies488 a h).2.1
  have c15618 := (publicBatchWrapper89_copies488 a h).2.2.1
  have c15619 := (publicBatchWrapper89_copies488 a h).2.2.2.1
  have c15620 := (publicBatchWrapper89_copies488 a h).2.2.2.2.1
  have c15621 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.1
  have c15622 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.1
  have c15623 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.1
  have c15624 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.1
  have c15625 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.1
  have c15626 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15627 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15628 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15629 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15630 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15631 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15632 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_239_10 := arithEq_of_rows h (row := 239) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_10
  simp only [← c15616, k3, ← c15617, k3, ← c15618] at e_239_10
  have e_239_11 := arithEq_of_rows h (row := 239) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_11
  simp only [← c15619, ← c15620, k3, ← c15621] at e_239_11
  have e_239_12 := arithEq_of_rows h (row := 239) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_12
  simp only [← c15628, ← c15629, k3, ← c15630] at e_239_12
  have e_240_4 := arithEq_of_rows h (row := 240) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_4
  simp only [← c15622, ← c15623, ← c15624] at e_240_4
  have e_240_5 := arithEq_of_rows h (row := 240) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_5
  simp only [← c15625, ← c15626, ← c15627] at e_240_5
  refine ⟨?_, ?_⟩
  · have hc := e_240_4
    simp only [e_239_11] at hc
    linear_combination c15631.trans k2 - hc
  · have hc := e_239_12
    simp only [e_239_10, e_240_5, e_239_11] at hc
    linear_combination c15632.trans k2 - hc

theorem publicBatchWrapper89_f2225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7008)) (a (.wire 170 7)) (a (.virt 18949)) (a (.virt 18950)) := by
  have c15633 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15634 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15635 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15636 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15637 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15638 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15639 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15640 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15641 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15642 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15643 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15644 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15645 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15646 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15647 := (publicBatchWrapper89_copies488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15648 := (publicBatchWrapper89_copies489 a h).1
  have c15649 := (publicBatchWrapper89_copies489 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_239_13 := arithEq_of_rows h (row := 239) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_13
  simp only [← c15633, k3, ← c15634, k3, ← c15635] at e_239_13
  have e_239_14 := arithEq_of_rows h (row := 239) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_14
  simp only [← c15636, ← c15637, k3, ← c15638] at e_239_14
  have e_239_15 := arithEq_of_rows h (row := 239) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_15
  simp only [← c15645, ← c15646, k3, ← c15647] at e_239_15
  have e_240_6 := arithEq_of_rows h (row := 240) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_6
  simp only [← c15639, ← c15640, ← c15641] at e_240_6
  have e_240_7 := arithEq_of_rows h (row := 240) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_7
  simp only [← c15642, ← c15643, ← c15644] at e_240_7
  refine ⟨?_, ?_⟩
  · have hc := e_240_6
    simp only [e_239_14] at hc
    linear_combination c15648.trans k2 - hc
  · have hc := e_239_15
    simp only [e_239_13, e_240_7, e_239_14] at hc
    linear_combination c15649.trans k2 - hc

theorem publicBatchWrapper89_f2226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 240 35) = band (a (.virt 18943)) (a (.virt 18945)) := by
  have c15650 := (publicBatchWrapper89_copies489 a h).2.2.1
  have c15651 := (publicBatchWrapper89_copies489 a h).2.2.2.1
  have c15652 := (publicBatchWrapper89_copies489 a h).2.2.2.2.1
  have e_240_8 := arithEq_of_rows h (row := 240) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_8
  simp only [← c15650, ← c15651, ← c15652] at e_240_8
  have hr := e_240_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 240 39) = band (a (.virt 18947)) (a (.virt 18949)) := by
  have c15653 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.1
  have c15654 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.1
  have c15655 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.1
  have e_240_9 := arithEq_of_rows h (row := 240) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_9
  simp only [← c15653, ← c15654, ← c15655] at e_240_9
  have hr := e_240_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 240 43) = band (a (.wire 240 35)) (a (.wire 240 39)) := by
  have c15656 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.1
  have c15657 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.1
  have c15658 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_240_10 := arithEq_of_rows h (row := 240) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_10
  simp only [← c15656, ← c15657, ← c15658] at e_240_10
  have hr := e_240_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 31) = bor (a (.wire 34 79)) (a (.wire 240 43)) := by
  have c15659 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15660 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15661 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15662 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15663 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15664 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_15 := arithEq_of_rows h (row := 231) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_15
  simp only [← c15659, ← c15660, ← c15661] at e_231_15
  have e_236_7 := arithEq_of_rows h (row := 236) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_7
  simp only [← c15662, ← c15663, k3, ← c15664] at e_236_7
  have hr := e_236_7
  simp only [e_231_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 31) = a (.virt 201) := by
  have c15665 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15665

theorem publicBatchWrapper89_f2231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7203)) (a (.wire 170 23)) (a (.virt 18951)) (a (.virt 18952)) := by
  have c15666 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15667 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15668 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15669 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15670 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15671 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15672 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15673 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15674 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15675 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15676 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15677 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15678 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15679 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15680 := (publicBatchWrapper89_copies490 a h).1
  have c15681 := (publicBatchWrapper89_copies490 a h).2.1
  have c15682 := (publicBatchWrapper89_copies490 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_239_16 := arithEq_of_rows h (row := 239) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_16
  simp only [← c15666, k3, ← c15667, k3, ← c15668] at e_239_16
  have e_239_17 := arithEq_of_rows h (row := 239) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_17
  simp only [← c15669, ← c15670, k3, ← c15671] at e_239_17
  have e_239_18 := arithEq_of_rows h (row := 239) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_18
  simp only [← c15678, ← c15679, k3, ← c15680] at e_239_18
  have e_240_11 := arithEq_of_rows h (row := 240) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_11
  simp only [← c15672, ← c15673, ← c15674] at e_240_11
  have e_240_12 := arithEq_of_rows h (row := 240) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_12
  simp only [← c15675, ← c15676, ← c15677] at e_240_12
  refine ⟨?_, ?_⟩
  · have hc := e_240_11
    simp only [e_239_17] at hc
    linear_combination c15681.trans k2 - hc
  · have hc := e_239_18
    simp only [e_239_16, e_240_12, e_239_17] at hc
    linear_combination c15682.trans k2 - hc

theorem publicBatchWrapper89_f2232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 35) = bor (a (.wire 36 43)) (a (.virt 18951)) := by
  have c15683 := (publicBatchWrapper89_copies490 a h).2.2.2.1
  have c15684 := (publicBatchWrapper89_copies490 a h).2.2.2.2.1
  have c15685 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.1
  have c15686 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.1
  have c15687 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.1
  have c15688 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_16 := arithEq_of_rows h (row := 231) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_16
  simp only [← c15683, ← c15684, ← c15685] at e_231_16
  have e_236_8 := arithEq_of_rows h (row := 236) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_8
  simp only [← c15686, ← c15687, k3, ← c15688] at e_236_8
  have hr := e_236_8
  simp only [e_231_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 35) = a (.virt 201) := by
  have c15689 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.1
  exact c15689

theorem publicBatchWrapper89_f2234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7204)) (a (.wire 170 31)) (a (.virt 18953)) (a (.virt 18954)) := by
  have c15690 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15691 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15692 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15693 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15694 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15695 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15696 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15697 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15698 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15699 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15700 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15701 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15702 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15703 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15704 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15705 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15706 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_239_19 := arithEq_of_rows h (row := 239) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_19
  simp only [← c15690, k3, ← c15691, k3, ← c15692] at e_239_19
  have e_240_13 := arithEq_of_rows h (row := 240) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_13
  simp only [← c15696, ← c15697, ← c15698] at e_240_13
  have e_240_14 := arithEq_of_rows h (row := 240) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_14
  simp only [← c15699, ← c15700, ← c15701] at e_240_14
  have e_241_0 := arithEq_of_rows h (row := 241) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_0
  simp only [← c15693, ← c15694, k3, ← c15695] at e_241_0
  have e_241_1 := arithEq_of_rows h (row := 241) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_1
  simp only [← c15702, ← c15703, k3, ← c15704] at e_241_1
  refine ⟨?_, ?_⟩
  · have hc := e_240_13
    simp only [e_241_0] at hc
    linear_combination c15705.trans k2 - hc
  · have hc := e_241_1
    simp only [e_239_19, e_240_14, e_241_0] at hc
    linear_combination c15706.trans k2 - hc

theorem publicBatchWrapper89_f2235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 39) = bor (a (.wire 36 43)) (a (.virt 18953)) := by
  have c15707 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15708 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15709 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15710 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15711 := (publicBatchWrapper89_copies490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15712 := (publicBatchWrapper89_copies491 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_17 := arithEq_of_rows h (row := 231) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_17
  simp only [← c15707, ← c15708, ← c15709] at e_231_17
  have e_236_9 := arithEq_of_rows h (row := 236) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_9
  simp only [← c15710, ← c15711, k3, ← c15712] at e_236_9
  have hr := e_236_9
  simp only [e_231_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 39) = a (.virt 201) := by
  have c15713 := (publicBatchWrapper89_copies491 a h).2.1
  exact c15713

theorem publicBatchWrapper89_f2237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7205)) (a (.wire 169 63)) (a (.virt 18955)) (a (.virt 18956)) := by
  have c15714 := (publicBatchWrapper89_copies491 a h).2.2.1
  have c15715 := (publicBatchWrapper89_copies491 a h).2.2.2.1
  have c15716 := (publicBatchWrapper89_copies491 a h).2.2.2.2.1
  have c15717 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.1
  have c15718 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.1
  have c15719 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.1
  have c15720 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.1
  have c15721 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.1
  have c15722 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15723 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15724 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15725 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15726 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15727 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15728 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15729 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15730 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_240_15 := arithEq_of_rows h (row := 240) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_15
  simp only [← c15720, ← c15721, ← c15722] at e_240_15
  have e_240_16 := arithEq_of_rows h (row := 240) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_16
  simp only [← c15723, ← c15724, ← c15725] at e_240_16
  have e_241_2 := arithEq_of_rows h (row := 241) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_2
  simp only [← c15714, k3, ← c15715, k3, ← c15716] at e_241_2
  have e_241_3 := arithEq_of_rows h (row := 241) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_3
  simp only [← c15717, ← c15718, k3, ← c15719] at e_241_3
  have e_241_4 := arithEq_of_rows h (row := 241) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_4
  simp only [← c15726, ← c15727, k3, ← c15728] at e_241_4
  refine ⟨?_, ?_⟩
  · have hc := e_240_15
    simp only [e_241_3] at hc
    linear_combination c15729.trans k2 - hc
  · have hc := e_241_4
    simp only [e_241_2, e_240_16, e_241_3] at hc
    linear_combination c15730.trans k2 - hc

theorem publicBatchWrapper89_f2238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7206)) (a (.wire 169 71)) (a (.virt 18957)) (a (.virt 18958)) := by
  have c15731 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15732 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15733 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15734 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15735 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15736 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15737 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15738 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15739 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15740 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15741 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15742 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15743 := (publicBatchWrapper89_copies491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15744 := (publicBatchWrapper89_copies492 a h).1
  have c15745 := (publicBatchWrapper89_copies492 a h).2.1
  have c15746 := (publicBatchWrapper89_copies492 a h).2.2.1
  have c15747 := (publicBatchWrapper89_copies492 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_240_17 := arithEq_of_rows h (row := 240) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_17
  simp only [← c15737, ← c15738, ← c15739] at e_240_17
  have e_240_18 := arithEq_of_rows h (row := 240) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_18
  simp only [← c15740, ← c15741, ← c15742] at e_240_18
  have e_241_5 := arithEq_of_rows h (row := 241) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_5
  simp only [← c15731, k3, ← c15732, k3, ← c15733] at e_241_5
  have e_241_6 := arithEq_of_rows h (row := 241) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_6
  simp only [← c15734, ← c15735, k3, ← c15736] at e_241_6
  have e_241_7 := arithEq_of_rows h (row := 241) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_7
  simp only [← c15743, ← c15744, k3, ← c15745] at e_241_7
  refine ⟨?_, ?_⟩
  · have hc := e_240_17
    simp only [e_241_6] at hc
    linear_combination c15746.trans k2 - hc
  · have hc := e_241_7
    simp only [e_241_5, e_240_18, e_241_6] at hc
    linear_combination c15747.trans k2 - hc

theorem publicBatchWrapper89_f2239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7207)) (a (.wire 169 79)) (a (.virt 18959)) (a (.virt 18960)) := by
  have c15748 := (publicBatchWrapper89_copies492 a h).2.2.2.2.1
  have c15749 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.1
  have c15750 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.1
  have c15751 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.1
  have c15752 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.1
  have c15753 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.1
  have c15754 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15755 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15756 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15757 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15758 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15759 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15760 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15761 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15762 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15763 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15764 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_240_19 := arithEq_of_rows h (row := 240) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_19
  simp only [← c15754, ← c15755, ← c15756] at e_240_19
  have e_241_8 := arithEq_of_rows h (row := 241) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_8
  simp only [← c15748, k3, ← c15749, k3, ← c15750] at e_241_8
  have e_241_9 := arithEq_of_rows h (row := 241) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_9
  simp only [← c15751, ← c15752, k3, ← c15753] at e_241_9
  have e_241_10 := arithEq_of_rows h (row := 241) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_10
  simp only [← c15760, ← c15761, k3, ← c15762] at e_241_10
  have e_242_0 := arithEq_of_rows h (row := 242) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_0
  simp only [← c15757, ← c15758, ← c15759] at e_242_0
  refine ⟨?_, ?_⟩
  · have hc := e_240_19
    simp only [e_241_9] at hc
    linear_combination c15763.trans k2 - hc
  · have hc := e_241_10
    simp only [e_241_8, e_242_0, e_241_9] at hc
    linear_combination c15764.trans k2 - hc

theorem publicBatchWrapper89_f2240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7208)) (a (.wire 170 7)) (a (.virt 18961)) (a (.virt 18962)) := by
  have c15765 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15766 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15767 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15768 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15769 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15770 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15771 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15772 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15773 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15774 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15775 := (publicBatchWrapper89_copies492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15776 := (publicBatchWrapper89_copies493 a h).1
  have c15777 := (publicBatchWrapper89_copies493 a h).2.1
  have c15778 := (publicBatchWrapper89_copies493 a h).2.2.1
  have c15779 := (publicBatchWrapper89_copies493 a h).2.2.2.1
  have c15780 := (publicBatchWrapper89_copies493 a h).2.2.2.2.1
  have c15781 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_241_11 := arithEq_of_rows h (row := 241) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_11
  simp only [← c15765, k3, ← c15766, k3, ← c15767] at e_241_11
  have e_241_12 := arithEq_of_rows h (row := 241) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_12
  simp only [← c15768, ← c15769, k3, ← c15770] at e_241_12
  have e_241_13 := arithEq_of_rows h (row := 241) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_13
  simp only [← c15777, ← c15778, k3, ← c15779] at e_241_13
  have e_242_1 := arithEq_of_rows h (row := 242) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_1
  simp only [← c15771, ← c15772, ← c15773] at e_242_1
  have e_242_2 := arithEq_of_rows h (row := 242) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_2
  simp only [← c15774, ← c15775, ← c15776] at e_242_2
  refine ⟨?_, ?_⟩
  · have hc := e_242_1
    simp only [e_241_12] at hc
    linear_combination c15780.trans k2 - hc
  · have hc := e_241_13
    simp only [e_241_11, e_242_2, e_241_12] at hc
    linear_combination c15781.trans k2 - hc

theorem publicBatchWrapper89_f2241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 242 15) = band (a (.virt 18955)) (a (.virt 18957)) := by
  have c15782 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.1
  have c15783 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.1
  have c15784 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.1
  have e_242_3 := arithEq_of_rows h (row := 242) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_3
  simp only [← c15782, ← c15783, ← c15784] at e_242_3
  have hr := e_242_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 242 19) = band (a (.virt 18959)) (a (.virt 18961)) := by
  have c15785 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.1
  have c15786 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15787 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_242_4 := arithEq_of_rows h (row := 242) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_4
  simp only [← c15785, ← c15786, ← c15787] at e_242_4
  have hr := e_242_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 242 23) = band (a (.wire 242 15)) (a (.wire 242 19)) := by
  have c15788 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15789 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15790 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_242_5 := arithEq_of_rows h (row := 242) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_5
  simp only [← c15788, ← c15789, ← c15790] at e_242_5
  have hr := e_242_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 43) = bor (a (.wire 36 43)) (a (.wire 242 23)) := by
  have c15791 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15792 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15793 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15794 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15795 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15796 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_18 := arithEq_of_rows h (row := 231) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_18
  simp only [← c15791, ← c15792, ← c15793] at e_231_18
  have e_236_10 := arithEq_of_rows h (row := 236) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_10
  simp only [← c15794, ← c15795, k3, ← c15796] at e_236_10
  have hr := e_236_10
  simp only [e_231_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 43) = a (.virt 201) := by
  have c15797 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15797

theorem publicBatchWrapper89_f2246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7403)) (a (.wire 170 23)) (a (.virt 18963)) (a (.virt 18964)) := by
  have c15798 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15799 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15800 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15801 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15802 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15803 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15804 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15805 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15806 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15807 := (publicBatchWrapper89_copies493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15808 := (publicBatchWrapper89_copies494 a h).1
  have c15809 := (publicBatchWrapper89_copies494 a h).2.1
  have c15810 := (publicBatchWrapper89_copies494 a h).2.2.1
  have c15811 := (publicBatchWrapper89_copies494 a h).2.2.2.1
  have c15812 := (publicBatchWrapper89_copies494 a h).2.2.2.2.1
  have c15813 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.1
  have c15814 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_241_14 := arithEq_of_rows h (row := 241) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_14
  simp only [← c15798, k3, ← c15799, k3, ← c15800] at e_241_14
  have e_241_15 := arithEq_of_rows h (row := 241) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_15
  simp only [← c15801, ← c15802, k3, ← c15803] at e_241_15
  have e_241_16 := arithEq_of_rows h (row := 241) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_16
  simp only [← c15810, ← c15811, k3, ← c15812] at e_241_16
  have e_242_6 := arithEq_of_rows h (row := 242) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_6
  simp only [← c15804, ← c15805, ← c15806] at e_242_6
  have e_242_7 := arithEq_of_rows h (row := 242) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_7
  simp only [← c15807, ← c15808, ← c15809] at e_242_7
  refine ⟨?_, ?_⟩
  · have hc := e_242_6
    simp only [e_241_15] at hc
    linear_combination c15813.trans k2 - hc
  · have hc := e_241_16
    simp only [e_241_14, e_242_7, e_241_15] at hc
    linear_combination c15814.trans k2 - hc

theorem publicBatchWrapper89_f2247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 47) = bor (a (.wire 38 7)) (a (.virt 18963)) := by
  have c15815 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.1
  have c15816 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.1
  have c15817 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.1
  have c15818 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15819 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15820 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_231_19 := arithEq_of_rows h (row := 231) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_19
  simp only [← c15815, ← c15816, ← c15817] at e_231_19
  have e_236_11 := arithEq_of_rows h (row := 236) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_11
  simp only [← c15818, ← c15819, k3, ← c15820] at e_236_11
  have hr := e_236_11
  simp only [e_231_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 47) = a (.virt 201) := by
  have c15821 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15821

theorem publicBatchWrapper89_f2249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7404)) (a (.wire 170 31)) (a (.virt 18965)) (a (.virt 18966)) := by
  have c15822 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15823 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15824 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15825 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15826 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15827 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15828 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15829 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15830 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15831 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15832 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15833 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15834 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15835 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15836 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15837 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15838 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_241_17 := arithEq_of_rows h (row := 241) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_17
  simp only [← c15822, k3, ← c15823, k3, ← c15824] at e_241_17
  have e_241_18 := arithEq_of_rows h (row := 241) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_18
  simp only [← c15825, ← c15826, k3, ← c15827] at e_241_18
  have e_241_19 := arithEq_of_rows h (row := 241) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_19
  simp only [← c15834, ← c15835, k3, ← c15836] at e_241_19
  have e_242_8 := arithEq_of_rows h (row := 242) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_8
  simp only [← c15828, ← c15829, ← c15830] at e_242_8
  have e_242_9 := arithEq_of_rows h (row := 242) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_9
  simp only [← c15831, ← c15832, ← c15833] at e_242_9
  refine ⟨?_, ?_⟩
  · have hc := e_242_8
    simp only [e_241_18] at hc
    linear_combination c15837.trans k2 - hc
  · have hc := e_241_19
    simp only [e_241_17, e_242_9, e_241_18] at hc
    linear_combination c15838.trans k2 - hc

theorem publicBatchWrapper89_f2250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 51) = bor (a (.wire 38 7)) (a (.virt 18965)) := by
  have c15839 := (publicBatchWrapper89_copies494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15840 := (publicBatchWrapper89_copies495 a h).1
  have c15841 := (publicBatchWrapper89_copies495 a h).2.1
  have c15842 := (publicBatchWrapper89_copies495 a h).2.2.1
  have c15843 := (publicBatchWrapper89_copies495 a h).2.2.2.1
  have c15844 := (publicBatchWrapper89_copies495 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_236_12 := arithEq_of_rows h (row := 236) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_12
  simp only [← c15842, ← c15843, k3, ← c15844] at e_236_12
  have e_243_0 := arithEq_of_rows h (row := 243) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_0
  simp only [← c15839, ← c15840, ← c15841] at e_243_0
  have hr := e_236_12
  simp only [e_243_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 51) = a (.virt 201) := by
  have c15845 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.1
  exact c15845

theorem publicBatchWrapper89_f2252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7405)) (a (.wire 169 63)) (a (.virt 18967)) (a (.virt 18968)) := by
  have c15846 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.1
  have c15847 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.1
  have c15848 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.1
  have c15849 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.1
  have c15850 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15851 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15852 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15853 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15854 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15855 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15856 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15857 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15858 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15859 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15860 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15861 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15862 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_242_10 := arithEq_of_rows h (row := 242) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_10
  simp only [← c15852, ← c15853, ← c15854] at e_242_10
  have e_242_11 := arithEq_of_rows h (row := 242) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_11
  simp only [← c15855, ← c15856, ← c15857] at e_242_11
  have e_244_0 := arithEq_of_rows h (row := 244) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_0
  simp only [← c15846, k3, ← c15847, k3, ← c15848] at e_244_0
  have e_244_1 := arithEq_of_rows h (row := 244) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_1
  simp only [← c15849, ← c15850, k3, ← c15851] at e_244_1
  have e_244_2 := arithEq_of_rows h (row := 244) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_2
  simp only [← c15858, ← c15859, k3, ← c15860] at e_244_2
  refine ⟨?_, ?_⟩
  · have hc := e_242_10
    simp only [e_244_1] at hc
    linear_combination c15861.trans k2 - hc
  · have hc := e_244_2
    simp only [e_244_0, e_242_11, e_244_1] at hc
    linear_combination c15862.trans k2 - hc

theorem publicBatchWrapper89_f2253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7406)) (a (.wire 169 71)) (a (.virt 18969)) (a (.virt 18970)) := by
  have c15863 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15864 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15865 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15866 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15867 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15868 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15869 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15870 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15871 := (publicBatchWrapper89_copies495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15872 := (publicBatchWrapper89_copies496 a h).1
  have c15873 := (publicBatchWrapper89_copies496 a h).2.1
  have c15874 := (publicBatchWrapper89_copies496 a h).2.2.1
  have c15875 := (publicBatchWrapper89_copies496 a h).2.2.2.1
  have c15876 := (publicBatchWrapper89_copies496 a h).2.2.2.2.1
  have c15877 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.1
  have c15878 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.1
  have c15879 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_242_12 := arithEq_of_rows h (row := 242) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_12
  simp only [← c15869, ← c15870, ← c15871] at e_242_12
  have e_242_13 := arithEq_of_rows h (row := 242) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_13
  simp only [← c15872, ← c15873, ← c15874] at e_242_13
  have e_244_3 := arithEq_of_rows h (row := 244) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_3
  simp only [← c15863, k3, ← c15864, k3, ← c15865] at e_244_3
  have e_244_4 := arithEq_of_rows h (row := 244) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_4
  simp only [← c15866, ← c15867, k3, ← c15868] at e_244_4
  have e_244_5 := arithEq_of_rows h (row := 244) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_5
  simp only [← c15875, ← c15876, k3, ← c15877] at e_244_5
  refine ⟨?_, ?_⟩
  · have hc := e_242_12
    simp only [e_244_4] at hc
    linear_combination c15878.trans k2 - hc
  · have hc := e_244_5
    simp only [e_244_3, e_242_13, e_244_4] at hc
    linear_combination c15879.trans k2 - hc

theorem publicBatchWrapper89_f2254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7407)) (a (.wire 169 79)) (a (.virt 18971)) (a (.virt 18972)) := by
  have c15880 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.1
  have c15881 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.1
  have c15882 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15883 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15884 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15885 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15886 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15887 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15888 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15889 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15890 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15891 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15892 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15893 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15894 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15895 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15896 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_242_14 := arithEq_of_rows h (row := 242) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_14
  simp only [← c15886, ← c15887, ← c15888] at e_242_14
  have e_242_15 := arithEq_of_rows h (row := 242) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_15
  simp only [← c15889, ← c15890, ← c15891] at e_242_15
  have e_244_6 := arithEq_of_rows h (row := 244) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_6
  simp only [← c15880, k3, ← c15881, k3, ← c15882] at e_244_6
  have e_244_7 := arithEq_of_rows h (row := 244) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_7
  simp only [← c15883, ← c15884, k3, ← c15885] at e_244_7
  have e_244_8 := arithEq_of_rows h (row := 244) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_8
  simp only [← c15892, ← c15893, k3, ← c15894] at e_244_8
  refine ⟨?_, ?_⟩
  · have hc := e_242_14
    simp only [e_244_7] at hc
    linear_combination c15895.trans k2 - hc
  · have hc := e_244_8
    simp only [e_244_6, e_242_15, e_244_7] at hc
    linear_combination c15896.trans k2 - hc

theorem publicBatchWrapper89_f2255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7408)) (a (.wire 170 7)) (a (.virt 18973)) (a (.virt 18974)) := by
  have c15897 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15898 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15899 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15900 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15901 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15902 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15903 := (publicBatchWrapper89_copies496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15904 := (publicBatchWrapper89_copies497 a h).1
  have c15905 := (publicBatchWrapper89_copies497 a h).2.1
  have c15906 := (publicBatchWrapper89_copies497 a h).2.2.1
  have c15907 := (publicBatchWrapper89_copies497 a h).2.2.2.1
  have c15908 := (publicBatchWrapper89_copies497 a h).2.2.2.2.1
  have c15909 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.1
  have c15910 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.1
  have c15911 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.1
  have c15912 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.1
  have c15913 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_242_16 := arithEq_of_rows h (row := 242) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_16
  simp only [← c15903, ← c15904, ← c15905] at e_242_16
  have e_242_17 := arithEq_of_rows h (row := 242) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_17
  simp only [← c15906, ← c15907, ← c15908] at e_242_17
  have e_244_9 := arithEq_of_rows h (row := 244) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_9
  simp only [← c15897, k3, ← c15898, k3, ← c15899] at e_244_9
  have e_244_10 := arithEq_of_rows h (row := 244) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_10
  simp only [← c15900, ← c15901, k3, ← c15902] at e_244_10
  have e_244_11 := arithEq_of_rows h (row := 244) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_11
  simp only [← c15909, ← c15910, k3, ← c15911] at e_244_11
  refine ⟨?_, ?_⟩
  · have hc := e_242_16
    simp only [e_244_10] at hc
    linear_combination c15912.trans k2 - hc
  · have hc := e_244_11
    simp only [e_244_9, e_242_17, e_244_10] at hc
    linear_combination c15913.trans k2 - hc

theorem publicBatchWrapper89_f2256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 242 75) = band (a (.virt 18967)) (a (.virt 18969)) := by
  have c15914 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15915 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15916 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_242_18 := arithEq_of_rows h (row := 242) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_18
  simp only [← c15914, ← c15915, ← c15916] at e_242_18
  have hr := e_242_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 242 79) = band (a (.virt 18971)) (a (.virt 18973)) := by
  have c15917 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15918 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15919 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_242_19 := arithEq_of_rows h (row := 242) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_19
  simp only [← c15917, ← c15918, ← c15919] at e_242_19
  have hr := e_242_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 245 3) = band (a (.wire 242 75)) (a (.wire 242 79)) := by
  have c15920 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15921 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15922 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_245_0 := arithEq_of_rows h (row := 245) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_0
  simp only [← c15920, ← c15921, ← c15922] at e_245_0
  have hr := e_245_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 55) = bor (a (.wire 38 7)) (a (.wire 245 3)) := by
  have c15923 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15924 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15925 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15926 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15927 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15928 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_236_13 := arithEq_of_rows h (row := 236) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_13
  simp only [← c15926, ← c15927, k3, ← c15928] at e_236_13
  have e_243_1 := arithEq_of_rows h (row := 243) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_1
  simp only [← c15923, ← c15924, ← c15925] at e_243_1
  have hr := e_236_13
  simp only [e_243_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 55) = a (.virt 201) := by
  have c15929 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15929

end Plonky2Spec.Generated
