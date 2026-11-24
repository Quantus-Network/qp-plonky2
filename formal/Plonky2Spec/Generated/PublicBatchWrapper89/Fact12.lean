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

theorem publicBatchWrapper89_f9173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 959 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7677)) := by
  have c59544 := (publicBatchWrapper89_copies1860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59545 := (publicBatchWrapper89_copies1860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59546 := (publicBatchWrapper89_copies1860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59547 := (publicBatchWrapper89_copies1860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59548 := (publicBatchWrapper89_copies1860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59549 := (publicBatchWrapper89_copies1860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_959_4 := arithEq_of_rows h (row := 959) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_4
  simp only [← c59544, ← c59545, ← c59546] at e_959_4
  have e_959_5 := arithEq_of_rows h (row := 959) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_5
  simp only [← c59547, ← c59548, k2, ← c59549] at e_959_5
  have hr := e_959_5
  simp only [e_959_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 959 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7678)) := by
  have c59550 := (publicBatchWrapper89_copies1860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59551 := (publicBatchWrapper89_copies1860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59552 := (publicBatchWrapper89_copies1861 a h).1
  have c59553 := (publicBatchWrapper89_copies1861 a h).2.1
  have c59554 := (publicBatchWrapper89_copies1861 a h).2.2.1
  have c59555 := (publicBatchWrapper89_copies1861 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_959_6 := arithEq_of_rows h (row := 959) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_6
  simp only [← c59550, ← c59551, ← c59552] at e_959_6
  have e_959_7 := arithEq_of_rows h (row := 959) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_7
  simp only [← c59553, ← c59554, k2, ← c59555] at e_959_7
  have hr := e_959_7
  simp only [e_959_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 959 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7679)) := by
  have c59556 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.1
  have c59557 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.1
  have c59558 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.1
  have c59559 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.1
  have c59560 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.1
  have c59561 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_959_8 := arithEq_of_rows h (row := 959) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_8
  simp only [← c59556, ← c59557, ← c59558] at e_959_8
  have e_959_9 := arithEq_of_rows h (row := 959) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_9
  simp only [← c59559, ← c59560, k2, ← c59561] at e_959_9
  have hr := e_959_9
  simp only [e_959_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 959 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7680)) := by
  have c59562 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59563 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59564 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59565 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59566 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59567 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_959_10 := arithEq_of_rows h (row := 959) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_10
  simp only [← c59562, ← c59563, ← c59564] at e_959_10
  have e_959_11 := arithEq_of_rows h (row := 959) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_11
  simp only [← c59565, ← c59566, k2, ← c59567] at e_959_11
  have hr := e_959_11
  simp only [e_959_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 959 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7681)) := by
  have c59568 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59569 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59570 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59571 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59572 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59573 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_959_12 := arithEq_of_rows h (row := 959) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_12
  simp only [← c59568, ← c59569, ← c59570] at e_959_12
  have e_959_13 := arithEq_of_rows h (row := 959) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_13
  simp only [← c59571, ← c59572, k2, ← c59573] at e_959_13
  have hr := e_959_13
  simp only [e_959_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 959 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7682)) := by
  have c59574 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59575 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59576 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59577 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59578 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59579 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_959_14 := arithEq_of_rows h (row := 959) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_14
  simp only [← c59574, ← c59575, ← c59576] at e_959_14
  have e_959_15 := arithEq_of_rows h (row := 959) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_15
  simp only [← c59577, ← c59578, k2, ← c59579] at e_959_15
  have hr := e_959_15
  simp only [e_959_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 959 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7683)) := by
  have c59580 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59581 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59582 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59583 := (publicBatchWrapper89_copies1861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59584 := (publicBatchWrapper89_copies1862 a h).1
  have c59585 := (publicBatchWrapper89_copies1862 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_959_16 := arithEq_of_rows h (row := 959) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_16
  simp only [← c59580, ← c59581, ← c59582] at e_959_16
  have e_959_17 := arithEq_of_rows h (row := 959) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_17
  simp only [← c59583, ← c59584, k2, ← c59585] at e_959_17
  have hr := e_959_17
  simp only [e_959_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 959 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7684)) := by
  have c59586 := (publicBatchWrapper89_copies1862 a h).2.2.1
  have c59587 := (publicBatchWrapper89_copies1862 a h).2.2.2.1
  have c59588 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.1
  have c59589 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.1
  have c59590 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.1
  have c59591 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_959_18 := arithEq_of_rows h (row := 959) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_18
  simp only [← c59586, ← c59587, ← c59588] at e_959_18
  have e_959_19 := arithEq_of_rows h (row := 959) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_19
  simp only [← c59589, ← c59590, k2, ← c59591] at e_959_19
  have hr := e_959_19
  simp only [e_959_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7685)) := by
  have c59592 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.1
  have c59593 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.1
  have c59594 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59595 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59596 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59597 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_0 := arithEq_of_rows h (row := 960) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_0
  simp only [← c59592, ← c59593, ← c59594] at e_960_0
  have e_960_1 := arithEq_of_rows h (row := 960) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_1
  simp only [← c59595, ← c59596, k2, ← c59597] at e_960_1
  have hr := e_960_1
  simp only [e_960_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7686)) := by
  have c59598 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59599 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59600 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59601 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59602 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59603 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_2 := arithEq_of_rows h (row := 960) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_2
  simp only [← c59598, ← c59599, ← c59600] at e_960_2
  have e_960_3 := arithEq_of_rows h (row := 960) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_3
  simp only [← c59601, ← c59602, k2, ← c59603] at e_960_3
  have hr := e_960_3
  simp only [e_960_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7687)) := by
  have c59604 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59605 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59606 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59607 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59608 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59609 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_4 := arithEq_of_rows h (row := 960) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_4
  simp only [← c59604, ← c59605, ← c59606] at e_960_4
  have e_960_5 := arithEq_of_rows h (row := 960) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_5
  simp only [← c59607, ← c59608, k2, ← c59609] at e_960_5
  have hr := e_960_5
  simp only [e_960_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7688)) := by
  have c59610 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59611 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59612 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59613 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59614 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59615 := (publicBatchWrapper89_copies1862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_6 := arithEq_of_rows h (row := 960) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_6
  simp only [← c59610, ← c59611, ← c59612] at e_960_6
  have e_960_7 := arithEq_of_rows h (row := 960) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_7
  simp only [← c59613, ← c59614, k2, ← c59615] at e_960_7
  have hr := e_960_7
  simp only [e_960_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7689)) := by
  have c59616 := (publicBatchWrapper89_copies1863 a h).1
  have c59617 := (publicBatchWrapper89_copies1863 a h).2.1
  have c59618 := (publicBatchWrapper89_copies1863 a h).2.2.1
  have c59619 := (publicBatchWrapper89_copies1863 a h).2.2.2.1
  have c59620 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.1
  have c59621 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_8 := arithEq_of_rows h (row := 960) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_8
  simp only [← c59616, ← c59617, ← c59618] at e_960_8
  have e_960_9 := arithEq_of_rows h (row := 960) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_9
  simp only [← c59619, ← c59620, k2, ← c59621] at e_960_9
  have hr := e_960_9
  simp only [e_960_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7690)) := by
  have c59622 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.1
  have c59623 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.1
  have c59624 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.1
  have c59625 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.1
  have c59626 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59627 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_10 := arithEq_of_rows h (row := 960) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_10
  simp only [← c59622, ← c59623, ← c59624] at e_960_10
  have e_960_11 := arithEq_of_rows h (row := 960) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_11
  simp only [← c59625, ← c59626, k2, ← c59627] at e_960_11
  have hr := e_960_11
  simp only [e_960_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7691)) := by
  have c59628 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59629 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59630 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59631 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59632 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59633 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_12 := arithEq_of_rows h (row := 960) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_12
  simp only [← c59628, ← c59629, ← c59630] at e_960_12
  have e_960_13 := arithEq_of_rows h (row := 960) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_13
  simp only [← c59631, ← c59632, k2, ← c59633] at e_960_13
  have hr := e_960_13
  simp only [e_960_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7692)) := by
  have c59634 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59635 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59636 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59637 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59638 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59639 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_14 := arithEq_of_rows h (row := 960) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_14
  simp only [← c59634, ← c59635, ← c59636] at e_960_14
  have e_960_15 := arithEq_of_rows h (row := 960) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_15
  simp only [← c59637, ← c59638, k2, ← c59639] at e_960_15
  have hr := e_960_15
  simp only [e_960_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7693)) := by
  have c59640 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59641 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59642 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59643 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59644 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59645 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_16 := arithEq_of_rows h (row := 960) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_16
  simp only [← c59640, ← c59641, ← c59642] at e_960_16
  have e_960_17 := arithEq_of_rows h (row := 960) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_17
  simp only [← c59643, ← c59644, k2, ← c59645] at e_960_17
  have hr := e_960_17
  simp only [e_960_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 960 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7694)) := by
  have c59646 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59647 := (publicBatchWrapper89_copies1863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59648 := (publicBatchWrapper89_copies1864 a h).1
  have c59649 := (publicBatchWrapper89_copies1864 a h).2.1
  have c59650 := (publicBatchWrapper89_copies1864 a h).2.2.1
  have c59651 := (publicBatchWrapper89_copies1864 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_960_18 := arithEq_of_rows h (row := 960) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_18
  simp only [← c59646, ← c59647, ← c59648] at e_960_18
  have e_960_19 := arithEq_of_rows h (row := 960) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_960_19
  simp only [← c59649, ← c59650, k2, ← c59651] at e_960_19
  have hr := e_960_19
  simp only [e_960_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7695)) := by
  have c59652 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.1
  have c59653 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.1
  have c59654 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.1
  have c59655 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.1
  have c59656 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.1
  have c59657 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_0 := arithEq_of_rows h (row := 961) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_0
  simp only [← c59652, ← c59653, ← c59654] at e_961_0
  have e_961_1 := arithEq_of_rows h (row := 961) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_1
  simp only [← c59655, ← c59656, k2, ← c59657] at e_961_1
  have hr := e_961_1
  simp only [e_961_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7696)) := by
  have c59658 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59659 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59660 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59661 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59662 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59663 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_2 := arithEq_of_rows h (row := 961) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_2
  simp only [← c59658, ← c59659, ← c59660] at e_961_2
  have e_961_3 := arithEq_of_rows h (row := 961) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_3
  simp only [← c59661, ← c59662, k2, ← c59663] at e_961_3
  have hr := e_961_3
  simp only [e_961_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7697)) := by
  have c59664 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59665 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59666 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59667 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59668 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59669 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_4 := arithEq_of_rows h (row := 961) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_4
  simp only [← c59664, ← c59665, ← c59666] at e_961_4
  have e_961_5 := arithEq_of_rows h (row := 961) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_5
  simp only [← c59667, ← c59668, k2, ← c59669] at e_961_5
  have hr := e_961_5
  simp only [e_961_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7698)) := by
  have c59670 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59671 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59672 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59673 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59674 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59675 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_6 := arithEq_of_rows h (row := 961) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_6
  simp only [← c59670, ← c59671, ← c59672] at e_961_6
  have e_961_7 := arithEq_of_rows h (row := 961) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_7
  simp only [← c59673, ← c59674, k2, ← c59675] at e_961_7
  have hr := e_961_7
  simp only [e_961_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7699)) := by
  have c59676 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59677 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59678 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59679 := (publicBatchWrapper89_copies1864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59680 := (publicBatchWrapper89_copies1865 a h).1
  have c59681 := (publicBatchWrapper89_copies1865 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_8 := arithEq_of_rows h (row := 961) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_8
  simp only [← c59676, ← c59677, ← c59678] at e_961_8
  have e_961_9 := arithEq_of_rows h (row := 961) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_9
  simp only [← c59679, ← c59680, k2, ← c59681] at e_961_9
  have hr := e_961_9
  simp only [e_961_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7700)) := by
  have c59682 := (publicBatchWrapper89_copies1865 a h).2.2.1
  have c59683 := (publicBatchWrapper89_copies1865 a h).2.2.2.1
  have c59684 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.1
  have c59685 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.1
  have c59686 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.1
  have c59687 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_10 := arithEq_of_rows h (row := 961) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_10
  simp only [← c59682, ← c59683, ← c59684] at e_961_10
  have e_961_11 := arithEq_of_rows h (row := 961) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_11
  simp only [← c59685, ← c59686, k2, ← c59687] at e_961_11
  have hr := e_961_11
  simp only [e_961_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7701)) := by
  have c59688 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.1
  have c59689 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.1
  have c59690 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59691 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59692 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59693 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_12 := arithEq_of_rows h (row := 961) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_12
  simp only [← c59688, ← c59689, ← c59690] at e_961_12
  have e_961_13 := arithEq_of_rows h (row := 961) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_13
  simp only [← c59691, ← c59692, k2, ← c59693] at e_961_13
  have hr := e_961_13
  simp only [e_961_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7702)) := by
  have c59694 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59695 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59696 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59697 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59698 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59699 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_14 := arithEq_of_rows h (row := 961) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_14
  simp only [← c59694, ← c59695, ← c59696] at e_961_14
  have e_961_15 := arithEq_of_rows h (row := 961) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_15
  simp only [← c59697, ← c59698, k2, ← c59699] at e_961_15
  have hr := e_961_15
  simp only [e_961_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7703)) := by
  have c59700 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59701 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59702 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59703 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59704 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59705 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_16 := arithEq_of_rows h (row := 961) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_16
  simp only [← c59700, ← c59701, ← c59702] at e_961_16
  have e_961_17 := arithEq_of_rows h (row := 961) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_17
  simp only [← c59703, ← c59704, k2, ← c59705] at e_961_17
  have hr := e_961_17
  simp only [e_961_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 961 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7704)) := by
  have c59706 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59707 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59708 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59709 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59710 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59711 := (publicBatchWrapper89_copies1865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_961_18 := arithEq_of_rows h (row := 961) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_18
  simp only [← c59706, ← c59707, ← c59708] at e_961_18
  have e_961_19 := arithEq_of_rows h (row := 961) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_961_19
  simp only [← c59709, ← c59710, k2, ← c59711] at e_961_19
  have hr := e_961_19
  simp only [e_961_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7705)) := by
  have c59712 := (publicBatchWrapper89_copies1866 a h).1
  have c59713 := (publicBatchWrapper89_copies1866 a h).2.1
  have c59714 := (publicBatchWrapper89_copies1866 a h).2.2.1
  have c59715 := (publicBatchWrapper89_copies1866 a h).2.2.2.1
  have c59716 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.1
  have c59717 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_0 := arithEq_of_rows h (row := 962) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_0
  simp only [← c59712, ← c59713, ← c59714] at e_962_0
  have e_962_1 := arithEq_of_rows h (row := 962) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_1
  simp only [← c59715, ← c59716, k2, ← c59717] at e_962_1
  have hr := e_962_1
  simp only [e_962_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7706)) := by
  have c59718 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.1
  have c59719 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.1
  have c59720 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.1
  have c59721 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.1
  have c59722 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59723 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_2 := arithEq_of_rows h (row := 962) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_2
  simp only [← c59718, ← c59719, ← c59720] at e_962_2
  have e_962_3 := arithEq_of_rows h (row := 962) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_3
  simp only [← c59721, ← c59722, k2, ← c59723] at e_962_3
  have hr := e_962_3
  simp only [e_962_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7707)) := by
  have c59724 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59725 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59726 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59727 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59728 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59729 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_4 := arithEq_of_rows h (row := 962) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_4
  simp only [← c59724, ← c59725, ← c59726] at e_962_4
  have e_962_5 := arithEq_of_rows h (row := 962) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_5
  simp only [← c59727, ← c59728, k2, ← c59729] at e_962_5
  have hr := e_962_5
  simp only [e_962_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7708)) := by
  have c59730 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59731 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59732 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59733 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59734 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59735 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_6 := arithEq_of_rows h (row := 962) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_6
  simp only [← c59730, ← c59731, ← c59732] at e_962_6
  have e_962_7 := arithEq_of_rows h (row := 962) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_7
  simp only [← c59733, ← c59734, k2, ← c59735] at e_962_7
  have hr := e_962_7
  simp only [e_962_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7709)) := by
  have c59736 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59737 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59738 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59739 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59740 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59741 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_8 := arithEq_of_rows h (row := 962) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_8
  simp only [← c59736, ← c59737, ← c59738] at e_962_8
  have e_962_9 := arithEq_of_rows h (row := 962) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_9
  simp only [← c59739, ← c59740, k2, ← c59741] at e_962_9
  have hr := e_962_9
  simp only [e_962_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7710)) := by
  have c59742 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59743 := (publicBatchWrapper89_copies1866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59744 := (publicBatchWrapper89_copies1867 a h).1
  have c59745 := (publicBatchWrapper89_copies1867 a h).2.1
  have c59746 := (publicBatchWrapper89_copies1867 a h).2.2.1
  have c59747 := (publicBatchWrapper89_copies1867 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_10 := arithEq_of_rows h (row := 962) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_10
  simp only [← c59742, ← c59743, ← c59744] at e_962_10
  have e_962_11 := arithEq_of_rows h (row := 962) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_11
  simp only [← c59745, ← c59746, k2, ← c59747] at e_962_11
  have hr := e_962_11
  simp only [e_962_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7711)) := by
  have c59748 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.1
  have c59749 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.1
  have c59750 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.1
  have c59751 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.1
  have c59752 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.1
  have c59753 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_12 := arithEq_of_rows h (row := 962) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_12
  simp only [← c59748, ← c59749, ← c59750] at e_962_12
  have e_962_13 := arithEq_of_rows h (row := 962) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_13
  simp only [← c59751, ← c59752, k2, ← c59753] at e_962_13
  have hr := e_962_13
  simp only [e_962_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7712)) := by
  have c59754 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59755 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59756 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59757 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59758 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59759 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_14 := arithEq_of_rows h (row := 962) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_14
  simp only [← c59754, ← c59755, ← c59756] at e_962_14
  have e_962_15 := arithEq_of_rows h (row := 962) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_15
  simp only [← c59757, ← c59758, k2, ← c59759] at e_962_15
  have hr := e_962_15
  simp only [e_962_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7713)) := by
  have c59760 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59761 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59762 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59763 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59764 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59765 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_16 := arithEq_of_rows h (row := 962) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_16
  simp only [← c59760, ← c59761, ← c59762] at e_962_16
  have e_962_17 := arithEq_of_rows h (row := 962) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_17
  simp only [← c59763, ← c59764, k2, ← c59765] at e_962_17
  have hr := e_962_17
  simp only [e_962_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 962 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7714)) := by
  have c59766 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59767 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59768 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59769 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59770 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59771 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_962_18 := arithEq_of_rows h (row := 962) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_18
  simp only [← c59766, ← c59767, ← c59768] at e_962_18
  have e_962_19 := arithEq_of_rows h (row := 962) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_962_19
  simp only [← c59769, ← c59770, k2, ← c59771] at e_962_19
  have hr := e_962_19
  simp only [e_962_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7715)) := by
  have c59772 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59773 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59774 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59775 := (publicBatchWrapper89_copies1867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59776 := (publicBatchWrapper89_copies1868 a h).1
  have c59777 := (publicBatchWrapper89_copies1868 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_0 := arithEq_of_rows h (row := 963) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_0
  simp only [← c59772, ← c59773, ← c59774] at e_963_0
  have e_963_1 := arithEq_of_rows h (row := 963) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_1
  simp only [← c59775, ← c59776, k2, ← c59777] at e_963_1
  have hr := e_963_1
  simp only [e_963_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7716)) := by
  have c59778 := (publicBatchWrapper89_copies1868 a h).2.2.1
  have c59779 := (publicBatchWrapper89_copies1868 a h).2.2.2.1
  have c59780 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.1
  have c59781 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.1
  have c59782 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.1
  have c59783 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_2 := arithEq_of_rows h (row := 963) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_2
  simp only [← c59778, ← c59779, ← c59780] at e_963_2
  have e_963_3 := arithEq_of_rows h (row := 963) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_3
  simp only [← c59781, ← c59782, k2, ← c59783] at e_963_3
  have hr := e_963_3
  simp only [e_963_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7717)) := by
  have c59784 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.1
  have c59785 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.1
  have c59786 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59787 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59788 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59789 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_4 := arithEq_of_rows h (row := 963) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_4
  simp only [← c59784, ← c59785, ← c59786] at e_963_4
  have e_963_5 := arithEq_of_rows h (row := 963) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_5
  simp only [← c59787, ← c59788, k2, ← c59789] at e_963_5
  have hr := e_963_5
  simp only [e_963_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7718)) := by
  have c59790 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59791 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59792 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59793 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59794 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59795 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_6 := arithEq_of_rows h (row := 963) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_6
  simp only [← c59790, ← c59791, ← c59792] at e_963_6
  have e_963_7 := arithEq_of_rows h (row := 963) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_7
  simp only [← c59793, ← c59794, k2, ← c59795] at e_963_7
  have hr := e_963_7
  simp only [e_963_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7719)) := by
  have c59796 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59797 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59798 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59799 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59800 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59801 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_8 := arithEq_of_rows h (row := 963) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_8
  simp only [← c59796, ← c59797, ← c59798] at e_963_8
  have e_963_9 := arithEq_of_rows h (row := 963) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_9
  simp only [← c59799, ← c59800, k2, ← c59801] at e_963_9
  have hr := e_963_9
  simp only [e_963_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7720)) := by
  have c59802 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59803 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59804 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59805 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59806 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59807 := (publicBatchWrapper89_copies1868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_10 := arithEq_of_rows h (row := 963) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_10
  simp only [← c59802, ← c59803, ← c59804] at e_963_10
  have e_963_11 := arithEq_of_rows h (row := 963) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_11
  simp only [← c59805, ← c59806, k2, ← c59807] at e_963_11
  have hr := e_963_11
  simp only [e_963_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7721)) := by
  have c59808 := (publicBatchWrapper89_copies1869 a h).1
  have c59809 := (publicBatchWrapper89_copies1869 a h).2.1
  have c59810 := (publicBatchWrapper89_copies1869 a h).2.2.1
  have c59811 := (publicBatchWrapper89_copies1869 a h).2.2.2.1
  have c59812 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.1
  have c59813 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_12 := arithEq_of_rows h (row := 963) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_12
  simp only [← c59808, ← c59809, ← c59810] at e_963_12
  have e_963_13 := arithEq_of_rows h (row := 963) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_13
  simp only [← c59811, ← c59812, k2, ← c59813] at e_963_13
  have hr := e_963_13
  simp only [e_963_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7722)) := by
  have c59814 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.1
  have c59815 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.1
  have c59816 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.1
  have c59817 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.1
  have c59818 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59819 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_14 := arithEq_of_rows h (row := 963) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_14
  simp only [← c59814, ← c59815, ← c59816] at e_963_14
  have e_963_15 := arithEq_of_rows h (row := 963) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_15
  simp only [← c59817, ← c59818, k2, ← c59819] at e_963_15
  have hr := e_963_15
  simp only [e_963_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7723)) := by
  have c59820 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59821 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59822 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59823 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59824 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59825 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_16 := arithEq_of_rows h (row := 963) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_16
  simp only [← c59820, ← c59821, ← c59822] at e_963_16
  have e_963_17 := arithEq_of_rows h (row := 963) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_17
  simp only [← c59823, ← c59824, k2, ← c59825] at e_963_17
  have hr := e_963_17
  simp only [e_963_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 963 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7724)) := by
  have c59826 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59827 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59828 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59829 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59830 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59831 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_963_18 := arithEq_of_rows h (row := 963) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_18
  simp only [← c59826, ← c59827, ← c59828] at e_963_18
  have e_963_19 := arithEq_of_rows h (row := 963) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_963_19
  simp only [← c59829, ← c59830, k2, ← c59831] at e_963_19
  have hr := e_963_19
  simp only [e_963_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7725)) := by
  have c59832 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59833 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59834 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59835 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59836 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59837 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_0 := arithEq_of_rows h (row := 964) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_0
  simp only [← c59832, ← c59833, ← c59834] at e_964_0
  have e_964_1 := arithEq_of_rows h (row := 964) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_1
  simp only [← c59835, ← c59836, k2, ← c59837] at e_964_1
  have hr := e_964_1
  simp only [e_964_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7726)) := by
  have c59838 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59839 := (publicBatchWrapper89_copies1869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59840 := (publicBatchWrapper89_copies1870 a h).1
  have c59841 := (publicBatchWrapper89_copies1870 a h).2.1
  have c59842 := (publicBatchWrapper89_copies1870 a h).2.2.1
  have c59843 := (publicBatchWrapper89_copies1870 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_2 := arithEq_of_rows h (row := 964) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_2
  simp only [← c59838, ← c59839, ← c59840] at e_964_2
  have e_964_3 := arithEq_of_rows h (row := 964) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_3
  simp only [← c59841, ← c59842, k2, ← c59843] at e_964_3
  have hr := e_964_3
  simp only [e_964_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7727)) := by
  have c59844 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.1
  have c59845 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.1
  have c59846 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.1
  have c59847 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.1
  have c59848 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.1
  have c59849 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_4 := arithEq_of_rows h (row := 964) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_4
  simp only [← c59844, ← c59845, ← c59846] at e_964_4
  have e_964_5 := arithEq_of_rows h (row := 964) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_5
  simp only [← c59847, ← c59848, k2, ← c59849] at e_964_5
  have hr := e_964_5
  simp only [e_964_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7728)) := by
  have c59850 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59851 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59852 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59853 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59854 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59855 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_6 := arithEq_of_rows h (row := 964) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_6
  simp only [← c59850, ← c59851, ← c59852] at e_964_6
  have e_964_7 := arithEq_of_rows h (row := 964) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_7
  simp only [← c59853, ← c59854, k2, ← c59855] at e_964_7
  have hr := e_964_7
  simp only [e_964_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7729)) := by
  have c59856 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59857 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59858 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59859 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59860 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59861 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_8 := arithEq_of_rows h (row := 964) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_8
  simp only [← c59856, ← c59857, ← c59858] at e_964_8
  have e_964_9 := arithEq_of_rows h (row := 964) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_9
  simp only [← c59859, ← c59860, k2, ← c59861] at e_964_9
  have hr := e_964_9
  simp only [e_964_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7730)) := by
  have c59862 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59863 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59864 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59865 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59866 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59867 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_10 := arithEq_of_rows h (row := 964) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_10
  simp only [← c59862, ← c59863, ← c59864] at e_964_10
  have e_964_11 := arithEq_of_rows h (row := 964) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_11
  simp only [← c59865, ← c59866, k2, ← c59867] at e_964_11
  have hr := e_964_11
  simp only [e_964_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7731)) := by
  have c59868 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59869 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59870 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59871 := (publicBatchWrapper89_copies1870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59872 := (publicBatchWrapper89_copies1871 a h).1
  have c59873 := (publicBatchWrapper89_copies1871 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_12 := arithEq_of_rows h (row := 964) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_12
  simp only [← c59868, ← c59869, ← c59870] at e_964_12
  have e_964_13 := arithEq_of_rows h (row := 964) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_13
  simp only [← c59871, ← c59872, k2, ← c59873] at e_964_13
  have hr := e_964_13
  simp only [e_964_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7732)) := by
  have c59874 := (publicBatchWrapper89_copies1871 a h).2.2.1
  have c59875 := (publicBatchWrapper89_copies1871 a h).2.2.2.1
  have c59876 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.1
  have c59877 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.1
  have c59878 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.1
  have c59879 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_14 := arithEq_of_rows h (row := 964) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_14
  simp only [← c59874, ← c59875, ← c59876] at e_964_14
  have e_964_15 := arithEq_of_rows h (row := 964) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_15
  simp only [← c59877, ← c59878, k2, ← c59879] at e_964_15
  have hr := e_964_15
  simp only [e_964_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7733)) := by
  have c59880 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.1
  have c59881 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.1
  have c59882 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59883 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59884 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59885 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_16 := arithEq_of_rows h (row := 964) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_16
  simp only [← c59880, ← c59881, ← c59882] at e_964_16
  have e_964_17 := arithEq_of_rows h (row := 964) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_17
  simp only [← c59883, ← c59884, k2, ← c59885] at e_964_17
  have hr := e_964_17
  simp only [e_964_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 964 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7734)) := by
  have c59886 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59887 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59888 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59889 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59890 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59891 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_964_18 := arithEq_of_rows h (row := 964) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_18
  simp only [← c59886, ← c59887, ← c59888] at e_964_18
  have e_964_19 := arithEq_of_rows h (row := 964) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_964_19
  simp only [← c59889, ← c59890, k2, ← c59891] at e_964_19
  have hr := e_964_19
  simp only [e_964_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7735)) := by
  have c59892 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59893 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59894 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59895 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59896 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59897 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_0 := arithEq_of_rows h (row := 965) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_0
  simp only [← c59892, ← c59893, ← c59894] at e_965_0
  have e_965_1 := arithEq_of_rows h (row := 965) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_1
  simp only [← c59895, ← c59896, k2, ← c59897] at e_965_1
  have hr := e_965_1
  simp only [e_965_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7736)) := by
  have c59898 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59899 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59900 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59901 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59902 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59903 := (publicBatchWrapper89_copies1871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_2 := arithEq_of_rows h (row := 965) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_2
  simp only [← c59898, ← c59899, ← c59900] at e_965_2
  have e_965_3 := arithEq_of_rows h (row := 965) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_3
  simp only [← c59901, ← c59902, k2, ← c59903] at e_965_3
  have hr := e_965_3
  simp only [e_965_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7737)) := by
  have c59904 := (publicBatchWrapper89_copies1872 a h).1
  have c59905 := (publicBatchWrapper89_copies1872 a h).2.1
  have c59906 := (publicBatchWrapper89_copies1872 a h).2.2.1
  have c59907 := (publicBatchWrapper89_copies1872 a h).2.2.2.1
  have c59908 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.1
  have c59909 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_4 := arithEq_of_rows h (row := 965) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_4
  simp only [← c59904, ← c59905, ← c59906] at e_965_4
  have e_965_5 := arithEq_of_rows h (row := 965) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_5
  simp only [← c59907, ← c59908, k2, ← c59909] at e_965_5
  have hr := e_965_5
  simp only [e_965_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7738)) := by
  have c59910 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.1
  have c59911 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.1
  have c59912 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.1
  have c59913 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.1
  have c59914 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59915 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_6 := arithEq_of_rows h (row := 965) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_6
  simp only [← c59910, ← c59911, ← c59912] at e_965_6
  have e_965_7 := arithEq_of_rows h (row := 965) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_7
  simp only [← c59913, ← c59914, k2, ← c59915] at e_965_7
  have hr := e_965_7
  simp only [e_965_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7739)) := by
  have c59916 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59917 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59918 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59919 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59920 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59921 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_8 := arithEq_of_rows h (row := 965) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_8
  simp only [← c59916, ← c59917, ← c59918] at e_965_8
  have e_965_9 := arithEq_of_rows h (row := 965) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_9
  simp only [← c59919, ← c59920, k2, ← c59921] at e_965_9
  have hr := e_965_9
  simp only [e_965_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7740)) := by
  have c59922 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59923 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59924 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59925 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59926 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59927 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_10 := arithEq_of_rows h (row := 965) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_10
  simp only [← c59922, ← c59923, ← c59924] at e_965_10
  have e_965_11 := arithEq_of_rows h (row := 965) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_11
  simp only [← c59925, ← c59926, k2, ← c59927] at e_965_11
  have hr := e_965_11
  simp only [e_965_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7741)) := by
  have c59928 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59929 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59930 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59931 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59932 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59933 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_12 := arithEq_of_rows h (row := 965) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_12
  simp only [← c59928, ← c59929, ← c59930] at e_965_12
  have e_965_13 := arithEq_of_rows h (row := 965) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_13
  simp only [← c59931, ← c59932, k2, ← c59933] at e_965_13
  have hr := e_965_13
  simp only [e_965_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7742)) := by
  have c59934 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59935 := (publicBatchWrapper89_copies1872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59936 := (publicBatchWrapper89_copies1873 a h).1
  have c59937 := (publicBatchWrapper89_copies1873 a h).2.1
  have c59938 := (publicBatchWrapper89_copies1873 a h).2.2.1
  have c59939 := (publicBatchWrapper89_copies1873 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_14 := arithEq_of_rows h (row := 965) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_14
  simp only [← c59934, ← c59935, ← c59936] at e_965_14
  have e_965_15 := arithEq_of_rows h (row := 965) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_15
  simp only [← c59937, ← c59938, k2, ← c59939] at e_965_15
  have hr := e_965_15
  simp only [e_965_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7743)) := by
  have c59940 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.1
  have c59941 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.1
  have c59942 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.1
  have c59943 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.1
  have c59944 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.1
  have c59945 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_16 := arithEq_of_rows h (row := 965) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_16
  simp only [← c59940, ← c59941, ← c59942] at e_965_16
  have e_965_17 := arithEq_of_rows h (row := 965) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_17
  simp only [← c59943, ← c59944, k2, ← c59945] at e_965_17
  have hr := e_965_17
  simp only [e_965_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 965 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7744)) := by
  have c59946 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59947 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59948 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59949 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59950 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59951 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_965_18 := arithEq_of_rows h (row := 965) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_18
  simp only [← c59946, ← c59947, ← c59948] at e_965_18
  have e_965_19 := arithEq_of_rows h (row := 965) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_965_19
  simp only [← c59949, ← c59950, k2, ← c59951] at e_965_19
  have hr := e_965_19
  simp only [e_965_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7745)) := by
  have c59952 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59953 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59954 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59955 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59956 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59957 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_0 := arithEq_of_rows h (row := 966) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_0
  simp only [← c59952, ← c59953, ← c59954] at e_966_0
  have e_966_1 := arithEq_of_rows h (row := 966) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_1
  simp only [← c59955, ← c59956, k2, ← c59957] at e_966_1
  have hr := e_966_1
  simp only [e_966_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7746)) := by
  have c59958 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59959 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59960 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59961 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59962 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59963 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_2 := arithEq_of_rows h (row := 966) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_2
  simp only [← c59958, ← c59959, ← c59960] at e_966_2
  have e_966_3 := arithEq_of_rows h (row := 966) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_3
  simp only [← c59961, ← c59962, k2, ← c59963] at e_966_3
  have hr := e_966_3
  simp only [e_966_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7747)) := by
  have c59964 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59965 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59966 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59967 := (publicBatchWrapper89_copies1873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59968 := (publicBatchWrapper89_copies1874 a h).1
  have c59969 := (publicBatchWrapper89_copies1874 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_4 := arithEq_of_rows h (row := 966) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_4
  simp only [← c59964, ← c59965, ← c59966] at e_966_4
  have e_966_5 := arithEq_of_rows h (row := 966) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_5
  simp only [← c59967, ← c59968, k2, ← c59969] at e_966_5
  have hr := e_966_5
  simp only [e_966_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7748)) := by
  have c59970 := (publicBatchWrapper89_copies1874 a h).2.2.1
  have c59971 := (publicBatchWrapper89_copies1874 a h).2.2.2.1
  have c59972 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.1
  have c59973 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.1
  have c59974 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.1
  have c59975 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_6 := arithEq_of_rows h (row := 966) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_6
  simp only [← c59970, ← c59971, ← c59972] at e_966_6
  have e_966_7 := arithEq_of_rows h (row := 966) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_7
  simp only [← c59973, ← c59974, k2, ← c59975] at e_966_7
  have hr := e_966_7
  simp only [e_966_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7749)) := by
  have c59976 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.1
  have c59977 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.1
  have c59978 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59979 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59980 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59981 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_8 := arithEq_of_rows h (row := 966) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_8
  simp only [← c59976, ← c59977, ← c59978] at e_966_8
  have e_966_9 := arithEq_of_rows h (row := 966) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_9
  simp only [← c59979, ← c59980, k2, ← c59981] at e_966_9
  have hr := e_966_9
  simp only [e_966_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7750)) := by
  have c59982 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59983 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59984 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59985 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59986 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59987 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_10 := arithEq_of_rows h (row := 966) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_10
  simp only [← c59982, ← c59983, ← c59984] at e_966_10
  have e_966_11 := arithEq_of_rows h (row := 966) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_11
  simp only [← c59985, ← c59986, k2, ← c59987] at e_966_11
  have hr := e_966_11
  simp only [e_966_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7751)) := by
  have c59988 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59989 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59990 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59991 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59992 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59993 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_12 := arithEq_of_rows h (row := 966) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_12
  simp only [← c59988, ← c59989, ← c59990] at e_966_12
  have e_966_13 := arithEq_of_rows h (row := 966) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_13
  simp only [← c59991, ← c59992, k2, ← c59993] at e_966_13
  have hr := e_966_13
  simp only [e_966_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7752)) := by
  have c59994 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59995 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59996 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59997 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59998 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59999 := (publicBatchWrapper89_copies1874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_14 := arithEq_of_rows h (row := 966) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_14
  simp only [← c59994, ← c59995, ← c59996] at e_966_14
  have e_966_15 := arithEq_of_rows h (row := 966) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_15
  simp only [← c59997, ← c59998, k2, ← c59999] at e_966_15
  have hr := e_966_15
  simp only [e_966_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7753)) := by
  have c60000 := (publicBatchWrapper89_copies1875 a h).1
  have c60001 := (publicBatchWrapper89_copies1875 a h).2.1
  have c60002 := (publicBatchWrapper89_copies1875 a h).2.2.1
  have c60003 := (publicBatchWrapper89_copies1875 a h).2.2.2.1
  have c60004 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.1
  have c60005 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_16 := arithEq_of_rows h (row := 966) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_16
  simp only [← c60000, ← c60001, ← c60002] at e_966_16
  have e_966_17 := arithEq_of_rows h (row := 966) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_17
  simp only [← c60003, ← c60004, k2, ← c60005] at e_966_17
  have hr := e_966_17
  simp only [e_966_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 966 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7754)) := by
  have c60006 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.1
  have c60007 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.1
  have c60008 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.1
  have c60009 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.1
  have c60010 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60011 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_966_18 := arithEq_of_rows h (row := 966) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_18
  simp only [← c60006, ← c60007, ← c60008] at e_966_18
  have e_966_19 := arithEq_of_rows h (row := 966) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_966_19
  simp only [← c60009, ← c60010, k2, ← c60011] at e_966_19
  have hr := e_966_19
  simp only [e_966_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7755)) := by
  have c60012 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60013 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60014 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60015 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60016 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60017 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_0 := arithEq_of_rows h (row := 967) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_0
  simp only [← c60012, ← c60013, ← c60014] at e_967_0
  have e_967_1 := arithEq_of_rows h (row := 967) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_1
  simp only [← c60015, ← c60016, k2, ← c60017] at e_967_1
  have hr := e_967_1
  simp only [e_967_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7756)) := by
  have c60018 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60019 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60020 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60021 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60022 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60023 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_2 := arithEq_of_rows h (row := 967) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_2
  simp only [← c60018, ← c60019, ← c60020] at e_967_2
  have e_967_3 := arithEq_of_rows h (row := 967) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_3
  simp only [← c60021, ← c60022, k2, ← c60023] at e_967_3
  have hr := e_967_3
  simp only [e_967_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7757)) := by
  have c60024 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60025 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60026 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60027 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60028 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60029 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_4 := arithEq_of_rows h (row := 967) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_4
  simp only [← c60024, ← c60025, ← c60026] at e_967_4
  have e_967_5 := arithEq_of_rows h (row := 967) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_5
  simp only [← c60027, ← c60028, k2, ← c60029] at e_967_5
  have hr := e_967_5
  simp only [e_967_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7758)) := by
  have c60030 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60031 := (publicBatchWrapper89_copies1875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60032 := (publicBatchWrapper89_copies1876 a h).1
  have c60033 := (publicBatchWrapper89_copies1876 a h).2.1
  have c60034 := (publicBatchWrapper89_copies1876 a h).2.2.1
  have c60035 := (publicBatchWrapper89_copies1876 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_6 := arithEq_of_rows h (row := 967) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_6
  simp only [← c60030, ← c60031, ← c60032] at e_967_6
  have e_967_7 := arithEq_of_rows h (row := 967) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_7
  simp only [← c60033, ← c60034, k2, ← c60035] at e_967_7
  have hr := e_967_7
  simp only [e_967_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7759)) := by
  have c60036 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.1
  have c60037 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.1
  have c60038 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.1
  have c60039 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.1
  have c60040 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.1
  have c60041 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_8 := arithEq_of_rows h (row := 967) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_8
  simp only [← c60036, ← c60037, ← c60038] at e_967_8
  have e_967_9 := arithEq_of_rows h (row := 967) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_9
  simp only [← c60039, ← c60040, k2, ← c60041] at e_967_9
  have hr := e_967_9
  simp only [e_967_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7760)) := by
  have c60042 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60043 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60044 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60045 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60046 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60047 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_10 := arithEq_of_rows h (row := 967) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_10
  simp only [← c60042, ← c60043, ← c60044] at e_967_10
  have e_967_11 := arithEq_of_rows h (row := 967) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_11
  simp only [← c60045, ← c60046, k2, ← c60047] at e_967_11
  have hr := e_967_11
  simp only [e_967_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7761)) := by
  have c60048 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60049 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60050 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60051 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60052 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60053 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_12 := arithEq_of_rows h (row := 967) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_12
  simp only [← c60048, ← c60049, ← c60050] at e_967_12
  have e_967_13 := arithEq_of_rows h (row := 967) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_13
  simp only [← c60051, ← c60052, k2, ← c60053] at e_967_13
  have hr := e_967_13
  simp only [e_967_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7762)) := by
  have c60054 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60055 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60056 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60057 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60058 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60059 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_14 := arithEq_of_rows h (row := 967) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_14
  simp only [← c60054, ← c60055, ← c60056] at e_967_14
  have e_967_15 := arithEq_of_rows h (row := 967) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_15
  simp only [← c60057, ← c60058, k2, ← c60059] at e_967_15
  have hr := e_967_15
  simp only [e_967_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7763)) := by
  have c60060 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60061 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60062 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60063 := (publicBatchWrapper89_copies1876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60064 := (publicBatchWrapper89_copies1877 a h).1
  have c60065 := (publicBatchWrapper89_copies1877 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_16 := arithEq_of_rows h (row := 967) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_16
  simp only [← c60060, ← c60061, ← c60062] at e_967_16
  have e_967_17 := arithEq_of_rows h (row := 967) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_17
  simp only [← c60063, ← c60064, k2, ← c60065] at e_967_17
  have hr := e_967_17
  simp only [e_967_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 967 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7764)) := by
  have c60066 := (publicBatchWrapper89_copies1877 a h).2.2.1
  have c60067 := (publicBatchWrapper89_copies1877 a h).2.2.2.1
  have c60068 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.1
  have c60069 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.1
  have c60070 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.1
  have c60071 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_967_18 := arithEq_of_rows h (row := 967) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_18
  simp only [← c60066, ← c60067, ← c60068] at e_967_18
  have e_967_19 := arithEq_of_rows h (row := 967) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_967_19
  simp only [← c60069, ← c60070, k2, ← c60071] at e_967_19
  have hr := e_967_19
  simp only [e_967_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7765)) := by
  have c60072 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.1
  have c60073 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.1
  have c60074 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60075 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60076 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60077 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_0 := arithEq_of_rows h (row := 968) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_0
  simp only [← c60072, ← c60073, ← c60074] at e_968_0
  have e_968_1 := arithEq_of_rows h (row := 968) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_1
  simp only [← c60075, ← c60076, k2, ← c60077] at e_968_1
  have hr := e_968_1
  simp only [e_968_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7766)) := by
  have c60078 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60079 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60080 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60081 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60082 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60083 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_2 := arithEq_of_rows h (row := 968) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_2
  simp only [← c60078, ← c60079, ← c60080] at e_968_2
  have e_968_3 := arithEq_of_rows h (row := 968) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_3
  simp only [← c60081, ← c60082, k2, ← c60083] at e_968_3
  have hr := e_968_3
  simp only [e_968_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7767)) := by
  have c60084 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60085 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60086 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60087 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60088 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60089 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_4 := arithEq_of_rows h (row := 968) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_4
  simp only [← c60084, ← c60085, ← c60086] at e_968_4
  have e_968_5 := arithEq_of_rows h (row := 968) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_5
  simp only [← c60087, ← c60088, k2, ← c60089] at e_968_5
  have hr := e_968_5
  simp only [e_968_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7768)) := by
  have c60090 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60091 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60092 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60093 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60094 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60095 := (publicBatchWrapper89_copies1877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_6 := arithEq_of_rows h (row := 968) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_6
  simp only [← c60090, ← c60091, ← c60092] at e_968_6
  have e_968_7 := arithEq_of_rows h (row := 968) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_7
  simp only [← c60093, ← c60094, k2, ← c60095] at e_968_7
  have hr := e_968_7
  simp only [e_968_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7769)) := by
  have c60096 := (publicBatchWrapper89_copies1878 a h).1
  have c60097 := (publicBatchWrapper89_copies1878 a h).2.1
  have c60098 := (publicBatchWrapper89_copies1878 a h).2.2.1
  have c60099 := (publicBatchWrapper89_copies1878 a h).2.2.2.1
  have c60100 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.1
  have c60101 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_8 := arithEq_of_rows h (row := 968) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_8
  simp only [← c60096, ← c60097, ← c60098] at e_968_8
  have e_968_9 := arithEq_of_rows h (row := 968) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_9
  simp only [← c60099, ← c60100, k2, ← c60101] at e_968_9
  have hr := e_968_9
  simp only [e_968_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7810)) := by
  have c60102 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.1
  have c60103 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.1
  have c60104 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.1
  have c60105 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.1
  have c60106 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60107 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_10 := arithEq_of_rows h (row := 968) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_10
  simp only [← c60102, ← c60103, ← c60104] at e_968_10
  have e_968_11 := arithEq_of_rows h (row := 968) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_11
  simp only [← c60105, ← c60106, k2, ← c60107] at e_968_11
  have hr := e_968_11
  simp only [e_968_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7811)) := by
  have c60108 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60109 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60110 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60111 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60112 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60113 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_12 := arithEq_of_rows h (row := 968) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_12
  simp only [← c60108, ← c60109, ← c60110] at e_968_12
  have e_968_13 := arithEq_of_rows h (row := 968) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_13
  simp only [← c60111, ← c60112, k2, ← c60113] at e_968_13
  have hr := e_968_13
  simp only [e_968_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7812)) := by
  have c60114 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60115 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60116 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60117 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60118 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60119 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_14 := arithEq_of_rows h (row := 968) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_14
  simp only [← c60114, ← c60115, ← c60116] at e_968_14
  have e_968_15 := arithEq_of_rows h (row := 968) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_15
  simp only [← c60117, ← c60118, k2, ← c60119] at e_968_15
  have hr := e_968_15
  simp only [e_968_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7813)) := by
  have c60120 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60121 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60122 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60123 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60124 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60125 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_16 := arithEq_of_rows h (row := 968) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_16
  simp only [← c60120, ← c60121, ← c60122] at e_968_16
  have e_968_17 := arithEq_of_rows h (row := 968) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_17
  simp only [← c60123, ← c60124, k2, ← c60125] at e_968_17
  have hr := e_968_17
  simp only [e_968_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 968 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7814)) := by
  have c60126 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60127 := (publicBatchWrapper89_copies1878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60128 := (publicBatchWrapper89_copies1879 a h).1
  have c60129 := (publicBatchWrapper89_copies1879 a h).2.1
  have c60130 := (publicBatchWrapper89_copies1879 a h).2.2.1
  have c60131 := (publicBatchWrapper89_copies1879 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_968_18 := arithEq_of_rows h (row := 968) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_18
  simp only [← c60126, ← c60127, ← c60128] at e_968_18
  have e_968_19 := arithEq_of_rows h (row := 968) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_968_19
  simp only [← c60129, ← c60130, k2, ← c60131] at e_968_19
  have hr := e_968_19
  simp only [e_968_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7815)) := by
  have c60132 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.1
  have c60133 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.1
  have c60134 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.1
  have c60135 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.1
  have c60136 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.1
  have c60137 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_0 := arithEq_of_rows h (row := 969) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_0
  simp only [← c60132, ← c60133, ← c60134] at e_969_0
  have e_969_1 := arithEq_of_rows h (row := 969) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_1
  simp only [← c60135, ← c60136, k2, ← c60137] at e_969_1
  have hr := e_969_1
  simp only [e_969_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7816)) := by
  have c60138 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60139 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60140 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60141 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60142 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60143 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_2 := arithEq_of_rows h (row := 969) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_2
  simp only [← c60138, ← c60139, ← c60140] at e_969_2
  have e_969_3 := arithEq_of_rows h (row := 969) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_3
  simp only [← c60141, ← c60142, k2, ← c60143] at e_969_3
  have hr := e_969_3
  simp only [e_969_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7817)) := by
  have c60144 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60145 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60146 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60147 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60148 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60149 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_4 := arithEq_of_rows h (row := 969) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_4
  simp only [← c60144, ← c60145, ← c60146] at e_969_4
  have e_969_5 := arithEq_of_rows h (row := 969) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_5
  simp only [← c60147, ← c60148, k2, ← c60149] at e_969_5
  have hr := e_969_5
  simp only [e_969_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7818)) := by
  have c60150 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60151 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60152 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60153 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60154 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60155 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_6 := arithEq_of_rows h (row := 969) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_6
  simp only [← c60150, ← c60151, ← c60152] at e_969_6
  have e_969_7 := arithEq_of_rows h (row := 969) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_7
  simp only [← c60153, ← c60154, k2, ← c60155] at e_969_7
  have hr := e_969_7
  simp only [e_969_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7819)) := by
  have c60156 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60157 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60158 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60159 := (publicBatchWrapper89_copies1879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60160 := (publicBatchWrapper89_copies1880 a h).1
  have c60161 := (publicBatchWrapper89_copies1880 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_8 := arithEq_of_rows h (row := 969) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_8
  simp only [← c60156, ← c60157, ← c60158] at e_969_8
  have e_969_9 := arithEq_of_rows h (row := 969) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_9
  simp only [← c60159, ← c60160, k2, ← c60161] at e_969_9
  have hr := e_969_9
  simp only [e_969_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7820)) := by
  have c60162 := (publicBatchWrapper89_copies1880 a h).2.2.1
  have c60163 := (publicBatchWrapper89_copies1880 a h).2.2.2.1
  have c60164 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.1
  have c60165 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.1
  have c60166 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.1
  have c60167 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_10 := arithEq_of_rows h (row := 969) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_10
  simp only [← c60162, ← c60163, ← c60164] at e_969_10
  have e_969_11 := arithEq_of_rows h (row := 969) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_11
  simp only [← c60165, ← c60166, k2, ← c60167] at e_969_11
  have hr := e_969_11
  simp only [e_969_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7821)) := by
  have c60168 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.1
  have c60169 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.1
  have c60170 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60171 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60172 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60173 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_12 := arithEq_of_rows h (row := 969) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_12
  simp only [← c60168, ← c60169, ← c60170] at e_969_12
  have e_969_13 := arithEq_of_rows h (row := 969) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_13
  simp only [← c60171, ← c60172, k2, ← c60173] at e_969_13
  have hr := e_969_13
  simp only [e_969_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7822)) := by
  have c60174 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60175 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60176 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60177 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60178 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60179 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_14 := arithEq_of_rows h (row := 969) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_14
  simp only [← c60174, ← c60175, ← c60176] at e_969_14
  have e_969_15 := arithEq_of_rows h (row := 969) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_15
  simp only [← c60177, ← c60178, k2, ← c60179] at e_969_15
  have hr := e_969_15
  simp only [e_969_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7823)) := by
  have c60180 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60181 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60182 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60183 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60184 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60185 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_16 := arithEq_of_rows h (row := 969) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_16
  simp only [← c60180, ← c60181, ← c60182] at e_969_16
  have e_969_17 := arithEq_of_rows h (row := 969) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_17
  simp only [← c60183, ← c60184, k2, ← c60185] at e_969_17
  have hr := e_969_17
  simp only [e_969_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 969 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7824)) := by
  have c60186 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60187 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60188 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60189 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60190 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60191 := (publicBatchWrapper89_copies1880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_969_18 := arithEq_of_rows h (row := 969) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_18
  simp only [← c60186, ← c60187, ← c60188] at e_969_18
  have e_969_19 := arithEq_of_rows h (row := 969) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_969_19
  simp only [← c60189, ← c60190, k2, ← c60191] at e_969_19
  have hr := e_969_19
  simp only [e_969_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7825)) := by
  have c60192 := (publicBatchWrapper89_copies1881 a h).1
  have c60193 := (publicBatchWrapper89_copies1881 a h).2.1
  have c60194 := (publicBatchWrapper89_copies1881 a h).2.2.1
  have c60195 := (publicBatchWrapper89_copies1881 a h).2.2.2.1
  have c60196 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.1
  have c60197 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_0 := arithEq_of_rows h (row := 970) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_0
  simp only [← c60192, ← c60193, ← c60194] at e_970_0
  have e_970_1 := arithEq_of_rows h (row := 970) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_1
  simp only [← c60195, ← c60196, k2, ← c60197] at e_970_1
  have hr := e_970_1
  simp only [e_970_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7826)) := by
  have c60198 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.1
  have c60199 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.1
  have c60200 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.1
  have c60201 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.1
  have c60202 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60203 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_2 := arithEq_of_rows h (row := 970) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_2
  simp only [← c60198, ← c60199, ← c60200] at e_970_2
  have e_970_3 := arithEq_of_rows h (row := 970) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_3
  simp only [← c60201, ← c60202, k2, ← c60203] at e_970_3
  have hr := e_970_3
  simp only [e_970_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7827)) := by
  have c60204 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60205 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60206 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60207 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60208 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60209 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_4 := arithEq_of_rows h (row := 970) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_4
  simp only [← c60204, ← c60205, ← c60206] at e_970_4
  have e_970_5 := arithEq_of_rows h (row := 970) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_5
  simp only [← c60207, ← c60208, k2, ← c60209] at e_970_5
  have hr := e_970_5
  simp only [e_970_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7828)) := by
  have c60210 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60211 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60212 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60213 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60214 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60215 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_6 := arithEq_of_rows h (row := 970) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_6
  simp only [← c60210, ← c60211, ← c60212] at e_970_6
  have e_970_7 := arithEq_of_rows h (row := 970) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_7
  simp only [← c60213, ← c60214, k2, ← c60215] at e_970_7
  have hr := e_970_7
  simp only [e_970_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7829)) := by
  have c60216 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60217 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60218 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60219 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60220 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60221 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_8 := arithEq_of_rows h (row := 970) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_8
  simp only [← c60216, ← c60217, ← c60218] at e_970_8
  have e_970_9 := arithEq_of_rows h (row := 970) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_9
  simp only [← c60219, ← c60220, k2, ← c60221] at e_970_9
  have hr := e_970_9
  simp only [e_970_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7830)) := by
  have c60222 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60223 := (publicBatchWrapper89_copies1881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60224 := (publicBatchWrapper89_copies1882 a h).1
  have c60225 := (publicBatchWrapper89_copies1882 a h).2.1
  have c60226 := (publicBatchWrapper89_copies1882 a h).2.2.1
  have c60227 := (publicBatchWrapper89_copies1882 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_10 := arithEq_of_rows h (row := 970) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_10
  simp only [← c60222, ← c60223, ← c60224] at e_970_10
  have e_970_11 := arithEq_of_rows h (row := 970) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_11
  simp only [← c60225, ← c60226, k2, ← c60227] at e_970_11
  have hr := e_970_11
  simp only [e_970_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7831)) := by
  have c60228 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.1
  have c60229 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.1
  have c60230 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.1
  have c60231 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.1
  have c60232 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.1
  have c60233 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_12 := arithEq_of_rows h (row := 970) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_12
  simp only [← c60228, ← c60229, ← c60230] at e_970_12
  have e_970_13 := arithEq_of_rows h (row := 970) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_13
  simp only [← c60231, ← c60232, k2, ← c60233] at e_970_13
  have hr := e_970_13
  simp only [e_970_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7832)) := by
  have c60234 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60235 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60236 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60237 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60238 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60239 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_14 := arithEq_of_rows h (row := 970) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_14
  simp only [← c60234, ← c60235, ← c60236] at e_970_14
  have e_970_15 := arithEq_of_rows h (row := 970) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_15
  simp only [← c60237, ← c60238, k2, ← c60239] at e_970_15
  have hr := e_970_15
  simp only [e_970_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7833)) := by
  have c60240 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60241 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60242 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60243 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60244 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60245 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_16 := arithEq_of_rows h (row := 970) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_16
  simp only [← c60240, ← c60241, ← c60242] at e_970_16
  have e_970_17 := arithEq_of_rows h (row := 970) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_17
  simp only [← c60243, ← c60244, k2, ← c60245] at e_970_17
  have hr := e_970_17
  simp only [e_970_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 970 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7834)) := by
  have c60246 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60247 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60248 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60249 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60250 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60251 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_970_18 := arithEq_of_rows h (row := 970) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_18
  simp only [← c60246, ← c60247, ← c60248] at e_970_18
  have e_970_19 := arithEq_of_rows h (row := 970) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_970_19
  simp only [← c60249, ← c60250, k2, ← c60251] at e_970_19
  have hr := e_970_19
  simp only [e_970_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7835)) := by
  have c60252 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60253 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60254 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60255 := (publicBatchWrapper89_copies1882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60256 := (publicBatchWrapper89_copies1883 a h).1
  have c60257 := (publicBatchWrapper89_copies1883 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_0 := arithEq_of_rows h (row := 971) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_0
  simp only [← c60252, ← c60253, ← c60254] at e_971_0
  have e_971_1 := arithEq_of_rows h (row := 971) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_1
  simp only [← c60255, ← c60256, k2, ← c60257] at e_971_1
  have hr := e_971_1
  simp only [e_971_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7836)) := by
  have c60258 := (publicBatchWrapper89_copies1883 a h).2.2.1
  have c60259 := (publicBatchWrapper89_copies1883 a h).2.2.2.1
  have c60260 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.1
  have c60261 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.1
  have c60262 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.1
  have c60263 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_2 := arithEq_of_rows h (row := 971) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_2
  simp only [← c60258, ← c60259, ← c60260] at e_971_2
  have e_971_3 := arithEq_of_rows h (row := 971) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_3
  simp only [← c60261, ← c60262, k2, ← c60263] at e_971_3
  have hr := e_971_3
  simp only [e_971_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7837)) := by
  have c60264 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.1
  have c60265 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.1
  have c60266 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60267 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60268 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60269 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_4 := arithEq_of_rows h (row := 971) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_4
  simp only [← c60264, ← c60265, ← c60266] at e_971_4
  have e_971_5 := arithEq_of_rows h (row := 971) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_5
  simp only [← c60267, ← c60268, k2, ← c60269] at e_971_5
  have hr := e_971_5
  simp only [e_971_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7838)) := by
  have c60270 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60271 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60272 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60273 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60274 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60275 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_6 := arithEq_of_rows h (row := 971) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_6
  simp only [← c60270, ← c60271, ← c60272] at e_971_6
  have e_971_7 := arithEq_of_rows h (row := 971) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_7
  simp only [← c60273, ← c60274, k2, ← c60275] at e_971_7
  have hr := e_971_7
  simp only [e_971_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7839)) := by
  have c60276 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60277 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60278 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60279 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60280 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60281 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_8 := arithEq_of_rows h (row := 971) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_8
  simp only [← c60276, ← c60277, ← c60278] at e_971_8
  have e_971_9 := arithEq_of_rows h (row := 971) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_9
  simp only [← c60279, ← c60280, k2, ← c60281] at e_971_9
  have hr := e_971_9
  simp only [e_971_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7840)) := by
  have c60282 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60283 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60284 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60285 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60286 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60287 := (publicBatchWrapper89_copies1883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_10 := arithEq_of_rows h (row := 971) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_10
  simp only [← c60282, ← c60283, ← c60284] at e_971_10
  have e_971_11 := arithEq_of_rows h (row := 971) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_11
  simp only [← c60285, ← c60286, k2, ← c60287] at e_971_11
  have hr := e_971_11
  simp only [e_971_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7841)) := by
  have c60288 := (publicBatchWrapper89_copies1884 a h).1
  have c60289 := (publicBatchWrapper89_copies1884 a h).2.1
  have c60290 := (publicBatchWrapper89_copies1884 a h).2.2.1
  have c60291 := (publicBatchWrapper89_copies1884 a h).2.2.2.1
  have c60292 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.1
  have c60293 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_12 := arithEq_of_rows h (row := 971) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_12
  simp only [← c60288, ← c60289, ← c60290] at e_971_12
  have e_971_13 := arithEq_of_rows h (row := 971) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_13
  simp only [← c60291, ← c60292, k2, ← c60293] at e_971_13
  have hr := e_971_13
  simp only [e_971_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7842)) := by
  have c60294 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.1
  have c60295 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.1
  have c60296 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.1
  have c60297 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.1
  have c60298 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60299 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_14 := arithEq_of_rows h (row := 971) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_14
  simp only [← c60294, ← c60295, ← c60296] at e_971_14
  have e_971_15 := arithEq_of_rows h (row := 971) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_15
  simp only [← c60297, ← c60298, k2, ← c60299] at e_971_15
  have hr := e_971_15
  simp only [e_971_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7843)) := by
  have c60300 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60301 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60302 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60303 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60304 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60305 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_16 := arithEq_of_rows h (row := 971) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_16
  simp only [← c60300, ← c60301, ← c60302] at e_971_16
  have e_971_17 := arithEq_of_rows h (row := 971) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_17
  simp only [← c60303, ← c60304, k2, ← c60305] at e_971_17
  have hr := e_971_17
  simp only [e_971_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 971 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7844)) := by
  have c60306 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60307 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60308 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60309 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60310 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60311 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_971_18 := arithEq_of_rows h (row := 971) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_18
  simp only [← c60306, ← c60307, ← c60308] at e_971_18
  have e_971_19 := arithEq_of_rows h (row := 971) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_971_19
  simp only [← c60309, ← c60310, k2, ← c60311] at e_971_19
  have hr := e_971_19
  simp only [e_971_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7845)) := by
  have c60312 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60313 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60314 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60315 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60316 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60317 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_0 := arithEq_of_rows h (row := 972) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_0
  simp only [← c60312, ← c60313, ← c60314] at e_972_0
  have e_972_1 := arithEq_of_rows h (row := 972) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_1
  simp only [← c60315, ← c60316, k2, ← c60317] at e_972_1
  have hr := e_972_1
  simp only [e_972_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7846)) := by
  have c60318 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60319 := (publicBatchWrapper89_copies1884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60320 := (publicBatchWrapper89_copies1885 a h).1
  have c60321 := (publicBatchWrapper89_copies1885 a h).2.1
  have c60322 := (publicBatchWrapper89_copies1885 a h).2.2.1
  have c60323 := (publicBatchWrapper89_copies1885 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_2 := arithEq_of_rows h (row := 972) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_2
  simp only [← c60318, ← c60319, ← c60320] at e_972_2
  have e_972_3 := arithEq_of_rows h (row := 972) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_3
  simp only [← c60321, ← c60322, k2, ← c60323] at e_972_3
  have hr := e_972_3
  simp only [e_972_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7847)) := by
  have c60324 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.1
  have c60325 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.1
  have c60326 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.1
  have c60327 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.1
  have c60328 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.1
  have c60329 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_4 := arithEq_of_rows h (row := 972) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_4
  simp only [← c60324, ← c60325, ← c60326] at e_972_4
  have e_972_5 := arithEq_of_rows h (row := 972) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_5
  simp only [← c60327, ← c60328, k2, ← c60329] at e_972_5
  have hr := e_972_5
  simp only [e_972_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7848)) := by
  have c60330 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60331 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60332 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60333 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60334 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60335 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_6 := arithEq_of_rows h (row := 972) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_6
  simp only [← c60330, ← c60331, ← c60332] at e_972_6
  have e_972_7 := arithEq_of_rows h (row := 972) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_7
  simp only [← c60333, ← c60334, k2, ← c60335] at e_972_7
  have hr := e_972_7
  simp only [e_972_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7849)) := by
  have c60336 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60337 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60338 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60339 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60340 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60341 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_8 := arithEq_of_rows h (row := 972) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_8
  simp only [← c60336, ← c60337, ← c60338] at e_972_8
  have e_972_9 := arithEq_of_rows h (row := 972) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_9
  simp only [← c60339, ← c60340, k2, ← c60341] at e_972_9
  have hr := e_972_9
  simp only [e_972_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7850)) := by
  have c60342 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60343 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60344 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60345 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60346 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60347 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_10 := arithEq_of_rows h (row := 972) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_10
  simp only [← c60342, ← c60343, ← c60344] at e_972_10
  have e_972_11 := arithEq_of_rows h (row := 972) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_11
  simp only [← c60345, ← c60346, k2, ← c60347] at e_972_11
  have hr := e_972_11
  simp only [e_972_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7851)) := by
  have c60348 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60349 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60350 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60351 := (publicBatchWrapper89_copies1885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60352 := (publicBatchWrapper89_copies1886 a h).1
  have c60353 := (publicBatchWrapper89_copies1886 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_12 := arithEq_of_rows h (row := 972) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_12
  simp only [← c60348, ← c60349, ← c60350] at e_972_12
  have e_972_13 := arithEq_of_rows h (row := 972) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_13
  simp only [← c60351, ← c60352, k2, ← c60353] at e_972_13
  have hr := e_972_13
  simp only [e_972_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7852)) := by
  have c60354 := (publicBatchWrapper89_copies1886 a h).2.2.1
  have c60355 := (publicBatchWrapper89_copies1886 a h).2.2.2.1
  have c60356 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.1
  have c60357 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.1
  have c60358 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.1
  have c60359 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_14 := arithEq_of_rows h (row := 972) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_14
  simp only [← c60354, ← c60355, ← c60356] at e_972_14
  have e_972_15 := arithEq_of_rows h (row := 972) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_15
  simp only [← c60357, ← c60358, k2, ← c60359] at e_972_15
  have hr := e_972_15
  simp only [e_972_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7853)) := by
  have c60360 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.1
  have c60361 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.1
  have c60362 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60363 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60364 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60365 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_16 := arithEq_of_rows h (row := 972) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_16
  simp only [← c60360, ← c60361, ← c60362] at e_972_16
  have e_972_17 := arithEq_of_rows h (row := 972) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_17
  simp only [← c60363, ← c60364, k2, ← c60365] at e_972_17
  have hr := e_972_17
  simp only [e_972_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 972 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7854)) := by
  have c60366 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60367 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60368 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60369 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60370 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60371 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_972_18 := arithEq_of_rows h (row := 972) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_18
  simp only [← c60366, ← c60367, ← c60368] at e_972_18
  have e_972_19 := arithEq_of_rows h (row := 972) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_972_19
  simp only [← c60369, ← c60370, k2, ← c60371] at e_972_19
  have hr := e_972_19
  simp only [e_972_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7855)) := by
  have c60372 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60373 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60374 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60375 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60376 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60377 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_0 := arithEq_of_rows h (row := 973) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_0
  simp only [← c60372, ← c60373, ← c60374] at e_973_0
  have e_973_1 := arithEq_of_rows h (row := 973) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_1
  simp only [← c60375, ← c60376, k2, ← c60377] at e_973_1
  have hr := e_973_1
  simp only [e_973_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7856)) := by
  have c60378 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60379 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60380 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60381 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60382 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60383 := (publicBatchWrapper89_copies1886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_2 := arithEq_of_rows h (row := 973) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_2
  simp only [← c60378, ← c60379, ← c60380] at e_973_2
  have e_973_3 := arithEq_of_rows h (row := 973) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_3
  simp only [← c60381, ← c60382, k2, ← c60383] at e_973_3
  have hr := e_973_3
  simp only [e_973_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7857)) := by
  have c60384 := (publicBatchWrapper89_copies1887 a h).1
  have c60385 := (publicBatchWrapper89_copies1887 a h).2.1
  have c60386 := (publicBatchWrapper89_copies1887 a h).2.2.1
  have c60387 := (publicBatchWrapper89_copies1887 a h).2.2.2.1
  have c60388 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.1
  have c60389 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_4 := arithEq_of_rows h (row := 973) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_4
  simp only [← c60384, ← c60385, ← c60386] at e_973_4
  have e_973_5 := arithEq_of_rows h (row := 973) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_5
  simp only [← c60387, ← c60388, k2, ← c60389] at e_973_5
  have hr := e_973_5
  simp only [e_973_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7858)) := by
  have c60390 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.1
  have c60391 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.1
  have c60392 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.1
  have c60393 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.1
  have c60394 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60395 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_6 := arithEq_of_rows h (row := 973) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_6
  simp only [← c60390, ← c60391, ← c60392] at e_973_6
  have e_973_7 := arithEq_of_rows h (row := 973) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_7
  simp only [← c60393, ← c60394, k2, ← c60395] at e_973_7
  have hr := e_973_7
  simp only [e_973_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7859)) := by
  have c60396 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60397 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60398 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60399 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60400 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60401 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_8 := arithEq_of_rows h (row := 973) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_8
  simp only [← c60396, ← c60397, ← c60398] at e_973_8
  have e_973_9 := arithEq_of_rows h (row := 973) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_9
  simp only [← c60399, ← c60400, k2, ← c60401] at e_973_9
  have hr := e_973_9
  simp only [e_973_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7860)) := by
  have c60402 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60403 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60404 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60405 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60406 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60407 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_10 := arithEq_of_rows h (row := 973) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_10
  simp only [← c60402, ← c60403, ← c60404] at e_973_10
  have e_973_11 := arithEq_of_rows h (row := 973) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_11
  simp only [← c60405, ← c60406, k2, ← c60407] at e_973_11
  have hr := e_973_11
  simp only [e_973_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7861)) := by
  have c60408 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60409 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60410 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60411 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60412 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60413 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_12 := arithEq_of_rows h (row := 973) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_12
  simp only [← c60408, ← c60409, ← c60410] at e_973_12
  have e_973_13 := arithEq_of_rows h (row := 973) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_13
  simp only [← c60411, ← c60412, k2, ← c60413] at e_973_13
  have hr := e_973_13
  simp only [e_973_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7862)) := by
  have c60414 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60415 := (publicBatchWrapper89_copies1887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60416 := (publicBatchWrapper89_copies1888 a h).1
  have c60417 := (publicBatchWrapper89_copies1888 a h).2.1
  have c60418 := (publicBatchWrapper89_copies1888 a h).2.2.1
  have c60419 := (publicBatchWrapper89_copies1888 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_14 := arithEq_of_rows h (row := 973) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_14
  simp only [← c60414, ← c60415, ← c60416] at e_973_14
  have e_973_15 := arithEq_of_rows h (row := 973) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_15
  simp only [← c60417, ← c60418, k2, ← c60419] at e_973_15
  have hr := e_973_15
  simp only [e_973_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7863)) := by
  have c60420 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.1
  have c60421 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.1
  have c60422 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.1
  have c60423 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.1
  have c60424 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.1
  have c60425 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_16 := arithEq_of_rows h (row := 973) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_16
  simp only [← c60420, ← c60421, ← c60422] at e_973_16
  have e_973_17 := arithEq_of_rows h (row := 973) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_17
  simp only [← c60423, ← c60424, k2, ← c60425] at e_973_17
  have hr := e_973_17
  simp only [e_973_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 973 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7864)) := by
  have c60426 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60427 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60428 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60429 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60430 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60431 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_973_18 := arithEq_of_rows h (row := 973) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_18
  simp only [← c60426, ← c60427, ← c60428] at e_973_18
  have e_973_19 := arithEq_of_rows h (row := 973) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_973_19
  simp only [← c60429, ← c60430, k2, ← c60431] at e_973_19
  have hr := e_973_19
  simp only [e_973_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7865)) := by
  have c60432 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60433 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60434 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60435 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60436 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60437 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_0 := arithEq_of_rows h (row := 974) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_0
  simp only [← c60432, ← c60433, ← c60434] at e_974_0
  have e_974_1 := arithEq_of_rows h (row := 974) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_1
  simp only [← c60435, ← c60436, k2, ← c60437] at e_974_1
  have hr := e_974_1
  simp only [e_974_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7866)) := by
  have c60438 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60439 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60440 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60441 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60442 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60443 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_2 := arithEq_of_rows h (row := 974) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_2
  simp only [← c60438, ← c60439, ← c60440] at e_974_2
  have e_974_3 := arithEq_of_rows h (row := 974) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_3
  simp only [← c60441, ← c60442, k2, ← c60443] at e_974_3
  have hr := e_974_3
  simp only [e_974_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7867)) := by
  have c60444 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60445 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60446 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60447 := (publicBatchWrapper89_copies1888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60448 := (publicBatchWrapper89_copies1889 a h).1
  have c60449 := (publicBatchWrapper89_copies1889 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_4 := arithEq_of_rows h (row := 974) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_4
  simp only [← c60444, ← c60445, ← c60446] at e_974_4
  have e_974_5 := arithEq_of_rows h (row := 974) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_5
  simp only [← c60447, ← c60448, k2, ← c60449] at e_974_5
  have hr := e_974_5
  simp only [e_974_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7868)) := by
  have c60450 := (publicBatchWrapper89_copies1889 a h).2.2.1
  have c60451 := (publicBatchWrapper89_copies1889 a h).2.2.2.1
  have c60452 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.1
  have c60453 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.1
  have c60454 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.1
  have c60455 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_6 := arithEq_of_rows h (row := 974) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_6
  simp only [← c60450, ← c60451, ← c60452] at e_974_6
  have e_974_7 := arithEq_of_rows h (row := 974) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_7
  simp only [← c60453, ← c60454, k2, ← c60455] at e_974_7
  have hr := e_974_7
  simp only [e_974_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7869)) := by
  have c60456 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.1
  have c60457 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.1
  have c60458 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60459 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60460 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60461 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_8 := arithEq_of_rows h (row := 974) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_8
  simp only [← c60456, ← c60457, ← c60458] at e_974_8
  have e_974_9 := arithEq_of_rows h (row := 974) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_9
  simp only [← c60459, ← c60460, k2, ← c60461] at e_974_9
  have hr := e_974_9
  simp only [e_974_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7870)) := by
  have c60462 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60463 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60464 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60465 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60466 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60467 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_10 := arithEq_of_rows h (row := 974) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_10
  simp only [← c60462, ← c60463, ← c60464] at e_974_10
  have e_974_11 := arithEq_of_rows h (row := 974) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_11
  simp only [← c60465, ← c60466, k2, ← c60467] at e_974_11
  have hr := e_974_11
  simp only [e_974_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7871)) := by
  have c60468 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60469 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60470 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60471 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60472 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60473 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_12 := arithEq_of_rows h (row := 974) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_12
  simp only [← c60468, ← c60469, ← c60470] at e_974_12
  have e_974_13 := arithEq_of_rows h (row := 974) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_13
  simp only [← c60471, ← c60472, k2, ← c60473] at e_974_13
  have hr := e_974_13
  simp only [e_974_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7872)) := by
  have c60474 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60475 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60476 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60477 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60478 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60479 := (publicBatchWrapper89_copies1889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_14 := arithEq_of_rows h (row := 974) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_14
  simp only [← c60474, ← c60475, ← c60476] at e_974_14
  have e_974_15 := arithEq_of_rows h (row := 974) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_15
  simp only [← c60477, ← c60478, k2, ← c60479] at e_974_15
  have hr := e_974_15
  simp only [e_974_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7873)) := by
  have c60480 := (publicBatchWrapper89_copies1890 a h).1
  have c60481 := (publicBatchWrapper89_copies1890 a h).2.1
  have c60482 := (publicBatchWrapper89_copies1890 a h).2.2.1
  have c60483 := (publicBatchWrapper89_copies1890 a h).2.2.2.1
  have c60484 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.1
  have c60485 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_16 := arithEq_of_rows h (row := 974) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_16
  simp only [← c60480, ← c60481, ← c60482] at e_974_16
  have e_974_17 := arithEq_of_rows h (row := 974) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_17
  simp only [← c60483, ← c60484, k2, ← c60485] at e_974_17
  have hr := e_974_17
  simp only [e_974_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 974 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7874)) := by
  have c60486 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.1
  have c60487 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.1
  have c60488 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.1
  have c60489 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.1
  have c60490 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60491 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_974_18 := arithEq_of_rows h (row := 974) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_18
  simp only [← c60486, ← c60487, ← c60488] at e_974_18
  have e_974_19 := arithEq_of_rows h (row := 974) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_974_19
  simp only [← c60489, ← c60490, k2, ← c60491] at e_974_19
  have hr := e_974_19
  simp only [e_974_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7875)) := by
  have c60492 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60493 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60494 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60495 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60496 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60497 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_0 := arithEq_of_rows h (row := 975) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_0
  simp only [← c60492, ← c60493, ← c60494] at e_975_0
  have e_975_1 := arithEq_of_rows h (row := 975) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_1
  simp only [← c60495, ← c60496, k2, ← c60497] at e_975_1
  have hr := e_975_1
  simp only [e_975_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7876)) := by
  have c60498 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60499 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60500 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60501 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60502 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60503 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_2 := arithEq_of_rows h (row := 975) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_2
  simp only [← c60498, ← c60499, ← c60500] at e_975_2
  have e_975_3 := arithEq_of_rows h (row := 975) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_3
  simp only [← c60501, ← c60502, k2, ← c60503] at e_975_3
  have hr := e_975_3
  simp only [e_975_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7877)) := by
  have c60504 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60505 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60506 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60507 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60508 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60509 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_4 := arithEq_of_rows h (row := 975) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_4
  simp only [← c60504, ← c60505, ← c60506] at e_975_4
  have e_975_5 := arithEq_of_rows h (row := 975) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_5
  simp only [← c60507, ← c60508, k2, ← c60509] at e_975_5
  have hr := e_975_5
  simp only [e_975_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7878)) := by
  have c60510 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60511 := (publicBatchWrapper89_copies1890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60512 := (publicBatchWrapper89_copies1891 a h).1
  have c60513 := (publicBatchWrapper89_copies1891 a h).2.1
  have c60514 := (publicBatchWrapper89_copies1891 a h).2.2.1
  have c60515 := (publicBatchWrapper89_copies1891 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_6 := arithEq_of_rows h (row := 975) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_6
  simp only [← c60510, ← c60511, ← c60512] at e_975_6
  have e_975_7 := arithEq_of_rows h (row := 975) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_7
  simp only [← c60513, ← c60514, k2, ← c60515] at e_975_7
  have hr := e_975_7
  simp only [e_975_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7879)) := by
  have c60516 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.1
  have c60517 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.1
  have c60518 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.1
  have c60519 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.1
  have c60520 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.1
  have c60521 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_8 := arithEq_of_rows h (row := 975) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_8
  simp only [← c60516, ← c60517, ← c60518] at e_975_8
  have e_975_9 := arithEq_of_rows h (row := 975) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_9
  simp only [← c60519, ← c60520, k2, ← c60521] at e_975_9
  have hr := e_975_9
  simp only [e_975_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7880)) := by
  have c60522 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60523 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60524 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60525 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60526 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60527 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_10 := arithEq_of_rows h (row := 975) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_10
  simp only [← c60522, ← c60523, ← c60524] at e_975_10
  have e_975_11 := arithEq_of_rows h (row := 975) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_11
  simp only [← c60525, ← c60526, k2, ← c60527] at e_975_11
  have hr := e_975_11
  simp only [e_975_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7881)) := by
  have c60528 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60529 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60530 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60531 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60532 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60533 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_12 := arithEq_of_rows h (row := 975) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_12
  simp only [← c60528, ← c60529, ← c60530] at e_975_12
  have e_975_13 := arithEq_of_rows h (row := 975) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_13
  simp only [← c60531, ← c60532, k2, ← c60533] at e_975_13
  have hr := e_975_13
  simp only [e_975_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7882)) := by
  have c60534 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60535 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60536 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60537 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60538 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60539 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_14 := arithEq_of_rows h (row := 975) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_14
  simp only [← c60534, ← c60535, ← c60536] at e_975_14
  have e_975_15 := arithEq_of_rows h (row := 975) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_15
  simp only [← c60537, ← c60538, k2, ← c60539] at e_975_15
  have hr := e_975_15
  simp only [e_975_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7883)) := by
  have c60540 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60541 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60542 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60543 := (publicBatchWrapper89_copies1891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60544 := (publicBatchWrapper89_copies1892 a h).1
  have c60545 := (publicBatchWrapper89_copies1892 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_16 := arithEq_of_rows h (row := 975) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_16
  simp only [← c60540, ← c60541, ← c60542] at e_975_16
  have e_975_17 := arithEq_of_rows h (row := 975) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_17
  simp only [← c60543, ← c60544, k2, ← c60545] at e_975_17
  have hr := e_975_17
  simp only [e_975_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 975 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7884)) := by
  have c60546 := (publicBatchWrapper89_copies1892 a h).2.2.1
  have c60547 := (publicBatchWrapper89_copies1892 a h).2.2.2.1
  have c60548 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.1
  have c60549 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.1
  have c60550 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.1
  have c60551 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_975_18 := arithEq_of_rows h (row := 975) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_18
  simp only [← c60546, ← c60547, ← c60548] at e_975_18
  have e_975_19 := arithEq_of_rows h (row := 975) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_975_19
  simp only [← c60549, ← c60550, k2, ← c60551] at e_975_19
  have hr := e_975_19
  simp only [e_975_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7885)) := by
  have c60552 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.1
  have c60553 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.1
  have c60554 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60555 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60556 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60557 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_0 := arithEq_of_rows h (row := 976) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_0
  simp only [← c60552, ← c60553, ← c60554] at e_976_0
  have e_976_1 := arithEq_of_rows h (row := 976) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_1
  simp only [← c60555, ← c60556, k2, ← c60557] at e_976_1
  have hr := e_976_1
  simp only [e_976_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7886)) := by
  have c60558 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60559 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60560 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60561 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60562 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60563 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_2 := arithEq_of_rows h (row := 976) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_2
  simp only [← c60558, ← c60559, ← c60560] at e_976_2
  have e_976_3 := arithEq_of_rows h (row := 976) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_3
  simp only [← c60561, ← c60562, k2, ← c60563] at e_976_3
  have hr := e_976_3
  simp only [e_976_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7887)) := by
  have c60564 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60565 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60566 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60567 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60568 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60569 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_4 := arithEq_of_rows h (row := 976) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_4
  simp only [← c60564, ← c60565, ← c60566] at e_976_4
  have e_976_5 := arithEq_of_rows h (row := 976) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_5
  simp only [← c60567, ← c60568, k2, ← c60569] at e_976_5
  have hr := e_976_5
  simp only [e_976_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7888)) := by
  have c60570 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60571 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60572 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60573 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60574 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60575 := (publicBatchWrapper89_copies1892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_6 := arithEq_of_rows h (row := 976) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_6
  simp only [← c60570, ← c60571, ← c60572] at e_976_6
  have e_976_7 := arithEq_of_rows h (row := 976) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_7
  simp only [← c60573, ← c60574, k2, ← c60575] at e_976_7
  have hr := e_976_7
  simp only [e_976_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7889)) := by
  have c60576 := (publicBatchWrapper89_copies1893 a h).1
  have c60577 := (publicBatchWrapper89_copies1893 a h).2.1
  have c60578 := (publicBatchWrapper89_copies1893 a h).2.2.1
  have c60579 := (publicBatchWrapper89_copies1893 a h).2.2.2.1
  have c60580 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.1
  have c60581 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_8 := arithEq_of_rows h (row := 976) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_8
  simp only [← c60576, ← c60577, ← c60578] at e_976_8
  have e_976_9 := arithEq_of_rows h (row := 976) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_9
  simp only [← c60579, ← c60580, k2, ← c60581] at e_976_9
  have hr := e_976_9
  simp only [e_976_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7890)) := by
  have c60582 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.1
  have c60583 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.1
  have c60584 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.1
  have c60585 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.1
  have c60586 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60587 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_10 := arithEq_of_rows h (row := 976) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_10
  simp only [← c60582, ← c60583, ← c60584] at e_976_10
  have e_976_11 := arithEq_of_rows h (row := 976) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_11
  simp only [← c60585, ← c60586, k2, ← c60587] at e_976_11
  have hr := e_976_11
  simp only [e_976_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7891)) := by
  have c60588 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60589 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60590 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60591 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60592 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60593 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_12 := arithEq_of_rows h (row := 976) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_12
  simp only [← c60588, ← c60589, ← c60590] at e_976_12
  have e_976_13 := arithEq_of_rows h (row := 976) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_13
  simp only [← c60591, ← c60592, k2, ← c60593] at e_976_13
  have hr := e_976_13
  simp only [e_976_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7892)) := by
  have c60594 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60595 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60596 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60597 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60598 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60599 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_14 := arithEq_of_rows h (row := 976) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_14
  simp only [← c60594, ← c60595, ← c60596] at e_976_14
  have e_976_15 := arithEq_of_rows h (row := 976) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_15
  simp only [← c60597, ← c60598, k2, ← c60599] at e_976_15
  have hr := e_976_15
  simp only [e_976_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7893)) := by
  have c60600 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60601 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60602 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60603 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60604 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60605 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_16 := arithEq_of_rows h (row := 976) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_16
  simp only [← c60600, ← c60601, ← c60602] at e_976_16
  have e_976_17 := arithEq_of_rows h (row := 976) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_17
  simp only [← c60603, ← c60604, k2, ← c60605] at e_976_17
  have hr := e_976_17
  simp only [e_976_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 976 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7894)) := by
  have c60606 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60607 := (publicBatchWrapper89_copies1893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60608 := (publicBatchWrapper89_copies1894 a h).1
  have c60609 := (publicBatchWrapper89_copies1894 a h).2.1
  have c60610 := (publicBatchWrapper89_copies1894 a h).2.2.1
  have c60611 := (publicBatchWrapper89_copies1894 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_976_18 := arithEq_of_rows h (row := 976) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_18
  simp only [← c60606, ← c60607, ← c60608] at e_976_18
  have e_976_19 := arithEq_of_rows h (row := 976) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_976_19
  simp only [← c60609, ← c60610, k2, ← c60611] at e_976_19
  have hr := e_976_19
  simp only [e_976_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7895)) := by
  have c60612 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.1
  have c60613 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.1
  have c60614 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.1
  have c60615 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.1
  have c60616 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.1
  have c60617 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_0 := arithEq_of_rows h (row := 977) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_0
  simp only [← c60612, ← c60613, ← c60614] at e_977_0
  have e_977_1 := arithEq_of_rows h (row := 977) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_1
  simp only [← c60615, ← c60616, k2, ← c60617] at e_977_1
  have hr := e_977_1
  simp only [e_977_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7896)) := by
  have c60618 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60619 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60620 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60621 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60622 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60623 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_2 := arithEq_of_rows h (row := 977) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_2
  simp only [← c60618, ← c60619, ← c60620] at e_977_2
  have e_977_3 := arithEq_of_rows h (row := 977) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_3
  simp only [← c60621, ← c60622, k2, ← c60623] at e_977_3
  have hr := e_977_3
  simp only [e_977_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7897)) := by
  have c60624 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60625 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60626 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60627 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60628 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60629 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_4 := arithEq_of_rows h (row := 977) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_4
  simp only [← c60624, ← c60625, ← c60626] at e_977_4
  have e_977_5 := arithEq_of_rows h (row := 977) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_5
  simp only [← c60627, ← c60628, k2, ← c60629] at e_977_5
  have hr := e_977_5
  simp only [e_977_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7898)) := by
  have c60630 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60631 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60632 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60633 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60634 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60635 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_6 := arithEq_of_rows h (row := 977) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_6
  simp only [← c60630, ← c60631, ← c60632] at e_977_6
  have e_977_7 := arithEq_of_rows h (row := 977) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_7
  simp only [← c60633, ← c60634, k2, ← c60635] at e_977_7
  have hr := e_977_7
  simp only [e_977_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7899)) := by
  have c60636 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60637 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60638 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60639 := (publicBatchWrapper89_copies1894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60640 := (publicBatchWrapper89_copies1895 a h).1
  have c60641 := (publicBatchWrapper89_copies1895 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_8 := arithEq_of_rows h (row := 977) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_8
  simp only [← c60636, ← c60637, ← c60638] at e_977_8
  have e_977_9 := arithEq_of_rows h (row := 977) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_9
  simp only [← c60639, ← c60640, k2, ← c60641] at e_977_9
  have hr := e_977_9
  simp only [e_977_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7900)) := by
  have c60642 := (publicBatchWrapper89_copies1895 a h).2.2.1
  have c60643 := (publicBatchWrapper89_copies1895 a h).2.2.2.1
  have c60644 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.1
  have c60645 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.1
  have c60646 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.1
  have c60647 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_10 := arithEq_of_rows h (row := 977) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_10
  simp only [← c60642, ← c60643, ← c60644] at e_977_10
  have e_977_11 := arithEq_of_rows h (row := 977) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_11
  simp only [← c60645, ← c60646, k2, ← c60647] at e_977_11
  have hr := e_977_11
  simp only [e_977_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7901)) := by
  have c60648 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.1
  have c60649 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.1
  have c60650 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60651 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60652 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60653 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_12 := arithEq_of_rows h (row := 977) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_12
  simp only [← c60648, ← c60649, ← c60650] at e_977_12
  have e_977_13 := arithEq_of_rows h (row := 977) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_13
  simp only [← c60651, ← c60652, k2, ← c60653] at e_977_13
  have hr := e_977_13
  simp only [e_977_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7902)) := by
  have c60654 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60655 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60656 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60657 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60658 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60659 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_14 := arithEq_of_rows h (row := 977) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_14
  simp only [← c60654, ← c60655, ← c60656] at e_977_14
  have e_977_15 := arithEq_of_rows h (row := 977) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_15
  simp only [← c60657, ← c60658, k2, ← c60659] at e_977_15
  have hr := e_977_15
  simp only [e_977_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7903)) := by
  have c60660 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60661 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60662 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60663 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60664 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60665 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_16 := arithEq_of_rows h (row := 977) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_16
  simp only [← c60660, ← c60661, ← c60662] at e_977_16
  have e_977_17 := arithEq_of_rows h (row := 977) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_17
  simp only [← c60663, ← c60664, k2, ← c60665] at e_977_17
  have hr := e_977_17
  simp only [e_977_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 977 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7904)) := by
  have c60666 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60667 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60668 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60669 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60670 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60671 := (publicBatchWrapper89_copies1895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_977_18 := arithEq_of_rows h (row := 977) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_18
  simp only [← c60666, ← c60667, ← c60668] at e_977_18
  have e_977_19 := arithEq_of_rows h (row := 977) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_977_19
  simp only [← c60669, ← c60670, k2, ← c60671] at e_977_19
  have hr := e_977_19
  simp only [e_977_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7905)) := by
  have c60672 := (publicBatchWrapper89_copies1896 a h).1
  have c60673 := (publicBatchWrapper89_copies1896 a h).2.1
  have c60674 := (publicBatchWrapper89_copies1896 a h).2.2.1
  have c60675 := (publicBatchWrapper89_copies1896 a h).2.2.2.1
  have c60676 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.1
  have c60677 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_0 := arithEq_of_rows h (row := 978) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_0
  simp only [← c60672, ← c60673, ← c60674] at e_978_0
  have e_978_1 := arithEq_of_rows h (row := 978) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_1
  simp only [← c60675, ← c60676, k2, ← c60677] at e_978_1
  have hr := e_978_1
  simp only [e_978_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7906)) := by
  have c60678 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.1
  have c60679 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.1
  have c60680 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.1
  have c60681 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.1
  have c60682 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60683 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_2 := arithEq_of_rows h (row := 978) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_2
  simp only [← c60678, ← c60679, ← c60680] at e_978_2
  have e_978_3 := arithEq_of_rows h (row := 978) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_3
  simp only [← c60681, ← c60682, k2, ← c60683] at e_978_3
  have hr := e_978_3
  simp only [e_978_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7907)) := by
  have c60684 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60685 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60686 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60687 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60688 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60689 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_4 := arithEq_of_rows h (row := 978) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_4
  simp only [← c60684, ← c60685, ← c60686] at e_978_4
  have e_978_5 := arithEq_of_rows h (row := 978) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_5
  simp only [← c60687, ← c60688, k2, ← c60689] at e_978_5
  have hr := e_978_5
  simp only [e_978_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7908)) := by
  have c60690 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60691 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60692 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60693 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60694 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60695 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_6 := arithEq_of_rows h (row := 978) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_6
  simp only [← c60690, ← c60691, ← c60692] at e_978_6
  have e_978_7 := arithEq_of_rows h (row := 978) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_7
  simp only [← c60693, ← c60694, k2, ← c60695] at e_978_7
  have hr := e_978_7
  simp only [e_978_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7909)) := by
  have c60696 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60697 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60698 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60699 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60700 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60701 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_8 := arithEq_of_rows h (row := 978) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_8
  simp only [← c60696, ← c60697, ← c60698] at e_978_8
  have e_978_9 := arithEq_of_rows h (row := 978) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_9
  simp only [← c60699, ← c60700, k2, ← c60701] at e_978_9
  have hr := e_978_9
  simp only [e_978_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7910)) := by
  have c60702 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60703 := (publicBatchWrapper89_copies1896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60704 := (publicBatchWrapper89_copies1897 a h).1
  have c60705 := (publicBatchWrapper89_copies1897 a h).2.1
  have c60706 := (publicBatchWrapper89_copies1897 a h).2.2.1
  have c60707 := (publicBatchWrapper89_copies1897 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_10 := arithEq_of_rows h (row := 978) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_10
  simp only [← c60702, ← c60703, ← c60704] at e_978_10
  have e_978_11 := arithEq_of_rows h (row := 978) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_11
  simp only [← c60705, ← c60706, k2, ← c60707] at e_978_11
  have hr := e_978_11
  simp only [e_978_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7911)) := by
  have c60708 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.1
  have c60709 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.1
  have c60710 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.1
  have c60711 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.1
  have c60712 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.1
  have c60713 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_12 := arithEq_of_rows h (row := 978) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_12
  simp only [← c60708, ← c60709, ← c60710] at e_978_12
  have e_978_13 := arithEq_of_rows h (row := 978) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_13
  simp only [← c60711, ← c60712, k2, ← c60713] at e_978_13
  have hr := e_978_13
  simp only [e_978_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7912)) := by
  have c60714 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60715 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60716 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60717 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60718 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60719 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_14 := arithEq_of_rows h (row := 978) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_14
  simp only [← c60714, ← c60715, ← c60716] at e_978_14
  have e_978_15 := arithEq_of_rows h (row := 978) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_15
  simp only [← c60717, ← c60718, k2, ← c60719] at e_978_15
  have hr := e_978_15
  simp only [e_978_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7913)) := by
  have c60720 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60721 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60722 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60723 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60724 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60725 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_16 := arithEq_of_rows h (row := 978) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_16
  simp only [← c60720, ← c60721, ← c60722] at e_978_16
  have e_978_17 := arithEq_of_rows h (row := 978) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_17
  simp only [← c60723, ← c60724, k2, ← c60725] at e_978_17
  have hr := e_978_17
  simp only [e_978_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 978 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7914)) := by
  have c60726 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60727 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60728 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60729 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60730 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60731 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_978_18 := arithEq_of_rows h (row := 978) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_18
  simp only [← c60726, ← c60727, ← c60728] at e_978_18
  have e_978_19 := arithEq_of_rows h (row := 978) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_978_19
  simp only [← c60729, ← c60730, k2, ← c60731] at e_978_19
  have hr := e_978_19
  simp only [e_978_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7915)) := by
  have c60732 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60733 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60734 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60735 := (publicBatchWrapper89_copies1897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60736 := (publicBatchWrapper89_copies1898 a h).1
  have c60737 := (publicBatchWrapper89_copies1898 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_0 := arithEq_of_rows h (row := 979) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_0
  simp only [← c60732, ← c60733, ← c60734] at e_979_0
  have e_979_1 := arithEq_of_rows h (row := 979) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_1
  simp only [← c60735, ← c60736, k2, ← c60737] at e_979_1
  have hr := e_979_1
  simp only [e_979_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7916)) := by
  have c60738 := (publicBatchWrapper89_copies1898 a h).2.2.1
  have c60739 := (publicBatchWrapper89_copies1898 a h).2.2.2.1
  have c60740 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.1
  have c60741 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.1
  have c60742 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.1
  have c60743 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_2 := arithEq_of_rows h (row := 979) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_2
  simp only [← c60738, ← c60739, ← c60740] at e_979_2
  have e_979_3 := arithEq_of_rows h (row := 979) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_3
  simp only [← c60741, ← c60742, k2, ← c60743] at e_979_3
  have hr := e_979_3
  simp only [e_979_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7917)) := by
  have c60744 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.1
  have c60745 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.1
  have c60746 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60747 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60748 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60749 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_4 := arithEq_of_rows h (row := 979) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_4
  simp only [← c60744, ← c60745, ← c60746] at e_979_4
  have e_979_5 := arithEq_of_rows h (row := 979) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_5
  simp only [← c60747, ← c60748, k2, ← c60749] at e_979_5
  have hr := e_979_5
  simp only [e_979_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7918)) := by
  have c60750 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60751 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60752 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60753 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60754 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60755 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_6 := arithEq_of_rows h (row := 979) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_6
  simp only [← c60750, ← c60751, ← c60752] at e_979_6
  have e_979_7 := arithEq_of_rows h (row := 979) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_7
  simp only [← c60753, ← c60754, k2, ← c60755] at e_979_7
  have hr := e_979_7
  simp only [e_979_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7919)) := by
  have c60756 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60757 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60758 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60759 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60760 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60761 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_8 := arithEq_of_rows h (row := 979) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_8
  simp only [← c60756, ← c60757, ← c60758] at e_979_8
  have e_979_9 := arithEq_of_rows h (row := 979) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_9
  simp only [← c60759, ← c60760, k2, ← c60761] at e_979_9
  have hr := e_979_9
  simp only [e_979_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7920)) := by
  have c60762 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60763 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60764 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60765 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60766 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60767 := (publicBatchWrapper89_copies1898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_10 := arithEq_of_rows h (row := 979) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_10
  simp only [← c60762, ← c60763, ← c60764] at e_979_10
  have e_979_11 := arithEq_of_rows h (row := 979) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_11
  simp only [← c60765, ← c60766, k2, ← c60767] at e_979_11
  have hr := e_979_11
  simp only [e_979_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7921)) := by
  have c60768 := (publicBatchWrapper89_copies1899 a h).1
  have c60769 := (publicBatchWrapper89_copies1899 a h).2.1
  have c60770 := (publicBatchWrapper89_copies1899 a h).2.2.1
  have c60771 := (publicBatchWrapper89_copies1899 a h).2.2.2.1
  have c60772 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.1
  have c60773 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_12 := arithEq_of_rows h (row := 979) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_12
  simp only [← c60768, ← c60769, ← c60770] at e_979_12
  have e_979_13 := arithEq_of_rows h (row := 979) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_13
  simp only [← c60771, ← c60772, k2, ← c60773] at e_979_13
  have hr := e_979_13
  simp only [e_979_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7922)) := by
  have c60774 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.1
  have c60775 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.1
  have c60776 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.1
  have c60777 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.1
  have c60778 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60779 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_14 := arithEq_of_rows h (row := 979) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_14
  simp only [← c60774, ← c60775, ← c60776] at e_979_14
  have e_979_15 := arithEq_of_rows h (row := 979) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_15
  simp only [← c60777, ← c60778, k2, ← c60779] at e_979_15
  have hr := e_979_15
  simp only [e_979_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7923)) := by
  have c60780 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60781 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60782 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60783 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60784 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60785 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_16 := arithEq_of_rows h (row := 979) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_16
  simp only [← c60780, ← c60781, ← c60782] at e_979_16
  have e_979_17 := arithEq_of_rows h (row := 979) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_17
  simp only [← c60783, ← c60784, k2, ← c60785] at e_979_17
  have hr := e_979_17
  simp only [e_979_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 979 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7924)) := by
  have c60786 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60787 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60788 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60789 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60790 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60791 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_979_18 := arithEq_of_rows h (row := 979) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_18
  simp only [← c60786, ← c60787, ← c60788] at e_979_18
  have e_979_19 := arithEq_of_rows h (row := 979) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_979_19
  simp only [← c60789, ← c60790, k2, ← c60791] at e_979_19
  have hr := e_979_19
  simp only [e_979_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7925)) := by
  have c60792 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60793 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60794 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60795 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60796 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60797 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_0 := arithEq_of_rows h (row := 980) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_0
  simp only [← c60792, ← c60793, ← c60794] at e_980_0
  have e_980_1 := arithEq_of_rows h (row := 980) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_1
  simp only [← c60795, ← c60796, k2, ← c60797] at e_980_1
  have hr := e_980_1
  simp only [e_980_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7926)) := by
  have c60798 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60799 := (publicBatchWrapper89_copies1899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60800 := (publicBatchWrapper89_copies1900 a h).1
  have c60801 := (publicBatchWrapper89_copies1900 a h).2.1
  have c60802 := (publicBatchWrapper89_copies1900 a h).2.2.1
  have c60803 := (publicBatchWrapper89_copies1900 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_2 := arithEq_of_rows h (row := 980) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_2
  simp only [← c60798, ← c60799, ← c60800] at e_980_2
  have e_980_3 := arithEq_of_rows h (row := 980) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_3
  simp only [← c60801, ← c60802, k2, ← c60803] at e_980_3
  have hr := e_980_3
  simp only [e_980_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7927)) := by
  have c60804 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.1
  have c60805 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.1
  have c60806 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.1
  have c60807 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.1
  have c60808 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.1
  have c60809 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_4 := arithEq_of_rows h (row := 980) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_4
  simp only [← c60804, ← c60805, ← c60806] at e_980_4
  have e_980_5 := arithEq_of_rows h (row := 980) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_5
  simp only [← c60807, ← c60808, k2, ← c60809] at e_980_5
  have hr := e_980_5
  simp only [e_980_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7928)) := by
  have c60810 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60811 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60812 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60813 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60814 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60815 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_6 := arithEq_of_rows h (row := 980) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_6
  simp only [← c60810, ← c60811, ← c60812] at e_980_6
  have e_980_7 := arithEq_of_rows h (row := 980) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_7
  simp only [← c60813, ← c60814, k2, ← c60815] at e_980_7
  have hr := e_980_7
  simp only [e_980_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7929)) := by
  have c60816 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60817 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60818 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60819 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60820 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60821 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_8 := arithEq_of_rows h (row := 980) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_8
  simp only [← c60816, ← c60817, ← c60818] at e_980_8
  have e_980_9 := arithEq_of_rows h (row := 980) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_9
  simp only [← c60819, ← c60820, k2, ← c60821] at e_980_9
  have hr := e_980_9
  simp only [e_980_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7930)) := by
  have c60822 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60823 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60824 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60825 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60826 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60827 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_10 := arithEq_of_rows h (row := 980) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_10
  simp only [← c60822, ← c60823, ← c60824] at e_980_10
  have e_980_11 := arithEq_of_rows h (row := 980) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_11
  simp only [← c60825, ← c60826, k2, ← c60827] at e_980_11
  have hr := e_980_11
  simp only [e_980_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7931)) := by
  have c60828 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60829 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60830 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60831 := (publicBatchWrapper89_copies1900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60832 := (publicBatchWrapper89_copies1901 a h).1
  have c60833 := (publicBatchWrapper89_copies1901 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_12 := arithEq_of_rows h (row := 980) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_12
  simp only [← c60828, ← c60829, ← c60830] at e_980_12
  have e_980_13 := arithEq_of_rows h (row := 980) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_13
  simp only [← c60831, ← c60832, k2, ← c60833] at e_980_13
  have hr := e_980_13
  simp only [e_980_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7932)) := by
  have c60834 := (publicBatchWrapper89_copies1901 a h).2.2.1
  have c60835 := (publicBatchWrapper89_copies1901 a h).2.2.2.1
  have c60836 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.1
  have c60837 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.1
  have c60838 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.1
  have c60839 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_14 := arithEq_of_rows h (row := 980) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_14
  simp only [← c60834, ← c60835, ← c60836] at e_980_14
  have e_980_15 := arithEq_of_rows h (row := 980) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_15
  simp only [← c60837, ← c60838, k2, ← c60839] at e_980_15
  have hr := e_980_15
  simp only [e_980_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7933)) := by
  have c60840 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.1
  have c60841 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.1
  have c60842 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60843 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60844 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60845 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_16 := arithEq_of_rows h (row := 980) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_16
  simp only [← c60840, ← c60841, ← c60842] at e_980_16
  have e_980_17 := arithEq_of_rows h (row := 980) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_17
  simp only [← c60843, ← c60844, k2, ← c60845] at e_980_17
  have hr := e_980_17
  simp only [e_980_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 980 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7934)) := by
  have c60846 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60847 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60848 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60849 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60850 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60851 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_980_18 := arithEq_of_rows h (row := 980) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_18
  simp only [← c60846, ← c60847, ← c60848] at e_980_18
  have e_980_19 := arithEq_of_rows h (row := 980) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_980_19
  simp only [← c60849, ← c60850, k2, ← c60851] at e_980_19
  have hr := e_980_19
  simp only [e_980_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7935)) := by
  have c60852 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60853 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60854 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60855 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60856 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60857 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_0 := arithEq_of_rows h (row := 981) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_0
  simp only [← c60852, ← c60853, ← c60854] at e_981_0
  have e_981_1 := arithEq_of_rows h (row := 981) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_1
  simp only [← c60855, ← c60856, k2, ← c60857] at e_981_1
  have hr := e_981_1
  simp only [e_981_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7936)) := by
  have c60858 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60859 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60860 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60861 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60862 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60863 := (publicBatchWrapper89_copies1901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_2 := arithEq_of_rows h (row := 981) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_2
  simp only [← c60858, ← c60859, ← c60860] at e_981_2
  have e_981_3 := arithEq_of_rows h (row := 981) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_3
  simp only [← c60861, ← c60862, k2, ← c60863] at e_981_3
  have hr := e_981_3
  simp only [e_981_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7937)) := by
  have c60864 := (publicBatchWrapper89_copies1902 a h).1
  have c60865 := (publicBatchWrapper89_copies1902 a h).2.1
  have c60866 := (publicBatchWrapper89_copies1902 a h).2.2.1
  have c60867 := (publicBatchWrapper89_copies1902 a h).2.2.2.1
  have c60868 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.1
  have c60869 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_4 := arithEq_of_rows h (row := 981) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_4
  simp only [← c60864, ← c60865, ← c60866] at e_981_4
  have e_981_5 := arithEq_of_rows h (row := 981) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_5
  simp only [← c60867, ← c60868, k2, ← c60869] at e_981_5
  have hr := e_981_5
  simp only [e_981_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7938)) := by
  have c60870 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.1
  have c60871 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.1
  have c60872 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.1
  have c60873 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.1
  have c60874 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60875 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_6 := arithEq_of_rows h (row := 981) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_6
  simp only [← c60870, ← c60871, ← c60872] at e_981_6
  have e_981_7 := arithEq_of_rows h (row := 981) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_7
  simp only [← c60873, ← c60874, k2, ← c60875] at e_981_7
  have hr := e_981_7
  simp only [e_981_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7939)) := by
  have c60876 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60877 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60878 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60879 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60880 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60881 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_8 := arithEq_of_rows h (row := 981) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_8
  simp only [← c60876, ← c60877, ← c60878] at e_981_8
  have e_981_9 := arithEq_of_rows h (row := 981) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_9
  simp only [← c60879, ← c60880, k2, ← c60881] at e_981_9
  have hr := e_981_9
  simp only [e_981_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7940)) := by
  have c60882 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60883 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60884 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60885 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60886 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60887 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_10 := arithEq_of_rows h (row := 981) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_10
  simp only [← c60882, ← c60883, ← c60884] at e_981_10
  have e_981_11 := arithEq_of_rows h (row := 981) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_11
  simp only [← c60885, ← c60886, k2, ← c60887] at e_981_11
  have hr := e_981_11
  simp only [e_981_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7941)) := by
  have c60888 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60889 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60890 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60891 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60892 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60893 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_12 := arithEq_of_rows h (row := 981) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_12
  simp only [← c60888, ← c60889, ← c60890] at e_981_12
  have e_981_13 := arithEq_of_rows h (row := 981) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_13
  simp only [← c60891, ← c60892, k2, ← c60893] at e_981_13
  have hr := e_981_13
  simp only [e_981_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7942)) := by
  have c60894 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60895 := (publicBatchWrapper89_copies1902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60896 := (publicBatchWrapper89_copies1903 a h).1
  have c60897 := (publicBatchWrapper89_copies1903 a h).2.1
  have c60898 := (publicBatchWrapper89_copies1903 a h).2.2.1
  have c60899 := (publicBatchWrapper89_copies1903 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_14 := arithEq_of_rows h (row := 981) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_14
  simp only [← c60894, ← c60895, ← c60896] at e_981_14
  have e_981_15 := arithEq_of_rows h (row := 981) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_15
  simp only [← c60897, ← c60898, k2, ← c60899] at e_981_15
  have hr := e_981_15
  simp only [e_981_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7943)) := by
  have c60900 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.1
  have c60901 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.1
  have c60902 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.1
  have c60903 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.1
  have c60904 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.1
  have c60905 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_16 := arithEq_of_rows h (row := 981) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_16
  simp only [← c60900, ← c60901, ← c60902] at e_981_16
  have e_981_17 := arithEq_of_rows h (row := 981) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_17
  simp only [← c60903, ← c60904, k2, ← c60905] at e_981_17
  have hr := e_981_17
  simp only [e_981_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 981 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7944)) := by
  have c60906 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60907 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60908 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60909 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60910 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60911 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_981_18 := arithEq_of_rows h (row := 981) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_18
  simp only [← c60906, ← c60907, ← c60908] at e_981_18
  have e_981_19 := arithEq_of_rows h (row := 981) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_981_19
  simp only [← c60909, ← c60910, k2, ← c60911] at e_981_19
  have hr := e_981_19
  simp only [e_981_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7945)) := by
  have c60912 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60913 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60914 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60915 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60916 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60917 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_0 := arithEq_of_rows h (row := 982) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_0
  simp only [← c60912, ← c60913, ← c60914] at e_982_0
  have e_982_1 := arithEq_of_rows h (row := 982) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_1
  simp only [← c60915, ← c60916, k2, ← c60917] at e_982_1
  have hr := e_982_1
  simp only [e_982_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7946)) := by
  have c60918 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60919 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60920 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60921 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60922 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60923 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_2 := arithEq_of_rows h (row := 982) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_2
  simp only [← c60918, ← c60919, ← c60920] at e_982_2
  have e_982_3 := arithEq_of_rows h (row := 982) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_3
  simp only [← c60921, ← c60922, k2, ← c60923] at e_982_3
  have hr := e_982_3
  simp only [e_982_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7947)) := by
  have c60924 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60925 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60926 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60927 := (publicBatchWrapper89_copies1903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60928 := (publicBatchWrapper89_copies1904 a h).1
  have c60929 := (publicBatchWrapper89_copies1904 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_4 := arithEq_of_rows h (row := 982) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_4
  simp only [← c60924, ← c60925, ← c60926] at e_982_4
  have e_982_5 := arithEq_of_rows h (row := 982) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_5
  simp only [← c60927, ← c60928, k2, ← c60929] at e_982_5
  have hr := e_982_5
  simp only [e_982_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7948)) := by
  have c60930 := (publicBatchWrapper89_copies1904 a h).2.2.1
  have c60931 := (publicBatchWrapper89_copies1904 a h).2.2.2.1
  have c60932 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.1
  have c60933 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.1
  have c60934 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.1
  have c60935 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_6 := arithEq_of_rows h (row := 982) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_6
  simp only [← c60930, ← c60931, ← c60932] at e_982_6
  have e_982_7 := arithEq_of_rows h (row := 982) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_7
  simp only [← c60933, ← c60934, k2, ← c60935] at e_982_7
  have hr := e_982_7
  simp only [e_982_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7949)) := by
  have c60936 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.1
  have c60937 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.1
  have c60938 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60939 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c60940 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60941 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_8 := arithEq_of_rows h (row := 982) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_8
  simp only [← c60936, ← c60937, ← c60938] at e_982_8
  have e_982_9 := arithEq_of_rows h (row := 982) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_9
  simp only [← c60939, ← c60940, k2, ← c60941] at e_982_9
  have hr := e_982_9
  simp only [e_982_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7950)) := by
  have c60942 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60943 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60944 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60945 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60946 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60947 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_10 := arithEq_of_rows h (row := 982) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_10
  simp only [← c60942, ← c60943, ← c60944] at e_982_10
  have e_982_11 := arithEq_of_rows h (row := 982) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_11
  simp only [← c60945, ← c60946, k2, ← c60947] at e_982_11
  have hr := e_982_11
  simp only [e_982_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7951)) := by
  have c60948 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60949 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60950 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60951 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60952 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60953 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_12 := arithEq_of_rows h (row := 982) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_12
  simp only [← c60948, ← c60949, ← c60950] at e_982_12
  have e_982_13 := arithEq_of_rows h (row := 982) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_13
  simp only [← c60951, ← c60952, k2, ← c60953] at e_982_13
  have hr := e_982_13
  simp only [e_982_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7952)) := by
  have c60954 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60955 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60956 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60957 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60958 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60959 := (publicBatchWrapper89_copies1904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_14 := arithEq_of_rows h (row := 982) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_14
  simp only [← c60954, ← c60955, ← c60956] at e_982_14
  have e_982_15 := arithEq_of_rows h (row := 982) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_15
  simp only [← c60957, ← c60958, k2, ← c60959] at e_982_15
  have hr := e_982_15
  simp only [e_982_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7953)) := by
  have c60960 := (publicBatchWrapper89_copies1905 a h).1
  have c60961 := (publicBatchWrapper89_copies1905 a h).2.1
  have c60962 := (publicBatchWrapper89_copies1905 a h).2.2.1
  have c60963 := (publicBatchWrapper89_copies1905 a h).2.2.2.1
  have c60964 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.1
  have c60965 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_16 := arithEq_of_rows h (row := 982) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_16
  simp only [← c60960, ← c60961, ← c60962] at e_982_16
  have e_982_17 := arithEq_of_rows h (row := 982) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_17
  simp only [← c60963, ← c60964, k2, ← c60965] at e_982_17
  have hr := e_982_17
  simp only [e_982_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 982 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7954)) := by
  have c60966 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.1
  have c60967 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.1
  have c60968 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.1
  have c60969 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.1
  have c60970 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.1
  have c60971 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_982_18 := arithEq_of_rows h (row := 982) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_18
  simp only [← c60966, ← c60967, ← c60968] at e_982_18
  have e_982_19 := arithEq_of_rows h (row := 982) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_982_19
  simp only [← c60969, ← c60970, k2, ← c60971] at e_982_19
  have hr := e_982_19
  simp only [e_982_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7955)) := by
  have c60972 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60973 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60974 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60975 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60976 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60977 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_0 := arithEq_of_rows h (row := 983) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_0
  simp only [← c60972, ← c60973, ← c60974] at e_983_0
  have e_983_1 := arithEq_of_rows h (row := 983) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_1
  simp only [← c60975, ← c60976, k2, ← c60977] at e_983_1
  have hr := e_983_1
  simp only [e_983_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7956)) := by
  have c60978 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60979 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60980 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60981 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60982 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60983 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_2 := arithEq_of_rows h (row := 983) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_2
  simp only [← c60978, ← c60979, ← c60980] at e_983_2
  have e_983_3 := arithEq_of_rows h (row := 983) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_3
  simp only [← c60981, ← c60982, k2, ← c60983] at e_983_3
  have hr := e_983_3
  simp only [e_983_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7957)) := by
  have c60984 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60985 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60986 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60987 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60988 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60989 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_4 := arithEq_of_rows h (row := 983) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_4
  simp only [← c60984, ← c60985, ← c60986] at e_983_4
  have e_983_5 := arithEq_of_rows h (row := 983) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_5
  simp only [← c60987, ← c60988, k2, ← c60989] at e_983_5
  have hr := e_983_5
  simp only [e_983_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7958)) := by
  have c60990 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60991 := (publicBatchWrapper89_copies1905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c60992 := (publicBatchWrapper89_copies1906 a h).1
  have c60993 := (publicBatchWrapper89_copies1906 a h).2.1
  have c60994 := (publicBatchWrapper89_copies1906 a h).2.2.1
  have c60995 := (publicBatchWrapper89_copies1906 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_6 := arithEq_of_rows h (row := 983) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_6
  simp only [← c60990, ← c60991, ← c60992] at e_983_6
  have e_983_7 := arithEq_of_rows h (row := 983) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_7
  simp only [← c60993, ← c60994, k2, ← c60995] at e_983_7
  have hr := e_983_7
  simp only [e_983_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7959)) := by
  have c60996 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.1
  have c60997 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.1
  have c60998 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.1
  have c60999 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.1
  have c61000 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.1
  have c61001 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_8 := arithEq_of_rows h (row := 983) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_8
  simp only [← c60996, ← c60997, ← c60998] at e_983_8
  have e_983_9 := arithEq_of_rows h (row := 983) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_9
  simp only [← c60999, ← c61000, k2, ← c61001] at e_983_9
  have hr := e_983_9
  simp only [e_983_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7960)) := by
  have c61002 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61003 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61004 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61005 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61006 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61007 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_10 := arithEq_of_rows h (row := 983) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_10
  simp only [← c61002, ← c61003, ← c61004] at e_983_10
  have e_983_11 := arithEq_of_rows h (row := 983) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_11
  simp only [← c61005, ← c61006, k2, ← c61007] at e_983_11
  have hr := e_983_11
  simp only [e_983_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7961)) := by
  have c61008 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61009 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61010 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61011 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61012 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61013 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_12 := arithEq_of_rows h (row := 983) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_12
  simp only [← c61008, ← c61009, ← c61010] at e_983_12
  have e_983_13 := arithEq_of_rows h (row := 983) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_13
  simp only [← c61011, ← c61012, k2, ← c61013] at e_983_13
  have hr := e_983_13
  simp only [e_983_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7962)) := by
  have c61014 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61015 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61016 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61017 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61018 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61019 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_14 := arithEq_of_rows h (row := 983) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_14
  simp only [← c61014, ← c61015, ← c61016] at e_983_14
  have e_983_15 := arithEq_of_rows h (row := 983) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_15
  simp only [← c61017, ← c61018, k2, ← c61019] at e_983_15
  have hr := e_983_15
  simp only [e_983_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7963)) := by
  have c61020 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61021 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61022 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61023 := (publicBatchWrapper89_copies1906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61024 := (publicBatchWrapper89_copies1907 a h).1
  have c61025 := (publicBatchWrapper89_copies1907 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_16 := arithEq_of_rows h (row := 983) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_16
  simp only [← c61020, ← c61021, ← c61022] at e_983_16
  have e_983_17 := arithEq_of_rows h (row := 983) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_17
  simp only [← c61023, ← c61024, k2, ← c61025] at e_983_17
  have hr := e_983_17
  simp only [e_983_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 983 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7964)) := by
  have c61026 := (publicBatchWrapper89_copies1907 a h).2.2.1
  have c61027 := (publicBatchWrapper89_copies1907 a h).2.2.2.1
  have c61028 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.1
  have c61029 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.1
  have c61030 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.1
  have c61031 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_983_18 := arithEq_of_rows h (row := 983) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_18
  simp only [← c61026, ← c61027, ← c61028] at e_983_18
  have e_983_19 := arithEq_of_rows h (row := 983) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_983_19
  simp only [← c61029, ← c61030, k2, ← c61031] at e_983_19
  have hr := e_983_19
  simp only [e_983_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7965)) := by
  have c61032 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.1
  have c61033 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.1
  have c61034 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61035 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61036 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61037 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_0 := arithEq_of_rows h (row := 984) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_0
  simp only [← c61032, ← c61033, ← c61034] at e_984_0
  have e_984_1 := arithEq_of_rows h (row := 984) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_1
  simp only [← c61035, ← c61036, k2, ← c61037] at e_984_1
  have hr := e_984_1
  simp only [e_984_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7966)) := by
  have c61038 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61039 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61040 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61041 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61042 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61043 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_2 := arithEq_of_rows h (row := 984) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_2
  simp only [← c61038, ← c61039, ← c61040] at e_984_2
  have e_984_3 := arithEq_of_rows h (row := 984) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_3
  simp only [← c61041, ← c61042, k2, ← c61043] at e_984_3
  have hr := e_984_3
  simp only [e_984_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7967)) := by
  have c61044 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61045 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61046 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61047 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61048 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61049 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_4 := arithEq_of_rows h (row := 984) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_4
  simp only [← c61044, ← c61045, ← c61046] at e_984_4
  have e_984_5 := arithEq_of_rows h (row := 984) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_5
  simp only [← c61047, ← c61048, k2, ← c61049] at e_984_5
  have hr := e_984_5
  simp only [e_984_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7968)) := by
  have c61050 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61051 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61052 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61053 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61054 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61055 := (publicBatchWrapper89_copies1907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_6 := arithEq_of_rows h (row := 984) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_6
  simp only [← c61050, ← c61051, ← c61052] at e_984_6
  have e_984_7 := arithEq_of_rows h (row := 984) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_7
  simp only [← c61053, ← c61054, k2, ← c61055] at e_984_7
  have hr := e_984_7
  simp only [e_984_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7969)) := by
  have c61056 := (publicBatchWrapper89_copies1908 a h).1
  have c61057 := (publicBatchWrapper89_copies1908 a h).2.1
  have c61058 := (publicBatchWrapper89_copies1908 a h).2.2.1
  have c61059 := (publicBatchWrapper89_copies1908 a h).2.2.2.1
  have c61060 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.1
  have c61061 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_8 := arithEq_of_rows h (row := 984) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_8
  simp only [← c61056, ← c61057, ← c61058] at e_984_8
  have e_984_9 := arithEq_of_rows h (row := 984) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_9
  simp only [← c61059, ← c61060, k2, ← c61061] at e_984_9
  have hr := e_984_9
  simp only [e_984_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8010)) := by
  have c61062 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.1
  have c61063 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.1
  have c61064 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.1
  have c61065 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.1
  have c61066 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61067 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_10 := arithEq_of_rows h (row := 984) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_10
  simp only [← c61062, ← c61063, ← c61064] at e_984_10
  have e_984_11 := arithEq_of_rows h (row := 984) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_11
  simp only [← c61065, ← c61066, k2, ← c61067] at e_984_11
  have hr := e_984_11
  simp only [e_984_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8011)) := by
  have c61068 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61069 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61070 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61071 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61072 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61073 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_12 := arithEq_of_rows h (row := 984) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_12
  simp only [← c61068, ← c61069, ← c61070] at e_984_12
  have e_984_13 := arithEq_of_rows h (row := 984) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_13
  simp only [← c61071, ← c61072, k2, ← c61073] at e_984_13
  have hr := e_984_13
  simp only [e_984_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8012)) := by
  have c61074 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61075 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61076 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61077 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61078 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61079 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_14 := arithEq_of_rows h (row := 984) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_14
  simp only [← c61074, ← c61075, ← c61076] at e_984_14
  have e_984_15 := arithEq_of_rows h (row := 984) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_15
  simp only [← c61077, ← c61078, k2, ← c61079] at e_984_15
  have hr := e_984_15
  simp only [e_984_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8013)) := by
  have c61080 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61081 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61082 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61083 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61084 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61085 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_16 := arithEq_of_rows h (row := 984) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_16
  simp only [← c61080, ← c61081, ← c61082] at e_984_16
  have e_984_17 := arithEq_of_rows h (row := 984) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_17
  simp only [← c61083, ← c61084, k2, ← c61085] at e_984_17
  have hr := e_984_17
  simp only [e_984_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 984 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8014)) := by
  have c61086 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61087 := (publicBatchWrapper89_copies1908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61088 := (publicBatchWrapper89_copies1909 a h).1
  have c61089 := (publicBatchWrapper89_copies1909 a h).2.1
  have c61090 := (publicBatchWrapper89_copies1909 a h).2.2.1
  have c61091 := (publicBatchWrapper89_copies1909 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_984_18 := arithEq_of_rows h (row := 984) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_18
  simp only [← c61086, ← c61087, ← c61088] at e_984_18
  have e_984_19 := arithEq_of_rows h (row := 984) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_984_19
  simp only [← c61089, ← c61090, k2, ← c61091] at e_984_19
  have hr := e_984_19
  simp only [e_984_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8015)) := by
  have c61092 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.1
  have c61093 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.1
  have c61094 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.1
  have c61095 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.1
  have c61096 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.1
  have c61097 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_0 := arithEq_of_rows h (row := 985) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_0
  simp only [← c61092, ← c61093, ← c61094] at e_985_0
  have e_985_1 := arithEq_of_rows h (row := 985) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_1
  simp only [← c61095, ← c61096, k2, ← c61097] at e_985_1
  have hr := e_985_1
  simp only [e_985_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8016)) := by
  have c61098 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61099 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61100 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61101 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61102 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61103 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_2 := arithEq_of_rows h (row := 985) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_2
  simp only [← c61098, ← c61099, ← c61100] at e_985_2
  have e_985_3 := arithEq_of_rows h (row := 985) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_3
  simp only [← c61101, ← c61102, k2, ← c61103] at e_985_3
  have hr := e_985_3
  simp only [e_985_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8017)) := by
  have c61104 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61105 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61106 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61107 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61108 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61109 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_4 := arithEq_of_rows h (row := 985) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_4
  simp only [← c61104, ← c61105, ← c61106] at e_985_4
  have e_985_5 := arithEq_of_rows h (row := 985) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_5
  simp only [← c61107, ← c61108, k2, ← c61109] at e_985_5
  have hr := e_985_5
  simp only [e_985_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8018)) := by
  have c61110 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61111 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61112 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61113 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61114 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61115 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_6 := arithEq_of_rows h (row := 985) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_6
  simp only [← c61110, ← c61111, ← c61112] at e_985_6
  have e_985_7 := arithEq_of_rows h (row := 985) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_7
  simp only [← c61113, ← c61114, k2, ← c61115] at e_985_7
  have hr := e_985_7
  simp only [e_985_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8019)) := by
  have c61116 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61117 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61118 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61119 := (publicBatchWrapper89_copies1909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61120 := (publicBatchWrapper89_copies1910 a h).1
  have c61121 := (publicBatchWrapper89_copies1910 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_8 := arithEq_of_rows h (row := 985) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_8
  simp only [← c61116, ← c61117, ← c61118] at e_985_8
  have e_985_9 := arithEq_of_rows h (row := 985) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_9
  simp only [← c61119, ← c61120, k2, ← c61121] at e_985_9
  have hr := e_985_9
  simp only [e_985_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8020)) := by
  have c61122 := (publicBatchWrapper89_copies1910 a h).2.2.1
  have c61123 := (publicBatchWrapper89_copies1910 a h).2.2.2.1
  have c61124 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.1
  have c61125 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.1
  have c61126 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.1
  have c61127 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_10 := arithEq_of_rows h (row := 985) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_10
  simp only [← c61122, ← c61123, ← c61124] at e_985_10
  have e_985_11 := arithEq_of_rows h (row := 985) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_11
  simp only [← c61125, ← c61126, k2, ← c61127] at e_985_11
  have hr := e_985_11
  simp only [e_985_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8021)) := by
  have c61128 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.1
  have c61129 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.1
  have c61130 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61131 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61132 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61133 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_12 := arithEq_of_rows h (row := 985) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_12
  simp only [← c61128, ← c61129, ← c61130] at e_985_12
  have e_985_13 := arithEq_of_rows h (row := 985) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_13
  simp only [← c61131, ← c61132, k2, ← c61133] at e_985_13
  have hr := e_985_13
  simp only [e_985_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8022)) := by
  have c61134 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61135 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61136 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61137 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61138 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61139 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_14 := arithEq_of_rows h (row := 985) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_14
  simp only [← c61134, ← c61135, ← c61136] at e_985_14
  have e_985_15 := arithEq_of_rows h (row := 985) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_15
  simp only [← c61137, ← c61138, k2, ← c61139] at e_985_15
  have hr := e_985_15
  simp only [e_985_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8023)) := by
  have c61140 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61141 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61142 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61143 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61144 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61145 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_16 := arithEq_of_rows h (row := 985) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_16
  simp only [← c61140, ← c61141, ← c61142] at e_985_16
  have e_985_17 := arithEq_of_rows h (row := 985) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_17
  simp only [← c61143, ← c61144, k2, ← c61145] at e_985_17
  have hr := e_985_17
  simp only [e_985_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 985 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8024)) := by
  have c61146 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61147 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61148 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61149 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61150 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61151 := (publicBatchWrapper89_copies1910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_985_18 := arithEq_of_rows h (row := 985) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_18
  simp only [← c61146, ← c61147, ← c61148] at e_985_18
  have e_985_19 := arithEq_of_rows h (row := 985) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_985_19
  simp only [← c61149, ← c61150, k2, ← c61151] at e_985_19
  have hr := e_985_19
  simp only [e_985_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8025)) := by
  have c61152 := (publicBatchWrapper89_copies1911 a h).1
  have c61153 := (publicBatchWrapper89_copies1911 a h).2.1
  have c61154 := (publicBatchWrapper89_copies1911 a h).2.2.1
  have c61155 := (publicBatchWrapper89_copies1911 a h).2.2.2.1
  have c61156 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.1
  have c61157 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_0 := arithEq_of_rows h (row := 986) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_0
  simp only [← c61152, ← c61153, ← c61154] at e_986_0
  have e_986_1 := arithEq_of_rows h (row := 986) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_1
  simp only [← c61155, ← c61156, k2, ← c61157] at e_986_1
  have hr := e_986_1
  simp only [e_986_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8026)) := by
  have c61158 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.1
  have c61159 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.1
  have c61160 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.1
  have c61161 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.1
  have c61162 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61163 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_2 := arithEq_of_rows h (row := 986) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_2
  simp only [← c61158, ← c61159, ← c61160] at e_986_2
  have e_986_3 := arithEq_of_rows h (row := 986) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_3
  simp only [← c61161, ← c61162, k2, ← c61163] at e_986_3
  have hr := e_986_3
  simp only [e_986_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8027)) := by
  have c61164 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61165 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61166 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61167 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61168 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61169 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_4 := arithEq_of_rows h (row := 986) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_4
  simp only [← c61164, ← c61165, ← c61166] at e_986_4
  have e_986_5 := arithEq_of_rows h (row := 986) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_5
  simp only [← c61167, ← c61168, k2, ← c61169] at e_986_5
  have hr := e_986_5
  simp only [e_986_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8028)) := by
  have c61170 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61171 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61172 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61173 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61174 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61175 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_6 := arithEq_of_rows h (row := 986) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_6
  simp only [← c61170, ← c61171, ← c61172] at e_986_6
  have e_986_7 := arithEq_of_rows h (row := 986) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_7
  simp only [← c61173, ← c61174, k2, ← c61175] at e_986_7
  have hr := e_986_7
  simp only [e_986_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8029)) := by
  have c61176 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61177 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61178 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61179 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61180 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61181 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_8 := arithEq_of_rows h (row := 986) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_8
  simp only [← c61176, ← c61177, ← c61178] at e_986_8
  have e_986_9 := arithEq_of_rows h (row := 986) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_9
  simp only [← c61179, ← c61180, k2, ← c61181] at e_986_9
  have hr := e_986_9
  simp only [e_986_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8030)) := by
  have c61182 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61183 := (publicBatchWrapper89_copies1911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61184 := (publicBatchWrapper89_copies1912 a h).1
  have c61185 := (publicBatchWrapper89_copies1912 a h).2.1
  have c61186 := (publicBatchWrapper89_copies1912 a h).2.2.1
  have c61187 := (publicBatchWrapper89_copies1912 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_10 := arithEq_of_rows h (row := 986) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_10
  simp only [← c61182, ← c61183, ← c61184] at e_986_10
  have e_986_11 := arithEq_of_rows h (row := 986) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_11
  simp only [← c61185, ← c61186, k2, ← c61187] at e_986_11
  have hr := e_986_11
  simp only [e_986_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8031)) := by
  have c61188 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.1
  have c61189 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.1
  have c61190 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.1
  have c61191 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.1
  have c61192 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.1
  have c61193 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_12 := arithEq_of_rows h (row := 986) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_12
  simp only [← c61188, ← c61189, ← c61190] at e_986_12
  have e_986_13 := arithEq_of_rows h (row := 986) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_13
  simp only [← c61191, ← c61192, k2, ← c61193] at e_986_13
  have hr := e_986_13
  simp only [e_986_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8032)) := by
  have c61194 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61195 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61196 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61197 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61198 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61199 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_14 := arithEq_of_rows h (row := 986) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_14
  simp only [← c61194, ← c61195, ← c61196] at e_986_14
  have e_986_15 := arithEq_of_rows h (row := 986) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_15
  simp only [← c61197, ← c61198, k2, ← c61199] at e_986_15
  have hr := e_986_15
  simp only [e_986_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8033)) := by
  have c61200 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61201 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61202 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61203 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61204 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61205 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_16 := arithEq_of_rows h (row := 986) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_16
  simp only [← c61200, ← c61201, ← c61202] at e_986_16
  have e_986_17 := arithEq_of_rows h (row := 986) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_17
  simp only [← c61203, ← c61204, k2, ← c61205] at e_986_17
  have hr := e_986_17
  simp only [e_986_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 986 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8034)) := by
  have c61206 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61207 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61208 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61209 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61210 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61211 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_986_18 := arithEq_of_rows h (row := 986) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_18
  simp only [← c61206, ← c61207, ← c61208] at e_986_18
  have e_986_19 := arithEq_of_rows h (row := 986) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_986_19
  simp only [← c61209, ← c61210, k2, ← c61211] at e_986_19
  have hr := e_986_19
  simp only [e_986_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8035)) := by
  have c61212 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61213 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61214 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61215 := (publicBatchWrapper89_copies1912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61216 := (publicBatchWrapper89_copies1913 a h).1
  have c61217 := (publicBatchWrapper89_copies1913 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_0 := arithEq_of_rows h (row := 987) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_0
  simp only [← c61212, ← c61213, ← c61214] at e_987_0
  have e_987_1 := arithEq_of_rows h (row := 987) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_1
  simp only [← c61215, ← c61216, k2, ← c61217] at e_987_1
  have hr := e_987_1
  simp only [e_987_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8036)) := by
  have c61218 := (publicBatchWrapper89_copies1913 a h).2.2.1
  have c61219 := (publicBatchWrapper89_copies1913 a h).2.2.2.1
  have c61220 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.1
  have c61221 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.1
  have c61222 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.1
  have c61223 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_2 := arithEq_of_rows h (row := 987) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_2
  simp only [← c61218, ← c61219, ← c61220] at e_987_2
  have e_987_3 := arithEq_of_rows h (row := 987) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_3
  simp only [← c61221, ← c61222, k2, ← c61223] at e_987_3
  have hr := e_987_3
  simp only [e_987_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8037)) := by
  have c61224 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.1
  have c61225 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.1
  have c61226 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61227 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61228 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61229 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_4 := arithEq_of_rows h (row := 987) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_4
  simp only [← c61224, ← c61225, ← c61226] at e_987_4
  have e_987_5 := arithEq_of_rows h (row := 987) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_5
  simp only [← c61227, ← c61228, k2, ← c61229] at e_987_5
  have hr := e_987_5
  simp only [e_987_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8038)) := by
  have c61230 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61231 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61232 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61233 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61234 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61235 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_6 := arithEq_of_rows h (row := 987) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_6
  simp only [← c61230, ← c61231, ← c61232] at e_987_6
  have e_987_7 := arithEq_of_rows h (row := 987) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_7
  simp only [← c61233, ← c61234, k2, ← c61235] at e_987_7
  have hr := e_987_7
  simp only [e_987_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8039)) := by
  have c61236 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61237 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61238 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61239 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61240 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61241 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_8 := arithEq_of_rows h (row := 987) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_8
  simp only [← c61236, ← c61237, ← c61238] at e_987_8
  have e_987_9 := arithEq_of_rows h (row := 987) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_9
  simp only [← c61239, ← c61240, k2, ← c61241] at e_987_9
  have hr := e_987_9
  simp only [e_987_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8040)) := by
  have c61242 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61243 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61244 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61245 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61246 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61247 := (publicBatchWrapper89_copies1913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_10 := arithEq_of_rows h (row := 987) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_10
  simp only [← c61242, ← c61243, ← c61244] at e_987_10
  have e_987_11 := arithEq_of_rows h (row := 987) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_11
  simp only [← c61245, ← c61246, k2, ← c61247] at e_987_11
  have hr := e_987_11
  simp only [e_987_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8041)) := by
  have c61248 := (publicBatchWrapper89_copies1914 a h).1
  have c61249 := (publicBatchWrapper89_copies1914 a h).2.1
  have c61250 := (publicBatchWrapper89_copies1914 a h).2.2.1
  have c61251 := (publicBatchWrapper89_copies1914 a h).2.2.2.1
  have c61252 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.1
  have c61253 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_12 := arithEq_of_rows h (row := 987) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_12
  simp only [← c61248, ← c61249, ← c61250] at e_987_12
  have e_987_13 := arithEq_of_rows h (row := 987) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_13
  simp only [← c61251, ← c61252, k2, ← c61253] at e_987_13
  have hr := e_987_13
  simp only [e_987_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8042)) := by
  have c61254 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.1
  have c61255 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.1
  have c61256 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.1
  have c61257 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.1
  have c61258 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61259 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_14 := arithEq_of_rows h (row := 987) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_14
  simp only [← c61254, ← c61255, ← c61256] at e_987_14
  have e_987_15 := arithEq_of_rows h (row := 987) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_15
  simp only [← c61257, ← c61258, k2, ← c61259] at e_987_15
  have hr := e_987_15
  simp only [e_987_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8043)) := by
  have c61260 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61261 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61262 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61263 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61264 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61265 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_16 := arithEq_of_rows h (row := 987) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_16
  simp only [← c61260, ← c61261, ← c61262] at e_987_16
  have e_987_17 := arithEq_of_rows h (row := 987) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_17
  simp only [← c61263, ← c61264, k2, ← c61265] at e_987_17
  have hr := e_987_17
  simp only [e_987_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 987 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8044)) := by
  have c61266 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61267 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61268 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61269 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61270 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61271 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_987_18 := arithEq_of_rows h (row := 987) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_18
  simp only [← c61266, ← c61267, ← c61268] at e_987_18
  have e_987_19 := arithEq_of_rows h (row := 987) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_987_19
  simp only [← c61269, ← c61270, k2, ← c61271] at e_987_19
  have hr := e_987_19
  simp only [e_987_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8045)) := by
  have c61272 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61273 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61274 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61275 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61276 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61277 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_0 := arithEq_of_rows h (row := 988) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_0
  simp only [← c61272, ← c61273, ← c61274] at e_988_0
  have e_988_1 := arithEq_of_rows h (row := 988) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_1
  simp only [← c61275, ← c61276, k2, ← c61277] at e_988_1
  have hr := e_988_1
  simp only [e_988_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8046)) := by
  have c61278 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61279 := (publicBatchWrapper89_copies1914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61280 := (publicBatchWrapper89_copies1915 a h).1
  have c61281 := (publicBatchWrapper89_copies1915 a h).2.1
  have c61282 := (publicBatchWrapper89_copies1915 a h).2.2.1
  have c61283 := (publicBatchWrapper89_copies1915 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_2 := arithEq_of_rows h (row := 988) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_2
  simp only [← c61278, ← c61279, ← c61280] at e_988_2
  have e_988_3 := arithEq_of_rows h (row := 988) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_3
  simp only [← c61281, ← c61282, k2, ← c61283] at e_988_3
  have hr := e_988_3
  simp only [e_988_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8047)) := by
  have c61284 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.1
  have c61285 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.1
  have c61286 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.1
  have c61287 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.1
  have c61288 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.1
  have c61289 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_4 := arithEq_of_rows h (row := 988) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_4
  simp only [← c61284, ← c61285, ← c61286] at e_988_4
  have e_988_5 := arithEq_of_rows h (row := 988) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_5
  simp only [← c61287, ← c61288, k2, ← c61289] at e_988_5
  have hr := e_988_5
  simp only [e_988_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8048)) := by
  have c61290 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61291 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61292 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61293 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61294 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61295 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_6 := arithEq_of_rows h (row := 988) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_6
  simp only [← c61290, ← c61291, ← c61292] at e_988_6
  have e_988_7 := arithEq_of_rows h (row := 988) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_7
  simp only [← c61293, ← c61294, k2, ← c61295] at e_988_7
  have hr := e_988_7
  simp only [e_988_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8049)) := by
  have c61296 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61297 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61298 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61299 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61300 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61301 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_8 := arithEq_of_rows h (row := 988) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_8
  simp only [← c61296, ← c61297, ← c61298] at e_988_8
  have e_988_9 := arithEq_of_rows h (row := 988) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_9
  simp only [← c61299, ← c61300, k2, ← c61301] at e_988_9
  have hr := e_988_9
  simp only [e_988_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8050)) := by
  have c61302 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61303 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61304 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61305 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61306 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61307 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_10 := arithEq_of_rows h (row := 988) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_10
  simp only [← c61302, ← c61303, ← c61304] at e_988_10
  have e_988_11 := arithEq_of_rows h (row := 988) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_11
  simp only [← c61305, ← c61306, k2, ← c61307] at e_988_11
  have hr := e_988_11
  simp only [e_988_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8051)) := by
  have c61308 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61309 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61310 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61311 := (publicBatchWrapper89_copies1915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61312 := (publicBatchWrapper89_copies1916 a h).1
  have c61313 := (publicBatchWrapper89_copies1916 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_12 := arithEq_of_rows h (row := 988) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_12
  simp only [← c61308, ← c61309, ← c61310] at e_988_12
  have e_988_13 := arithEq_of_rows h (row := 988) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_13
  simp only [← c61311, ← c61312, k2, ← c61313] at e_988_13
  have hr := e_988_13
  simp only [e_988_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8052)) := by
  have c61314 := (publicBatchWrapper89_copies1916 a h).2.2.1
  have c61315 := (publicBatchWrapper89_copies1916 a h).2.2.2.1
  have c61316 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.1
  have c61317 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.1
  have c61318 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.1
  have c61319 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_14 := arithEq_of_rows h (row := 988) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_14
  simp only [← c61314, ← c61315, ← c61316] at e_988_14
  have e_988_15 := arithEq_of_rows h (row := 988) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_15
  simp only [← c61317, ← c61318, k2, ← c61319] at e_988_15
  have hr := e_988_15
  simp only [e_988_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8053)) := by
  have c61320 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.1
  have c61321 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.1
  have c61322 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61323 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61324 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61325 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_16 := arithEq_of_rows h (row := 988) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_16
  simp only [← c61320, ← c61321, ← c61322] at e_988_16
  have e_988_17 := arithEq_of_rows h (row := 988) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_17
  simp only [← c61323, ← c61324, k2, ← c61325] at e_988_17
  have hr := e_988_17
  simp only [e_988_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 988 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8054)) := by
  have c61326 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61327 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61328 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61329 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61330 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61331 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_988_18 := arithEq_of_rows h (row := 988) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_18
  simp only [← c61326, ← c61327, ← c61328] at e_988_18
  have e_988_19 := arithEq_of_rows h (row := 988) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_988_19
  simp only [← c61329, ← c61330, k2, ← c61331] at e_988_19
  have hr := e_988_19
  simp only [e_988_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8055)) := by
  have c61332 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61333 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61334 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61335 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61336 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61337 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_0 := arithEq_of_rows h (row := 989) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_0
  simp only [← c61332, ← c61333, ← c61334] at e_989_0
  have e_989_1 := arithEq_of_rows h (row := 989) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_1
  simp only [← c61335, ← c61336, k2, ← c61337] at e_989_1
  have hr := e_989_1
  simp only [e_989_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8056)) := by
  have c61338 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61339 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61340 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61341 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61342 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61343 := (publicBatchWrapper89_copies1916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_2 := arithEq_of_rows h (row := 989) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_2
  simp only [← c61338, ← c61339, ← c61340] at e_989_2
  have e_989_3 := arithEq_of_rows h (row := 989) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_3
  simp only [← c61341, ← c61342, k2, ← c61343] at e_989_3
  have hr := e_989_3
  simp only [e_989_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8057)) := by
  have c61344 := (publicBatchWrapper89_copies1917 a h).1
  have c61345 := (publicBatchWrapper89_copies1917 a h).2.1
  have c61346 := (publicBatchWrapper89_copies1917 a h).2.2.1
  have c61347 := (publicBatchWrapper89_copies1917 a h).2.2.2.1
  have c61348 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.1
  have c61349 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_4 := arithEq_of_rows h (row := 989) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_4
  simp only [← c61344, ← c61345, ← c61346] at e_989_4
  have e_989_5 := arithEq_of_rows h (row := 989) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_5
  simp only [← c61347, ← c61348, k2, ← c61349] at e_989_5
  have hr := e_989_5
  simp only [e_989_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8058)) := by
  have c61350 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.1
  have c61351 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.1
  have c61352 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.1
  have c61353 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.1
  have c61354 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61355 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_6 := arithEq_of_rows h (row := 989) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_6
  simp only [← c61350, ← c61351, ← c61352] at e_989_6
  have e_989_7 := arithEq_of_rows h (row := 989) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_7
  simp only [← c61353, ← c61354, k2, ← c61355] at e_989_7
  have hr := e_989_7
  simp only [e_989_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8059)) := by
  have c61356 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61357 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61358 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61359 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61360 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61361 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_8 := arithEq_of_rows h (row := 989) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_8
  simp only [← c61356, ← c61357, ← c61358] at e_989_8
  have e_989_9 := arithEq_of_rows h (row := 989) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_9
  simp only [← c61359, ← c61360, k2, ← c61361] at e_989_9
  have hr := e_989_9
  simp only [e_989_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8060)) := by
  have c61362 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61363 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61364 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61365 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61366 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61367 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_10 := arithEq_of_rows h (row := 989) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_10
  simp only [← c61362, ← c61363, ← c61364] at e_989_10
  have e_989_11 := arithEq_of_rows h (row := 989) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_11
  simp only [← c61365, ← c61366, k2, ← c61367] at e_989_11
  have hr := e_989_11
  simp only [e_989_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8061)) := by
  have c61368 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61369 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61370 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61371 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61372 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61373 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_12 := arithEq_of_rows h (row := 989) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_12
  simp only [← c61368, ← c61369, ← c61370] at e_989_12
  have e_989_13 := arithEq_of_rows h (row := 989) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_13
  simp only [← c61371, ← c61372, k2, ← c61373] at e_989_13
  have hr := e_989_13
  simp only [e_989_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8062)) := by
  have c61374 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61375 := (publicBatchWrapper89_copies1917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61376 := (publicBatchWrapper89_copies1918 a h).1
  have c61377 := (publicBatchWrapper89_copies1918 a h).2.1
  have c61378 := (publicBatchWrapper89_copies1918 a h).2.2.1
  have c61379 := (publicBatchWrapper89_copies1918 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_14 := arithEq_of_rows h (row := 989) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_14
  simp only [← c61374, ← c61375, ← c61376] at e_989_14
  have e_989_15 := arithEq_of_rows h (row := 989) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_15
  simp only [← c61377, ← c61378, k2, ← c61379] at e_989_15
  have hr := e_989_15
  simp only [e_989_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8063)) := by
  have c61380 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.1
  have c61381 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.1
  have c61382 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.1
  have c61383 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.1
  have c61384 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.1
  have c61385 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_16 := arithEq_of_rows h (row := 989) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_16
  simp only [← c61380, ← c61381, ← c61382] at e_989_16
  have e_989_17 := arithEq_of_rows h (row := 989) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_17
  simp only [← c61383, ← c61384, k2, ← c61385] at e_989_17
  have hr := e_989_17
  simp only [e_989_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 989 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8064)) := by
  have c61386 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61387 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61388 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61389 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61390 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61391 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_989_18 := arithEq_of_rows h (row := 989) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_18
  simp only [← c61386, ← c61387, ← c61388] at e_989_18
  have e_989_19 := arithEq_of_rows h (row := 989) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_989_19
  simp only [← c61389, ← c61390, k2, ← c61391] at e_989_19
  have hr := e_989_19
  simp only [e_989_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8065)) := by
  have c61392 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61393 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61394 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61395 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61396 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61397 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_0 := arithEq_of_rows h (row := 990) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_0
  simp only [← c61392, ← c61393, ← c61394] at e_990_0
  have e_990_1 := arithEq_of_rows h (row := 990) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_1
  simp only [← c61395, ← c61396, k2, ← c61397] at e_990_1
  have hr := e_990_1
  simp only [e_990_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8066)) := by
  have c61398 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61399 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61400 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61401 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61402 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61403 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_2 := arithEq_of_rows h (row := 990) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_2
  simp only [← c61398, ← c61399, ← c61400] at e_990_2
  have e_990_3 := arithEq_of_rows h (row := 990) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_3
  simp only [← c61401, ← c61402, k2, ← c61403] at e_990_3
  have hr := e_990_3
  simp only [e_990_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8067)) := by
  have c61404 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61405 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61406 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61407 := (publicBatchWrapper89_copies1918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61408 := (publicBatchWrapper89_copies1919 a h).1
  have c61409 := (publicBatchWrapper89_copies1919 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_4 := arithEq_of_rows h (row := 990) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_4
  simp only [← c61404, ← c61405, ← c61406] at e_990_4
  have e_990_5 := arithEq_of_rows h (row := 990) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_5
  simp only [← c61407, ← c61408, k2, ← c61409] at e_990_5
  have hr := e_990_5
  simp only [e_990_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8068)) := by
  have c61410 := (publicBatchWrapper89_copies1919 a h).2.2.1
  have c61411 := (publicBatchWrapper89_copies1919 a h).2.2.2.1
  have c61412 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.1
  have c61413 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.1
  have c61414 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.1
  have c61415 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_6 := arithEq_of_rows h (row := 990) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_6
  simp only [← c61410, ← c61411, ← c61412] at e_990_6
  have e_990_7 := arithEq_of_rows h (row := 990) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_7
  simp only [← c61413, ← c61414, k2, ← c61415] at e_990_7
  have hr := e_990_7
  simp only [e_990_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8069)) := by
  have c61416 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.1
  have c61417 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.1
  have c61418 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61419 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61420 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61421 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_8 := arithEq_of_rows h (row := 990) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_8
  simp only [← c61416, ← c61417, ← c61418] at e_990_8
  have e_990_9 := arithEq_of_rows h (row := 990) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_9
  simp only [← c61419, ← c61420, k2, ← c61421] at e_990_9
  have hr := e_990_9
  simp only [e_990_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8070)) := by
  have c61422 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61423 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61424 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61425 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61426 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61427 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_10 := arithEq_of_rows h (row := 990) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_10
  simp only [← c61422, ← c61423, ← c61424] at e_990_10
  have e_990_11 := arithEq_of_rows h (row := 990) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_11
  simp only [← c61425, ← c61426, k2, ← c61427] at e_990_11
  have hr := e_990_11
  simp only [e_990_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8071)) := by
  have c61428 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61429 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61430 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61431 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61432 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61433 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_12 := arithEq_of_rows h (row := 990) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_12
  simp only [← c61428, ← c61429, ← c61430] at e_990_12
  have e_990_13 := arithEq_of_rows h (row := 990) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_13
  simp only [← c61431, ← c61432, k2, ← c61433] at e_990_13
  have hr := e_990_13
  simp only [e_990_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8072)) := by
  have c61434 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61435 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61436 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61437 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61438 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61439 := (publicBatchWrapper89_copies1919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_14 := arithEq_of_rows h (row := 990) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_14
  simp only [← c61434, ← c61435, ← c61436] at e_990_14
  have e_990_15 := arithEq_of_rows h (row := 990) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_15
  simp only [← c61437, ← c61438, k2, ← c61439] at e_990_15
  have hr := e_990_15
  simp only [e_990_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8073)) := by
  have c61440 := (publicBatchWrapper89_copies1920 a h).1
  have c61441 := (publicBatchWrapper89_copies1920 a h).2.1
  have c61442 := (publicBatchWrapper89_copies1920 a h).2.2.1
  have c61443 := (publicBatchWrapper89_copies1920 a h).2.2.2.1
  have c61444 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.1
  have c61445 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_16 := arithEq_of_rows h (row := 990) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_16
  simp only [← c61440, ← c61441, ← c61442] at e_990_16
  have e_990_17 := arithEq_of_rows h (row := 990) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_17
  simp only [← c61443, ← c61444, k2, ← c61445] at e_990_17
  have hr := e_990_17
  simp only [e_990_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 990 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8074)) := by
  have c61446 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.1
  have c61447 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.1
  have c61448 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.1
  have c61449 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.1
  have c61450 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61451 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_990_18 := arithEq_of_rows h (row := 990) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_18
  simp only [← c61446, ← c61447, ← c61448] at e_990_18
  have e_990_19 := arithEq_of_rows h (row := 990) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_990_19
  simp only [← c61449, ← c61450, k2, ← c61451] at e_990_19
  have hr := e_990_19
  simp only [e_990_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8075)) := by
  have c61452 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61453 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61454 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61455 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61456 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61457 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_0 := arithEq_of_rows h (row := 991) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_0
  simp only [← c61452, ← c61453, ← c61454] at e_991_0
  have e_991_1 := arithEq_of_rows h (row := 991) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_1
  simp only [← c61455, ← c61456, k2, ← c61457] at e_991_1
  have hr := e_991_1
  simp only [e_991_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8076)) := by
  have c61458 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61459 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61460 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61461 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61462 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61463 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_2 := arithEq_of_rows h (row := 991) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_2
  simp only [← c61458, ← c61459, ← c61460] at e_991_2
  have e_991_3 := arithEq_of_rows h (row := 991) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_3
  simp only [← c61461, ← c61462, k2, ← c61463] at e_991_3
  have hr := e_991_3
  simp only [e_991_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8077)) := by
  have c61464 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61465 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61466 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61467 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61468 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61469 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_4 := arithEq_of_rows h (row := 991) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_4
  simp only [← c61464, ← c61465, ← c61466] at e_991_4
  have e_991_5 := arithEq_of_rows h (row := 991) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_5
  simp only [← c61467, ← c61468, k2, ← c61469] at e_991_5
  have hr := e_991_5
  simp only [e_991_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8078)) := by
  have c61470 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61471 := (publicBatchWrapper89_copies1920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61472 := (publicBatchWrapper89_copies1921 a h).1
  have c61473 := (publicBatchWrapper89_copies1921 a h).2.1
  have c61474 := (publicBatchWrapper89_copies1921 a h).2.2.1
  have c61475 := (publicBatchWrapper89_copies1921 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_6 := arithEq_of_rows h (row := 991) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_6
  simp only [← c61470, ← c61471, ← c61472] at e_991_6
  have e_991_7 := arithEq_of_rows h (row := 991) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_7
  simp only [← c61473, ← c61474, k2, ← c61475] at e_991_7
  have hr := e_991_7
  simp only [e_991_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8079)) := by
  have c61476 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.1
  have c61477 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.1
  have c61478 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.1
  have c61479 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.1
  have c61480 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.1
  have c61481 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_8 := arithEq_of_rows h (row := 991) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_8
  simp only [← c61476, ← c61477, ← c61478] at e_991_8
  have e_991_9 := arithEq_of_rows h (row := 991) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_9
  simp only [← c61479, ← c61480, k2, ← c61481] at e_991_9
  have hr := e_991_9
  simp only [e_991_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8080)) := by
  have c61482 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61483 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61484 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61485 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61486 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61487 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_10 := arithEq_of_rows h (row := 991) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_10
  simp only [← c61482, ← c61483, ← c61484] at e_991_10
  have e_991_11 := arithEq_of_rows h (row := 991) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_11
  simp only [← c61485, ← c61486, k2, ← c61487] at e_991_11
  have hr := e_991_11
  simp only [e_991_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8081)) := by
  have c61488 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61489 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61490 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61491 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61492 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61493 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_12 := arithEq_of_rows h (row := 991) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_12
  simp only [← c61488, ← c61489, ← c61490] at e_991_12
  have e_991_13 := arithEq_of_rows h (row := 991) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_13
  simp only [← c61491, ← c61492, k2, ← c61493] at e_991_13
  have hr := e_991_13
  simp only [e_991_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8082)) := by
  have c61494 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61495 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61496 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61497 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61498 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61499 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_14 := arithEq_of_rows h (row := 991) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_14
  simp only [← c61494, ← c61495, ← c61496] at e_991_14
  have e_991_15 := arithEq_of_rows h (row := 991) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_15
  simp only [← c61497, ← c61498, k2, ← c61499] at e_991_15
  have hr := e_991_15
  simp only [e_991_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8083)) := by
  have c61500 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61501 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61502 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61503 := (publicBatchWrapper89_copies1921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61504 := (publicBatchWrapper89_copies1922 a h).1
  have c61505 := (publicBatchWrapper89_copies1922 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_16 := arithEq_of_rows h (row := 991) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_16
  simp only [← c61500, ← c61501, ← c61502] at e_991_16
  have e_991_17 := arithEq_of_rows h (row := 991) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_17
  simp only [← c61503, ← c61504, k2, ← c61505] at e_991_17
  have hr := e_991_17
  simp only [e_991_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 991 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8084)) := by
  have c61506 := (publicBatchWrapper89_copies1922 a h).2.2.1
  have c61507 := (publicBatchWrapper89_copies1922 a h).2.2.2.1
  have c61508 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.1
  have c61509 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.1
  have c61510 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.1
  have c61511 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_991_18 := arithEq_of_rows h (row := 991) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_18
  simp only [← c61506, ← c61507, ← c61508] at e_991_18
  have e_991_19 := arithEq_of_rows h (row := 991) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_991_19
  simp only [← c61509, ← c61510, k2, ← c61511] at e_991_19
  have hr := e_991_19
  simp only [e_991_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8085)) := by
  have c61512 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.1
  have c61513 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.1
  have c61514 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61515 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61516 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61517 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_0 := arithEq_of_rows h (row := 992) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_0
  simp only [← c61512, ← c61513, ← c61514] at e_992_0
  have e_992_1 := arithEq_of_rows h (row := 992) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_1
  simp only [← c61515, ← c61516, k2, ← c61517] at e_992_1
  have hr := e_992_1
  simp only [e_992_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8086)) := by
  have c61518 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61519 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61520 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61521 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61522 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61523 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_2 := arithEq_of_rows h (row := 992) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_2
  simp only [← c61518, ← c61519, ← c61520] at e_992_2
  have e_992_3 := arithEq_of_rows h (row := 992) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_3
  simp only [← c61521, ← c61522, k2, ← c61523] at e_992_3
  have hr := e_992_3
  simp only [e_992_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8087)) := by
  have c61524 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61525 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61526 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61527 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61528 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61529 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_4 := arithEq_of_rows h (row := 992) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_4
  simp only [← c61524, ← c61525, ← c61526] at e_992_4
  have e_992_5 := arithEq_of_rows h (row := 992) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_5
  simp only [← c61527, ← c61528, k2, ← c61529] at e_992_5
  have hr := e_992_5
  simp only [e_992_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8088)) := by
  have c61530 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61531 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61532 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61533 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61534 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61535 := (publicBatchWrapper89_copies1922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_6 := arithEq_of_rows h (row := 992) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_6
  simp only [← c61530, ← c61531, ← c61532] at e_992_6
  have e_992_7 := arithEq_of_rows h (row := 992) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_7
  simp only [← c61533, ← c61534, k2, ← c61535] at e_992_7
  have hr := e_992_7
  simp only [e_992_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8089)) := by
  have c61536 := (publicBatchWrapper89_copies1923 a h).1
  have c61537 := (publicBatchWrapper89_copies1923 a h).2.1
  have c61538 := (publicBatchWrapper89_copies1923 a h).2.2.1
  have c61539 := (publicBatchWrapper89_copies1923 a h).2.2.2.1
  have c61540 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.1
  have c61541 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_8 := arithEq_of_rows h (row := 992) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_8
  simp only [← c61536, ← c61537, ← c61538] at e_992_8
  have e_992_9 := arithEq_of_rows h (row := 992) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_9
  simp only [← c61539, ← c61540, k2, ← c61541] at e_992_9
  have hr := e_992_9
  simp only [e_992_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8090)) := by
  have c61542 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.1
  have c61543 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.1
  have c61544 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.1
  have c61545 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.1
  have c61546 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61547 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_10 := arithEq_of_rows h (row := 992) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_10
  simp only [← c61542, ← c61543, ← c61544] at e_992_10
  have e_992_11 := arithEq_of_rows h (row := 992) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_11
  simp only [← c61545, ← c61546, k2, ← c61547] at e_992_11
  have hr := e_992_11
  simp only [e_992_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8091)) := by
  have c61548 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61549 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61550 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61551 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61552 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61553 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_12 := arithEq_of_rows h (row := 992) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_12
  simp only [← c61548, ← c61549, ← c61550] at e_992_12
  have e_992_13 := arithEq_of_rows h (row := 992) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_13
  simp only [← c61551, ← c61552, k2, ← c61553] at e_992_13
  have hr := e_992_13
  simp only [e_992_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8092)) := by
  have c61554 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61555 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61556 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61557 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61558 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61559 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_14 := arithEq_of_rows h (row := 992) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_14
  simp only [← c61554, ← c61555, ← c61556] at e_992_14
  have e_992_15 := arithEq_of_rows h (row := 992) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_15
  simp only [← c61557, ← c61558, k2, ← c61559] at e_992_15
  have hr := e_992_15
  simp only [e_992_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8093)) := by
  have c61560 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61561 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61562 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61563 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61564 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61565 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_16 := arithEq_of_rows h (row := 992) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_16
  simp only [← c61560, ← c61561, ← c61562] at e_992_16
  have e_992_17 := arithEq_of_rows h (row := 992) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_17
  simp only [← c61563, ← c61564, k2, ← c61565] at e_992_17
  have hr := e_992_17
  simp only [e_992_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 992 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8094)) := by
  have c61566 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61567 := (publicBatchWrapper89_copies1923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61568 := (publicBatchWrapper89_copies1924 a h).1
  have c61569 := (publicBatchWrapper89_copies1924 a h).2.1
  have c61570 := (publicBatchWrapper89_copies1924 a h).2.2.1
  have c61571 := (publicBatchWrapper89_copies1924 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_992_18 := arithEq_of_rows h (row := 992) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_18
  simp only [← c61566, ← c61567, ← c61568] at e_992_18
  have e_992_19 := arithEq_of_rows h (row := 992) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_992_19
  simp only [← c61569, ← c61570, k2, ← c61571] at e_992_19
  have hr := e_992_19
  simp only [e_992_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8095)) := by
  have c61572 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.1
  have c61573 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.1
  have c61574 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.1
  have c61575 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.1
  have c61576 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.1
  have c61577 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_0 := arithEq_of_rows h (row := 993) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_0
  simp only [← c61572, ← c61573, ← c61574] at e_993_0
  have e_993_1 := arithEq_of_rows h (row := 993) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_1
  simp only [← c61575, ← c61576, k2, ← c61577] at e_993_1
  have hr := e_993_1
  simp only [e_993_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8096)) := by
  have c61578 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61579 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61580 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61581 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61582 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61583 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_2 := arithEq_of_rows h (row := 993) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_2
  simp only [← c61578, ← c61579, ← c61580] at e_993_2
  have e_993_3 := arithEq_of_rows h (row := 993) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_3
  simp only [← c61581, ← c61582, k2, ← c61583] at e_993_3
  have hr := e_993_3
  simp only [e_993_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8097)) := by
  have c61584 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61585 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61586 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61587 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61588 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61589 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_4 := arithEq_of_rows h (row := 993) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_4
  simp only [← c61584, ← c61585, ← c61586] at e_993_4
  have e_993_5 := arithEq_of_rows h (row := 993) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_5
  simp only [← c61587, ← c61588, k2, ← c61589] at e_993_5
  have hr := e_993_5
  simp only [e_993_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8098)) := by
  have c61590 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61591 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61592 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61593 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61594 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61595 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_6 := arithEq_of_rows h (row := 993) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_6
  simp only [← c61590, ← c61591, ← c61592] at e_993_6
  have e_993_7 := arithEq_of_rows h (row := 993) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_7
  simp only [← c61593, ← c61594, k2, ← c61595] at e_993_7
  have hr := e_993_7
  simp only [e_993_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8099)) := by
  have c61596 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61597 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61598 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61599 := (publicBatchWrapper89_copies1924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61600 := (publicBatchWrapper89_copies1925 a h).1
  have c61601 := (publicBatchWrapper89_copies1925 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_8 := arithEq_of_rows h (row := 993) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_8
  simp only [← c61596, ← c61597, ← c61598] at e_993_8
  have e_993_9 := arithEq_of_rows h (row := 993) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_9
  simp only [← c61599, ← c61600, k2, ← c61601] at e_993_9
  have hr := e_993_9
  simp only [e_993_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8100)) := by
  have c61602 := (publicBatchWrapper89_copies1925 a h).2.2.1
  have c61603 := (publicBatchWrapper89_copies1925 a h).2.2.2.1
  have c61604 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.1
  have c61605 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.1
  have c61606 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.1
  have c61607 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_10 := arithEq_of_rows h (row := 993) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_10
  simp only [← c61602, ← c61603, ← c61604] at e_993_10
  have e_993_11 := arithEq_of_rows h (row := 993) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_11
  simp only [← c61605, ← c61606, k2, ← c61607] at e_993_11
  have hr := e_993_11
  simp only [e_993_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8101)) := by
  have c61608 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.1
  have c61609 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.1
  have c61610 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61611 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61612 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61613 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_12 := arithEq_of_rows h (row := 993) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_12
  simp only [← c61608, ← c61609, ← c61610] at e_993_12
  have e_993_13 := arithEq_of_rows h (row := 993) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_13
  simp only [← c61611, ← c61612, k2, ← c61613] at e_993_13
  have hr := e_993_13
  simp only [e_993_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8102)) := by
  have c61614 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61615 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61616 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61617 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61618 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61619 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_14 := arithEq_of_rows h (row := 993) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_14
  simp only [← c61614, ← c61615, ← c61616] at e_993_14
  have e_993_15 := arithEq_of_rows h (row := 993) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_15
  simp only [← c61617, ← c61618, k2, ← c61619] at e_993_15
  have hr := e_993_15
  simp only [e_993_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8103)) := by
  have c61620 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61621 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61622 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61623 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61624 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61625 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_16 := arithEq_of_rows h (row := 993) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_16
  simp only [← c61620, ← c61621, ← c61622] at e_993_16
  have e_993_17 := arithEq_of_rows h (row := 993) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_17
  simp only [← c61623, ← c61624, k2, ← c61625] at e_993_17
  have hr := e_993_17
  simp only [e_993_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 993 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8104)) := by
  have c61626 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61627 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61628 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61629 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61630 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61631 := (publicBatchWrapper89_copies1925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_993_18 := arithEq_of_rows h (row := 993) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_18
  simp only [← c61626, ← c61627, ← c61628] at e_993_18
  have e_993_19 := arithEq_of_rows h (row := 993) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_993_19
  simp only [← c61629, ← c61630, k2, ← c61631] at e_993_19
  have hr := e_993_19
  simp only [e_993_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8105)) := by
  have c61632 := (publicBatchWrapper89_copies1926 a h).1
  have c61633 := (publicBatchWrapper89_copies1926 a h).2.1
  have c61634 := (publicBatchWrapper89_copies1926 a h).2.2.1
  have c61635 := (publicBatchWrapper89_copies1926 a h).2.2.2.1
  have c61636 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.1
  have c61637 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_0 := arithEq_of_rows h (row := 994) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_0
  simp only [← c61632, ← c61633, ← c61634] at e_994_0
  have e_994_1 := arithEq_of_rows h (row := 994) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_1
  simp only [← c61635, ← c61636, k2, ← c61637] at e_994_1
  have hr := e_994_1
  simp only [e_994_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8106)) := by
  have c61638 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.1
  have c61639 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.1
  have c61640 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.1
  have c61641 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.1
  have c61642 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61643 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_2 := arithEq_of_rows h (row := 994) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_2
  simp only [← c61638, ← c61639, ← c61640] at e_994_2
  have e_994_3 := arithEq_of_rows h (row := 994) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_3
  simp only [← c61641, ← c61642, k2, ← c61643] at e_994_3
  have hr := e_994_3
  simp only [e_994_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8107)) := by
  have c61644 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61645 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61646 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61647 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61648 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61649 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_4 := arithEq_of_rows h (row := 994) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_4
  simp only [← c61644, ← c61645, ← c61646] at e_994_4
  have e_994_5 := arithEq_of_rows h (row := 994) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_5
  simp only [← c61647, ← c61648, k2, ← c61649] at e_994_5
  have hr := e_994_5
  simp only [e_994_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8108)) := by
  have c61650 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61651 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61652 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61653 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61654 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61655 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_6 := arithEq_of_rows h (row := 994) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_6
  simp only [← c61650, ← c61651, ← c61652] at e_994_6
  have e_994_7 := arithEq_of_rows h (row := 994) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_7
  simp only [← c61653, ← c61654, k2, ← c61655] at e_994_7
  have hr := e_994_7
  simp only [e_994_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8109)) := by
  have c61656 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61657 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61658 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61659 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61660 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61661 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_8 := arithEq_of_rows h (row := 994) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_8
  simp only [← c61656, ← c61657, ← c61658] at e_994_8
  have e_994_9 := arithEq_of_rows h (row := 994) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_9
  simp only [← c61659, ← c61660, k2, ← c61661] at e_994_9
  have hr := e_994_9
  simp only [e_994_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8110)) := by
  have c61662 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61663 := (publicBatchWrapper89_copies1926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61664 := (publicBatchWrapper89_copies1927 a h).1
  have c61665 := (publicBatchWrapper89_copies1927 a h).2.1
  have c61666 := (publicBatchWrapper89_copies1927 a h).2.2.1
  have c61667 := (publicBatchWrapper89_copies1927 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_10 := arithEq_of_rows h (row := 994) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_10
  simp only [← c61662, ← c61663, ← c61664] at e_994_10
  have e_994_11 := arithEq_of_rows h (row := 994) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_11
  simp only [← c61665, ← c61666, k2, ← c61667] at e_994_11
  have hr := e_994_11
  simp only [e_994_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8111)) := by
  have c61668 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.1
  have c61669 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.1
  have c61670 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.1
  have c61671 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.1
  have c61672 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.1
  have c61673 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_12 := arithEq_of_rows h (row := 994) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_12
  simp only [← c61668, ← c61669, ← c61670] at e_994_12
  have e_994_13 := arithEq_of_rows h (row := 994) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_13
  simp only [← c61671, ← c61672, k2, ← c61673] at e_994_13
  have hr := e_994_13
  simp only [e_994_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8112)) := by
  have c61674 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61675 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61676 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61677 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61678 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61679 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_14 := arithEq_of_rows h (row := 994) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_14
  simp only [← c61674, ← c61675, ← c61676] at e_994_14
  have e_994_15 := arithEq_of_rows h (row := 994) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_15
  simp only [← c61677, ← c61678, k2, ← c61679] at e_994_15
  have hr := e_994_15
  simp only [e_994_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8113)) := by
  have c61680 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61681 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61682 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61683 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61684 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61685 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_16 := arithEq_of_rows h (row := 994) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_16
  simp only [← c61680, ← c61681, ← c61682] at e_994_16
  have e_994_17 := arithEq_of_rows h (row := 994) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_17
  simp only [← c61683, ← c61684, k2, ← c61685] at e_994_17
  have hr := e_994_17
  simp only [e_994_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 994 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8114)) := by
  have c61686 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61687 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61688 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61689 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61690 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61691 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_994_18 := arithEq_of_rows h (row := 994) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_18
  simp only [← c61686, ← c61687, ← c61688] at e_994_18
  have e_994_19 := arithEq_of_rows h (row := 994) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_994_19
  simp only [← c61689, ← c61690, k2, ← c61691] at e_994_19
  have hr := e_994_19
  simp only [e_994_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8115)) := by
  have c61692 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61693 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61694 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61695 := (publicBatchWrapper89_copies1927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61696 := (publicBatchWrapper89_copies1928 a h).1
  have c61697 := (publicBatchWrapper89_copies1928 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_0 := arithEq_of_rows h (row := 995) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_0
  simp only [← c61692, ← c61693, ← c61694] at e_995_0
  have e_995_1 := arithEq_of_rows h (row := 995) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_1
  simp only [← c61695, ← c61696, k2, ← c61697] at e_995_1
  have hr := e_995_1
  simp only [e_995_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8116)) := by
  have c61698 := (publicBatchWrapper89_copies1928 a h).2.2.1
  have c61699 := (publicBatchWrapper89_copies1928 a h).2.2.2.1
  have c61700 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.1
  have c61701 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.1
  have c61702 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.1
  have c61703 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_2 := arithEq_of_rows h (row := 995) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_2
  simp only [← c61698, ← c61699, ← c61700] at e_995_2
  have e_995_3 := arithEq_of_rows h (row := 995) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_3
  simp only [← c61701, ← c61702, k2, ← c61703] at e_995_3
  have hr := e_995_3
  simp only [e_995_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8117)) := by
  have c61704 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.1
  have c61705 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.1
  have c61706 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61707 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61708 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61709 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_4 := arithEq_of_rows h (row := 995) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_4
  simp only [← c61704, ← c61705, ← c61706] at e_995_4
  have e_995_5 := arithEq_of_rows h (row := 995) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_5
  simp only [← c61707, ← c61708, k2, ← c61709] at e_995_5
  have hr := e_995_5
  simp only [e_995_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8118)) := by
  have c61710 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61711 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61712 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61713 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61714 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61715 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_6 := arithEq_of_rows h (row := 995) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_6
  simp only [← c61710, ← c61711, ← c61712] at e_995_6
  have e_995_7 := arithEq_of_rows h (row := 995) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_7
  simp only [← c61713, ← c61714, k2, ← c61715] at e_995_7
  have hr := e_995_7
  simp only [e_995_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8119)) := by
  have c61716 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61717 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61718 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61719 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61720 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61721 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_8 := arithEq_of_rows h (row := 995) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_8
  simp only [← c61716, ← c61717, ← c61718] at e_995_8
  have e_995_9 := arithEq_of_rows h (row := 995) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_9
  simp only [← c61719, ← c61720, k2, ← c61721] at e_995_9
  have hr := e_995_9
  simp only [e_995_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8120)) := by
  have c61722 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61723 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61724 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61725 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61726 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61727 := (publicBatchWrapper89_copies1928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_10 := arithEq_of_rows h (row := 995) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_10
  simp only [← c61722, ← c61723, ← c61724] at e_995_10
  have e_995_11 := arithEq_of_rows h (row := 995) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_11
  simp only [← c61725, ← c61726, k2, ← c61727] at e_995_11
  have hr := e_995_11
  simp only [e_995_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8121)) := by
  have c61728 := (publicBatchWrapper89_copies1929 a h).1
  have c61729 := (publicBatchWrapper89_copies1929 a h).2.1
  have c61730 := (publicBatchWrapper89_copies1929 a h).2.2.1
  have c61731 := (publicBatchWrapper89_copies1929 a h).2.2.2.1
  have c61732 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.1
  have c61733 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_12 := arithEq_of_rows h (row := 995) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_12
  simp only [← c61728, ← c61729, ← c61730] at e_995_12
  have e_995_13 := arithEq_of_rows h (row := 995) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_13
  simp only [← c61731, ← c61732, k2, ← c61733] at e_995_13
  have hr := e_995_13
  simp only [e_995_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8122)) := by
  have c61734 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.1
  have c61735 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.1
  have c61736 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.1
  have c61737 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.1
  have c61738 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61739 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_14 := arithEq_of_rows h (row := 995) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_14
  simp only [← c61734, ← c61735, ← c61736] at e_995_14
  have e_995_15 := arithEq_of_rows h (row := 995) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_15
  simp only [← c61737, ← c61738, k2, ← c61739] at e_995_15
  have hr := e_995_15
  simp only [e_995_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8123)) := by
  have c61740 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61741 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61742 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61743 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61744 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61745 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_16 := arithEq_of_rows h (row := 995) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_16
  simp only [← c61740, ← c61741, ← c61742] at e_995_16
  have e_995_17 := arithEq_of_rows h (row := 995) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_17
  simp only [← c61743, ← c61744, k2, ← c61745] at e_995_17
  have hr := e_995_17
  simp only [e_995_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 995 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8124)) := by
  have c61746 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61747 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61748 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61749 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61750 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61751 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_995_18 := arithEq_of_rows h (row := 995) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_18
  simp only [← c61746, ← c61747, ← c61748] at e_995_18
  have e_995_19 := arithEq_of_rows h (row := 995) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_995_19
  simp only [← c61749, ← c61750, k2, ← c61751] at e_995_19
  have hr := e_995_19
  simp only [e_995_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8125)) := by
  have c61752 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61753 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61754 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61755 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61756 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61757 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_0 := arithEq_of_rows h (row := 996) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_0
  simp only [← c61752, ← c61753, ← c61754] at e_996_0
  have e_996_1 := arithEq_of_rows h (row := 996) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_1
  simp only [← c61755, ← c61756, k2, ← c61757] at e_996_1
  have hr := e_996_1
  simp only [e_996_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8126)) := by
  have c61758 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61759 := (publicBatchWrapper89_copies1929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61760 := (publicBatchWrapper89_copies1930 a h).1
  have c61761 := (publicBatchWrapper89_copies1930 a h).2.1
  have c61762 := (publicBatchWrapper89_copies1930 a h).2.2.1
  have c61763 := (publicBatchWrapper89_copies1930 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_2 := arithEq_of_rows h (row := 996) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_2
  simp only [← c61758, ← c61759, ← c61760] at e_996_2
  have e_996_3 := arithEq_of_rows h (row := 996) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_3
  simp only [← c61761, ← c61762, k2, ← c61763] at e_996_3
  have hr := e_996_3
  simp only [e_996_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8127)) := by
  have c61764 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.1
  have c61765 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.1
  have c61766 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.1
  have c61767 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.1
  have c61768 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.1
  have c61769 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_4 := arithEq_of_rows h (row := 996) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_4
  simp only [← c61764, ← c61765, ← c61766] at e_996_4
  have e_996_5 := arithEq_of_rows h (row := 996) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_5
  simp only [← c61767, ← c61768, k2, ← c61769] at e_996_5
  have hr := e_996_5
  simp only [e_996_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8128)) := by
  have c61770 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61771 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61772 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61773 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61774 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61775 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_6 := arithEq_of_rows h (row := 996) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_6
  simp only [← c61770, ← c61771, ← c61772] at e_996_6
  have e_996_7 := arithEq_of_rows h (row := 996) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_7
  simp only [← c61773, ← c61774, k2, ← c61775] at e_996_7
  have hr := e_996_7
  simp only [e_996_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8129)) := by
  have c61776 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61777 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61778 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61779 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61780 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61781 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_8 := arithEq_of_rows h (row := 996) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_8
  simp only [← c61776, ← c61777, ← c61778] at e_996_8
  have e_996_9 := arithEq_of_rows h (row := 996) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_9
  simp only [← c61779, ← c61780, k2, ← c61781] at e_996_9
  have hr := e_996_9
  simp only [e_996_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8130)) := by
  have c61782 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61783 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61784 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61785 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61786 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61787 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_10 := arithEq_of_rows h (row := 996) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_10
  simp only [← c61782, ← c61783, ← c61784] at e_996_10
  have e_996_11 := arithEq_of_rows h (row := 996) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_11
  simp only [← c61785, ← c61786, k2, ← c61787] at e_996_11
  have hr := e_996_11
  simp only [e_996_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8131)) := by
  have c61788 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61789 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61790 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61791 := (publicBatchWrapper89_copies1930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61792 := (publicBatchWrapper89_copies1931 a h).1
  have c61793 := (publicBatchWrapper89_copies1931 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_12 := arithEq_of_rows h (row := 996) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_12
  simp only [← c61788, ← c61789, ← c61790] at e_996_12
  have e_996_13 := arithEq_of_rows h (row := 996) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_13
  simp only [← c61791, ← c61792, k2, ← c61793] at e_996_13
  have hr := e_996_13
  simp only [e_996_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8132)) := by
  have c61794 := (publicBatchWrapper89_copies1931 a h).2.2.1
  have c61795 := (publicBatchWrapper89_copies1931 a h).2.2.2.1
  have c61796 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.1
  have c61797 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.1
  have c61798 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.1
  have c61799 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_14 := arithEq_of_rows h (row := 996) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_14
  simp only [← c61794, ← c61795, ← c61796] at e_996_14
  have e_996_15 := arithEq_of_rows h (row := 996) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_15
  simp only [← c61797, ← c61798, k2, ← c61799] at e_996_15
  have hr := e_996_15
  simp only [e_996_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8133)) := by
  have c61800 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.1
  have c61801 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.1
  have c61802 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61803 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61804 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61805 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_16 := arithEq_of_rows h (row := 996) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_16
  simp only [← c61800, ← c61801, ← c61802] at e_996_16
  have e_996_17 := arithEq_of_rows h (row := 996) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_17
  simp only [← c61803, ← c61804, k2, ← c61805] at e_996_17
  have hr := e_996_17
  simp only [e_996_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 996 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8134)) := by
  have c61806 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61807 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61808 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61809 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61810 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61811 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_996_18 := arithEq_of_rows h (row := 996) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_18
  simp only [← c61806, ← c61807, ← c61808] at e_996_18
  have e_996_19 := arithEq_of_rows h (row := 996) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_996_19
  simp only [← c61809, ← c61810, k2, ← c61811] at e_996_19
  have hr := e_996_19
  simp only [e_996_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8135)) := by
  have c61812 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61813 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61814 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61815 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61816 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61817 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_0 := arithEq_of_rows h (row := 997) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_0
  simp only [← c61812, ← c61813, ← c61814] at e_997_0
  have e_997_1 := arithEq_of_rows h (row := 997) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_1
  simp only [← c61815, ← c61816, k2, ← c61817] at e_997_1
  have hr := e_997_1
  simp only [e_997_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8136)) := by
  have c61818 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61819 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61820 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61821 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61822 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61823 := (publicBatchWrapper89_copies1931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_2 := arithEq_of_rows h (row := 997) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_2
  simp only [← c61818, ← c61819, ← c61820] at e_997_2
  have e_997_3 := arithEq_of_rows h (row := 997) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_3
  simp only [← c61821, ← c61822, k2, ← c61823] at e_997_3
  have hr := e_997_3
  simp only [e_997_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8137)) := by
  have c61824 := (publicBatchWrapper89_copies1932 a h).1
  have c61825 := (publicBatchWrapper89_copies1932 a h).2.1
  have c61826 := (publicBatchWrapper89_copies1932 a h).2.2.1
  have c61827 := (publicBatchWrapper89_copies1932 a h).2.2.2.1
  have c61828 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.1
  have c61829 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_4 := arithEq_of_rows h (row := 997) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_4
  simp only [← c61824, ← c61825, ← c61826] at e_997_4
  have e_997_5 := arithEq_of_rows h (row := 997) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_5
  simp only [← c61827, ← c61828, k2, ← c61829] at e_997_5
  have hr := e_997_5
  simp only [e_997_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8138)) := by
  have c61830 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.1
  have c61831 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.1
  have c61832 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.1
  have c61833 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.1
  have c61834 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61835 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_6 := arithEq_of_rows h (row := 997) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_6
  simp only [← c61830, ← c61831, ← c61832] at e_997_6
  have e_997_7 := arithEq_of_rows h (row := 997) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_7
  simp only [← c61833, ← c61834, k2, ← c61835] at e_997_7
  have hr := e_997_7
  simp only [e_997_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8139)) := by
  have c61836 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61837 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61838 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61839 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61840 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61841 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_8 := arithEq_of_rows h (row := 997) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_8
  simp only [← c61836, ← c61837, ← c61838] at e_997_8
  have e_997_9 := arithEq_of_rows h (row := 997) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_9
  simp only [← c61839, ← c61840, k2, ← c61841] at e_997_9
  have hr := e_997_9
  simp only [e_997_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8140)) := by
  have c61842 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61843 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61844 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61845 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61846 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61847 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_10 := arithEq_of_rows h (row := 997) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_10
  simp only [← c61842, ← c61843, ← c61844] at e_997_10
  have e_997_11 := arithEq_of_rows h (row := 997) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_11
  simp only [← c61845, ← c61846, k2, ← c61847] at e_997_11
  have hr := e_997_11
  simp only [e_997_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8141)) := by
  have c61848 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61849 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61850 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61851 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61852 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61853 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_12 := arithEq_of_rows h (row := 997) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_12
  simp only [← c61848, ← c61849, ← c61850] at e_997_12
  have e_997_13 := arithEq_of_rows h (row := 997) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_13
  simp only [← c61851, ← c61852, k2, ← c61853] at e_997_13
  have hr := e_997_13
  simp only [e_997_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8142)) := by
  have c61854 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61855 := (publicBatchWrapper89_copies1932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61856 := (publicBatchWrapper89_copies1933 a h).1
  have c61857 := (publicBatchWrapper89_copies1933 a h).2.1
  have c61858 := (publicBatchWrapper89_copies1933 a h).2.2.1
  have c61859 := (publicBatchWrapper89_copies1933 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_14 := arithEq_of_rows h (row := 997) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_14
  simp only [← c61854, ← c61855, ← c61856] at e_997_14
  have e_997_15 := arithEq_of_rows h (row := 997) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_15
  simp only [← c61857, ← c61858, k2, ← c61859] at e_997_15
  have hr := e_997_15
  simp only [e_997_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8143)) := by
  have c61860 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.1
  have c61861 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.1
  have c61862 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.1
  have c61863 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.1
  have c61864 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.1
  have c61865 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_16 := arithEq_of_rows h (row := 997) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_16
  simp only [← c61860, ← c61861, ← c61862] at e_997_16
  have e_997_17 := arithEq_of_rows h (row := 997) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_17
  simp only [← c61863, ← c61864, k2, ← c61865] at e_997_17
  have hr := e_997_17
  simp only [e_997_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 997 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8144)) := by
  have c61866 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61867 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61868 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61869 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61870 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61871 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_997_18 := arithEq_of_rows h (row := 997) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_18
  simp only [← c61866, ← c61867, ← c61868] at e_997_18
  have e_997_19 := arithEq_of_rows h (row := 997) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_997_19
  simp only [← c61869, ← c61870, k2, ← c61871] at e_997_19
  have hr := e_997_19
  simp only [e_997_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8145)) := by
  have c61872 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61873 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61874 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61875 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61876 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61877 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_0 := arithEq_of_rows h (row := 998) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_0
  simp only [← c61872, ← c61873, ← c61874] at e_998_0
  have e_998_1 := arithEq_of_rows h (row := 998) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_1
  simp only [← c61875, ← c61876, k2, ← c61877] at e_998_1
  have hr := e_998_1
  simp only [e_998_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8146)) := by
  have c61878 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61879 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61880 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61881 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61882 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61883 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_2 := arithEq_of_rows h (row := 998) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_2
  simp only [← c61878, ← c61879, ← c61880] at e_998_2
  have e_998_3 := arithEq_of_rows h (row := 998) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_3
  simp only [← c61881, ← c61882, k2, ← c61883] at e_998_3
  have hr := e_998_3
  simp only [e_998_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8147)) := by
  have c61884 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61885 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61886 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61887 := (publicBatchWrapper89_copies1933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61888 := (publicBatchWrapper89_copies1934 a h).1
  have c61889 := (publicBatchWrapper89_copies1934 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_4 := arithEq_of_rows h (row := 998) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_4
  simp only [← c61884, ← c61885, ← c61886] at e_998_4
  have e_998_5 := arithEq_of_rows h (row := 998) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_5
  simp only [← c61887, ← c61888, k2, ← c61889] at e_998_5
  have hr := e_998_5
  simp only [e_998_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8148)) := by
  have c61890 := (publicBatchWrapper89_copies1934 a h).2.2.1
  have c61891 := (publicBatchWrapper89_copies1934 a h).2.2.2.1
  have c61892 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.1
  have c61893 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.1
  have c61894 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.1
  have c61895 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_6 := arithEq_of_rows h (row := 998) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_6
  simp only [← c61890, ← c61891, ← c61892] at e_998_6
  have e_998_7 := arithEq_of_rows h (row := 998) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_7
  simp only [← c61893, ← c61894, k2, ← c61895] at e_998_7
  have hr := e_998_7
  simp only [e_998_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8149)) := by
  have c61896 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.1
  have c61897 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.1
  have c61898 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61899 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61900 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61901 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_8 := arithEq_of_rows h (row := 998) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_8
  simp only [← c61896, ← c61897, ← c61898] at e_998_8
  have e_998_9 := arithEq_of_rows h (row := 998) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_9
  simp only [← c61899, ← c61900, k2, ← c61901] at e_998_9
  have hr := e_998_9
  simp only [e_998_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8150)) := by
  have c61902 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61903 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61904 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61905 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61906 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61907 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_10 := arithEq_of_rows h (row := 998) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_10
  simp only [← c61902, ← c61903, ← c61904] at e_998_10
  have e_998_11 := arithEq_of_rows h (row := 998) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_11
  simp only [← c61905, ← c61906, k2, ← c61907] at e_998_11
  have hr := e_998_11
  simp only [e_998_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8151)) := by
  have c61908 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61909 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61910 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61911 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61912 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61913 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_12 := arithEq_of_rows h (row := 998) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_12
  simp only [← c61908, ← c61909, ← c61910] at e_998_12
  have e_998_13 := arithEq_of_rows h (row := 998) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_13
  simp only [← c61911, ← c61912, k2, ← c61913] at e_998_13
  have hr := e_998_13
  simp only [e_998_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8152)) := by
  have c61914 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61915 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61916 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61917 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61918 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61919 := (publicBatchWrapper89_copies1934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_14 := arithEq_of_rows h (row := 998) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_14
  simp only [← c61914, ← c61915, ← c61916] at e_998_14
  have e_998_15 := arithEq_of_rows h (row := 998) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_15
  simp only [← c61917, ← c61918, k2, ← c61919] at e_998_15
  have hr := e_998_15
  simp only [e_998_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8153)) := by
  have c61920 := (publicBatchWrapper89_copies1935 a h).1
  have c61921 := (publicBatchWrapper89_copies1935 a h).2.1
  have c61922 := (publicBatchWrapper89_copies1935 a h).2.2.1
  have c61923 := (publicBatchWrapper89_copies1935 a h).2.2.2.1
  have c61924 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.1
  have c61925 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_16 := arithEq_of_rows h (row := 998) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_16
  simp only [← c61920, ← c61921, ← c61922] at e_998_16
  have e_998_17 := arithEq_of_rows h (row := 998) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_17
  simp only [← c61923, ← c61924, k2, ← c61925] at e_998_17
  have hr := e_998_17
  simp only [e_998_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 998 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8154)) := by
  have c61926 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.1
  have c61927 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.1
  have c61928 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.1
  have c61929 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.1
  have c61930 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61931 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_998_18 := arithEq_of_rows h (row := 998) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_18
  simp only [← c61926, ← c61927, ← c61928] at e_998_18
  have e_998_19 := arithEq_of_rows h (row := 998) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_998_19
  simp only [← c61929, ← c61930, k2, ← c61931] at e_998_19
  have hr := e_998_19
  simp only [e_998_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8155)) := by
  have c61932 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61933 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61934 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61935 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61936 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61937 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_0 := arithEq_of_rows h (row := 999) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_0
  simp only [← c61932, ← c61933, ← c61934] at e_999_0
  have e_999_1 := arithEq_of_rows h (row := 999) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_1
  simp only [← c61935, ← c61936, k2, ← c61937] at e_999_1
  have hr := e_999_1
  simp only [e_999_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8156)) := by
  have c61938 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61939 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61940 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61941 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61942 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61943 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_2 := arithEq_of_rows h (row := 999) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_2
  simp only [← c61938, ← c61939, ← c61940] at e_999_2
  have e_999_3 := arithEq_of_rows h (row := 999) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_3
  simp only [← c61941, ← c61942, k2, ← c61943] at e_999_3
  have hr := e_999_3
  simp only [e_999_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8157)) := by
  have c61944 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61945 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61946 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61947 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61948 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61949 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_4 := arithEq_of_rows h (row := 999) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_4
  simp only [← c61944, ← c61945, ← c61946] at e_999_4
  have e_999_5 := arithEq_of_rows h (row := 999) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_5
  simp only [← c61947, ← c61948, k2, ← c61949] at e_999_5
  have hr := e_999_5
  simp only [e_999_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8158)) := by
  have c61950 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61951 := (publicBatchWrapper89_copies1935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61952 := (publicBatchWrapper89_copies1936 a h).1
  have c61953 := (publicBatchWrapper89_copies1936 a h).2.1
  have c61954 := (publicBatchWrapper89_copies1936 a h).2.2.1
  have c61955 := (publicBatchWrapper89_copies1936 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_6 := arithEq_of_rows h (row := 999) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_6
  simp only [← c61950, ← c61951, ← c61952] at e_999_6
  have e_999_7 := arithEq_of_rows h (row := 999) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_7
  simp only [← c61953, ← c61954, k2, ← c61955] at e_999_7
  have hr := e_999_7
  simp only [e_999_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8159)) := by
  have c61956 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.1
  have c61957 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.1
  have c61958 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.1
  have c61959 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.1
  have c61960 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.1
  have c61961 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_8 := arithEq_of_rows h (row := 999) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_8
  simp only [← c61956, ← c61957, ← c61958] at e_999_8
  have e_999_9 := arithEq_of_rows h (row := 999) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_9
  simp only [← c61959, ← c61960, k2, ← c61961] at e_999_9
  have hr := e_999_9
  simp only [e_999_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8160)) := by
  have c61962 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61963 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61964 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61965 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61966 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61967 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_10 := arithEq_of_rows h (row := 999) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_10
  simp only [← c61962, ← c61963, ← c61964] at e_999_10
  have e_999_11 := arithEq_of_rows h (row := 999) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_11
  simp only [← c61965, ← c61966, k2, ← c61967] at e_999_11
  have hr := e_999_11
  simp only [e_999_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8161)) := by
  have c61968 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61969 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61970 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61971 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61972 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61973 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_12 := arithEq_of_rows h (row := 999) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_12
  simp only [← c61968, ← c61969, ← c61970] at e_999_12
  have e_999_13 := arithEq_of_rows h (row := 999) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_13
  simp only [← c61971, ← c61972, k2, ← c61973] at e_999_13
  have hr := e_999_13
  simp only [e_999_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8162)) := by
  have c61974 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61975 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61976 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61977 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61978 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61979 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_14 := arithEq_of_rows h (row := 999) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_14
  simp only [← c61974, ← c61975, ← c61976] at e_999_14
  have e_999_15 := arithEq_of_rows h (row := 999) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_15
  simp only [← c61977, ← c61978, k2, ← c61979] at e_999_15
  have hr := e_999_15
  simp only [e_999_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8163)) := by
  have c61980 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61981 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61982 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61983 := (publicBatchWrapper89_copies1936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c61984 := (publicBatchWrapper89_copies1937 a h).1
  have c61985 := (publicBatchWrapper89_copies1937 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_16 := arithEq_of_rows h (row := 999) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_16
  simp only [← c61980, ← c61981, ← c61982] at e_999_16
  have e_999_17 := arithEq_of_rows h (row := 999) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_17
  simp only [← c61983, ← c61984, k2, ← c61985] at e_999_17
  have hr := e_999_17
  simp only [e_999_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 999 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8164)) := by
  have c61986 := (publicBatchWrapper89_copies1937 a h).2.2.1
  have c61987 := (publicBatchWrapper89_copies1937 a h).2.2.2.1
  have c61988 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.1
  have c61989 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.1
  have c61990 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.1
  have c61991 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_999_18 := arithEq_of_rows h (row := 999) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_18
  simp only [← c61986, ← c61987, ← c61988] at e_999_18
  have e_999_19 := arithEq_of_rows h (row := 999) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_999_19
  simp only [← c61989, ← c61990, k2, ← c61991] at e_999_19
  have hr := e_999_19
  simp only [e_999_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8165)) := by
  have c61992 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.1
  have c61993 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.1
  have c61994 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.1
  have c61995 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c61996 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61997 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_0 := arithEq_of_rows h (row := 1000) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_0
  simp only [← c61992, ← c61993, ← c61994] at e_1000_0
  have e_1000_1 := arithEq_of_rows h (row := 1000) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_1
  simp only [← c61995, ← c61996, k2, ← c61997] at e_1000_1
  have hr := e_1000_1
  simp only [e_1000_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8166)) := by
  have c61998 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61999 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62000 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62001 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62002 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62003 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_2 := arithEq_of_rows h (row := 1000) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_2
  simp only [← c61998, ← c61999, ← c62000] at e_1000_2
  have e_1000_3 := arithEq_of_rows h (row := 1000) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_3
  simp only [← c62001, ← c62002, k2, ← c62003] at e_1000_3
  have hr := e_1000_3
  simp only [e_1000_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8167)) := by
  have c62004 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62005 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62006 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62007 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62008 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62009 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_4 := arithEq_of_rows h (row := 1000) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_4
  simp only [← c62004, ← c62005, ← c62006] at e_1000_4
  have e_1000_5 := arithEq_of_rows h (row := 1000) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_5
  simp only [← c62007, ← c62008, k2, ← c62009] at e_1000_5
  have hr := e_1000_5
  simp only [e_1000_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8168)) := by
  have c62010 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62011 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62012 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62013 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62014 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62015 := (publicBatchWrapper89_copies1937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_6 := arithEq_of_rows h (row := 1000) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_6
  simp only [← c62010, ← c62011, ← c62012] at e_1000_6
  have e_1000_7 := arithEq_of_rows h (row := 1000) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_7
  simp only [← c62013, ← c62014, k2, ← c62015] at e_1000_7
  have hr := e_1000_7
  simp only [e_1000_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8169)) := by
  have c62016 := (publicBatchWrapper89_copies1938 a h).1
  have c62017 := (publicBatchWrapper89_copies1938 a h).2.1
  have c62018 := (publicBatchWrapper89_copies1938 a h).2.2.1
  have c62019 := (publicBatchWrapper89_copies1938 a h).2.2.2.1
  have c62020 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.1
  have c62021 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_8 := arithEq_of_rows h (row := 1000) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_8
  simp only [← c62016, ← c62017, ← c62018] at e_1000_8
  have e_1000_9 := arithEq_of_rows h (row := 1000) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_9
  simp only [← c62019, ← c62020, k2, ← c62021] at e_1000_9
  have hr := e_1000_9
  simp only [e_1000_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8210)) := by
  have c62022 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.1
  have c62023 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.1
  have c62024 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.1
  have c62025 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.1
  have c62026 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62027 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_10 := arithEq_of_rows h (row := 1000) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_10
  simp only [← c62022, ← c62023, ← c62024] at e_1000_10
  have e_1000_11 := arithEq_of_rows h (row := 1000) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_11
  simp only [← c62025, ← c62026, k2, ← c62027] at e_1000_11
  have hr := e_1000_11
  simp only [e_1000_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8211)) := by
  have c62028 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62029 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62030 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62031 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62032 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62033 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_12 := arithEq_of_rows h (row := 1000) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_12
  simp only [← c62028, ← c62029, ← c62030] at e_1000_12
  have e_1000_13 := arithEq_of_rows h (row := 1000) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_13
  simp only [← c62031, ← c62032, k2, ← c62033] at e_1000_13
  have hr := e_1000_13
  simp only [e_1000_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8212)) := by
  have c62034 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62035 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62036 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62037 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62038 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62039 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_14 := arithEq_of_rows h (row := 1000) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_14
  simp only [← c62034, ← c62035, ← c62036] at e_1000_14
  have e_1000_15 := arithEq_of_rows h (row := 1000) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_15
  simp only [← c62037, ← c62038, k2, ← c62039] at e_1000_15
  have hr := e_1000_15
  simp only [e_1000_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8213)) := by
  have c62040 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62041 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62042 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62043 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62044 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62045 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_16 := arithEq_of_rows h (row := 1000) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_16
  simp only [← c62040, ← c62041, ← c62042] at e_1000_16
  have e_1000_17 := arithEq_of_rows h (row := 1000) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_17
  simp only [← c62043, ← c62044, k2, ← c62045] at e_1000_17
  have hr := e_1000_17
  simp only [e_1000_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1000 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8214)) := by
  have c62046 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62047 := (publicBatchWrapper89_copies1938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62048 := (publicBatchWrapper89_copies1939 a h).1
  have c62049 := (publicBatchWrapper89_copies1939 a h).2.1
  have c62050 := (publicBatchWrapper89_copies1939 a h).2.2.1
  have c62051 := (publicBatchWrapper89_copies1939 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1000_18 := arithEq_of_rows h (row := 1000) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_18
  simp only [← c62046, ← c62047, ← c62048] at e_1000_18
  have e_1000_19 := arithEq_of_rows h (row := 1000) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1000_19
  simp only [← c62049, ← c62050, k2, ← c62051] at e_1000_19
  have hr := e_1000_19
  simp only [e_1000_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8215)) := by
  have c62052 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.1
  have c62053 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.1
  have c62054 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.1
  have c62055 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.1
  have c62056 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.1
  have c62057 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_0 := arithEq_of_rows h (row := 1001) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_0
  simp only [← c62052, ← c62053, ← c62054] at e_1001_0
  have e_1001_1 := arithEq_of_rows h (row := 1001) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_1
  simp only [← c62055, ← c62056, k2, ← c62057] at e_1001_1
  have hr := e_1001_1
  simp only [e_1001_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8216)) := by
  have c62058 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62059 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62060 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62061 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62062 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62063 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_2 := arithEq_of_rows h (row := 1001) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_2
  simp only [← c62058, ← c62059, ← c62060] at e_1001_2
  have e_1001_3 := arithEq_of_rows h (row := 1001) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_3
  simp only [← c62061, ← c62062, k2, ← c62063] at e_1001_3
  have hr := e_1001_3
  simp only [e_1001_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8217)) := by
  have c62064 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62065 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62066 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62067 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62068 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62069 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_4 := arithEq_of_rows h (row := 1001) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_4
  simp only [← c62064, ← c62065, ← c62066] at e_1001_4
  have e_1001_5 := arithEq_of_rows h (row := 1001) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_5
  simp only [← c62067, ← c62068, k2, ← c62069] at e_1001_5
  have hr := e_1001_5
  simp only [e_1001_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8218)) := by
  have c62070 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62071 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62072 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62073 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62074 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62075 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_6 := arithEq_of_rows h (row := 1001) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_6
  simp only [← c62070, ← c62071, ← c62072] at e_1001_6
  have e_1001_7 := arithEq_of_rows h (row := 1001) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_7
  simp only [← c62073, ← c62074, k2, ← c62075] at e_1001_7
  have hr := e_1001_7
  simp only [e_1001_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8219)) := by
  have c62076 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62077 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62078 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62079 := (publicBatchWrapper89_copies1939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62080 := (publicBatchWrapper89_copies1940 a h).1
  have c62081 := (publicBatchWrapper89_copies1940 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_8 := arithEq_of_rows h (row := 1001) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_8
  simp only [← c62076, ← c62077, ← c62078] at e_1001_8
  have e_1001_9 := arithEq_of_rows h (row := 1001) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_9
  simp only [← c62079, ← c62080, k2, ← c62081] at e_1001_9
  have hr := e_1001_9
  simp only [e_1001_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8220)) := by
  have c62082 := (publicBatchWrapper89_copies1940 a h).2.2.1
  have c62083 := (publicBatchWrapper89_copies1940 a h).2.2.2.1
  have c62084 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.1
  have c62085 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.1
  have c62086 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.1
  have c62087 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_10 := arithEq_of_rows h (row := 1001) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_10
  simp only [← c62082, ← c62083, ← c62084] at e_1001_10
  have e_1001_11 := arithEq_of_rows h (row := 1001) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_11
  simp only [← c62085, ← c62086, k2, ← c62087] at e_1001_11
  have hr := e_1001_11
  simp only [e_1001_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8221)) := by
  have c62088 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.1
  have c62089 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.1
  have c62090 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62091 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62092 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62093 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_12 := arithEq_of_rows h (row := 1001) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_12
  simp only [← c62088, ← c62089, ← c62090] at e_1001_12
  have e_1001_13 := arithEq_of_rows h (row := 1001) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_13
  simp only [← c62091, ← c62092, k2, ← c62093] at e_1001_13
  have hr := e_1001_13
  simp only [e_1001_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8222)) := by
  have c62094 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62095 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62096 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62097 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62098 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62099 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_14 := arithEq_of_rows h (row := 1001) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_14
  simp only [← c62094, ← c62095, ← c62096] at e_1001_14
  have e_1001_15 := arithEq_of_rows h (row := 1001) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_15
  simp only [← c62097, ← c62098, k2, ← c62099] at e_1001_15
  have hr := e_1001_15
  simp only [e_1001_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8223)) := by
  have c62100 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62101 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62102 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62103 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62104 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62105 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_16 := arithEq_of_rows h (row := 1001) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_16
  simp only [← c62100, ← c62101, ← c62102] at e_1001_16
  have e_1001_17 := arithEq_of_rows h (row := 1001) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_17
  simp only [← c62103, ← c62104, k2, ← c62105] at e_1001_17
  have hr := e_1001_17
  simp only [e_1001_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1001 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8224)) := by
  have c62106 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62107 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62108 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62109 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62110 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62111 := (publicBatchWrapper89_copies1940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1001_18 := arithEq_of_rows h (row := 1001) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_18
  simp only [← c62106, ← c62107, ← c62108] at e_1001_18
  have e_1001_19 := arithEq_of_rows h (row := 1001) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1001_19
  simp only [← c62109, ← c62110, k2, ← c62111] at e_1001_19
  have hr := e_1001_19
  simp only [e_1001_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8225)) := by
  have c62112 := (publicBatchWrapper89_copies1941 a h).1
  have c62113 := (publicBatchWrapper89_copies1941 a h).2.1
  have c62114 := (publicBatchWrapper89_copies1941 a h).2.2.1
  have c62115 := (publicBatchWrapper89_copies1941 a h).2.2.2.1
  have c62116 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.1
  have c62117 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_0 := arithEq_of_rows h (row := 1002) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_0
  simp only [← c62112, ← c62113, ← c62114] at e_1002_0
  have e_1002_1 := arithEq_of_rows h (row := 1002) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_1
  simp only [← c62115, ← c62116, k2, ← c62117] at e_1002_1
  have hr := e_1002_1
  simp only [e_1002_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8226)) := by
  have c62118 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.1
  have c62119 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.1
  have c62120 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.1
  have c62121 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.1
  have c62122 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62123 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_2 := arithEq_of_rows h (row := 1002) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_2
  simp only [← c62118, ← c62119, ← c62120] at e_1002_2
  have e_1002_3 := arithEq_of_rows h (row := 1002) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_3
  simp only [← c62121, ← c62122, k2, ← c62123] at e_1002_3
  have hr := e_1002_3
  simp only [e_1002_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8227)) := by
  have c62124 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62125 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62126 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62127 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62128 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62129 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_4 := arithEq_of_rows h (row := 1002) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_4
  simp only [← c62124, ← c62125, ← c62126] at e_1002_4
  have e_1002_5 := arithEq_of_rows h (row := 1002) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_5
  simp only [← c62127, ← c62128, k2, ← c62129] at e_1002_5
  have hr := e_1002_5
  simp only [e_1002_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8228)) := by
  have c62130 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62131 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62132 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62133 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62134 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62135 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_6 := arithEq_of_rows h (row := 1002) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_6
  simp only [← c62130, ← c62131, ← c62132] at e_1002_6
  have e_1002_7 := arithEq_of_rows h (row := 1002) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_7
  simp only [← c62133, ← c62134, k2, ← c62135] at e_1002_7
  have hr := e_1002_7
  simp only [e_1002_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8229)) := by
  have c62136 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62137 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62138 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62139 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62140 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62141 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_8 := arithEq_of_rows h (row := 1002) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_8
  simp only [← c62136, ← c62137, ← c62138] at e_1002_8
  have e_1002_9 := arithEq_of_rows h (row := 1002) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_9
  simp only [← c62139, ← c62140, k2, ← c62141] at e_1002_9
  have hr := e_1002_9
  simp only [e_1002_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8230)) := by
  have c62142 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62143 := (publicBatchWrapper89_copies1941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62144 := (publicBatchWrapper89_copies1942 a h).1
  have c62145 := (publicBatchWrapper89_copies1942 a h).2.1
  have c62146 := (publicBatchWrapper89_copies1942 a h).2.2.1
  have c62147 := (publicBatchWrapper89_copies1942 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_10 := arithEq_of_rows h (row := 1002) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_10
  simp only [← c62142, ← c62143, ← c62144] at e_1002_10
  have e_1002_11 := arithEq_of_rows h (row := 1002) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_11
  simp only [← c62145, ← c62146, k2, ← c62147] at e_1002_11
  have hr := e_1002_11
  simp only [e_1002_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8231)) := by
  have c62148 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.1
  have c62149 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.1
  have c62150 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.1
  have c62151 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.1
  have c62152 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.1
  have c62153 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_12 := arithEq_of_rows h (row := 1002) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_12
  simp only [← c62148, ← c62149, ← c62150] at e_1002_12
  have e_1002_13 := arithEq_of_rows h (row := 1002) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_13
  simp only [← c62151, ← c62152, k2, ← c62153] at e_1002_13
  have hr := e_1002_13
  simp only [e_1002_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8232)) := by
  have c62154 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62155 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62156 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62157 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62158 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62159 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_14 := arithEq_of_rows h (row := 1002) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_14
  simp only [← c62154, ← c62155, ← c62156] at e_1002_14
  have e_1002_15 := arithEq_of_rows h (row := 1002) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_15
  simp only [← c62157, ← c62158, k2, ← c62159] at e_1002_15
  have hr := e_1002_15
  simp only [e_1002_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8233)) := by
  have c62160 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62161 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62162 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62163 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62164 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62165 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_16 := arithEq_of_rows h (row := 1002) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_16
  simp only [← c62160, ← c62161, ← c62162] at e_1002_16
  have e_1002_17 := arithEq_of_rows h (row := 1002) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_17
  simp only [← c62163, ← c62164, k2, ← c62165] at e_1002_17
  have hr := e_1002_17
  simp only [e_1002_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1002 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8234)) := by
  have c62166 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62167 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62168 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62169 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62170 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62171 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1002_18 := arithEq_of_rows h (row := 1002) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_18
  simp only [← c62166, ← c62167, ← c62168] at e_1002_18
  have e_1002_19 := arithEq_of_rows h (row := 1002) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1002_19
  simp only [← c62169, ← c62170, k2, ← c62171] at e_1002_19
  have hr := e_1002_19
  simp only [e_1002_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8235)) := by
  have c62172 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62173 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62174 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62175 := (publicBatchWrapper89_copies1942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62176 := (publicBatchWrapper89_copies1943 a h).1
  have c62177 := (publicBatchWrapper89_copies1943 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_0 := arithEq_of_rows h (row := 1003) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_0
  simp only [← c62172, ← c62173, ← c62174] at e_1003_0
  have e_1003_1 := arithEq_of_rows h (row := 1003) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_1
  simp only [← c62175, ← c62176, k2, ← c62177] at e_1003_1
  have hr := e_1003_1
  simp only [e_1003_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8236)) := by
  have c62178 := (publicBatchWrapper89_copies1943 a h).2.2.1
  have c62179 := (publicBatchWrapper89_copies1943 a h).2.2.2.1
  have c62180 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.1
  have c62181 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.1
  have c62182 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.1
  have c62183 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_2 := arithEq_of_rows h (row := 1003) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_2
  simp only [← c62178, ← c62179, ← c62180] at e_1003_2
  have e_1003_3 := arithEq_of_rows h (row := 1003) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_3
  simp only [← c62181, ← c62182, k2, ← c62183] at e_1003_3
  have hr := e_1003_3
  simp only [e_1003_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8237)) := by
  have c62184 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.1
  have c62185 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.1
  have c62186 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62187 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62188 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62189 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_4 := arithEq_of_rows h (row := 1003) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_4
  simp only [← c62184, ← c62185, ← c62186] at e_1003_4
  have e_1003_5 := arithEq_of_rows h (row := 1003) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_5
  simp only [← c62187, ← c62188, k2, ← c62189] at e_1003_5
  have hr := e_1003_5
  simp only [e_1003_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8238)) := by
  have c62190 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62191 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62192 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62193 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62194 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62195 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_6 := arithEq_of_rows h (row := 1003) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_6
  simp only [← c62190, ← c62191, ← c62192] at e_1003_6
  have e_1003_7 := arithEq_of_rows h (row := 1003) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_7
  simp only [← c62193, ← c62194, k2, ← c62195] at e_1003_7
  have hr := e_1003_7
  simp only [e_1003_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8239)) := by
  have c62196 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62197 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62198 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62199 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62200 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62201 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_8 := arithEq_of_rows h (row := 1003) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_8
  simp only [← c62196, ← c62197, ← c62198] at e_1003_8
  have e_1003_9 := arithEq_of_rows h (row := 1003) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_9
  simp only [← c62199, ← c62200, k2, ← c62201] at e_1003_9
  have hr := e_1003_9
  simp only [e_1003_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8240)) := by
  have c62202 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62203 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62204 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62205 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62206 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62207 := (publicBatchWrapper89_copies1943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_10 := arithEq_of_rows h (row := 1003) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_10
  simp only [← c62202, ← c62203, ← c62204] at e_1003_10
  have e_1003_11 := arithEq_of_rows h (row := 1003) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_11
  simp only [← c62205, ← c62206, k2, ← c62207] at e_1003_11
  have hr := e_1003_11
  simp only [e_1003_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8241)) := by
  have c62208 := (publicBatchWrapper89_copies1944 a h).1
  have c62209 := (publicBatchWrapper89_copies1944 a h).2.1
  have c62210 := (publicBatchWrapper89_copies1944 a h).2.2.1
  have c62211 := (publicBatchWrapper89_copies1944 a h).2.2.2.1
  have c62212 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.1
  have c62213 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_12 := arithEq_of_rows h (row := 1003) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_12
  simp only [← c62208, ← c62209, ← c62210] at e_1003_12
  have e_1003_13 := arithEq_of_rows h (row := 1003) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_13
  simp only [← c62211, ← c62212, k2, ← c62213] at e_1003_13
  have hr := e_1003_13
  simp only [e_1003_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8242)) := by
  have c62214 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.1
  have c62215 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.1
  have c62216 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.1
  have c62217 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.1
  have c62218 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62219 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_14 := arithEq_of_rows h (row := 1003) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_14
  simp only [← c62214, ← c62215, ← c62216] at e_1003_14
  have e_1003_15 := arithEq_of_rows h (row := 1003) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_15
  simp only [← c62217, ← c62218, k2, ← c62219] at e_1003_15
  have hr := e_1003_15
  simp only [e_1003_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8243)) := by
  have c62220 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62221 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62222 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62223 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62224 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62225 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_16 := arithEq_of_rows h (row := 1003) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_16
  simp only [← c62220, ← c62221, ← c62222] at e_1003_16
  have e_1003_17 := arithEq_of_rows h (row := 1003) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_17
  simp only [← c62223, ← c62224, k2, ← c62225] at e_1003_17
  have hr := e_1003_17
  simp only [e_1003_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1003 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8244)) := by
  have c62226 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62227 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62228 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62229 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62230 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62231 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1003_18 := arithEq_of_rows h (row := 1003) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_18
  simp only [← c62226, ← c62227, ← c62228] at e_1003_18
  have e_1003_19 := arithEq_of_rows h (row := 1003) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1003_19
  simp only [← c62229, ← c62230, k2, ← c62231] at e_1003_19
  have hr := e_1003_19
  simp only [e_1003_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8245)) := by
  have c62232 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62233 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62234 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62235 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62236 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62237 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_0 := arithEq_of_rows h (row := 1004) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_0
  simp only [← c62232, ← c62233, ← c62234] at e_1004_0
  have e_1004_1 := arithEq_of_rows h (row := 1004) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_1
  simp only [← c62235, ← c62236, k2, ← c62237] at e_1004_1
  have hr := e_1004_1
  simp only [e_1004_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8246)) := by
  have c62238 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62239 := (publicBatchWrapper89_copies1944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62240 := (publicBatchWrapper89_copies1945 a h).1
  have c62241 := (publicBatchWrapper89_copies1945 a h).2.1
  have c62242 := (publicBatchWrapper89_copies1945 a h).2.2.1
  have c62243 := (publicBatchWrapper89_copies1945 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_2 := arithEq_of_rows h (row := 1004) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_2
  simp only [← c62238, ← c62239, ← c62240] at e_1004_2
  have e_1004_3 := arithEq_of_rows h (row := 1004) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_3
  simp only [← c62241, ← c62242, k2, ← c62243] at e_1004_3
  have hr := e_1004_3
  simp only [e_1004_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8247)) := by
  have c62244 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.1
  have c62245 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.1
  have c62246 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.1
  have c62247 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.1
  have c62248 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.1
  have c62249 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_4 := arithEq_of_rows h (row := 1004) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_4
  simp only [← c62244, ← c62245, ← c62246] at e_1004_4
  have e_1004_5 := arithEq_of_rows h (row := 1004) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_5
  simp only [← c62247, ← c62248, k2, ← c62249] at e_1004_5
  have hr := e_1004_5
  simp only [e_1004_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8248)) := by
  have c62250 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62251 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62252 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62253 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62254 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62255 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_6 := arithEq_of_rows h (row := 1004) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_6
  simp only [← c62250, ← c62251, ← c62252] at e_1004_6
  have e_1004_7 := arithEq_of_rows h (row := 1004) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_7
  simp only [← c62253, ← c62254, k2, ← c62255] at e_1004_7
  have hr := e_1004_7
  simp only [e_1004_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8249)) := by
  have c62256 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62257 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62258 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62259 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62260 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62261 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_8 := arithEq_of_rows h (row := 1004) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_8
  simp only [← c62256, ← c62257, ← c62258] at e_1004_8
  have e_1004_9 := arithEq_of_rows h (row := 1004) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_9
  simp only [← c62259, ← c62260, k2, ← c62261] at e_1004_9
  have hr := e_1004_9
  simp only [e_1004_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8250)) := by
  have c62262 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62263 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62264 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62265 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62266 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62267 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_10 := arithEq_of_rows h (row := 1004) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_10
  simp only [← c62262, ← c62263, ← c62264] at e_1004_10
  have e_1004_11 := arithEq_of_rows h (row := 1004) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_11
  simp only [← c62265, ← c62266, k2, ← c62267] at e_1004_11
  have hr := e_1004_11
  simp only [e_1004_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8251)) := by
  have c62268 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62269 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62270 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62271 := (publicBatchWrapper89_copies1945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62272 := (publicBatchWrapper89_copies1946 a h).1
  have c62273 := (publicBatchWrapper89_copies1946 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_12 := arithEq_of_rows h (row := 1004) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_12
  simp only [← c62268, ← c62269, ← c62270] at e_1004_12
  have e_1004_13 := arithEq_of_rows h (row := 1004) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_13
  simp only [← c62271, ← c62272, k2, ← c62273] at e_1004_13
  have hr := e_1004_13
  simp only [e_1004_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8252)) := by
  have c62274 := (publicBatchWrapper89_copies1946 a h).2.2.1
  have c62275 := (publicBatchWrapper89_copies1946 a h).2.2.2.1
  have c62276 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.1
  have c62277 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.1
  have c62278 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.1
  have c62279 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_14 := arithEq_of_rows h (row := 1004) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_14
  simp only [← c62274, ← c62275, ← c62276] at e_1004_14
  have e_1004_15 := arithEq_of_rows h (row := 1004) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_15
  simp only [← c62277, ← c62278, k2, ← c62279] at e_1004_15
  have hr := e_1004_15
  simp only [e_1004_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8253)) := by
  have c62280 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.1
  have c62281 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.1
  have c62282 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62283 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62284 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62285 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_16 := arithEq_of_rows h (row := 1004) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_16
  simp only [← c62280, ← c62281, ← c62282] at e_1004_16
  have e_1004_17 := arithEq_of_rows h (row := 1004) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_17
  simp only [← c62283, ← c62284, k2, ← c62285] at e_1004_17
  have hr := e_1004_17
  simp only [e_1004_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1004 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8254)) := by
  have c62286 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62287 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62288 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62289 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62290 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62291 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1004_18 := arithEq_of_rows h (row := 1004) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_18
  simp only [← c62286, ← c62287, ← c62288] at e_1004_18
  have e_1004_19 := arithEq_of_rows h (row := 1004) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1004_19
  simp only [← c62289, ← c62290, k2, ← c62291] at e_1004_19
  have hr := e_1004_19
  simp only [e_1004_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8255)) := by
  have c62292 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62293 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62294 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62295 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62296 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62297 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_0 := arithEq_of_rows h (row := 1005) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_0
  simp only [← c62292, ← c62293, ← c62294] at e_1005_0
  have e_1005_1 := arithEq_of_rows h (row := 1005) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_1
  simp only [← c62295, ← c62296, k2, ← c62297] at e_1005_1
  have hr := e_1005_1
  simp only [e_1005_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8256)) := by
  have c62298 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62299 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62300 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62301 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62302 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62303 := (publicBatchWrapper89_copies1946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_2 := arithEq_of_rows h (row := 1005) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_2
  simp only [← c62298, ← c62299, ← c62300] at e_1005_2
  have e_1005_3 := arithEq_of_rows h (row := 1005) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_3
  simp only [← c62301, ← c62302, k2, ← c62303] at e_1005_3
  have hr := e_1005_3
  simp only [e_1005_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8257)) := by
  have c62304 := (publicBatchWrapper89_copies1947 a h).1
  have c62305 := (publicBatchWrapper89_copies1947 a h).2.1
  have c62306 := (publicBatchWrapper89_copies1947 a h).2.2.1
  have c62307 := (publicBatchWrapper89_copies1947 a h).2.2.2.1
  have c62308 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.1
  have c62309 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_4 := arithEq_of_rows h (row := 1005) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_4
  simp only [← c62304, ← c62305, ← c62306] at e_1005_4
  have e_1005_5 := arithEq_of_rows h (row := 1005) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_5
  simp only [← c62307, ← c62308, k2, ← c62309] at e_1005_5
  have hr := e_1005_5
  simp only [e_1005_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8258)) := by
  have c62310 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.1
  have c62311 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.1
  have c62312 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.1
  have c62313 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.1
  have c62314 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62315 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_6 := arithEq_of_rows h (row := 1005) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_6
  simp only [← c62310, ← c62311, ← c62312] at e_1005_6
  have e_1005_7 := arithEq_of_rows h (row := 1005) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_7
  simp only [← c62313, ← c62314, k2, ← c62315] at e_1005_7
  have hr := e_1005_7
  simp only [e_1005_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8259)) := by
  have c62316 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62317 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62318 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62319 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62320 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62321 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_8 := arithEq_of_rows h (row := 1005) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_8
  simp only [← c62316, ← c62317, ← c62318] at e_1005_8
  have e_1005_9 := arithEq_of_rows h (row := 1005) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_9
  simp only [← c62319, ← c62320, k2, ← c62321] at e_1005_9
  have hr := e_1005_9
  simp only [e_1005_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8260)) := by
  have c62322 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62323 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62324 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62325 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62326 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62327 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_10 := arithEq_of_rows h (row := 1005) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_10
  simp only [← c62322, ← c62323, ← c62324] at e_1005_10
  have e_1005_11 := arithEq_of_rows h (row := 1005) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_11
  simp only [← c62325, ← c62326, k2, ← c62327] at e_1005_11
  have hr := e_1005_11
  simp only [e_1005_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8261)) := by
  have c62328 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62329 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62330 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62331 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62332 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62333 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_12 := arithEq_of_rows h (row := 1005) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_12
  simp only [← c62328, ← c62329, ← c62330] at e_1005_12
  have e_1005_13 := arithEq_of_rows h (row := 1005) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_13
  simp only [← c62331, ← c62332, k2, ← c62333] at e_1005_13
  have hr := e_1005_13
  simp only [e_1005_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8262)) := by
  have c62334 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62335 := (publicBatchWrapper89_copies1947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62336 := (publicBatchWrapper89_copies1948 a h).1
  have c62337 := (publicBatchWrapper89_copies1948 a h).2.1
  have c62338 := (publicBatchWrapper89_copies1948 a h).2.2.1
  have c62339 := (publicBatchWrapper89_copies1948 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_14 := arithEq_of_rows h (row := 1005) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_14
  simp only [← c62334, ← c62335, ← c62336] at e_1005_14
  have e_1005_15 := arithEq_of_rows h (row := 1005) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_15
  simp only [← c62337, ← c62338, k2, ← c62339] at e_1005_15
  have hr := e_1005_15
  simp only [e_1005_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8263)) := by
  have c62340 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.1
  have c62341 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.1
  have c62342 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.1
  have c62343 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.1
  have c62344 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.1
  have c62345 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_16 := arithEq_of_rows h (row := 1005) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_16
  simp only [← c62340, ← c62341, ← c62342] at e_1005_16
  have e_1005_17 := arithEq_of_rows h (row := 1005) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_17
  simp only [← c62343, ← c62344, k2, ← c62345] at e_1005_17
  have hr := e_1005_17
  simp only [e_1005_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1005 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8264)) := by
  have c62346 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62347 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62348 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62349 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62350 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62351 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1005_18 := arithEq_of_rows h (row := 1005) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_18
  simp only [← c62346, ← c62347, ← c62348] at e_1005_18
  have e_1005_19 := arithEq_of_rows h (row := 1005) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1005_19
  simp only [← c62349, ← c62350, k2, ← c62351] at e_1005_19
  have hr := e_1005_19
  simp only [e_1005_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8265)) := by
  have c62352 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62353 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62354 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62355 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62356 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62357 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_0 := arithEq_of_rows h (row := 1006) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_0
  simp only [← c62352, ← c62353, ← c62354] at e_1006_0
  have e_1006_1 := arithEq_of_rows h (row := 1006) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_1
  simp only [← c62355, ← c62356, k2, ← c62357] at e_1006_1
  have hr := e_1006_1
  simp only [e_1006_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8266)) := by
  have c62358 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62359 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62360 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62361 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62362 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62363 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_2 := arithEq_of_rows h (row := 1006) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_2
  simp only [← c62358, ← c62359, ← c62360] at e_1006_2
  have e_1006_3 := arithEq_of_rows h (row := 1006) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_3
  simp only [← c62361, ← c62362, k2, ← c62363] at e_1006_3
  have hr := e_1006_3
  simp only [e_1006_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8267)) := by
  have c62364 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62365 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62366 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62367 := (publicBatchWrapper89_copies1948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62368 := (publicBatchWrapper89_copies1949 a h).1
  have c62369 := (publicBatchWrapper89_copies1949 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_4 := arithEq_of_rows h (row := 1006) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_4
  simp only [← c62364, ← c62365, ← c62366] at e_1006_4
  have e_1006_5 := arithEq_of_rows h (row := 1006) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_5
  simp only [← c62367, ← c62368, k2, ← c62369] at e_1006_5
  have hr := e_1006_5
  simp only [e_1006_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8268)) := by
  have c62370 := (publicBatchWrapper89_copies1949 a h).2.2.1
  have c62371 := (publicBatchWrapper89_copies1949 a h).2.2.2.1
  have c62372 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.1
  have c62373 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.1
  have c62374 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.1
  have c62375 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_6 := arithEq_of_rows h (row := 1006) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_6
  simp only [← c62370, ← c62371, ← c62372] at e_1006_6
  have e_1006_7 := arithEq_of_rows h (row := 1006) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_7
  simp only [← c62373, ← c62374, k2, ← c62375] at e_1006_7
  have hr := e_1006_7
  simp only [e_1006_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8269)) := by
  have c62376 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.1
  have c62377 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.1
  have c62378 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62379 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62380 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62381 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_8 := arithEq_of_rows h (row := 1006) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_8
  simp only [← c62376, ← c62377, ← c62378] at e_1006_8
  have e_1006_9 := arithEq_of_rows h (row := 1006) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_9
  simp only [← c62379, ← c62380, k2, ← c62381] at e_1006_9
  have hr := e_1006_9
  simp only [e_1006_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8270)) := by
  have c62382 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62383 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62384 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62385 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62386 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62387 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_10 := arithEq_of_rows h (row := 1006) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_10
  simp only [← c62382, ← c62383, ← c62384] at e_1006_10
  have e_1006_11 := arithEq_of_rows h (row := 1006) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_11
  simp only [← c62385, ← c62386, k2, ← c62387] at e_1006_11
  have hr := e_1006_11
  simp only [e_1006_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8271)) := by
  have c62388 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62389 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62390 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62391 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62392 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62393 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_12 := arithEq_of_rows h (row := 1006) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_12
  simp only [← c62388, ← c62389, ← c62390] at e_1006_12
  have e_1006_13 := arithEq_of_rows h (row := 1006) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_13
  simp only [← c62391, ← c62392, k2, ← c62393] at e_1006_13
  have hr := e_1006_13
  simp only [e_1006_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8272)) := by
  have c62394 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62395 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62396 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62397 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62398 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62399 := (publicBatchWrapper89_copies1949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_14 := arithEq_of_rows h (row := 1006) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_14
  simp only [← c62394, ← c62395, ← c62396] at e_1006_14
  have e_1006_15 := arithEq_of_rows h (row := 1006) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_15
  simp only [← c62397, ← c62398, k2, ← c62399] at e_1006_15
  have hr := e_1006_15
  simp only [e_1006_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8273)) := by
  have c62400 := (publicBatchWrapper89_copies1950 a h).1
  have c62401 := (publicBatchWrapper89_copies1950 a h).2.1
  have c62402 := (publicBatchWrapper89_copies1950 a h).2.2.1
  have c62403 := (publicBatchWrapper89_copies1950 a h).2.2.2.1
  have c62404 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.1
  have c62405 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_16 := arithEq_of_rows h (row := 1006) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_16
  simp only [← c62400, ← c62401, ← c62402] at e_1006_16
  have e_1006_17 := arithEq_of_rows h (row := 1006) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_17
  simp only [← c62403, ← c62404, k2, ← c62405] at e_1006_17
  have hr := e_1006_17
  simp only [e_1006_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1006 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8274)) := by
  have c62406 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.1
  have c62407 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.1
  have c62408 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.1
  have c62409 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.1
  have c62410 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62411 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1006_18 := arithEq_of_rows h (row := 1006) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_18
  simp only [← c62406, ← c62407, ← c62408] at e_1006_18
  have e_1006_19 := arithEq_of_rows h (row := 1006) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1006_19
  simp only [← c62409, ← c62410, k2, ← c62411] at e_1006_19
  have hr := e_1006_19
  simp only [e_1006_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8275)) := by
  have c62412 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62413 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62414 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62415 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62416 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62417 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_0 := arithEq_of_rows h (row := 1007) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_0
  simp only [← c62412, ← c62413, ← c62414] at e_1007_0
  have e_1007_1 := arithEq_of_rows h (row := 1007) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_1
  simp only [← c62415, ← c62416, k2, ← c62417] at e_1007_1
  have hr := e_1007_1
  simp only [e_1007_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8276)) := by
  have c62418 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62419 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62420 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62421 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62422 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62423 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_2 := arithEq_of_rows h (row := 1007) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_2
  simp only [← c62418, ← c62419, ← c62420] at e_1007_2
  have e_1007_3 := arithEq_of_rows h (row := 1007) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_3
  simp only [← c62421, ← c62422, k2, ← c62423] at e_1007_3
  have hr := e_1007_3
  simp only [e_1007_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8277)) := by
  have c62424 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62425 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62426 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62427 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62428 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62429 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_4 := arithEq_of_rows h (row := 1007) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_4
  simp only [← c62424, ← c62425, ← c62426] at e_1007_4
  have e_1007_5 := arithEq_of_rows h (row := 1007) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_5
  simp only [← c62427, ← c62428, k2, ← c62429] at e_1007_5
  have hr := e_1007_5
  simp only [e_1007_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8278)) := by
  have c62430 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62431 := (publicBatchWrapper89_copies1950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62432 := (publicBatchWrapper89_copies1951 a h).1
  have c62433 := (publicBatchWrapper89_copies1951 a h).2.1
  have c62434 := (publicBatchWrapper89_copies1951 a h).2.2.1
  have c62435 := (publicBatchWrapper89_copies1951 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_6 := arithEq_of_rows h (row := 1007) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_6
  simp only [← c62430, ← c62431, ← c62432] at e_1007_6
  have e_1007_7 := arithEq_of_rows h (row := 1007) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_7
  simp only [← c62433, ← c62434, k2, ← c62435] at e_1007_7
  have hr := e_1007_7
  simp only [e_1007_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8279)) := by
  have c62436 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.1
  have c62437 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.1
  have c62438 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.1
  have c62439 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.1
  have c62440 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.1
  have c62441 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_8 := arithEq_of_rows h (row := 1007) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_8
  simp only [← c62436, ← c62437, ← c62438] at e_1007_8
  have e_1007_9 := arithEq_of_rows h (row := 1007) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_9
  simp only [← c62439, ← c62440, k2, ← c62441] at e_1007_9
  have hr := e_1007_9
  simp only [e_1007_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8280)) := by
  have c62442 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62443 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62444 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62445 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62446 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62447 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_10 := arithEq_of_rows h (row := 1007) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_10
  simp only [← c62442, ← c62443, ← c62444] at e_1007_10
  have e_1007_11 := arithEq_of_rows h (row := 1007) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_11
  simp only [← c62445, ← c62446, k2, ← c62447] at e_1007_11
  have hr := e_1007_11
  simp only [e_1007_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8281)) := by
  have c62448 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62449 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62450 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62451 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62452 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62453 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_12 := arithEq_of_rows h (row := 1007) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_12
  simp only [← c62448, ← c62449, ← c62450] at e_1007_12
  have e_1007_13 := arithEq_of_rows h (row := 1007) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_13
  simp only [← c62451, ← c62452, k2, ← c62453] at e_1007_13
  have hr := e_1007_13
  simp only [e_1007_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8282)) := by
  have c62454 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62455 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62456 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62457 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62458 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62459 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_14 := arithEq_of_rows h (row := 1007) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_14
  simp only [← c62454, ← c62455, ← c62456] at e_1007_14
  have e_1007_15 := arithEq_of_rows h (row := 1007) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_15
  simp only [← c62457, ← c62458, k2, ← c62459] at e_1007_15
  have hr := e_1007_15
  simp only [e_1007_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8283)) := by
  have c62460 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62461 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62462 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62463 := (publicBatchWrapper89_copies1951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62464 := (publicBatchWrapper89_copies1952 a h).1
  have c62465 := (publicBatchWrapper89_copies1952 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_16 := arithEq_of_rows h (row := 1007) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_16
  simp only [← c62460, ← c62461, ← c62462] at e_1007_16
  have e_1007_17 := arithEq_of_rows h (row := 1007) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_17
  simp only [← c62463, ← c62464, k2, ← c62465] at e_1007_17
  have hr := e_1007_17
  simp only [e_1007_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1007 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8284)) := by
  have c62466 := (publicBatchWrapper89_copies1952 a h).2.2.1
  have c62467 := (publicBatchWrapper89_copies1952 a h).2.2.2.1
  have c62468 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.1
  have c62469 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.1
  have c62470 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.1
  have c62471 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1007_18 := arithEq_of_rows h (row := 1007) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_18
  simp only [← c62466, ← c62467, ← c62468] at e_1007_18
  have e_1007_19 := arithEq_of_rows h (row := 1007) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1007_19
  simp only [← c62469, ← c62470, k2, ← c62471] at e_1007_19
  have hr := e_1007_19
  simp only [e_1007_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8285)) := by
  have c62472 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.1
  have c62473 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.1
  have c62474 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62475 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62476 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62477 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_0 := arithEq_of_rows h (row := 1008) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_0
  simp only [← c62472, ← c62473, ← c62474] at e_1008_0
  have e_1008_1 := arithEq_of_rows h (row := 1008) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_1
  simp only [← c62475, ← c62476, k2, ← c62477] at e_1008_1
  have hr := e_1008_1
  simp only [e_1008_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8286)) := by
  have c62478 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62479 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62480 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62481 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62482 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62483 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_2 := arithEq_of_rows h (row := 1008) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_2
  simp only [← c62478, ← c62479, ← c62480] at e_1008_2
  have e_1008_3 := arithEq_of_rows h (row := 1008) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_3
  simp only [← c62481, ← c62482, k2, ← c62483] at e_1008_3
  have hr := e_1008_3
  simp only [e_1008_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8287)) := by
  have c62484 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62485 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62486 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62487 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62488 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62489 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_4 := arithEq_of_rows h (row := 1008) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_4
  simp only [← c62484, ← c62485, ← c62486] at e_1008_4
  have e_1008_5 := arithEq_of_rows h (row := 1008) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_5
  simp only [← c62487, ← c62488, k2, ← c62489] at e_1008_5
  have hr := e_1008_5
  simp only [e_1008_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8288)) := by
  have c62490 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62491 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62492 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62493 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62494 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62495 := (publicBatchWrapper89_copies1952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_6 := arithEq_of_rows h (row := 1008) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_6
  simp only [← c62490, ← c62491, ← c62492] at e_1008_6
  have e_1008_7 := arithEq_of_rows h (row := 1008) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_7
  simp only [← c62493, ← c62494, k2, ← c62495] at e_1008_7
  have hr := e_1008_7
  simp only [e_1008_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8289)) := by
  have c62496 := (publicBatchWrapper89_copies1953 a h).1
  have c62497 := (publicBatchWrapper89_copies1953 a h).2.1
  have c62498 := (publicBatchWrapper89_copies1953 a h).2.2.1
  have c62499 := (publicBatchWrapper89_copies1953 a h).2.2.2.1
  have c62500 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.1
  have c62501 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_8 := arithEq_of_rows h (row := 1008) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_8
  simp only [← c62496, ← c62497, ← c62498] at e_1008_8
  have e_1008_9 := arithEq_of_rows h (row := 1008) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_9
  simp only [← c62499, ← c62500, k2, ← c62501] at e_1008_9
  have hr := e_1008_9
  simp only [e_1008_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8290)) := by
  have c62502 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.1
  have c62503 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.1
  have c62504 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.1
  have c62505 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.1
  have c62506 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62507 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_10 := arithEq_of_rows h (row := 1008) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_10
  simp only [← c62502, ← c62503, ← c62504] at e_1008_10
  have e_1008_11 := arithEq_of_rows h (row := 1008) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_11
  simp only [← c62505, ← c62506, k2, ← c62507] at e_1008_11
  have hr := e_1008_11
  simp only [e_1008_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8291)) := by
  have c62508 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62509 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62510 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62511 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62512 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62513 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_12 := arithEq_of_rows h (row := 1008) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_12
  simp only [← c62508, ← c62509, ← c62510] at e_1008_12
  have e_1008_13 := arithEq_of_rows h (row := 1008) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_13
  simp only [← c62511, ← c62512, k2, ← c62513] at e_1008_13
  have hr := e_1008_13
  simp only [e_1008_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8292)) := by
  have c62514 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62515 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62516 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62517 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62518 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62519 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_14 := arithEq_of_rows h (row := 1008) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_14
  simp only [← c62514, ← c62515, ← c62516] at e_1008_14
  have e_1008_15 := arithEq_of_rows h (row := 1008) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_15
  simp only [← c62517, ← c62518, k2, ← c62519] at e_1008_15
  have hr := e_1008_15
  simp only [e_1008_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8293)) := by
  have c62520 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62521 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62522 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62523 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62524 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62525 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_16 := arithEq_of_rows h (row := 1008) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_16
  simp only [← c62520, ← c62521, ← c62522] at e_1008_16
  have e_1008_17 := arithEq_of_rows h (row := 1008) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_17
  simp only [← c62523, ← c62524, k2, ← c62525] at e_1008_17
  have hr := e_1008_17
  simp only [e_1008_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1008 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8294)) := by
  have c62526 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62527 := (publicBatchWrapper89_copies1953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62528 := (publicBatchWrapper89_copies1954 a h).1
  have c62529 := (publicBatchWrapper89_copies1954 a h).2.1
  have c62530 := (publicBatchWrapper89_copies1954 a h).2.2.1
  have c62531 := (publicBatchWrapper89_copies1954 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1008_18 := arithEq_of_rows h (row := 1008) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_18
  simp only [← c62526, ← c62527, ← c62528] at e_1008_18
  have e_1008_19 := arithEq_of_rows h (row := 1008) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1008_19
  simp only [← c62529, ← c62530, k2, ← c62531] at e_1008_19
  have hr := e_1008_19
  simp only [e_1008_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8295)) := by
  have c62532 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.1
  have c62533 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.1
  have c62534 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.1
  have c62535 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.1
  have c62536 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.1
  have c62537 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_0 := arithEq_of_rows h (row := 1009) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_0
  simp only [← c62532, ← c62533, ← c62534] at e_1009_0
  have e_1009_1 := arithEq_of_rows h (row := 1009) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_1
  simp only [← c62535, ← c62536, k2, ← c62537] at e_1009_1
  have hr := e_1009_1
  simp only [e_1009_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8296)) := by
  have c62538 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62539 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62540 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62541 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62542 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62543 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_2 := arithEq_of_rows h (row := 1009) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_2
  simp only [← c62538, ← c62539, ← c62540] at e_1009_2
  have e_1009_3 := arithEq_of_rows h (row := 1009) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_3
  simp only [← c62541, ← c62542, k2, ← c62543] at e_1009_3
  have hr := e_1009_3
  simp only [e_1009_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8297)) := by
  have c62544 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62545 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62546 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62547 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62548 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62549 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_4 := arithEq_of_rows h (row := 1009) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_4
  simp only [← c62544, ← c62545, ← c62546] at e_1009_4
  have e_1009_5 := arithEq_of_rows h (row := 1009) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_5
  simp only [← c62547, ← c62548, k2, ← c62549] at e_1009_5
  have hr := e_1009_5
  simp only [e_1009_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8298)) := by
  have c62550 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62551 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62552 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62553 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62554 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62555 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_6 := arithEq_of_rows h (row := 1009) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_6
  simp only [← c62550, ← c62551, ← c62552] at e_1009_6
  have e_1009_7 := arithEq_of_rows h (row := 1009) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_7
  simp only [← c62553, ← c62554, k2, ← c62555] at e_1009_7
  have hr := e_1009_7
  simp only [e_1009_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8299)) := by
  have c62556 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62557 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62558 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62559 := (publicBatchWrapper89_copies1954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62560 := (publicBatchWrapper89_copies1955 a h).1
  have c62561 := (publicBatchWrapper89_copies1955 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_8 := arithEq_of_rows h (row := 1009) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_8
  simp only [← c62556, ← c62557, ← c62558] at e_1009_8
  have e_1009_9 := arithEq_of_rows h (row := 1009) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_9
  simp only [← c62559, ← c62560, k2, ← c62561] at e_1009_9
  have hr := e_1009_9
  simp only [e_1009_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8300)) := by
  have c62562 := (publicBatchWrapper89_copies1955 a h).2.2.1
  have c62563 := (publicBatchWrapper89_copies1955 a h).2.2.2.1
  have c62564 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.1
  have c62565 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.1
  have c62566 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.1
  have c62567 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_10 := arithEq_of_rows h (row := 1009) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_10
  simp only [← c62562, ← c62563, ← c62564] at e_1009_10
  have e_1009_11 := arithEq_of_rows h (row := 1009) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_11
  simp only [← c62565, ← c62566, k2, ← c62567] at e_1009_11
  have hr := e_1009_11
  simp only [e_1009_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8301)) := by
  have c62568 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.1
  have c62569 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.1
  have c62570 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62571 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62572 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62573 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_12 := arithEq_of_rows h (row := 1009) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_12
  simp only [← c62568, ← c62569, ← c62570] at e_1009_12
  have e_1009_13 := arithEq_of_rows h (row := 1009) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_13
  simp only [← c62571, ← c62572, k2, ← c62573] at e_1009_13
  have hr := e_1009_13
  simp only [e_1009_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8302)) := by
  have c62574 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62575 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62576 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62577 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62578 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62579 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_14 := arithEq_of_rows h (row := 1009) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_14
  simp only [← c62574, ← c62575, ← c62576] at e_1009_14
  have e_1009_15 := arithEq_of_rows h (row := 1009) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_15
  simp only [← c62577, ← c62578, k2, ← c62579] at e_1009_15
  have hr := e_1009_15
  simp only [e_1009_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8303)) := by
  have c62580 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62581 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62582 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62583 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62584 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62585 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_16 := arithEq_of_rows h (row := 1009) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_16
  simp only [← c62580, ← c62581, ← c62582] at e_1009_16
  have e_1009_17 := arithEq_of_rows h (row := 1009) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_17
  simp only [← c62583, ← c62584, k2, ← c62585] at e_1009_17
  have hr := e_1009_17
  simp only [e_1009_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1009 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8304)) := by
  have c62586 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62587 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62588 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62589 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62590 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62591 := (publicBatchWrapper89_copies1955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1009_18 := arithEq_of_rows h (row := 1009) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_18
  simp only [← c62586, ← c62587, ← c62588] at e_1009_18
  have e_1009_19 := arithEq_of_rows h (row := 1009) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1009_19
  simp only [← c62589, ← c62590, k2, ← c62591] at e_1009_19
  have hr := e_1009_19
  simp only [e_1009_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8305)) := by
  have c62592 := (publicBatchWrapper89_copies1956 a h).1
  have c62593 := (publicBatchWrapper89_copies1956 a h).2.1
  have c62594 := (publicBatchWrapper89_copies1956 a h).2.2.1
  have c62595 := (publicBatchWrapper89_copies1956 a h).2.2.2.1
  have c62596 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.1
  have c62597 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_0 := arithEq_of_rows h (row := 1010) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_0
  simp only [← c62592, ← c62593, ← c62594] at e_1010_0
  have e_1010_1 := arithEq_of_rows h (row := 1010) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_1
  simp only [← c62595, ← c62596, k2, ← c62597] at e_1010_1
  have hr := e_1010_1
  simp only [e_1010_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8306)) := by
  have c62598 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.1
  have c62599 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.1
  have c62600 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.1
  have c62601 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.1
  have c62602 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62603 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_2 := arithEq_of_rows h (row := 1010) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_2
  simp only [← c62598, ← c62599, ← c62600] at e_1010_2
  have e_1010_3 := arithEq_of_rows h (row := 1010) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_3
  simp only [← c62601, ← c62602, k2, ← c62603] at e_1010_3
  have hr := e_1010_3
  simp only [e_1010_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8307)) := by
  have c62604 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62605 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62606 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62607 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62608 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62609 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_4 := arithEq_of_rows h (row := 1010) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_4
  simp only [← c62604, ← c62605, ← c62606] at e_1010_4
  have e_1010_5 := arithEq_of_rows h (row := 1010) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_5
  simp only [← c62607, ← c62608, k2, ← c62609] at e_1010_5
  have hr := e_1010_5
  simp only [e_1010_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8308)) := by
  have c62610 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62611 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62612 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62613 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62614 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62615 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_6 := arithEq_of_rows h (row := 1010) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_6
  simp only [← c62610, ← c62611, ← c62612] at e_1010_6
  have e_1010_7 := arithEq_of_rows h (row := 1010) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_7
  simp only [← c62613, ← c62614, k2, ← c62615] at e_1010_7
  have hr := e_1010_7
  simp only [e_1010_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8309)) := by
  have c62616 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62617 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62618 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62619 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62620 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62621 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_8 := arithEq_of_rows h (row := 1010) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_8
  simp only [← c62616, ← c62617, ← c62618] at e_1010_8
  have e_1010_9 := arithEq_of_rows h (row := 1010) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_9
  simp only [← c62619, ← c62620, k2, ← c62621] at e_1010_9
  have hr := e_1010_9
  simp only [e_1010_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8310)) := by
  have c62622 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62623 := (publicBatchWrapper89_copies1956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62624 := (publicBatchWrapper89_copies1957 a h).1
  have c62625 := (publicBatchWrapper89_copies1957 a h).2.1
  have c62626 := (publicBatchWrapper89_copies1957 a h).2.2.1
  have c62627 := (publicBatchWrapper89_copies1957 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_10 := arithEq_of_rows h (row := 1010) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_10
  simp only [← c62622, ← c62623, ← c62624] at e_1010_10
  have e_1010_11 := arithEq_of_rows h (row := 1010) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_11
  simp only [← c62625, ← c62626, k2, ← c62627] at e_1010_11
  have hr := e_1010_11
  simp only [e_1010_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8311)) := by
  have c62628 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.1
  have c62629 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.1
  have c62630 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.1
  have c62631 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.1
  have c62632 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.1
  have c62633 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_12 := arithEq_of_rows h (row := 1010) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_12
  simp only [← c62628, ← c62629, ← c62630] at e_1010_12
  have e_1010_13 := arithEq_of_rows h (row := 1010) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_13
  simp only [← c62631, ← c62632, k2, ← c62633] at e_1010_13
  have hr := e_1010_13
  simp only [e_1010_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8312)) := by
  have c62634 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62635 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62636 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62637 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62638 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62639 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_14 := arithEq_of_rows h (row := 1010) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_14
  simp only [← c62634, ← c62635, ← c62636] at e_1010_14
  have e_1010_15 := arithEq_of_rows h (row := 1010) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_15
  simp only [← c62637, ← c62638, k2, ← c62639] at e_1010_15
  have hr := e_1010_15
  simp only [e_1010_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8313)) := by
  have c62640 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62641 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62642 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62643 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62644 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62645 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_16 := arithEq_of_rows h (row := 1010) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_16
  simp only [← c62640, ← c62641, ← c62642] at e_1010_16
  have e_1010_17 := arithEq_of_rows h (row := 1010) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_17
  simp only [← c62643, ← c62644, k2, ← c62645] at e_1010_17
  have hr := e_1010_17
  simp only [e_1010_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1010 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8314)) := by
  have c62646 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62647 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62648 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62649 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62650 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62651 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1010_18 := arithEq_of_rows h (row := 1010) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_18
  simp only [← c62646, ← c62647, ← c62648] at e_1010_18
  have e_1010_19 := arithEq_of_rows h (row := 1010) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1010_19
  simp only [← c62649, ← c62650, k2, ← c62651] at e_1010_19
  have hr := e_1010_19
  simp only [e_1010_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8315)) := by
  have c62652 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62653 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62654 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62655 := (publicBatchWrapper89_copies1957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62656 := (publicBatchWrapper89_copies1958 a h).1
  have c62657 := (publicBatchWrapper89_copies1958 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_0 := arithEq_of_rows h (row := 1011) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_0
  simp only [← c62652, ← c62653, ← c62654] at e_1011_0
  have e_1011_1 := arithEq_of_rows h (row := 1011) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_1
  simp only [← c62655, ← c62656, k2, ← c62657] at e_1011_1
  have hr := e_1011_1
  simp only [e_1011_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8316)) := by
  have c62658 := (publicBatchWrapper89_copies1958 a h).2.2.1
  have c62659 := (publicBatchWrapper89_copies1958 a h).2.2.2.1
  have c62660 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.1
  have c62661 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.1
  have c62662 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.1
  have c62663 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_2 := arithEq_of_rows h (row := 1011) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_2
  simp only [← c62658, ← c62659, ← c62660] at e_1011_2
  have e_1011_3 := arithEq_of_rows h (row := 1011) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_3
  simp only [← c62661, ← c62662, k2, ← c62663] at e_1011_3
  have hr := e_1011_3
  simp only [e_1011_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8317)) := by
  have c62664 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.1
  have c62665 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.1
  have c62666 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62667 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62668 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62669 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_4 := arithEq_of_rows h (row := 1011) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_4
  simp only [← c62664, ← c62665, ← c62666] at e_1011_4
  have e_1011_5 := arithEq_of_rows h (row := 1011) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_5
  simp only [← c62667, ← c62668, k2, ← c62669] at e_1011_5
  have hr := e_1011_5
  simp only [e_1011_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8318)) := by
  have c62670 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62671 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62672 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62673 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62674 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62675 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_6 := arithEq_of_rows h (row := 1011) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_6
  simp only [← c62670, ← c62671, ← c62672] at e_1011_6
  have e_1011_7 := arithEq_of_rows h (row := 1011) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_7
  simp only [← c62673, ← c62674, k2, ← c62675] at e_1011_7
  have hr := e_1011_7
  simp only [e_1011_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8319)) := by
  have c62676 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62677 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62678 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62679 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62680 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62681 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_8 := arithEq_of_rows h (row := 1011) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_8
  simp only [← c62676, ← c62677, ← c62678] at e_1011_8
  have e_1011_9 := arithEq_of_rows h (row := 1011) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_9
  simp only [← c62679, ← c62680, k2, ← c62681] at e_1011_9
  have hr := e_1011_9
  simp only [e_1011_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8320)) := by
  have c62682 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62683 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62684 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62685 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62686 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62687 := (publicBatchWrapper89_copies1958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_10 := arithEq_of_rows h (row := 1011) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_10
  simp only [← c62682, ← c62683, ← c62684] at e_1011_10
  have e_1011_11 := arithEq_of_rows h (row := 1011) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_11
  simp only [← c62685, ← c62686, k2, ← c62687] at e_1011_11
  have hr := e_1011_11
  simp only [e_1011_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8321)) := by
  have c62688 := (publicBatchWrapper89_copies1959 a h).1
  have c62689 := (publicBatchWrapper89_copies1959 a h).2.1
  have c62690 := (publicBatchWrapper89_copies1959 a h).2.2.1
  have c62691 := (publicBatchWrapper89_copies1959 a h).2.2.2.1
  have c62692 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.1
  have c62693 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_12 := arithEq_of_rows h (row := 1011) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_12
  simp only [← c62688, ← c62689, ← c62690] at e_1011_12
  have e_1011_13 := arithEq_of_rows h (row := 1011) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_13
  simp only [← c62691, ← c62692, k2, ← c62693] at e_1011_13
  have hr := e_1011_13
  simp only [e_1011_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8322)) := by
  have c62694 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.1
  have c62695 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.1
  have c62696 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.1
  have c62697 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.1
  have c62698 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62699 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_14 := arithEq_of_rows h (row := 1011) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_14
  simp only [← c62694, ← c62695, ← c62696] at e_1011_14
  have e_1011_15 := arithEq_of_rows h (row := 1011) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_15
  simp only [← c62697, ← c62698, k2, ← c62699] at e_1011_15
  have hr := e_1011_15
  simp only [e_1011_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8323)) := by
  have c62700 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62701 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62702 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62703 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62704 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62705 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_16 := arithEq_of_rows h (row := 1011) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_16
  simp only [← c62700, ← c62701, ← c62702] at e_1011_16
  have e_1011_17 := arithEq_of_rows h (row := 1011) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_17
  simp only [← c62703, ← c62704, k2, ← c62705] at e_1011_17
  have hr := e_1011_17
  simp only [e_1011_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1011 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8324)) := by
  have c62706 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62707 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62708 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62709 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62710 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62711 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1011_18 := arithEq_of_rows h (row := 1011) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_18
  simp only [← c62706, ← c62707, ← c62708] at e_1011_18
  have e_1011_19 := arithEq_of_rows h (row := 1011) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1011_19
  simp only [← c62709, ← c62710, k2, ← c62711] at e_1011_19
  have hr := e_1011_19
  simp only [e_1011_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8325)) := by
  have c62712 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62713 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62714 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62715 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62716 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62717 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_0 := arithEq_of_rows h (row := 1012) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_0
  simp only [← c62712, ← c62713, ← c62714] at e_1012_0
  have e_1012_1 := arithEq_of_rows h (row := 1012) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_1
  simp only [← c62715, ← c62716, k2, ← c62717] at e_1012_1
  have hr := e_1012_1
  simp only [e_1012_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8326)) := by
  have c62718 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62719 := (publicBatchWrapper89_copies1959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62720 := (publicBatchWrapper89_copies1960 a h).1
  have c62721 := (publicBatchWrapper89_copies1960 a h).2.1
  have c62722 := (publicBatchWrapper89_copies1960 a h).2.2.1
  have c62723 := (publicBatchWrapper89_copies1960 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_2 := arithEq_of_rows h (row := 1012) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_2
  simp only [← c62718, ← c62719, ← c62720] at e_1012_2
  have e_1012_3 := arithEq_of_rows h (row := 1012) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_3
  simp only [← c62721, ← c62722, k2, ← c62723] at e_1012_3
  have hr := e_1012_3
  simp only [e_1012_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8327)) := by
  have c62724 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.1
  have c62725 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.1
  have c62726 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.1
  have c62727 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.1
  have c62728 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.1
  have c62729 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_4 := arithEq_of_rows h (row := 1012) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_4
  simp only [← c62724, ← c62725, ← c62726] at e_1012_4
  have e_1012_5 := arithEq_of_rows h (row := 1012) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_5
  simp only [← c62727, ← c62728, k2, ← c62729] at e_1012_5
  have hr := e_1012_5
  simp only [e_1012_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8328)) := by
  have c62730 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62731 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62732 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62733 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62734 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62735 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_6 := arithEq_of_rows h (row := 1012) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_6
  simp only [← c62730, ← c62731, ← c62732] at e_1012_6
  have e_1012_7 := arithEq_of_rows h (row := 1012) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_7
  simp only [← c62733, ← c62734, k2, ← c62735] at e_1012_7
  have hr := e_1012_7
  simp only [e_1012_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8329)) := by
  have c62736 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62737 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62738 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62739 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62740 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62741 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_8 := arithEq_of_rows h (row := 1012) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_8
  simp only [← c62736, ← c62737, ← c62738] at e_1012_8
  have e_1012_9 := arithEq_of_rows h (row := 1012) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_9
  simp only [← c62739, ← c62740, k2, ← c62741] at e_1012_9
  have hr := e_1012_9
  simp only [e_1012_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8330)) := by
  have c62742 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62743 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62744 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62745 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62746 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62747 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_10 := arithEq_of_rows h (row := 1012) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_10
  simp only [← c62742, ← c62743, ← c62744] at e_1012_10
  have e_1012_11 := arithEq_of_rows h (row := 1012) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_11
  simp only [← c62745, ← c62746, k2, ← c62747] at e_1012_11
  have hr := e_1012_11
  simp only [e_1012_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8331)) := by
  have c62748 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62749 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62750 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62751 := (publicBatchWrapper89_copies1960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62752 := (publicBatchWrapper89_copies1961 a h).1
  have c62753 := (publicBatchWrapper89_copies1961 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_12 := arithEq_of_rows h (row := 1012) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_12
  simp only [← c62748, ← c62749, ← c62750] at e_1012_12
  have e_1012_13 := arithEq_of_rows h (row := 1012) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_13
  simp only [← c62751, ← c62752, k2, ← c62753] at e_1012_13
  have hr := e_1012_13
  simp only [e_1012_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8332)) := by
  have c62754 := (publicBatchWrapper89_copies1961 a h).2.2.1
  have c62755 := (publicBatchWrapper89_copies1961 a h).2.2.2.1
  have c62756 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.1
  have c62757 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.1
  have c62758 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.1
  have c62759 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_14 := arithEq_of_rows h (row := 1012) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_14
  simp only [← c62754, ← c62755, ← c62756] at e_1012_14
  have e_1012_15 := arithEq_of_rows h (row := 1012) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_15
  simp only [← c62757, ← c62758, k2, ← c62759] at e_1012_15
  have hr := e_1012_15
  simp only [e_1012_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8333)) := by
  have c62760 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.1
  have c62761 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.1
  have c62762 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62763 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62764 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62765 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_16 := arithEq_of_rows h (row := 1012) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_16
  simp only [← c62760, ← c62761, ← c62762] at e_1012_16
  have e_1012_17 := arithEq_of_rows h (row := 1012) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_17
  simp only [← c62763, ← c62764, k2, ← c62765] at e_1012_17
  have hr := e_1012_17
  simp only [e_1012_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1012 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8334)) := by
  have c62766 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62767 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62768 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62769 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62770 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62771 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1012_18 := arithEq_of_rows h (row := 1012) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_18
  simp only [← c62766, ← c62767, ← c62768] at e_1012_18
  have e_1012_19 := arithEq_of_rows h (row := 1012) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1012_19
  simp only [← c62769, ← c62770, k2, ← c62771] at e_1012_19
  have hr := e_1012_19
  simp only [e_1012_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8335)) := by
  have c62772 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62773 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62774 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62775 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62776 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62777 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_0 := arithEq_of_rows h (row := 1013) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_0
  simp only [← c62772, ← c62773, ← c62774] at e_1013_0
  have e_1013_1 := arithEq_of_rows h (row := 1013) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_1
  simp only [← c62775, ← c62776, k2, ← c62777] at e_1013_1
  have hr := e_1013_1
  simp only [e_1013_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8336)) := by
  have c62778 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62779 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62780 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62781 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62782 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62783 := (publicBatchWrapper89_copies1961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_2 := arithEq_of_rows h (row := 1013) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_2
  simp only [← c62778, ← c62779, ← c62780] at e_1013_2
  have e_1013_3 := arithEq_of_rows h (row := 1013) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_3
  simp only [← c62781, ← c62782, k2, ← c62783] at e_1013_3
  have hr := e_1013_3
  simp only [e_1013_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8337)) := by
  have c62784 := (publicBatchWrapper89_copies1962 a h).1
  have c62785 := (publicBatchWrapper89_copies1962 a h).2.1
  have c62786 := (publicBatchWrapper89_copies1962 a h).2.2.1
  have c62787 := (publicBatchWrapper89_copies1962 a h).2.2.2.1
  have c62788 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.1
  have c62789 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_4 := arithEq_of_rows h (row := 1013) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_4
  simp only [← c62784, ← c62785, ← c62786] at e_1013_4
  have e_1013_5 := arithEq_of_rows h (row := 1013) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_5
  simp only [← c62787, ← c62788, k2, ← c62789] at e_1013_5
  have hr := e_1013_5
  simp only [e_1013_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8338)) := by
  have c62790 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.1
  have c62791 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.1
  have c62792 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.1
  have c62793 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.1
  have c62794 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62795 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_6 := arithEq_of_rows h (row := 1013) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_6
  simp only [← c62790, ← c62791, ← c62792] at e_1013_6
  have e_1013_7 := arithEq_of_rows h (row := 1013) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_7
  simp only [← c62793, ← c62794, k2, ← c62795] at e_1013_7
  have hr := e_1013_7
  simp only [e_1013_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8339)) := by
  have c62796 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62797 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62798 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62799 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62800 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62801 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_8 := arithEq_of_rows h (row := 1013) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_8
  simp only [← c62796, ← c62797, ← c62798] at e_1013_8
  have e_1013_9 := arithEq_of_rows h (row := 1013) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_9
  simp only [← c62799, ← c62800, k2, ← c62801] at e_1013_9
  have hr := e_1013_9
  simp only [e_1013_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8340)) := by
  have c62802 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62803 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62804 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62805 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62806 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62807 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_10 := arithEq_of_rows h (row := 1013) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_10
  simp only [← c62802, ← c62803, ← c62804] at e_1013_10
  have e_1013_11 := arithEq_of_rows h (row := 1013) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_11
  simp only [← c62805, ← c62806, k2, ← c62807] at e_1013_11
  have hr := e_1013_11
  simp only [e_1013_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8341)) := by
  have c62808 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62809 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62810 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62811 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62812 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62813 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_12 := arithEq_of_rows h (row := 1013) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_12
  simp only [← c62808, ← c62809, ← c62810] at e_1013_12
  have e_1013_13 := arithEq_of_rows h (row := 1013) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_13
  simp only [← c62811, ← c62812, k2, ← c62813] at e_1013_13
  have hr := e_1013_13
  simp only [e_1013_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8342)) := by
  have c62814 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62815 := (publicBatchWrapper89_copies1962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62816 := (publicBatchWrapper89_copies1963 a h).1
  have c62817 := (publicBatchWrapper89_copies1963 a h).2.1
  have c62818 := (publicBatchWrapper89_copies1963 a h).2.2.1
  have c62819 := (publicBatchWrapper89_copies1963 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_14 := arithEq_of_rows h (row := 1013) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_14
  simp only [← c62814, ← c62815, ← c62816] at e_1013_14
  have e_1013_15 := arithEq_of_rows h (row := 1013) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_15
  simp only [← c62817, ← c62818, k2, ← c62819] at e_1013_15
  have hr := e_1013_15
  simp only [e_1013_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8343)) := by
  have c62820 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.1
  have c62821 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.1
  have c62822 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.1
  have c62823 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.1
  have c62824 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.1
  have c62825 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_16 := arithEq_of_rows h (row := 1013) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_16
  simp only [← c62820, ← c62821, ← c62822] at e_1013_16
  have e_1013_17 := arithEq_of_rows h (row := 1013) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_17
  simp only [← c62823, ← c62824, k2, ← c62825] at e_1013_17
  have hr := e_1013_17
  simp only [e_1013_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1013 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8344)) := by
  have c62826 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62827 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62828 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62829 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62830 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62831 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1013_18 := arithEq_of_rows h (row := 1013) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_18
  simp only [← c62826, ← c62827, ← c62828] at e_1013_18
  have e_1013_19 := arithEq_of_rows h (row := 1013) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1013_19
  simp only [← c62829, ← c62830, k2, ← c62831] at e_1013_19
  have hr := e_1013_19
  simp only [e_1013_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8345)) := by
  have c62832 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62833 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62834 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62835 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62836 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62837 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_0 := arithEq_of_rows h (row := 1014) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_0
  simp only [← c62832, ← c62833, ← c62834] at e_1014_0
  have e_1014_1 := arithEq_of_rows h (row := 1014) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_1
  simp only [← c62835, ← c62836, k2, ← c62837] at e_1014_1
  have hr := e_1014_1
  simp only [e_1014_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8346)) := by
  have c62838 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62839 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62840 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62841 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62842 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62843 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_2 := arithEq_of_rows h (row := 1014) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_2
  simp only [← c62838, ← c62839, ← c62840] at e_1014_2
  have e_1014_3 := arithEq_of_rows h (row := 1014) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_3
  simp only [← c62841, ← c62842, k2, ← c62843] at e_1014_3
  have hr := e_1014_3
  simp only [e_1014_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8347)) := by
  have c62844 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62845 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62846 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62847 := (publicBatchWrapper89_copies1963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62848 := (publicBatchWrapper89_copies1964 a h).1
  have c62849 := (publicBatchWrapper89_copies1964 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_4 := arithEq_of_rows h (row := 1014) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_4
  simp only [← c62844, ← c62845, ← c62846] at e_1014_4
  have e_1014_5 := arithEq_of_rows h (row := 1014) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_5
  simp only [← c62847, ← c62848, k2, ← c62849] at e_1014_5
  have hr := e_1014_5
  simp only [e_1014_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8348)) := by
  have c62850 := (publicBatchWrapper89_copies1964 a h).2.2.1
  have c62851 := (publicBatchWrapper89_copies1964 a h).2.2.2.1
  have c62852 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.1
  have c62853 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.1
  have c62854 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.1
  have c62855 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_6 := arithEq_of_rows h (row := 1014) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_6
  simp only [← c62850, ← c62851, ← c62852] at e_1014_6
  have e_1014_7 := arithEq_of_rows h (row := 1014) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_7
  simp only [← c62853, ← c62854, k2, ← c62855] at e_1014_7
  have hr := e_1014_7
  simp only [e_1014_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8349)) := by
  have c62856 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.1
  have c62857 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.1
  have c62858 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62859 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62860 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62861 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_8 := arithEq_of_rows h (row := 1014) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_8
  simp only [← c62856, ← c62857, ← c62858] at e_1014_8
  have e_1014_9 := arithEq_of_rows h (row := 1014) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_9
  simp only [← c62859, ← c62860, k2, ← c62861] at e_1014_9
  have hr := e_1014_9
  simp only [e_1014_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8350)) := by
  have c62862 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62863 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62864 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62865 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62866 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62867 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_10 := arithEq_of_rows h (row := 1014) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_10
  simp only [← c62862, ← c62863, ← c62864] at e_1014_10
  have e_1014_11 := arithEq_of_rows h (row := 1014) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_11
  simp only [← c62865, ← c62866, k2, ← c62867] at e_1014_11
  have hr := e_1014_11
  simp only [e_1014_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8351)) := by
  have c62868 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62869 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62870 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62871 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62872 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62873 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_12 := arithEq_of_rows h (row := 1014) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_12
  simp only [← c62868, ← c62869, ← c62870] at e_1014_12
  have e_1014_13 := arithEq_of_rows h (row := 1014) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_13
  simp only [← c62871, ← c62872, k2, ← c62873] at e_1014_13
  have hr := e_1014_13
  simp only [e_1014_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8352)) := by
  have c62874 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62875 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62876 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62877 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62878 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62879 := (publicBatchWrapper89_copies1964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_14 := arithEq_of_rows h (row := 1014) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_14
  simp only [← c62874, ← c62875, ← c62876] at e_1014_14
  have e_1014_15 := arithEq_of_rows h (row := 1014) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_15
  simp only [← c62877, ← c62878, k2, ← c62879] at e_1014_15
  have hr := e_1014_15
  simp only [e_1014_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8353)) := by
  have c62880 := (publicBatchWrapper89_copies1965 a h).1
  have c62881 := (publicBatchWrapper89_copies1965 a h).2.1
  have c62882 := (publicBatchWrapper89_copies1965 a h).2.2.1
  have c62883 := (publicBatchWrapper89_copies1965 a h).2.2.2.1
  have c62884 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.1
  have c62885 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_16 := arithEq_of_rows h (row := 1014) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_16
  simp only [← c62880, ← c62881, ← c62882] at e_1014_16
  have e_1014_17 := arithEq_of_rows h (row := 1014) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_17
  simp only [← c62883, ← c62884, k2, ← c62885] at e_1014_17
  have hr := e_1014_17
  simp only [e_1014_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1014 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8354)) := by
  have c62886 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.1
  have c62887 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.1
  have c62888 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.1
  have c62889 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.1
  have c62890 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62891 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1014_18 := arithEq_of_rows h (row := 1014) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_18
  simp only [← c62886, ← c62887, ← c62888] at e_1014_18
  have e_1014_19 := arithEq_of_rows h (row := 1014) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1014_19
  simp only [← c62889, ← c62890, k2, ← c62891] at e_1014_19
  have hr := e_1014_19
  simp only [e_1014_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8355)) := by
  have c62892 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62893 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62894 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62895 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62896 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62897 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_0 := arithEq_of_rows h (row := 1015) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_0
  simp only [← c62892, ← c62893, ← c62894] at e_1015_0
  have e_1015_1 := arithEq_of_rows h (row := 1015) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_1
  simp only [← c62895, ← c62896, k2, ← c62897] at e_1015_1
  have hr := e_1015_1
  simp only [e_1015_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8356)) := by
  have c62898 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62899 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62900 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62901 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62902 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62903 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_2 := arithEq_of_rows h (row := 1015) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_2
  simp only [← c62898, ← c62899, ← c62900] at e_1015_2
  have e_1015_3 := arithEq_of_rows h (row := 1015) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_3
  simp only [← c62901, ← c62902, k2, ← c62903] at e_1015_3
  have hr := e_1015_3
  simp only [e_1015_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8357)) := by
  have c62904 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62905 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62906 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62907 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62908 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62909 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_4 := arithEq_of_rows h (row := 1015) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_4
  simp only [← c62904, ← c62905, ← c62906] at e_1015_4
  have e_1015_5 := arithEq_of_rows h (row := 1015) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_5
  simp only [← c62907, ← c62908, k2, ← c62909] at e_1015_5
  have hr := e_1015_5
  simp only [e_1015_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8358)) := by
  have c62910 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62911 := (publicBatchWrapper89_copies1965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62912 := (publicBatchWrapper89_copies1966 a h).1
  have c62913 := (publicBatchWrapper89_copies1966 a h).2.1
  have c62914 := (publicBatchWrapper89_copies1966 a h).2.2.1
  have c62915 := (publicBatchWrapper89_copies1966 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_6 := arithEq_of_rows h (row := 1015) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_6
  simp only [← c62910, ← c62911, ← c62912] at e_1015_6
  have e_1015_7 := arithEq_of_rows h (row := 1015) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_7
  simp only [← c62913, ← c62914, k2, ← c62915] at e_1015_7
  have hr := e_1015_7
  simp only [e_1015_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8359)) := by
  have c62916 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.1
  have c62917 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.1
  have c62918 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.1
  have c62919 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.1
  have c62920 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.1
  have c62921 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_8 := arithEq_of_rows h (row := 1015) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_8
  simp only [← c62916, ← c62917, ← c62918] at e_1015_8
  have e_1015_9 := arithEq_of_rows h (row := 1015) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_9
  simp only [← c62919, ← c62920, k2, ← c62921] at e_1015_9
  have hr := e_1015_9
  simp only [e_1015_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8360)) := by
  have c62922 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62923 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62924 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62925 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62926 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62927 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_10 := arithEq_of_rows h (row := 1015) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_10
  simp only [← c62922, ← c62923, ← c62924] at e_1015_10
  have e_1015_11 := arithEq_of_rows h (row := 1015) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_11
  simp only [← c62925, ← c62926, k2, ← c62927] at e_1015_11
  have hr := e_1015_11
  simp only [e_1015_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8361)) := by
  have c62928 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62929 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62930 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62931 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62932 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62933 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_12 := arithEq_of_rows h (row := 1015) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_12
  simp only [← c62928, ← c62929, ← c62930] at e_1015_12
  have e_1015_13 := arithEq_of_rows h (row := 1015) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_13
  simp only [← c62931, ← c62932, k2, ← c62933] at e_1015_13
  have hr := e_1015_13
  simp only [e_1015_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8362)) := by
  have c62934 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62935 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62936 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62937 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62938 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62939 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_14 := arithEq_of_rows h (row := 1015) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_14
  simp only [← c62934, ← c62935, ← c62936] at e_1015_14
  have e_1015_15 := arithEq_of_rows h (row := 1015) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_15
  simp only [← c62937, ← c62938, k2, ← c62939] at e_1015_15
  have hr := e_1015_15
  simp only [e_1015_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8363)) := by
  have c62940 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62941 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62942 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62943 := (publicBatchWrapper89_copies1966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c62944 := (publicBatchWrapper89_copies1967 a h).1
  have c62945 := (publicBatchWrapper89_copies1967 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_16 := arithEq_of_rows h (row := 1015) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_16
  simp only [← c62940, ← c62941, ← c62942] at e_1015_16
  have e_1015_17 := arithEq_of_rows h (row := 1015) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_17
  simp only [← c62943, ← c62944, k2, ← c62945] at e_1015_17
  have hr := e_1015_17
  simp only [e_1015_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1015 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8364)) := by
  have c62946 := (publicBatchWrapper89_copies1967 a h).2.2.1
  have c62947 := (publicBatchWrapper89_copies1967 a h).2.2.2.1
  have c62948 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.1
  have c62949 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.1
  have c62950 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.1
  have c62951 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1015_18 := arithEq_of_rows h (row := 1015) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_18
  simp only [← c62946, ← c62947, ← c62948] at e_1015_18
  have e_1015_19 := arithEq_of_rows h (row := 1015) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1015_19
  simp only [← c62949, ← c62950, k2, ← c62951] at e_1015_19
  have hr := e_1015_19
  simp only [e_1015_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8365)) := by
  have c62952 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.1
  have c62953 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.1
  have c62954 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62955 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c62956 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62957 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_0 := arithEq_of_rows h (row := 1016) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_0
  simp only [← c62952, ← c62953, ← c62954] at e_1016_0
  have e_1016_1 := arithEq_of_rows h (row := 1016) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_1
  simp only [← c62955, ← c62956, k2, ← c62957] at e_1016_1
  have hr := e_1016_1
  simp only [e_1016_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8366)) := by
  have c62958 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62959 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62960 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62961 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62962 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62963 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_2 := arithEq_of_rows h (row := 1016) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_2
  simp only [← c62958, ← c62959, ← c62960] at e_1016_2
  have e_1016_3 := arithEq_of_rows h (row := 1016) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_3
  simp only [← c62961, ← c62962, k2, ← c62963] at e_1016_3
  have hr := e_1016_3
  simp only [e_1016_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8367)) := by
  have c62964 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62965 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62966 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62967 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62968 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62969 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_4 := arithEq_of_rows h (row := 1016) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_4
  simp only [← c62964, ← c62965, ← c62966] at e_1016_4
  have e_1016_5 := arithEq_of_rows h (row := 1016) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_5
  simp only [← c62967, ← c62968, k2, ← c62969] at e_1016_5
  have hr := e_1016_5
  simp only [e_1016_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8368)) := by
  have c62970 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62971 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62972 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62973 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62974 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62975 := (publicBatchWrapper89_copies1967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_6 := arithEq_of_rows h (row := 1016) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_6
  simp only [← c62970, ← c62971, ← c62972] at e_1016_6
  have e_1016_7 := arithEq_of_rows h (row := 1016) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_7
  simp only [← c62973, ← c62974, k2, ← c62975] at e_1016_7
  have hr := e_1016_7
  simp only [e_1016_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8369)) := by
  have c62976 := (publicBatchWrapper89_copies1968 a h).1
  have c62977 := (publicBatchWrapper89_copies1968 a h).2.1
  have c62978 := (publicBatchWrapper89_copies1968 a h).2.2.1
  have c62979 := (publicBatchWrapper89_copies1968 a h).2.2.2.1
  have c62980 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.1
  have c62981 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_8 := arithEq_of_rows h (row := 1016) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_8
  simp only [← c62976, ← c62977, ← c62978] at e_1016_8
  have e_1016_9 := arithEq_of_rows h (row := 1016) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_9
  simp only [← c62979, ← c62980, k2, ← c62981] at e_1016_9
  have hr := e_1016_9
  simp only [e_1016_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8410)) := by
  have c62982 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.1
  have c62983 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.1
  have c62984 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.1
  have c62985 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.1
  have c62986 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.1
  have c62987 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_10 := arithEq_of_rows h (row := 1016) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_10
  simp only [← c62982, ← c62983, ← c62984] at e_1016_10
  have e_1016_11 := arithEq_of_rows h (row := 1016) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_11
  simp only [← c62985, ← c62986, k2, ← c62987] at e_1016_11
  have hr := e_1016_11
  simp only [e_1016_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8411)) := by
  have c62988 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62989 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62990 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62991 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62992 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62993 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_12 := arithEq_of_rows h (row := 1016) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_12
  simp only [← c62988, ← c62989, ← c62990] at e_1016_12
  have e_1016_13 := arithEq_of_rows h (row := 1016) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_13
  simp only [← c62991, ← c62992, k2, ← c62993] at e_1016_13
  have hr := e_1016_13
  simp only [e_1016_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8412)) := by
  have c62994 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62995 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62996 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62997 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62998 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62999 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_14 := arithEq_of_rows h (row := 1016) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_14
  simp only [← c62994, ← c62995, ← c62996] at e_1016_14
  have e_1016_15 := arithEq_of_rows h (row := 1016) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_15
  simp only [← c62997, ← c62998, k2, ← c62999] at e_1016_15
  have hr := e_1016_15
  simp only [e_1016_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8413)) := by
  have c63000 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63001 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63002 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63003 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63004 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63005 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_16 := arithEq_of_rows h (row := 1016) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_16
  simp only [← c63000, ← c63001, ← c63002] at e_1016_16
  have e_1016_17 := arithEq_of_rows h (row := 1016) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_17
  simp only [← c63003, ← c63004, k2, ← c63005] at e_1016_17
  have hr := e_1016_17
  simp only [e_1016_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1016 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8414)) := by
  have c63006 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63007 := (publicBatchWrapper89_copies1968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63008 := (publicBatchWrapper89_copies1969 a h).1
  have c63009 := (publicBatchWrapper89_copies1969 a h).2.1
  have c63010 := (publicBatchWrapper89_copies1969 a h).2.2.1
  have c63011 := (publicBatchWrapper89_copies1969 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1016_18 := arithEq_of_rows h (row := 1016) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_18
  simp only [← c63006, ← c63007, ← c63008] at e_1016_18
  have e_1016_19 := arithEq_of_rows h (row := 1016) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1016_19
  simp only [← c63009, ← c63010, k2, ← c63011] at e_1016_19
  have hr := e_1016_19
  simp only [e_1016_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8415)) := by
  have c63012 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.1
  have c63013 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.1
  have c63014 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.1
  have c63015 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.1
  have c63016 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.1
  have c63017 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_0 := arithEq_of_rows h (row := 1017) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_0
  simp only [← c63012, ← c63013, ← c63014] at e_1017_0
  have e_1017_1 := arithEq_of_rows h (row := 1017) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_1
  simp only [← c63015, ← c63016, k2, ← c63017] at e_1017_1
  have hr := e_1017_1
  simp only [e_1017_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8416)) := by
  have c63018 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63019 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63020 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63021 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63022 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63023 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_2 := arithEq_of_rows h (row := 1017) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_2
  simp only [← c63018, ← c63019, ← c63020] at e_1017_2
  have e_1017_3 := arithEq_of_rows h (row := 1017) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_3
  simp only [← c63021, ← c63022, k2, ← c63023] at e_1017_3
  have hr := e_1017_3
  simp only [e_1017_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8417)) := by
  have c63024 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63025 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63026 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63027 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63028 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63029 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_4 := arithEq_of_rows h (row := 1017) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_4
  simp only [← c63024, ← c63025, ← c63026] at e_1017_4
  have e_1017_5 := arithEq_of_rows h (row := 1017) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_5
  simp only [← c63027, ← c63028, k2, ← c63029] at e_1017_5
  have hr := e_1017_5
  simp only [e_1017_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8418)) := by
  have c63030 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63031 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63032 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63033 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63034 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63035 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_6 := arithEq_of_rows h (row := 1017) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_6
  simp only [← c63030, ← c63031, ← c63032] at e_1017_6
  have e_1017_7 := arithEq_of_rows h (row := 1017) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_7
  simp only [← c63033, ← c63034, k2, ← c63035] at e_1017_7
  have hr := e_1017_7
  simp only [e_1017_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8419)) := by
  have c63036 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63037 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63038 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63039 := (publicBatchWrapper89_copies1969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63040 := (publicBatchWrapper89_copies1970 a h).1
  have c63041 := (publicBatchWrapper89_copies1970 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_8 := arithEq_of_rows h (row := 1017) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_8
  simp only [← c63036, ← c63037, ← c63038] at e_1017_8
  have e_1017_9 := arithEq_of_rows h (row := 1017) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_9
  simp only [← c63039, ← c63040, k2, ← c63041] at e_1017_9
  have hr := e_1017_9
  simp only [e_1017_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8420)) := by
  have c63042 := (publicBatchWrapper89_copies1970 a h).2.2.1
  have c63043 := (publicBatchWrapper89_copies1970 a h).2.2.2.1
  have c63044 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.1
  have c63045 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.1
  have c63046 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.1
  have c63047 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_10 := arithEq_of_rows h (row := 1017) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_10
  simp only [← c63042, ← c63043, ← c63044] at e_1017_10
  have e_1017_11 := arithEq_of_rows h (row := 1017) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_11
  simp only [← c63045, ← c63046, k2, ← c63047] at e_1017_11
  have hr := e_1017_11
  simp only [e_1017_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8421)) := by
  have c63048 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.1
  have c63049 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.1
  have c63050 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63051 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63052 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63053 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_12 := arithEq_of_rows h (row := 1017) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_12
  simp only [← c63048, ← c63049, ← c63050] at e_1017_12
  have e_1017_13 := arithEq_of_rows h (row := 1017) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_13
  simp only [← c63051, ← c63052, k2, ← c63053] at e_1017_13
  have hr := e_1017_13
  simp only [e_1017_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8422)) := by
  have c63054 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63055 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63056 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63057 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63058 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63059 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_14 := arithEq_of_rows h (row := 1017) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_14
  simp only [← c63054, ← c63055, ← c63056] at e_1017_14
  have e_1017_15 := arithEq_of_rows h (row := 1017) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_15
  simp only [← c63057, ← c63058, k2, ← c63059] at e_1017_15
  have hr := e_1017_15
  simp only [e_1017_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8423)) := by
  have c63060 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63061 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63062 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63063 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63064 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63065 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_16 := arithEq_of_rows h (row := 1017) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_16
  simp only [← c63060, ← c63061, ← c63062] at e_1017_16
  have e_1017_17 := arithEq_of_rows h (row := 1017) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_17
  simp only [← c63063, ← c63064, k2, ← c63065] at e_1017_17
  have hr := e_1017_17
  simp only [e_1017_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1017 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8424)) := by
  have c63066 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63067 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63068 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63069 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63070 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63071 := (publicBatchWrapper89_copies1970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1017_18 := arithEq_of_rows h (row := 1017) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_18
  simp only [← c63066, ← c63067, ← c63068] at e_1017_18
  have e_1017_19 := arithEq_of_rows h (row := 1017) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1017_19
  simp only [← c63069, ← c63070, k2, ← c63071] at e_1017_19
  have hr := e_1017_19
  simp only [e_1017_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8425)) := by
  have c63072 := (publicBatchWrapper89_copies1971 a h).1
  have c63073 := (publicBatchWrapper89_copies1971 a h).2.1
  have c63074 := (publicBatchWrapper89_copies1971 a h).2.2.1
  have c63075 := (publicBatchWrapper89_copies1971 a h).2.2.2.1
  have c63076 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.1
  have c63077 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_0 := arithEq_of_rows h (row := 1018) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_0
  simp only [← c63072, ← c63073, ← c63074] at e_1018_0
  have e_1018_1 := arithEq_of_rows h (row := 1018) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_1
  simp only [← c63075, ← c63076, k2, ← c63077] at e_1018_1
  have hr := e_1018_1
  simp only [e_1018_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8426)) := by
  have c63078 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.1
  have c63079 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.1
  have c63080 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.1
  have c63081 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.1
  have c63082 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63083 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_2 := arithEq_of_rows h (row := 1018) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_2
  simp only [← c63078, ← c63079, ← c63080] at e_1018_2
  have e_1018_3 := arithEq_of_rows h (row := 1018) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_3
  simp only [← c63081, ← c63082, k2, ← c63083] at e_1018_3
  have hr := e_1018_3
  simp only [e_1018_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8427)) := by
  have c63084 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63085 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63086 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63087 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63088 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63089 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_4 := arithEq_of_rows h (row := 1018) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_4
  simp only [← c63084, ← c63085, ← c63086] at e_1018_4
  have e_1018_5 := arithEq_of_rows h (row := 1018) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_5
  simp only [← c63087, ← c63088, k2, ← c63089] at e_1018_5
  have hr := e_1018_5
  simp only [e_1018_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8428)) := by
  have c63090 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63091 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63092 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63093 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63094 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63095 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_6 := arithEq_of_rows h (row := 1018) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_6
  simp only [← c63090, ← c63091, ← c63092] at e_1018_6
  have e_1018_7 := arithEq_of_rows h (row := 1018) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_7
  simp only [← c63093, ← c63094, k2, ← c63095] at e_1018_7
  have hr := e_1018_7
  simp only [e_1018_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8429)) := by
  have c63096 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63097 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63098 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63099 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63100 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63101 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_8 := arithEq_of_rows h (row := 1018) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_8
  simp only [← c63096, ← c63097, ← c63098] at e_1018_8
  have e_1018_9 := arithEq_of_rows h (row := 1018) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_9
  simp only [← c63099, ← c63100, k2, ← c63101] at e_1018_9
  have hr := e_1018_9
  simp only [e_1018_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8430)) := by
  have c63102 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63103 := (publicBatchWrapper89_copies1971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63104 := (publicBatchWrapper89_copies1972 a h).1
  have c63105 := (publicBatchWrapper89_copies1972 a h).2.1
  have c63106 := (publicBatchWrapper89_copies1972 a h).2.2.1
  have c63107 := (publicBatchWrapper89_copies1972 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_10 := arithEq_of_rows h (row := 1018) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_10
  simp only [← c63102, ← c63103, ← c63104] at e_1018_10
  have e_1018_11 := arithEq_of_rows h (row := 1018) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_11
  simp only [← c63105, ← c63106, k2, ← c63107] at e_1018_11
  have hr := e_1018_11
  simp only [e_1018_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8431)) := by
  have c63108 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.1
  have c63109 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.1
  have c63110 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.1
  have c63111 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.1
  have c63112 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.1
  have c63113 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_12 := arithEq_of_rows h (row := 1018) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_12
  simp only [← c63108, ← c63109, ← c63110] at e_1018_12
  have e_1018_13 := arithEq_of_rows h (row := 1018) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_13
  simp only [← c63111, ← c63112, k2, ← c63113] at e_1018_13
  have hr := e_1018_13
  simp only [e_1018_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8432)) := by
  have c63114 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63115 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63116 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63117 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63118 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63119 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_14 := arithEq_of_rows h (row := 1018) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_14
  simp only [← c63114, ← c63115, ← c63116] at e_1018_14
  have e_1018_15 := arithEq_of_rows h (row := 1018) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_15
  simp only [← c63117, ← c63118, k2, ← c63119] at e_1018_15
  have hr := e_1018_15
  simp only [e_1018_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8433)) := by
  have c63120 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63121 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63122 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63123 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63124 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63125 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_16 := arithEq_of_rows h (row := 1018) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_16
  simp only [← c63120, ← c63121, ← c63122] at e_1018_16
  have e_1018_17 := arithEq_of_rows h (row := 1018) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_17
  simp only [← c63123, ← c63124, k2, ← c63125] at e_1018_17
  have hr := e_1018_17
  simp only [e_1018_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1018 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8434)) := by
  have c63126 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63127 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63128 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63129 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63130 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63131 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1018_18 := arithEq_of_rows h (row := 1018) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_18
  simp only [← c63126, ← c63127, ← c63128] at e_1018_18
  have e_1018_19 := arithEq_of_rows h (row := 1018) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1018_19
  simp only [← c63129, ← c63130, k2, ← c63131] at e_1018_19
  have hr := e_1018_19
  simp only [e_1018_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8435)) := by
  have c63132 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63133 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63134 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63135 := (publicBatchWrapper89_copies1972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63136 := (publicBatchWrapper89_copies1973 a h).1
  have c63137 := (publicBatchWrapper89_copies1973 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_0 := arithEq_of_rows h (row := 1019) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_0
  simp only [← c63132, ← c63133, ← c63134] at e_1019_0
  have e_1019_1 := arithEq_of_rows h (row := 1019) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_1
  simp only [← c63135, ← c63136, k2, ← c63137] at e_1019_1
  have hr := e_1019_1
  simp only [e_1019_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8436)) := by
  have c63138 := (publicBatchWrapper89_copies1973 a h).2.2.1
  have c63139 := (publicBatchWrapper89_copies1973 a h).2.2.2.1
  have c63140 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.1
  have c63141 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.1
  have c63142 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.1
  have c63143 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_2 := arithEq_of_rows h (row := 1019) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_2
  simp only [← c63138, ← c63139, ← c63140] at e_1019_2
  have e_1019_3 := arithEq_of_rows h (row := 1019) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_3
  simp only [← c63141, ← c63142, k2, ← c63143] at e_1019_3
  have hr := e_1019_3
  simp only [e_1019_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8437)) := by
  have c63144 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.1
  have c63145 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.1
  have c63146 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63147 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63148 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63149 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_4 := arithEq_of_rows h (row := 1019) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_4
  simp only [← c63144, ← c63145, ← c63146] at e_1019_4
  have e_1019_5 := arithEq_of_rows h (row := 1019) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_5
  simp only [← c63147, ← c63148, k2, ← c63149] at e_1019_5
  have hr := e_1019_5
  simp only [e_1019_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8438)) := by
  have c63150 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63151 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63152 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63153 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63154 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63155 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_6 := arithEq_of_rows h (row := 1019) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_6
  simp only [← c63150, ← c63151, ← c63152] at e_1019_6
  have e_1019_7 := arithEq_of_rows h (row := 1019) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_7
  simp only [← c63153, ← c63154, k2, ← c63155] at e_1019_7
  have hr := e_1019_7
  simp only [e_1019_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8439)) := by
  have c63156 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63157 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63158 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63159 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63160 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63161 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_8 := arithEq_of_rows h (row := 1019) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_8
  simp only [← c63156, ← c63157, ← c63158] at e_1019_8
  have e_1019_9 := arithEq_of_rows h (row := 1019) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_9
  simp only [← c63159, ← c63160, k2, ← c63161] at e_1019_9
  have hr := e_1019_9
  simp only [e_1019_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8440)) := by
  have c63162 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63163 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63164 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63165 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63166 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63167 := (publicBatchWrapper89_copies1973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_10 := arithEq_of_rows h (row := 1019) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_10
  simp only [← c63162, ← c63163, ← c63164] at e_1019_10
  have e_1019_11 := arithEq_of_rows h (row := 1019) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_11
  simp only [← c63165, ← c63166, k2, ← c63167] at e_1019_11
  have hr := e_1019_11
  simp only [e_1019_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8441)) := by
  have c63168 := (publicBatchWrapper89_copies1974 a h).1
  have c63169 := (publicBatchWrapper89_copies1974 a h).2.1
  have c63170 := (publicBatchWrapper89_copies1974 a h).2.2.1
  have c63171 := (publicBatchWrapper89_copies1974 a h).2.2.2.1
  have c63172 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.1
  have c63173 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_12 := arithEq_of_rows h (row := 1019) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_12
  simp only [← c63168, ← c63169, ← c63170] at e_1019_12
  have e_1019_13 := arithEq_of_rows h (row := 1019) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_13
  simp only [← c63171, ← c63172, k2, ← c63173] at e_1019_13
  have hr := e_1019_13
  simp only [e_1019_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8442)) := by
  have c63174 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.1
  have c63175 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.1
  have c63176 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.1
  have c63177 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.1
  have c63178 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63179 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_14 := arithEq_of_rows h (row := 1019) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_14
  simp only [← c63174, ← c63175, ← c63176] at e_1019_14
  have e_1019_15 := arithEq_of_rows h (row := 1019) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_15
  simp only [← c63177, ← c63178, k2, ← c63179] at e_1019_15
  have hr := e_1019_15
  simp only [e_1019_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8443)) := by
  have c63180 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63181 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63182 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63183 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63184 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63185 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_16 := arithEq_of_rows h (row := 1019) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_16
  simp only [← c63180, ← c63181, ← c63182] at e_1019_16
  have e_1019_17 := arithEq_of_rows h (row := 1019) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_17
  simp only [← c63183, ← c63184, k2, ← c63185] at e_1019_17
  have hr := e_1019_17
  simp only [e_1019_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1019 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8444)) := by
  have c63186 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63187 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63188 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63189 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63190 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63191 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1019_18 := arithEq_of_rows h (row := 1019) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_18
  simp only [← c63186, ← c63187, ← c63188] at e_1019_18
  have e_1019_19 := arithEq_of_rows h (row := 1019) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1019_19
  simp only [← c63189, ← c63190, k2, ← c63191] at e_1019_19
  have hr := e_1019_19
  simp only [e_1019_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8445)) := by
  have c63192 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63193 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63194 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63195 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63196 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63197 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_0 := arithEq_of_rows h (row := 1020) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_0
  simp only [← c63192, ← c63193, ← c63194] at e_1020_0
  have e_1020_1 := arithEq_of_rows h (row := 1020) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_1
  simp only [← c63195, ← c63196, k2, ← c63197] at e_1020_1
  have hr := e_1020_1
  simp only [e_1020_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8446)) := by
  have c63198 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63199 := (publicBatchWrapper89_copies1974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63200 := (publicBatchWrapper89_copies1975 a h).1
  have c63201 := (publicBatchWrapper89_copies1975 a h).2.1
  have c63202 := (publicBatchWrapper89_copies1975 a h).2.2.1
  have c63203 := (publicBatchWrapper89_copies1975 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_2 := arithEq_of_rows h (row := 1020) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_2
  simp only [← c63198, ← c63199, ← c63200] at e_1020_2
  have e_1020_3 := arithEq_of_rows h (row := 1020) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_3
  simp only [← c63201, ← c63202, k2, ← c63203] at e_1020_3
  have hr := e_1020_3
  simp only [e_1020_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8447)) := by
  have c63204 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.1
  have c63205 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.1
  have c63206 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.1
  have c63207 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.1
  have c63208 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.1
  have c63209 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_4 := arithEq_of_rows h (row := 1020) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_4
  simp only [← c63204, ← c63205, ← c63206] at e_1020_4
  have e_1020_5 := arithEq_of_rows h (row := 1020) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_5
  simp only [← c63207, ← c63208, k2, ← c63209] at e_1020_5
  have hr := e_1020_5
  simp only [e_1020_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8448)) := by
  have c63210 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63211 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63212 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63213 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63214 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63215 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_6 := arithEq_of_rows h (row := 1020) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_6
  simp only [← c63210, ← c63211, ← c63212] at e_1020_6
  have e_1020_7 := arithEq_of_rows h (row := 1020) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_7
  simp only [← c63213, ← c63214, k2, ← c63215] at e_1020_7
  have hr := e_1020_7
  simp only [e_1020_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8449)) := by
  have c63216 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63217 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63218 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63219 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63220 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63221 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_8 := arithEq_of_rows h (row := 1020) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_8
  simp only [← c63216, ← c63217, ← c63218] at e_1020_8
  have e_1020_9 := arithEq_of_rows h (row := 1020) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_9
  simp only [← c63219, ← c63220, k2, ← c63221] at e_1020_9
  have hr := e_1020_9
  simp only [e_1020_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8450)) := by
  have c63222 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63223 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63224 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63225 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63226 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63227 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_10 := arithEq_of_rows h (row := 1020) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_10
  simp only [← c63222, ← c63223, ← c63224] at e_1020_10
  have e_1020_11 := arithEq_of_rows h (row := 1020) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_11
  simp only [← c63225, ← c63226, k2, ← c63227] at e_1020_11
  have hr := e_1020_11
  simp only [e_1020_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8451)) := by
  have c63228 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63229 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63230 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63231 := (publicBatchWrapper89_copies1975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63232 := (publicBatchWrapper89_copies1976 a h).1
  have c63233 := (publicBatchWrapper89_copies1976 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_12 := arithEq_of_rows h (row := 1020) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_12
  simp only [← c63228, ← c63229, ← c63230] at e_1020_12
  have e_1020_13 := arithEq_of_rows h (row := 1020) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_13
  simp only [← c63231, ← c63232, k2, ← c63233] at e_1020_13
  have hr := e_1020_13
  simp only [e_1020_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8452)) := by
  have c63234 := (publicBatchWrapper89_copies1976 a h).2.2.1
  have c63235 := (publicBatchWrapper89_copies1976 a h).2.2.2.1
  have c63236 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.1
  have c63237 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.1
  have c63238 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.1
  have c63239 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_14 := arithEq_of_rows h (row := 1020) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_14
  simp only [← c63234, ← c63235, ← c63236] at e_1020_14
  have e_1020_15 := arithEq_of_rows h (row := 1020) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_15
  simp only [← c63237, ← c63238, k2, ← c63239] at e_1020_15
  have hr := e_1020_15
  simp only [e_1020_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8453)) := by
  have c63240 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.1
  have c63241 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.1
  have c63242 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63243 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63244 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63245 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_16 := arithEq_of_rows h (row := 1020) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_16
  simp only [← c63240, ← c63241, ← c63242] at e_1020_16
  have e_1020_17 := arithEq_of_rows h (row := 1020) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_17
  simp only [← c63243, ← c63244, k2, ← c63245] at e_1020_17
  have hr := e_1020_17
  simp only [e_1020_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8454)) := by
  have c63246 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63247 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63248 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63249 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63250 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63251 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_18 := arithEq_of_rows h (row := 1020) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_18
  simp only [← c63246, ← c63247, ← c63248] at e_1020_18
  have e_1020_19 := arithEq_of_rows h (row := 1020) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_19
  simp only [← c63249, ← c63250, k2, ← c63251] at e_1020_19
  have hr := e_1020_19
  simp only [e_1020_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8455)) := by
  have c63252 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63253 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63254 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63255 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63256 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63257 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_0 := arithEq_of_rows h (row := 1021) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_0
  simp only [← c63252, ← c63253, ← c63254] at e_1021_0
  have e_1021_1 := arithEq_of_rows h (row := 1021) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_1
  simp only [← c63255, ← c63256, k2, ← c63257] at e_1021_1
  have hr := e_1021_1
  simp only [e_1021_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8456)) := by
  have c63258 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63259 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63260 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63261 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63262 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63263 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_2 := arithEq_of_rows h (row := 1021) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_2
  simp only [← c63258, ← c63259, ← c63260] at e_1021_2
  have e_1021_3 := arithEq_of_rows h (row := 1021) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_3
  simp only [← c63261, ← c63262, k2, ← c63263] at e_1021_3
  have hr := e_1021_3
  simp only [e_1021_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8457)) := by
  have c63264 := (publicBatchWrapper89_copies1977 a h).1
  have c63265 := (publicBatchWrapper89_copies1977 a h).2.1
  have c63266 := (publicBatchWrapper89_copies1977 a h).2.2.1
  have c63267 := (publicBatchWrapper89_copies1977 a h).2.2.2.1
  have c63268 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.1
  have c63269 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_4 := arithEq_of_rows h (row := 1021) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_4
  simp only [← c63264, ← c63265, ← c63266] at e_1021_4
  have e_1021_5 := arithEq_of_rows h (row := 1021) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_5
  simp only [← c63267, ← c63268, k2, ← c63269] at e_1021_5
  have hr := e_1021_5
  simp only [e_1021_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8458)) := by
  have c63270 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.1
  have c63271 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.1
  have c63272 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.1
  have c63273 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.1
  have c63274 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63275 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_6 := arithEq_of_rows h (row := 1021) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_6
  simp only [← c63270, ← c63271, ← c63272] at e_1021_6
  have e_1021_7 := arithEq_of_rows h (row := 1021) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_7
  simp only [← c63273, ← c63274, k2, ← c63275] at e_1021_7
  have hr := e_1021_7
  simp only [e_1021_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8459)) := by
  have c63276 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63277 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63278 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63279 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63280 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63281 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_8 := arithEq_of_rows h (row := 1021) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_8
  simp only [← c63276, ← c63277, ← c63278] at e_1021_8
  have e_1021_9 := arithEq_of_rows h (row := 1021) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_9
  simp only [← c63279, ← c63280, k2, ← c63281] at e_1021_9
  have hr := e_1021_9
  simp only [e_1021_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8460)) := by
  have c63282 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63283 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63284 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63285 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63286 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63287 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_10 := arithEq_of_rows h (row := 1021) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_10
  simp only [← c63282, ← c63283, ← c63284] at e_1021_10
  have e_1021_11 := arithEq_of_rows h (row := 1021) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_11
  simp only [← c63285, ← c63286, k2, ← c63287] at e_1021_11
  have hr := e_1021_11
  simp only [e_1021_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8461)) := by
  have c63288 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63289 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63290 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63291 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63292 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63293 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_12 := arithEq_of_rows h (row := 1021) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_12
  simp only [← c63288, ← c63289, ← c63290] at e_1021_12
  have e_1021_13 := arithEq_of_rows h (row := 1021) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_13
  simp only [← c63291, ← c63292, k2, ← c63293] at e_1021_13
  have hr := e_1021_13
  simp only [e_1021_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8462)) := by
  have c63294 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63295 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63296 := (publicBatchWrapper89_copies1978 a h).1
  have c63297 := (publicBatchWrapper89_copies1978 a h).2.1
  have c63298 := (publicBatchWrapper89_copies1978 a h).2.2.1
  have c63299 := (publicBatchWrapper89_copies1978 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_14 := arithEq_of_rows h (row := 1021) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_14
  simp only [← c63294, ← c63295, ← c63296] at e_1021_14
  have e_1021_15 := arithEq_of_rows h (row := 1021) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_15
  simp only [← c63297, ← c63298, k2, ← c63299] at e_1021_15
  have hr := e_1021_15
  simp only [e_1021_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8463)) := by
  have c63300 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.1
  have c63301 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.1
  have c63302 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.1
  have c63303 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.1
  have c63304 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.1
  have c63305 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_16 := arithEq_of_rows h (row := 1021) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_16
  simp only [← c63300, ← c63301, ← c63302] at e_1021_16
  have e_1021_17 := arithEq_of_rows h (row := 1021) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_17
  simp only [← c63303, ← c63304, k2, ← c63305] at e_1021_17
  have hr := e_1021_17
  simp only [e_1021_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8464)) := by
  have c63306 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63307 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63308 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63309 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63310 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63311 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_18 := arithEq_of_rows h (row := 1021) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_18
  simp only [← c63306, ← c63307, ← c63308] at e_1021_18
  have e_1021_19 := arithEq_of_rows h (row := 1021) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_19
  simp only [← c63309, ← c63310, k2, ← c63311] at e_1021_19
  have hr := e_1021_19
  simp only [e_1021_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8465)) := by
  have c63312 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63313 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63314 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63315 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63316 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63317 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_0 := arithEq_of_rows h (row := 1022) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_0
  simp only [← c63312, ← c63313, ← c63314] at e_1022_0
  have e_1022_1 := arithEq_of_rows h (row := 1022) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_1
  simp only [← c63315, ← c63316, k2, ← c63317] at e_1022_1
  have hr := e_1022_1
  simp only [e_1022_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8466)) := by
  have c63318 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63319 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63320 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63321 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63322 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63323 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_2 := arithEq_of_rows h (row := 1022) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_2
  simp only [← c63318, ← c63319, ← c63320] at e_1022_2
  have e_1022_3 := arithEq_of_rows h (row := 1022) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_3
  simp only [← c63321, ← c63322, k2, ← c63323] at e_1022_3
  have hr := e_1022_3
  simp only [e_1022_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8467)) := by
  have c63324 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63325 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63326 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63327 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63328 := (publicBatchWrapper89_copies1979 a h).1
  have c63329 := (publicBatchWrapper89_copies1979 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_4 := arithEq_of_rows h (row := 1022) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_4
  simp only [← c63324, ← c63325, ← c63326] at e_1022_4
  have e_1022_5 := arithEq_of_rows h (row := 1022) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_5
  simp only [← c63327, ← c63328, k2, ← c63329] at e_1022_5
  have hr := e_1022_5
  simp only [e_1022_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8468)) := by
  have c63330 := (publicBatchWrapper89_copies1979 a h).2.2.1
  have c63331 := (publicBatchWrapper89_copies1979 a h).2.2.2.1
  have c63332 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.1
  have c63333 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.1
  have c63334 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.1
  have c63335 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_6 := arithEq_of_rows h (row := 1022) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_6
  simp only [← c63330, ← c63331, ← c63332] at e_1022_6
  have e_1022_7 := arithEq_of_rows h (row := 1022) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_7
  simp only [← c63333, ← c63334, k2, ← c63335] at e_1022_7
  have hr := e_1022_7
  simp only [e_1022_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8469)) := by
  have c63336 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.1
  have c63337 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.1
  have c63338 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63339 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63340 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63341 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_8 := arithEq_of_rows h (row := 1022) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_8
  simp only [← c63336, ← c63337, ← c63338] at e_1022_8
  have e_1022_9 := arithEq_of_rows h (row := 1022) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_9
  simp only [← c63339, ← c63340, k2, ← c63341] at e_1022_9
  have hr := e_1022_9
  simp only [e_1022_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8470)) := by
  have c63342 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63343 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63344 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63345 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63346 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63347 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_10 := arithEq_of_rows h (row := 1022) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_10
  simp only [← c63342, ← c63343, ← c63344] at e_1022_10
  have e_1022_11 := arithEq_of_rows h (row := 1022) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_11
  simp only [← c63345, ← c63346, k2, ← c63347] at e_1022_11
  have hr := e_1022_11
  simp only [e_1022_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8471)) := by
  have c63348 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63349 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63350 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63351 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63352 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63353 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_12 := arithEq_of_rows h (row := 1022) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_12
  simp only [← c63348, ← c63349, ← c63350] at e_1022_12
  have e_1022_13 := arithEq_of_rows h (row := 1022) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_13
  simp only [← c63351, ← c63352, k2, ← c63353] at e_1022_13
  have hr := e_1022_13
  simp only [e_1022_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8472)) := by
  have c63354 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63355 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63356 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63357 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63358 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63359 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_14 := arithEq_of_rows h (row := 1022) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_14
  simp only [← c63354, ← c63355, ← c63356] at e_1022_14
  have e_1022_15 := arithEq_of_rows h (row := 1022) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_15
  simp only [← c63357, ← c63358, k2, ← c63359] at e_1022_15
  have hr := e_1022_15
  simp only [e_1022_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8473)) := by
  have c63360 := (publicBatchWrapper89_copies1980 a h).1
  have c63361 := (publicBatchWrapper89_copies1980 a h).2.1
  have c63362 := (publicBatchWrapper89_copies1980 a h).2.2.1
  have c63363 := (publicBatchWrapper89_copies1980 a h).2.2.2.1
  have c63364 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.1
  have c63365 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_16 := arithEq_of_rows h (row := 1022) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_16
  simp only [← c63360, ← c63361, ← c63362] at e_1022_16
  have e_1022_17 := arithEq_of_rows h (row := 1022) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_17
  simp only [← c63363, ← c63364, k2, ← c63365] at e_1022_17
  have hr := e_1022_17
  simp only [e_1022_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8474)) := by
  have c63366 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.1
  have c63367 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.1
  have c63368 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.1
  have c63369 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.1
  have c63370 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63371 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_18 := arithEq_of_rows h (row := 1022) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_18
  simp only [← c63366, ← c63367, ← c63368] at e_1022_18
  have e_1022_19 := arithEq_of_rows h (row := 1022) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_19
  simp only [← c63369, ← c63370, k2, ← c63371] at e_1022_19
  have hr := e_1022_19
  simp only [e_1022_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8475)) := by
  have c63372 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63373 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63374 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63375 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63376 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63377 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_0 := arithEq_of_rows h (row := 1023) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_0
  simp only [← c63372, ← c63373, ← c63374] at e_1023_0
  have e_1023_1 := arithEq_of_rows h (row := 1023) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_1
  simp only [← c63375, ← c63376, k2, ← c63377] at e_1023_1
  have hr := e_1023_1
  simp only [e_1023_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8476)) := by
  have c63378 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63379 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63380 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63381 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63382 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63383 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_2 := arithEq_of_rows h (row := 1023) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_2
  simp only [← c63378, ← c63379, ← c63380] at e_1023_2
  have e_1023_3 := arithEq_of_rows h (row := 1023) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_3
  simp only [← c63381, ← c63382, k2, ← c63383] at e_1023_3
  have hr := e_1023_3
  simp only [e_1023_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8477)) := by
  have c63384 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63385 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63386 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63387 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63388 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63389 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_4 := arithEq_of_rows h (row := 1023) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_4
  simp only [← c63384, ← c63385, ← c63386] at e_1023_4
  have e_1023_5 := arithEq_of_rows h (row := 1023) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_5
  simp only [← c63387, ← c63388, k2, ← c63389] at e_1023_5
  have hr := e_1023_5
  simp only [e_1023_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8478)) := by
  have c63390 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63391 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63392 := (publicBatchWrapper89_copies1981 a h).1
  have c63393 := (publicBatchWrapper89_copies1981 a h).2.1
  have c63394 := (publicBatchWrapper89_copies1981 a h).2.2.1
  have c63395 := (publicBatchWrapper89_copies1981 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_6 := arithEq_of_rows h (row := 1023) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_6
  simp only [← c63390, ← c63391, ← c63392] at e_1023_6
  have e_1023_7 := arithEq_of_rows h (row := 1023) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_7
  simp only [← c63393, ← c63394, k2, ← c63395] at e_1023_7
  have hr := e_1023_7
  simp only [e_1023_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8479)) := by
  have c63396 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.1
  have c63397 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.1
  have c63398 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.1
  have c63399 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.1
  have c63400 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.1
  have c63401 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_8 := arithEq_of_rows h (row := 1023) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_8
  simp only [← c63396, ← c63397, ← c63398] at e_1023_8
  have e_1023_9 := arithEq_of_rows h (row := 1023) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_9
  simp only [← c63399, ← c63400, k2, ← c63401] at e_1023_9
  have hr := e_1023_9
  simp only [e_1023_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8480)) := by
  have c63402 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63403 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63404 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63405 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63406 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63407 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_10 := arithEq_of_rows h (row := 1023) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_10
  simp only [← c63402, ← c63403, ← c63404] at e_1023_10
  have e_1023_11 := arithEq_of_rows h (row := 1023) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_11
  simp only [← c63405, ← c63406, k2, ← c63407] at e_1023_11
  have hr := e_1023_11
  simp only [e_1023_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8481)) := by
  have c63408 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63409 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63410 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63411 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63412 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63413 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_12 := arithEq_of_rows h (row := 1023) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_12
  simp only [← c63408, ← c63409, ← c63410] at e_1023_12
  have e_1023_13 := arithEq_of_rows h (row := 1023) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_13
  simp only [← c63411, ← c63412, k2, ← c63413] at e_1023_13
  have hr := e_1023_13
  simp only [e_1023_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8482)) := by
  have c63414 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63415 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63416 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63417 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63418 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63419 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_14 := arithEq_of_rows h (row := 1023) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_14
  simp only [← c63414, ← c63415, ← c63416] at e_1023_14
  have e_1023_15 := arithEq_of_rows h (row := 1023) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_15
  simp only [← c63417, ← c63418, k2, ← c63419] at e_1023_15
  have hr := e_1023_15
  simp only [e_1023_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8483)) := by
  have c63420 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63421 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63422 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63423 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63424 := (publicBatchWrapper89_copies1982 a h).1
  have c63425 := (publicBatchWrapper89_copies1982 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_16 := arithEq_of_rows h (row := 1023) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_16
  simp only [← c63420, ← c63421, ← c63422] at e_1023_16
  have e_1023_17 := arithEq_of_rows h (row := 1023) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_17
  simp only [← c63423, ← c63424, k2, ← c63425] at e_1023_17
  have hr := e_1023_17
  simp only [e_1023_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8484)) := by
  have c63426 := (publicBatchWrapper89_copies1982 a h).2.2.1
  have c63427 := (publicBatchWrapper89_copies1982 a h).2.2.2.1
  have c63428 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.1
  have c63429 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.1
  have c63430 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.1
  have c63431 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_18 := arithEq_of_rows h (row := 1023) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_18
  simp only [← c63426, ← c63427, ← c63428] at e_1023_18
  have e_1023_19 := arithEq_of_rows h (row := 1023) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_19
  simp only [← c63429, ← c63430, k2, ← c63431] at e_1023_19
  have hr := e_1023_19
  simp only [e_1023_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8485)) := by
  have c63432 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.1
  have c63433 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.1
  have c63434 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63435 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63436 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63437 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_0 := arithEq_of_rows h (row := 1024) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_0
  simp only [← c63432, ← c63433, ← c63434] at e_1024_0
  have e_1024_1 := arithEq_of_rows h (row := 1024) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_1
  simp only [← c63435, ← c63436, k2, ← c63437] at e_1024_1
  have hr := e_1024_1
  simp only [e_1024_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8486)) := by
  have c63438 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63439 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63440 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63441 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63442 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63443 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_2 := arithEq_of_rows h (row := 1024) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_2
  simp only [← c63438, ← c63439, ← c63440] at e_1024_2
  have e_1024_3 := arithEq_of_rows h (row := 1024) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_3
  simp only [← c63441, ← c63442, k2, ← c63443] at e_1024_3
  have hr := e_1024_3
  simp only [e_1024_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8487)) := by
  have c63444 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63445 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63446 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63447 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63448 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63449 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_4 := arithEq_of_rows h (row := 1024) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_4
  simp only [← c63444, ← c63445, ← c63446] at e_1024_4
  have e_1024_5 := arithEq_of_rows h (row := 1024) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_5
  simp only [← c63447, ← c63448, k2, ← c63449] at e_1024_5
  have hr := e_1024_5
  simp only [e_1024_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8488)) := by
  have c63450 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63451 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63452 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63453 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63454 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63455 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_6 := arithEq_of_rows h (row := 1024) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_6
  simp only [← c63450, ← c63451, ← c63452] at e_1024_6
  have e_1024_7 := arithEq_of_rows h (row := 1024) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_7
  simp only [← c63453, ← c63454, k2, ← c63455] at e_1024_7
  have hr := e_1024_7
  simp only [e_1024_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8489)) := by
  have c63456 := (publicBatchWrapper89_copies1983 a h).1
  have c63457 := (publicBatchWrapper89_copies1983 a h).2.1
  have c63458 := (publicBatchWrapper89_copies1983 a h).2.2.1
  have c63459 := (publicBatchWrapper89_copies1983 a h).2.2.2.1
  have c63460 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.1
  have c63461 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_8 := arithEq_of_rows h (row := 1024) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_8
  simp only [← c63456, ← c63457, ← c63458] at e_1024_8
  have e_1024_9 := arithEq_of_rows h (row := 1024) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_9
  simp only [← c63459, ← c63460, k2, ← c63461] at e_1024_9
  have hr := e_1024_9
  simp only [e_1024_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8490)) := by
  have c63462 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.1
  have c63463 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.1
  have c63464 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.1
  have c63465 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.1
  have c63466 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63467 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_10 := arithEq_of_rows h (row := 1024) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_10
  simp only [← c63462, ← c63463, ← c63464] at e_1024_10
  have e_1024_11 := arithEq_of_rows h (row := 1024) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_11
  simp only [← c63465, ← c63466, k2, ← c63467] at e_1024_11
  have hr := e_1024_11
  simp only [e_1024_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8491)) := by
  have c63468 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63469 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63470 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63471 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63472 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63473 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_12 := arithEq_of_rows h (row := 1024) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_12
  simp only [← c63468, ← c63469, ← c63470] at e_1024_12
  have e_1024_13 := arithEq_of_rows h (row := 1024) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_13
  simp only [← c63471, ← c63472, k2, ← c63473] at e_1024_13
  have hr := e_1024_13
  simp only [e_1024_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8492)) := by
  have c63474 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63475 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63476 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63477 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63478 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63479 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_14 := arithEq_of_rows h (row := 1024) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_14
  simp only [← c63474, ← c63475, ← c63476] at e_1024_14
  have e_1024_15 := arithEq_of_rows h (row := 1024) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_15
  simp only [← c63477, ← c63478, k2, ← c63479] at e_1024_15
  have hr := e_1024_15
  simp only [e_1024_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8493)) := by
  have c63480 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63481 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63482 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63483 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63484 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63485 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_16 := arithEq_of_rows h (row := 1024) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_16
  simp only [← c63480, ← c63481, ← c63482] at e_1024_16
  have e_1024_17 := arithEq_of_rows h (row := 1024) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_17
  simp only [← c63483, ← c63484, k2, ← c63485] at e_1024_17
  have hr := e_1024_17
  simp only [e_1024_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8494)) := by
  have c63486 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63487 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63488 := (publicBatchWrapper89_copies1984 a h).1
  have c63489 := (publicBatchWrapper89_copies1984 a h).2.1
  have c63490 := (publicBatchWrapper89_copies1984 a h).2.2.1
  have c63491 := (publicBatchWrapper89_copies1984 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_18 := arithEq_of_rows h (row := 1024) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_18
  simp only [← c63486, ← c63487, ← c63488] at e_1024_18
  have e_1024_19 := arithEq_of_rows h (row := 1024) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_19
  simp only [← c63489, ← c63490, k2, ← c63491] at e_1024_19
  have hr := e_1024_19
  simp only [e_1024_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8495)) := by
  have c63492 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.1
  have c63493 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.1
  have c63494 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.1
  have c63495 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.1
  have c63496 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.1
  have c63497 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_0 := arithEq_of_rows h (row := 1025) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_0
  simp only [← c63492, ← c63493, ← c63494] at e_1025_0
  have e_1025_1 := arithEq_of_rows h (row := 1025) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_1
  simp only [← c63495, ← c63496, k2, ← c63497] at e_1025_1
  have hr := e_1025_1
  simp only [e_1025_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8496)) := by
  have c63498 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63499 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63500 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63501 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63502 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63503 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_2 := arithEq_of_rows h (row := 1025) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_2
  simp only [← c63498, ← c63499, ← c63500] at e_1025_2
  have e_1025_3 := arithEq_of_rows h (row := 1025) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_3
  simp only [← c63501, ← c63502, k2, ← c63503] at e_1025_3
  have hr := e_1025_3
  simp only [e_1025_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8497)) := by
  have c63504 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63505 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63506 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63507 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63508 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63509 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_4 := arithEq_of_rows h (row := 1025) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_4
  simp only [← c63504, ← c63505, ← c63506] at e_1025_4
  have e_1025_5 := arithEq_of_rows h (row := 1025) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_5
  simp only [← c63507, ← c63508, k2, ← c63509] at e_1025_5
  have hr := e_1025_5
  simp only [e_1025_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8498)) := by
  have c63510 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63511 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63512 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63513 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63514 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63515 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_6 := arithEq_of_rows h (row := 1025) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_6
  simp only [← c63510, ← c63511, ← c63512] at e_1025_6
  have e_1025_7 := arithEq_of_rows h (row := 1025) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_7
  simp only [← c63513, ← c63514, k2, ← c63515] at e_1025_7
  have hr := e_1025_7
  simp only [e_1025_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8499)) := by
  have c63516 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63517 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63518 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63519 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63520 := (publicBatchWrapper89_copies1985 a h).1
  have c63521 := (publicBatchWrapper89_copies1985 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_8 := arithEq_of_rows h (row := 1025) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_8
  simp only [← c63516, ← c63517, ← c63518] at e_1025_8
  have e_1025_9 := arithEq_of_rows h (row := 1025) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_9
  simp only [← c63519, ← c63520, k2, ← c63521] at e_1025_9
  have hr := e_1025_9
  simp only [e_1025_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8500)) := by
  have c63522 := (publicBatchWrapper89_copies1985 a h).2.2.1
  have c63523 := (publicBatchWrapper89_copies1985 a h).2.2.2.1
  have c63524 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.1
  have c63525 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.1
  have c63526 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.1
  have c63527 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_10 := arithEq_of_rows h (row := 1025) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_10
  simp only [← c63522, ← c63523, ← c63524] at e_1025_10
  have e_1025_11 := arithEq_of_rows h (row := 1025) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_11
  simp only [← c63525, ← c63526, k2, ← c63527] at e_1025_11
  have hr := e_1025_11
  simp only [e_1025_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8501)) := by
  have c63528 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.1
  have c63529 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.1
  have c63530 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63531 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63532 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63533 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_12 := arithEq_of_rows h (row := 1025) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_12
  simp only [← c63528, ← c63529, ← c63530] at e_1025_12
  have e_1025_13 := arithEq_of_rows h (row := 1025) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_13
  simp only [← c63531, ← c63532, k2, ← c63533] at e_1025_13
  have hr := e_1025_13
  simp only [e_1025_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8502)) := by
  have c63534 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63535 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63536 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63537 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63538 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63539 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_14 := arithEq_of_rows h (row := 1025) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_14
  simp only [← c63534, ← c63535, ← c63536] at e_1025_14
  have e_1025_15 := arithEq_of_rows h (row := 1025) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_15
  simp only [← c63537, ← c63538, k2, ← c63539] at e_1025_15
  have hr := e_1025_15
  simp only [e_1025_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8503)) := by
  have c63540 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63541 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63542 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63543 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63544 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63545 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_16 := arithEq_of_rows h (row := 1025) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_16
  simp only [← c63540, ← c63541, ← c63542] at e_1025_16
  have e_1025_17 := arithEq_of_rows h (row := 1025) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_17
  simp only [← c63543, ← c63544, k2, ← c63545] at e_1025_17
  have hr := e_1025_17
  simp only [e_1025_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8504)) := by
  have c63546 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63547 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63548 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63549 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63550 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63551 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_18 := arithEq_of_rows h (row := 1025) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_18
  simp only [← c63546, ← c63547, ← c63548] at e_1025_18
  have e_1025_19 := arithEq_of_rows h (row := 1025) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_19
  simp only [← c63549, ← c63550, k2, ← c63551] at e_1025_19
  have hr := e_1025_19
  simp only [e_1025_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8505)) := by
  have c63552 := (publicBatchWrapper89_copies1986 a h).1
  have c63553 := (publicBatchWrapper89_copies1986 a h).2.1
  have c63554 := (publicBatchWrapper89_copies1986 a h).2.2.1
  have c63555 := (publicBatchWrapper89_copies1986 a h).2.2.2.1
  have c63556 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.1
  have c63557 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_0 := arithEq_of_rows h (row := 1026) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_0
  simp only [← c63552, ← c63553, ← c63554] at e_1026_0
  have e_1026_1 := arithEq_of_rows h (row := 1026) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_1
  simp only [← c63555, ← c63556, k2, ← c63557] at e_1026_1
  have hr := e_1026_1
  simp only [e_1026_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8506)) := by
  have c63558 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.1
  have c63559 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.1
  have c63560 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.1
  have c63561 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.1
  have c63562 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63563 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_2 := arithEq_of_rows h (row := 1026) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_2
  simp only [← c63558, ← c63559, ← c63560] at e_1026_2
  have e_1026_3 := arithEq_of_rows h (row := 1026) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_3
  simp only [← c63561, ← c63562, k2, ← c63563] at e_1026_3
  have hr := e_1026_3
  simp only [e_1026_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8507)) := by
  have c63564 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63565 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63566 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63567 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63568 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63569 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_4 := arithEq_of_rows h (row := 1026) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_4
  simp only [← c63564, ← c63565, ← c63566] at e_1026_4
  have e_1026_5 := arithEq_of_rows h (row := 1026) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_5
  simp only [← c63567, ← c63568, k2, ← c63569] at e_1026_5
  have hr := e_1026_5
  simp only [e_1026_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8508)) := by
  have c63570 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63571 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63572 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63573 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63574 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63575 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_6 := arithEq_of_rows h (row := 1026) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_6
  simp only [← c63570, ← c63571, ← c63572] at e_1026_6
  have e_1026_7 := arithEq_of_rows h (row := 1026) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_7
  simp only [← c63573, ← c63574, k2, ← c63575] at e_1026_7
  have hr := e_1026_7
  simp only [e_1026_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8509)) := by
  have c63576 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63577 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63578 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63579 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63580 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63581 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_8 := arithEq_of_rows h (row := 1026) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_8
  simp only [← c63576, ← c63577, ← c63578] at e_1026_8
  have e_1026_9 := arithEq_of_rows h (row := 1026) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_9
  simp only [← c63579, ← c63580, k2, ← c63581] at e_1026_9
  have hr := e_1026_9
  simp only [e_1026_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8510)) := by
  have c63582 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63583 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63584 := (publicBatchWrapper89_copies1987 a h).1
  have c63585 := (publicBatchWrapper89_copies1987 a h).2.1
  have c63586 := (publicBatchWrapper89_copies1987 a h).2.2.1
  have c63587 := (publicBatchWrapper89_copies1987 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_10 := arithEq_of_rows h (row := 1026) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_10
  simp only [← c63582, ← c63583, ← c63584] at e_1026_10
  have e_1026_11 := arithEq_of_rows h (row := 1026) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_11
  simp only [← c63585, ← c63586, k2, ← c63587] at e_1026_11
  have hr := e_1026_11
  simp only [e_1026_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8511)) := by
  have c63588 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.1
  have c63589 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.1
  have c63590 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.1
  have c63591 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.1
  have c63592 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.1
  have c63593 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_12 := arithEq_of_rows h (row := 1026) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_12
  simp only [← c63588, ← c63589, ← c63590] at e_1026_12
  have e_1026_13 := arithEq_of_rows h (row := 1026) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_13
  simp only [← c63591, ← c63592, k2, ← c63593] at e_1026_13
  have hr := e_1026_13
  simp only [e_1026_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8512)) := by
  have c63594 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63595 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63596 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63597 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63598 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63599 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_14 := arithEq_of_rows h (row := 1026) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_14
  simp only [← c63594, ← c63595, ← c63596] at e_1026_14
  have e_1026_15 := arithEq_of_rows h (row := 1026) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_15
  simp only [← c63597, ← c63598, k2, ← c63599] at e_1026_15
  have hr := e_1026_15
  simp only [e_1026_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8513)) := by
  have c63600 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63601 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63602 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63603 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63604 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63605 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_16 := arithEq_of_rows h (row := 1026) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_16
  simp only [← c63600, ← c63601, ← c63602] at e_1026_16
  have e_1026_17 := arithEq_of_rows h (row := 1026) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_17
  simp only [← c63603, ← c63604, k2, ← c63605] at e_1026_17
  have hr := e_1026_17
  simp only [e_1026_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8514)) := by
  have c63606 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63607 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63608 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63609 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63610 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63611 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_18 := arithEq_of_rows h (row := 1026) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_18
  simp only [← c63606, ← c63607, ← c63608] at e_1026_18
  have e_1026_19 := arithEq_of_rows h (row := 1026) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_19
  simp only [← c63609, ← c63610, k2, ← c63611] at e_1026_19
  have hr := e_1026_19
  simp only [e_1026_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8515)) := by
  have c63612 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63613 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63614 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63615 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63616 := (publicBatchWrapper89_copies1988 a h).1
  have c63617 := (publicBatchWrapper89_copies1988 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_0 := arithEq_of_rows h (row := 1027) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_0
  simp only [← c63612, ← c63613, ← c63614] at e_1027_0
  have e_1027_1 := arithEq_of_rows h (row := 1027) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_1
  simp only [← c63615, ← c63616, k2, ← c63617] at e_1027_1
  have hr := e_1027_1
  simp only [e_1027_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8516)) := by
  have c63618 := (publicBatchWrapper89_copies1988 a h).2.2.1
  have c63619 := (publicBatchWrapper89_copies1988 a h).2.2.2.1
  have c63620 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.1
  have c63621 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.1
  have c63622 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.1
  have c63623 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_2 := arithEq_of_rows h (row := 1027) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_2
  simp only [← c63618, ← c63619, ← c63620] at e_1027_2
  have e_1027_3 := arithEq_of_rows h (row := 1027) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_3
  simp only [← c63621, ← c63622, k2, ← c63623] at e_1027_3
  have hr := e_1027_3
  simp only [e_1027_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8517)) := by
  have c63624 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.1
  have c63625 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.1
  have c63626 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63627 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63628 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63629 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_4 := arithEq_of_rows h (row := 1027) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_4
  simp only [← c63624, ← c63625, ← c63626] at e_1027_4
  have e_1027_5 := arithEq_of_rows h (row := 1027) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_5
  simp only [← c63627, ← c63628, k2, ← c63629] at e_1027_5
  have hr := e_1027_5
  simp only [e_1027_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8518)) := by
  have c63630 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63631 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63632 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63633 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63634 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63635 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_6 := arithEq_of_rows h (row := 1027) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_6
  simp only [← c63630, ← c63631, ← c63632] at e_1027_6
  have e_1027_7 := arithEq_of_rows h (row := 1027) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_7
  simp only [← c63633, ← c63634, k2, ← c63635] at e_1027_7
  have hr := e_1027_7
  simp only [e_1027_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8519)) := by
  have c63636 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63637 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63638 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63639 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63640 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63641 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_8 := arithEq_of_rows h (row := 1027) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_8
  simp only [← c63636, ← c63637, ← c63638] at e_1027_8
  have e_1027_9 := arithEq_of_rows h (row := 1027) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_9
  simp only [← c63639, ← c63640, k2, ← c63641] at e_1027_9
  have hr := e_1027_9
  simp only [e_1027_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8520)) := by
  have c63642 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63643 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63644 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63645 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63646 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63647 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_10 := arithEq_of_rows h (row := 1027) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_10
  simp only [← c63642, ← c63643, ← c63644] at e_1027_10
  have e_1027_11 := arithEq_of_rows h (row := 1027) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_11
  simp only [← c63645, ← c63646, k2, ← c63647] at e_1027_11
  have hr := e_1027_11
  simp only [e_1027_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8521)) := by
  have c63648 := (publicBatchWrapper89_copies1989 a h).1
  have c63649 := (publicBatchWrapper89_copies1989 a h).2.1
  have c63650 := (publicBatchWrapper89_copies1989 a h).2.2.1
  have c63651 := (publicBatchWrapper89_copies1989 a h).2.2.2.1
  have c63652 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.1
  have c63653 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_12 := arithEq_of_rows h (row := 1027) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_12
  simp only [← c63648, ← c63649, ← c63650] at e_1027_12
  have e_1027_13 := arithEq_of_rows h (row := 1027) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_13
  simp only [← c63651, ← c63652, k2, ← c63653] at e_1027_13
  have hr := e_1027_13
  simp only [e_1027_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8522)) := by
  have c63654 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.1
  have c63655 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.1
  have c63656 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.1
  have c63657 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.1
  have c63658 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63659 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_14 := arithEq_of_rows h (row := 1027) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_14
  simp only [← c63654, ← c63655, ← c63656] at e_1027_14
  have e_1027_15 := arithEq_of_rows h (row := 1027) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_15
  simp only [← c63657, ← c63658, k2, ← c63659] at e_1027_15
  have hr := e_1027_15
  simp only [e_1027_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8523)) := by
  have c63660 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63661 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63662 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63663 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63664 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63665 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_16 := arithEq_of_rows h (row := 1027) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_16
  simp only [← c63660, ← c63661, ← c63662] at e_1027_16
  have e_1027_17 := arithEq_of_rows h (row := 1027) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_17
  simp only [← c63663, ← c63664, k2, ← c63665] at e_1027_17
  have hr := e_1027_17
  simp only [e_1027_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8524)) := by
  have c63666 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63667 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63668 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63669 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63670 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63671 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_18 := arithEq_of_rows h (row := 1027) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_18
  simp only [← c63666, ← c63667, ← c63668] at e_1027_18
  have e_1027_19 := arithEq_of_rows h (row := 1027) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_19
  simp only [← c63669, ← c63670, k2, ← c63671] at e_1027_19
  have hr := e_1027_19
  simp only [e_1027_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8525)) := by
  have c63672 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63673 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63674 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63675 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63676 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63677 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_0 := arithEq_of_rows h (row := 1028) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_0
  simp only [← c63672, ← c63673, ← c63674] at e_1028_0
  have e_1028_1 := arithEq_of_rows h (row := 1028) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_1
  simp only [← c63675, ← c63676, k2, ← c63677] at e_1028_1
  have hr := e_1028_1
  simp only [e_1028_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8526)) := by
  have c63678 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63679 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63680 := (publicBatchWrapper89_copies1990 a h).1
  have c63681 := (publicBatchWrapper89_copies1990 a h).2.1
  have c63682 := (publicBatchWrapper89_copies1990 a h).2.2.1
  have c63683 := (publicBatchWrapper89_copies1990 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_2 := arithEq_of_rows h (row := 1028) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_2
  simp only [← c63678, ← c63679, ← c63680] at e_1028_2
  have e_1028_3 := arithEq_of_rows h (row := 1028) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_3
  simp only [← c63681, ← c63682, k2, ← c63683] at e_1028_3
  have hr := e_1028_3
  simp only [e_1028_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8527)) := by
  have c63684 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.1
  have c63685 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.1
  have c63686 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.1
  have c63687 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.1
  have c63688 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.1
  have c63689 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_4 := arithEq_of_rows h (row := 1028) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_4
  simp only [← c63684, ← c63685, ← c63686] at e_1028_4
  have e_1028_5 := arithEq_of_rows h (row := 1028) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_5
  simp only [← c63687, ← c63688, k2, ← c63689] at e_1028_5
  have hr := e_1028_5
  simp only [e_1028_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8528)) := by
  have c63690 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63691 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63692 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63693 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63694 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63695 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_6 := arithEq_of_rows h (row := 1028) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_6
  simp only [← c63690, ← c63691, ← c63692] at e_1028_6
  have e_1028_7 := arithEq_of_rows h (row := 1028) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_7
  simp only [← c63693, ← c63694, k2, ← c63695] at e_1028_7
  have hr := e_1028_7
  simp only [e_1028_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8529)) := by
  have c63696 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63697 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63698 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63699 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63700 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63701 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_8 := arithEq_of_rows h (row := 1028) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_8
  simp only [← c63696, ← c63697, ← c63698] at e_1028_8
  have e_1028_9 := arithEq_of_rows h (row := 1028) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_9
  simp only [← c63699, ← c63700, k2, ← c63701] at e_1028_9
  have hr := e_1028_9
  simp only [e_1028_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8530)) := by
  have c63702 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63703 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63704 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63705 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63706 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63707 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_10 := arithEq_of_rows h (row := 1028) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_10
  simp only [← c63702, ← c63703, ← c63704] at e_1028_10
  have e_1028_11 := arithEq_of_rows h (row := 1028) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_11
  simp only [← c63705, ← c63706, k2, ← c63707] at e_1028_11
  have hr := e_1028_11
  simp only [e_1028_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8531)) := by
  have c63708 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63709 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63710 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63711 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63712 := (publicBatchWrapper89_copies1991 a h).1
  have c63713 := (publicBatchWrapper89_copies1991 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_12 := arithEq_of_rows h (row := 1028) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_12
  simp only [← c63708, ← c63709, ← c63710] at e_1028_12
  have e_1028_13 := arithEq_of_rows h (row := 1028) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_13
  simp only [← c63711, ← c63712, k2, ← c63713] at e_1028_13
  have hr := e_1028_13
  simp only [e_1028_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8532)) := by
  have c63714 := (publicBatchWrapper89_copies1991 a h).2.2.1
  have c63715 := (publicBatchWrapper89_copies1991 a h).2.2.2.1
  have c63716 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.1
  have c63717 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.1
  have c63718 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.1
  have c63719 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_14 := arithEq_of_rows h (row := 1028) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_14
  simp only [← c63714, ← c63715, ← c63716] at e_1028_14
  have e_1028_15 := arithEq_of_rows h (row := 1028) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_15
  simp only [← c63717, ← c63718, k2, ← c63719] at e_1028_15
  have hr := e_1028_15
  simp only [e_1028_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8533)) := by
  have c63720 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.1
  have c63721 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.1
  have c63722 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63723 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63724 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63725 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_16 := arithEq_of_rows h (row := 1028) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_16
  simp only [← c63720, ← c63721, ← c63722] at e_1028_16
  have e_1028_17 := arithEq_of_rows h (row := 1028) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_17
  simp only [← c63723, ← c63724, k2, ← c63725] at e_1028_17
  have hr := e_1028_17
  simp only [e_1028_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8534)) := by
  have c63726 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63727 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63728 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63729 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63730 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63731 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_18 := arithEq_of_rows h (row := 1028) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_18
  simp only [← c63726, ← c63727, ← c63728] at e_1028_18
  have e_1028_19 := arithEq_of_rows h (row := 1028) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_19
  simp only [← c63729, ← c63730, k2, ← c63731] at e_1028_19
  have hr := e_1028_19
  simp only [e_1028_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8535)) := by
  have c63732 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63733 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63734 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63735 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63736 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63737 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_0 := arithEq_of_rows h (row := 1029) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_0
  simp only [← c63732, ← c63733, ← c63734] at e_1029_0
  have e_1029_1 := arithEq_of_rows h (row := 1029) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_1
  simp only [← c63735, ← c63736, k2, ← c63737] at e_1029_1
  have hr := e_1029_1
  simp only [e_1029_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8536)) := by
  have c63738 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63739 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63740 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63741 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63742 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63743 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_2 := arithEq_of_rows h (row := 1029) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_2
  simp only [← c63738, ← c63739, ← c63740] at e_1029_2
  have e_1029_3 := arithEq_of_rows h (row := 1029) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_3
  simp only [← c63741, ← c63742, k2, ← c63743] at e_1029_3
  have hr := e_1029_3
  simp only [e_1029_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8537)) := by
  have c63744 := (publicBatchWrapper89_copies1992 a h).1
  have c63745 := (publicBatchWrapper89_copies1992 a h).2.1
  have c63746 := (publicBatchWrapper89_copies1992 a h).2.2.1
  have c63747 := (publicBatchWrapper89_copies1992 a h).2.2.2.1
  have c63748 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.1
  have c63749 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_4 := arithEq_of_rows h (row := 1029) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_4
  simp only [← c63744, ← c63745, ← c63746] at e_1029_4
  have e_1029_5 := arithEq_of_rows h (row := 1029) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_5
  simp only [← c63747, ← c63748, k2, ← c63749] at e_1029_5
  have hr := e_1029_5
  simp only [e_1029_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8538)) := by
  have c63750 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.1
  have c63751 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.1
  have c63752 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.1
  have c63753 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.1
  have c63754 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63755 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_6 := arithEq_of_rows h (row := 1029) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_6
  simp only [← c63750, ← c63751, ← c63752] at e_1029_6
  have e_1029_7 := arithEq_of_rows h (row := 1029) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_7
  simp only [← c63753, ← c63754, k2, ← c63755] at e_1029_7
  have hr := e_1029_7
  simp only [e_1029_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8539)) := by
  have c63756 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63757 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63758 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63759 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63760 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63761 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_8 := arithEq_of_rows h (row := 1029) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_8
  simp only [← c63756, ← c63757, ← c63758] at e_1029_8
  have e_1029_9 := arithEq_of_rows h (row := 1029) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_9
  simp only [← c63759, ← c63760, k2, ← c63761] at e_1029_9
  have hr := e_1029_9
  simp only [e_1029_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8540)) := by
  have c63762 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63763 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63764 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63765 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63766 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63767 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_10 := arithEq_of_rows h (row := 1029) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_10
  simp only [← c63762, ← c63763, ← c63764] at e_1029_10
  have e_1029_11 := arithEq_of_rows h (row := 1029) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_11
  simp only [← c63765, ← c63766, k2, ← c63767] at e_1029_11
  have hr := e_1029_11
  simp only [e_1029_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8541)) := by
  have c63768 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63769 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63770 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63771 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63772 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63773 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_12 := arithEq_of_rows h (row := 1029) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_12
  simp only [← c63768, ← c63769, ← c63770] at e_1029_12
  have e_1029_13 := arithEq_of_rows h (row := 1029) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_13
  simp only [← c63771, ← c63772, k2, ← c63773] at e_1029_13
  have hr := e_1029_13
  simp only [e_1029_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8542)) := by
  have c63774 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63775 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63776 := (publicBatchWrapper89_copies1993 a h).1
  have c63777 := (publicBatchWrapper89_copies1993 a h).2.1
  have c63778 := (publicBatchWrapper89_copies1993 a h).2.2.1
  have c63779 := (publicBatchWrapper89_copies1993 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_14 := arithEq_of_rows h (row := 1029) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_14
  simp only [← c63774, ← c63775, ← c63776] at e_1029_14
  have e_1029_15 := arithEq_of_rows h (row := 1029) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_15
  simp only [← c63777, ← c63778, k2, ← c63779] at e_1029_15
  have hr := e_1029_15
  simp only [e_1029_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8543)) := by
  have c63780 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.1
  have c63781 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.1
  have c63782 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.1
  have c63783 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.1
  have c63784 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.1
  have c63785 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_16 := arithEq_of_rows h (row := 1029) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_16
  simp only [← c63780, ← c63781, ← c63782] at e_1029_16
  have e_1029_17 := arithEq_of_rows h (row := 1029) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_17
  simp only [← c63783, ← c63784, k2, ← c63785] at e_1029_17
  have hr := e_1029_17
  simp only [e_1029_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8544)) := by
  have c63786 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63787 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63788 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63789 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63790 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63791 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_18 := arithEq_of_rows h (row := 1029) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_18
  simp only [← c63786, ← c63787, ← c63788] at e_1029_18
  have e_1029_19 := arithEq_of_rows h (row := 1029) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_19
  simp only [← c63789, ← c63790, k2, ← c63791] at e_1029_19
  have hr := e_1029_19
  simp only [e_1029_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8545)) := by
  have c63792 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63793 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63794 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63795 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63796 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63797 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_0 := arithEq_of_rows h (row := 1030) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_0
  simp only [← c63792, ← c63793, ← c63794] at e_1030_0
  have e_1030_1 := arithEq_of_rows h (row := 1030) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_1
  simp only [← c63795, ← c63796, k2, ← c63797] at e_1030_1
  have hr := e_1030_1
  simp only [e_1030_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8546)) := by
  have c63798 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63799 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63800 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63801 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63802 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63803 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_2 := arithEq_of_rows h (row := 1030) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_2
  simp only [← c63798, ← c63799, ← c63800] at e_1030_2
  have e_1030_3 := arithEq_of_rows h (row := 1030) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_3
  simp only [← c63801, ← c63802, k2, ← c63803] at e_1030_3
  have hr := e_1030_3
  simp only [e_1030_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8547)) := by
  have c63804 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63805 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63806 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63807 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63808 := (publicBatchWrapper89_copies1994 a h).1
  have c63809 := (publicBatchWrapper89_copies1994 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_4 := arithEq_of_rows h (row := 1030) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_4
  simp only [← c63804, ← c63805, ← c63806] at e_1030_4
  have e_1030_5 := arithEq_of_rows h (row := 1030) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_5
  simp only [← c63807, ← c63808, k2, ← c63809] at e_1030_5
  have hr := e_1030_5
  simp only [e_1030_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8548)) := by
  have c63810 := (publicBatchWrapper89_copies1994 a h).2.2.1
  have c63811 := (publicBatchWrapper89_copies1994 a h).2.2.2.1
  have c63812 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.1
  have c63813 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.1
  have c63814 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.1
  have c63815 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_6 := arithEq_of_rows h (row := 1030) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_6
  simp only [← c63810, ← c63811, ← c63812] at e_1030_6
  have e_1030_7 := arithEq_of_rows h (row := 1030) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_7
  simp only [← c63813, ← c63814, k2, ← c63815] at e_1030_7
  have hr := e_1030_7
  simp only [e_1030_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8549)) := by
  have c63816 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.1
  have c63817 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.1
  have c63818 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63819 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63820 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63821 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_8 := arithEq_of_rows h (row := 1030) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_8
  simp only [← c63816, ← c63817, ← c63818] at e_1030_8
  have e_1030_9 := arithEq_of_rows h (row := 1030) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_9
  simp only [← c63819, ← c63820, k2, ← c63821] at e_1030_9
  have hr := e_1030_9
  simp only [e_1030_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8550)) := by
  have c63822 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63823 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63824 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63825 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63826 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63827 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_10 := arithEq_of_rows h (row := 1030) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_10
  simp only [← c63822, ← c63823, ← c63824] at e_1030_10
  have e_1030_11 := arithEq_of_rows h (row := 1030) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_11
  simp only [← c63825, ← c63826, k2, ← c63827] at e_1030_11
  have hr := e_1030_11
  simp only [e_1030_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8551)) := by
  have c63828 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63829 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63830 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63831 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63832 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63833 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_12 := arithEq_of_rows h (row := 1030) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_12
  simp only [← c63828, ← c63829, ← c63830] at e_1030_12
  have e_1030_13 := arithEq_of_rows h (row := 1030) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_13
  simp only [← c63831, ← c63832, k2, ← c63833] at e_1030_13
  have hr := e_1030_13
  simp only [e_1030_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8552)) := by
  have c63834 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63835 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63836 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63837 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63838 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63839 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_14 := arithEq_of_rows h (row := 1030) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_14
  simp only [← c63834, ← c63835, ← c63836] at e_1030_14
  have e_1030_15 := arithEq_of_rows h (row := 1030) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_15
  simp only [← c63837, ← c63838, k2, ← c63839] at e_1030_15
  have hr := e_1030_15
  simp only [e_1030_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8553)) := by
  have c63840 := (publicBatchWrapper89_copies1995 a h).1
  have c63841 := (publicBatchWrapper89_copies1995 a h).2.1
  have c63842 := (publicBatchWrapper89_copies1995 a h).2.2.1
  have c63843 := (publicBatchWrapper89_copies1995 a h).2.2.2.1
  have c63844 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.1
  have c63845 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_16 := arithEq_of_rows h (row := 1030) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_16
  simp only [← c63840, ← c63841, ← c63842] at e_1030_16
  have e_1030_17 := arithEq_of_rows h (row := 1030) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_17
  simp only [← c63843, ← c63844, k2, ← c63845] at e_1030_17
  have hr := e_1030_17
  simp only [e_1030_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8554)) := by
  have c63846 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.1
  have c63847 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.1
  have c63848 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.1
  have c63849 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.1
  have c63850 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63851 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_18 := arithEq_of_rows h (row := 1030) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_18
  simp only [← c63846, ← c63847, ← c63848] at e_1030_18
  have e_1030_19 := arithEq_of_rows h (row := 1030) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_19
  simp only [← c63849, ← c63850, k2, ← c63851] at e_1030_19
  have hr := e_1030_19
  simp only [e_1030_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8555)) := by
  have c63852 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63853 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63854 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63855 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63856 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63857 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_0 := arithEq_of_rows h (row := 1031) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_0
  simp only [← c63852, ← c63853, ← c63854] at e_1031_0
  have e_1031_1 := arithEq_of_rows h (row := 1031) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_1
  simp only [← c63855, ← c63856, k2, ← c63857] at e_1031_1
  have hr := e_1031_1
  simp only [e_1031_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8556)) := by
  have c63858 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63859 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63860 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63861 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63862 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63863 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_2 := arithEq_of_rows h (row := 1031) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_2
  simp only [← c63858, ← c63859, ← c63860] at e_1031_2
  have e_1031_3 := arithEq_of_rows h (row := 1031) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_3
  simp only [← c63861, ← c63862, k2, ← c63863] at e_1031_3
  have hr := e_1031_3
  simp only [e_1031_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8557)) := by
  have c63864 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63865 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63866 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63867 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63868 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63869 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_4 := arithEq_of_rows h (row := 1031) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_4
  simp only [← c63864, ← c63865, ← c63866] at e_1031_4
  have e_1031_5 := arithEq_of_rows h (row := 1031) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_5
  simp only [← c63867, ← c63868, k2, ← c63869] at e_1031_5
  have hr := e_1031_5
  simp only [e_1031_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8558)) := by
  have c63870 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63871 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63872 := (publicBatchWrapper89_copies1996 a h).1
  have c63873 := (publicBatchWrapper89_copies1996 a h).2.1
  have c63874 := (publicBatchWrapper89_copies1996 a h).2.2.1
  have c63875 := (publicBatchWrapper89_copies1996 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_6 := arithEq_of_rows h (row := 1031) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_6
  simp only [← c63870, ← c63871, ← c63872] at e_1031_6
  have e_1031_7 := arithEq_of_rows h (row := 1031) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_7
  simp only [← c63873, ← c63874, k2, ← c63875] at e_1031_7
  have hr := e_1031_7
  simp only [e_1031_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8559)) := by
  have c63876 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.1
  have c63877 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.1
  have c63878 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.1
  have c63879 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.1
  have c63880 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.1
  have c63881 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_8 := arithEq_of_rows h (row := 1031) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_8
  simp only [← c63876, ← c63877, ← c63878] at e_1031_8
  have e_1031_9 := arithEq_of_rows h (row := 1031) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_9
  simp only [← c63879, ← c63880, k2, ← c63881] at e_1031_9
  have hr := e_1031_9
  simp only [e_1031_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8560)) := by
  have c63882 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63883 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63884 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63885 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63886 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63887 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_10 := arithEq_of_rows h (row := 1031) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_10
  simp only [← c63882, ← c63883, ← c63884] at e_1031_10
  have e_1031_11 := arithEq_of_rows h (row := 1031) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_11
  simp only [← c63885, ← c63886, k2, ← c63887] at e_1031_11
  have hr := e_1031_11
  simp only [e_1031_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8561)) := by
  have c63888 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63889 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63890 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63891 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63892 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63893 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_12 := arithEq_of_rows h (row := 1031) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_12
  simp only [← c63888, ← c63889, ← c63890] at e_1031_12
  have e_1031_13 := arithEq_of_rows h (row := 1031) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_13
  simp only [← c63891, ← c63892, k2, ← c63893] at e_1031_13
  have hr := e_1031_13
  simp only [e_1031_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8562)) := by
  have c63894 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63895 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63896 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63897 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63898 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63899 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_14 := arithEq_of_rows h (row := 1031) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_14
  simp only [← c63894, ← c63895, ← c63896] at e_1031_14
  have e_1031_15 := arithEq_of_rows h (row := 1031) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_15
  simp only [← c63897, ← c63898, k2, ← c63899] at e_1031_15
  have hr := e_1031_15
  simp only [e_1031_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8563)) := by
  have c63900 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63901 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63902 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63903 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63904 := (publicBatchWrapper89_copies1997 a h).1
  have c63905 := (publicBatchWrapper89_copies1997 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_16 := arithEq_of_rows h (row := 1031) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_16
  simp only [← c63900, ← c63901, ← c63902] at e_1031_16
  have e_1031_17 := arithEq_of_rows h (row := 1031) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_17
  simp only [← c63903, ← c63904, k2, ← c63905] at e_1031_17
  have hr := e_1031_17
  simp only [e_1031_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8564)) := by
  have c63906 := (publicBatchWrapper89_copies1997 a h).2.2.1
  have c63907 := (publicBatchWrapper89_copies1997 a h).2.2.2.1
  have c63908 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.1
  have c63909 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.1
  have c63910 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.1
  have c63911 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_18 := arithEq_of_rows h (row := 1031) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_18
  simp only [← c63906, ← c63907, ← c63908] at e_1031_18
  have e_1031_19 := arithEq_of_rows h (row := 1031) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_19
  simp only [← c63909, ← c63910, k2, ← c63911] at e_1031_19
  have hr := e_1031_19
  simp only [e_1031_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8565)) := by
  have c63912 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.1
  have c63913 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.1
  have c63914 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63915 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63916 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63917 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_0 := arithEq_of_rows h (row := 1032) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_0
  simp only [← c63912, ← c63913, ← c63914] at e_1032_0
  have e_1032_1 := arithEq_of_rows h (row := 1032) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_1
  simp only [← c63915, ← c63916, k2, ← c63917] at e_1032_1
  have hr := e_1032_1
  simp only [e_1032_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8566)) := by
  have c63918 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63919 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63920 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63921 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63922 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63923 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_2 := arithEq_of_rows h (row := 1032) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_2
  simp only [← c63918, ← c63919, ← c63920] at e_1032_2
  have e_1032_3 := arithEq_of_rows h (row := 1032) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_3
  simp only [← c63921, ← c63922, k2, ← c63923] at e_1032_3
  have hr := e_1032_3
  simp only [e_1032_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8567)) := by
  have c63924 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63925 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63926 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63927 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63928 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63929 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_4 := arithEq_of_rows h (row := 1032) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_4
  simp only [← c63924, ← c63925, ← c63926] at e_1032_4
  have e_1032_5 := arithEq_of_rows h (row := 1032) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_5
  simp only [← c63927, ← c63928, k2, ← c63929] at e_1032_5
  have hr := e_1032_5
  simp only [e_1032_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8568)) := by
  have c63930 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63931 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63932 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63933 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63934 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63935 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_6 := arithEq_of_rows h (row := 1032) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_6
  simp only [← c63930, ← c63931, ← c63932] at e_1032_6
  have e_1032_7 := arithEq_of_rows h (row := 1032) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_7
  simp only [← c63933, ← c63934, k2, ← c63935] at e_1032_7
  have hr := e_1032_7
  simp only [e_1032_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8569)) := by
  have c63936 := (publicBatchWrapper89_copies1998 a h).1
  have c63937 := (publicBatchWrapper89_copies1998 a h).2.1
  have c63938 := (publicBatchWrapper89_copies1998 a h).2.2.1
  have c63939 := (publicBatchWrapper89_copies1998 a h).2.2.2.1
  have c63940 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.1
  have c63941 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_8 := arithEq_of_rows h (row := 1032) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_8
  simp only [← c63936, ← c63937, ← c63938] at e_1032_8
  have e_1032_9 := arithEq_of_rows h (row := 1032) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_9
  simp only [← c63939, ← c63940, k2, ← c63941] at e_1032_9
  have hr := e_1032_9
  simp only [e_1032_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8610)) := by
  have c63942 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.1
  have c63943 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.1
  have c63944 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.1
  have c63945 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.1
  have c63946 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63947 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_10 := arithEq_of_rows h (row := 1032) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_10
  simp only [← c63942, ← c63943, ← c63944] at e_1032_10
  have e_1032_11 := arithEq_of_rows h (row := 1032) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_11
  simp only [← c63945, ← c63946, k2, ← c63947] at e_1032_11
  have hr := e_1032_11
  simp only [e_1032_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8611)) := by
  have c63948 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63949 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63950 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63951 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63952 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63953 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_12 := arithEq_of_rows h (row := 1032) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_12
  simp only [← c63948, ← c63949, ← c63950] at e_1032_12
  have e_1032_13 := arithEq_of_rows h (row := 1032) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_13
  simp only [← c63951, ← c63952, k2, ← c63953] at e_1032_13
  have hr := e_1032_13
  simp only [e_1032_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8612)) := by
  have c63954 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63955 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63956 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63957 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63958 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63959 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_14 := arithEq_of_rows h (row := 1032) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_14
  simp only [← c63954, ← c63955, ← c63956] at e_1032_14
  have e_1032_15 := arithEq_of_rows h (row := 1032) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_15
  simp only [← c63957, ← c63958, k2, ← c63959] at e_1032_15
  have hr := e_1032_15
  simp only [e_1032_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8613)) := by
  have c63960 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63961 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63962 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63963 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63964 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63965 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_16 := arithEq_of_rows h (row := 1032) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_16
  simp only [← c63960, ← c63961, ← c63962] at e_1032_16
  have e_1032_17 := arithEq_of_rows h (row := 1032) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_17
  simp only [← c63963, ← c63964, k2, ← c63965] at e_1032_17
  have hr := e_1032_17
  simp only [e_1032_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8614)) := by
  have c63966 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63967 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63968 := (publicBatchWrapper89_copies1999 a h).1
  have c63969 := (publicBatchWrapper89_copies1999 a h).2.1
  have c63970 := (publicBatchWrapper89_copies1999 a h).2.2.1
  have c63971 := (publicBatchWrapper89_copies1999 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_18 := arithEq_of_rows h (row := 1032) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_18
  simp only [← c63966, ← c63967, ← c63968] at e_1032_18
  have e_1032_19 := arithEq_of_rows h (row := 1032) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_19
  simp only [← c63969, ← c63970, k2, ← c63971] at e_1032_19
  have hr := e_1032_19
  simp only [e_1032_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8615)) := by
  have c63972 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.1
  have c63973 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.1
  have c63974 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.1
  have c63975 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.1
  have c63976 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.1
  have c63977 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_0 := arithEq_of_rows h (row := 1033) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_0
  simp only [← c63972, ← c63973, ← c63974] at e_1033_0
  have e_1033_1 := arithEq_of_rows h (row := 1033) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_1
  simp only [← c63975, ← c63976, k2, ← c63977] at e_1033_1
  have hr := e_1033_1
  simp only [e_1033_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8616)) := by
  have c63978 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63979 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63980 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63981 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63982 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63983 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_2 := arithEq_of_rows h (row := 1033) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_2
  simp only [← c63978, ← c63979, ← c63980] at e_1033_2
  have e_1033_3 := arithEq_of_rows h (row := 1033) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_3
  simp only [← c63981, ← c63982, k2, ← c63983] at e_1033_3
  have hr := e_1033_3
  simp only [e_1033_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8617)) := by
  have c63984 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63985 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63986 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63987 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63988 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63989 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_4 := arithEq_of_rows h (row := 1033) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_4
  simp only [← c63984, ← c63985, ← c63986] at e_1033_4
  have e_1033_5 := arithEq_of_rows h (row := 1033) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_5
  simp only [← c63987, ← c63988, k2, ← c63989] at e_1033_5
  have hr := e_1033_5
  simp only [e_1033_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8618)) := by
  have c63990 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63991 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63992 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63993 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63994 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63995 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_6 := arithEq_of_rows h (row := 1033) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_6
  simp only [← c63990, ← c63991, ← c63992] at e_1033_6
  have e_1033_7 := arithEq_of_rows h (row := 1033) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_7
  simp only [← c63993, ← c63994, k2, ← c63995] at e_1033_7
  have hr := e_1033_7
  simp only [e_1033_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8619)) := by
  have c63996 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63997 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63998 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63999 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64000 := (publicBatchWrapper89_copies2000 a h).1
  have c64001 := (publicBatchWrapper89_copies2000 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_8 := arithEq_of_rows h (row := 1033) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_8
  simp only [← c63996, ← c63997, ← c63998] at e_1033_8
  have e_1033_9 := arithEq_of_rows h (row := 1033) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_9
  simp only [← c63999, ← c64000, k2, ← c64001] at e_1033_9
  have hr := e_1033_9
  simp only [e_1033_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8620)) := by
  have c64002 := (publicBatchWrapper89_copies2000 a h).2.2.1
  have c64003 := (publicBatchWrapper89_copies2000 a h).2.2.2.1
  have c64004 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.1
  have c64005 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.1
  have c64006 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.1
  have c64007 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_10 := arithEq_of_rows h (row := 1033) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_10
  simp only [← c64002, ← c64003, ← c64004] at e_1033_10
  have e_1033_11 := arithEq_of_rows h (row := 1033) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_11
  simp only [← c64005, ← c64006, k2, ← c64007] at e_1033_11
  have hr := e_1033_11
  simp only [e_1033_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8621)) := by
  have c64008 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.1
  have c64009 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.1
  have c64010 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64011 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64012 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64013 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_12 := arithEq_of_rows h (row := 1033) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_12
  simp only [← c64008, ← c64009, ← c64010] at e_1033_12
  have e_1033_13 := arithEq_of_rows h (row := 1033) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_13
  simp only [← c64011, ← c64012, k2, ← c64013] at e_1033_13
  have hr := e_1033_13
  simp only [e_1033_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8622)) := by
  have c64014 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64015 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64016 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64017 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64018 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64019 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_14 := arithEq_of_rows h (row := 1033) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_14
  simp only [← c64014, ← c64015, ← c64016] at e_1033_14
  have e_1033_15 := arithEq_of_rows h (row := 1033) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_15
  simp only [← c64017, ← c64018, k2, ← c64019] at e_1033_15
  have hr := e_1033_15
  simp only [e_1033_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8623)) := by
  have c64020 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64021 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64022 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64023 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64024 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64025 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_16 := arithEq_of_rows h (row := 1033) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_16
  simp only [← c64020, ← c64021, ← c64022] at e_1033_16
  have e_1033_17 := arithEq_of_rows h (row := 1033) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_17
  simp only [← c64023, ← c64024, k2, ← c64025] at e_1033_17
  have hr := e_1033_17
  simp only [e_1033_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8624)) := by
  have c64026 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64027 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64028 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64029 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64030 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64031 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_18 := arithEq_of_rows h (row := 1033) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_18
  simp only [← c64026, ← c64027, ← c64028] at e_1033_18
  have e_1033_19 := arithEq_of_rows h (row := 1033) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_19
  simp only [← c64029, ← c64030, k2, ← c64031] at e_1033_19
  have hr := e_1033_19
  simp only [e_1033_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8625)) := by
  have c64032 := (publicBatchWrapper89_copies2001 a h).1
  have c64033 := (publicBatchWrapper89_copies2001 a h).2.1
  have c64034 := (publicBatchWrapper89_copies2001 a h).2.2.1
  have c64035 := (publicBatchWrapper89_copies2001 a h).2.2.2.1
  have c64036 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.1
  have c64037 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_0 := arithEq_of_rows h (row := 1034) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_0
  simp only [← c64032, ← c64033, ← c64034] at e_1034_0
  have e_1034_1 := arithEq_of_rows h (row := 1034) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_1
  simp only [← c64035, ← c64036, k2, ← c64037] at e_1034_1
  have hr := e_1034_1
  simp only [e_1034_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8626)) := by
  have c64038 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.1
  have c64039 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.1
  have c64040 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.1
  have c64041 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.1
  have c64042 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64043 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_2 := arithEq_of_rows h (row := 1034) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_2
  simp only [← c64038, ← c64039, ← c64040] at e_1034_2
  have e_1034_3 := arithEq_of_rows h (row := 1034) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_3
  simp only [← c64041, ← c64042, k2, ← c64043] at e_1034_3
  have hr := e_1034_3
  simp only [e_1034_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8627)) := by
  have c64044 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64045 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64046 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64047 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64048 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64049 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_4 := arithEq_of_rows h (row := 1034) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_4
  simp only [← c64044, ← c64045, ← c64046] at e_1034_4
  have e_1034_5 := arithEq_of_rows h (row := 1034) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_5
  simp only [← c64047, ← c64048, k2, ← c64049] at e_1034_5
  have hr := e_1034_5
  simp only [e_1034_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8628)) := by
  have c64050 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64051 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64052 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64053 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64054 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64055 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_6 := arithEq_of_rows h (row := 1034) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_6
  simp only [← c64050, ← c64051, ← c64052] at e_1034_6
  have e_1034_7 := arithEq_of_rows h (row := 1034) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_7
  simp only [← c64053, ← c64054, k2, ← c64055] at e_1034_7
  have hr := e_1034_7
  simp only [e_1034_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8629)) := by
  have c64056 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64057 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64058 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64059 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64060 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64061 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_8 := arithEq_of_rows h (row := 1034) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_8
  simp only [← c64056, ← c64057, ← c64058] at e_1034_8
  have e_1034_9 := arithEq_of_rows h (row := 1034) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_9
  simp only [← c64059, ← c64060, k2, ← c64061] at e_1034_9
  have hr := e_1034_9
  simp only [e_1034_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8630)) := by
  have c64062 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64063 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64064 := (publicBatchWrapper89_copies2002 a h).1
  have c64065 := (publicBatchWrapper89_copies2002 a h).2.1
  have c64066 := (publicBatchWrapper89_copies2002 a h).2.2.1
  have c64067 := (publicBatchWrapper89_copies2002 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_10 := arithEq_of_rows h (row := 1034) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_10
  simp only [← c64062, ← c64063, ← c64064] at e_1034_10
  have e_1034_11 := arithEq_of_rows h (row := 1034) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_11
  simp only [← c64065, ← c64066, k2, ← c64067] at e_1034_11
  have hr := e_1034_11
  simp only [e_1034_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8631)) := by
  have c64068 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.1
  have c64069 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.1
  have c64070 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.1
  have c64071 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.1
  have c64072 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.1
  have c64073 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_12 := arithEq_of_rows h (row := 1034) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_12
  simp only [← c64068, ← c64069, ← c64070] at e_1034_12
  have e_1034_13 := arithEq_of_rows h (row := 1034) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_13
  simp only [← c64071, ← c64072, k2, ← c64073] at e_1034_13
  have hr := e_1034_13
  simp only [e_1034_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8632)) := by
  have c64074 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64075 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64076 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64077 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64078 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64079 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_14 := arithEq_of_rows h (row := 1034) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_14
  simp only [← c64074, ← c64075, ← c64076] at e_1034_14
  have e_1034_15 := arithEq_of_rows h (row := 1034) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_15
  simp only [← c64077, ← c64078, k2, ← c64079] at e_1034_15
  have hr := e_1034_15
  simp only [e_1034_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8633)) := by
  have c64080 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64081 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64082 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64083 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64084 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64085 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_16 := arithEq_of_rows h (row := 1034) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_16
  simp only [← c64080, ← c64081, ← c64082] at e_1034_16
  have e_1034_17 := arithEq_of_rows h (row := 1034) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_17
  simp only [← c64083, ← c64084, k2, ← c64085] at e_1034_17
  have hr := e_1034_17
  simp only [e_1034_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8634)) := by
  have c64086 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64087 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64088 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64089 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64090 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64091 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_18 := arithEq_of_rows h (row := 1034) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_18
  simp only [← c64086, ← c64087, ← c64088] at e_1034_18
  have e_1034_19 := arithEq_of_rows h (row := 1034) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_19
  simp only [← c64089, ← c64090, k2, ← c64091] at e_1034_19
  have hr := e_1034_19
  simp only [e_1034_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8635)) := by
  have c64092 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64093 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64094 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64095 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64096 := (publicBatchWrapper89_copies2003 a h).1
  have c64097 := (publicBatchWrapper89_copies2003 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_0 := arithEq_of_rows h (row := 1035) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_0
  simp only [← c64092, ← c64093, ← c64094] at e_1035_0
  have e_1035_1 := arithEq_of_rows h (row := 1035) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_1
  simp only [← c64095, ← c64096, k2, ← c64097] at e_1035_1
  have hr := e_1035_1
  simp only [e_1035_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8636)) := by
  have c64098 := (publicBatchWrapper89_copies2003 a h).2.2.1
  have c64099 := (publicBatchWrapper89_copies2003 a h).2.2.2.1
  have c64100 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.1
  have c64101 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.1
  have c64102 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.1
  have c64103 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_2 := arithEq_of_rows h (row := 1035) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_2
  simp only [← c64098, ← c64099, ← c64100] at e_1035_2
  have e_1035_3 := arithEq_of_rows h (row := 1035) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_3
  simp only [← c64101, ← c64102, k2, ← c64103] at e_1035_3
  have hr := e_1035_3
  simp only [e_1035_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8637)) := by
  have c64104 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.1
  have c64105 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.1
  have c64106 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64107 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64108 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64109 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_4 := arithEq_of_rows h (row := 1035) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_4
  simp only [← c64104, ← c64105, ← c64106] at e_1035_4
  have e_1035_5 := arithEq_of_rows h (row := 1035) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_5
  simp only [← c64107, ← c64108, k2, ← c64109] at e_1035_5
  have hr := e_1035_5
  simp only [e_1035_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8638)) := by
  have c64110 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64111 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64112 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64113 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64114 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64115 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_6 := arithEq_of_rows h (row := 1035) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_6
  simp only [← c64110, ← c64111, ← c64112] at e_1035_6
  have e_1035_7 := arithEq_of_rows h (row := 1035) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_7
  simp only [← c64113, ← c64114, k2, ← c64115] at e_1035_7
  have hr := e_1035_7
  simp only [e_1035_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8639)) := by
  have c64116 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64117 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64118 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64119 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64120 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64121 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_8 := arithEq_of_rows h (row := 1035) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_8
  simp only [← c64116, ← c64117, ← c64118] at e_1035_8
  have e_1035_9 := arithEq_of_rows h (row := 1035) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_9
  simp only [← c64119, ← c64120, k2, ← c64121] at e_1035_9
  have hr := e_1035_9
  simp only [e_1035_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8640)) := by
  have c64122 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64123 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64124 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64125 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64126 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64127 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_10 := arithEq_of_rows h (row := 1035) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_10
  simp only [← c64122, ← c64123, ← c64124] at e_1035_10
  have e_1035_11 := arithEq_of_rows h (row := 1035) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_11
  simp only [← c64125, ← c64126, k2, ← c64127] at e_1035_11
  have hr := e_1035_11
  simp only [e_1035_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8641)) := by
  have c64128 := (publicBatchWrapper89_copies2004 a h).1
  have c64129 := (publicBatchWrapper89_copies2004 a h).2.1
  have c64130 := (publicBatchWrapper89_copies2004 a h).2.2.1
  have c64131 := (publicBatchWrapper89_copies2004 a h).2.2.2.1
  have c64132 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.1
  have c64133 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_12 := arithEq_of_rows h (row := 1035) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_12
  simp only [← c64128, ← c64129, ← c64130] at e_1035_12
  have e_1035_13 := arithEq_of_rows h (row := 1035) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_13
  simp only [← c64131, ← c64132, k2, ← c64133] at e_1035_13
  have hr := e_1035_13
  simp only [e_1035_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8642)) := by
  have c64134 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.1
  have c64135 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.1
  have c64136 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.1
  have c64137 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.1
  have c64138 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64139 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_14 := arithEq_of_rows h (row := 1035) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_14
  simp only [← c64134, ← c64135, ← c64136] at e_1035_14
  have e_1035_15 := arithEq_of_rows h (row := 1035) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_15
  simp only [← c64137, ← c64138, k2, ← c64139] at e_1035_15
  have hr := e_1035_15
  simp only [e_1035_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8643)) := by
  have c64140 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64141 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64142 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64143 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64144 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64145 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_16 := arithEq_of_rows h (row := 1035) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_16
  simp only [← c64140, ← c64141, ← c64142] at e_1035_16
  have e_1035_17 := arithEq_of_rows h (row := 1035) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_17
  simp only [← c64143, ← c64144, k2, ← c64145] at e_1035_17
  have hr := e_1035_17
  simp only [e_1035_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8644)) := by
  have c64146 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64147 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64148 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64149 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64150 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64151 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_18 := arithEq_of_rows h (row := 1035) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_18
  simp only [← c64146, ← c64147, ← c64148] at e_1035_18
  have e_1035_19 := arithEq_of_rows h (row := 1035) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_19
  simp only [← c64149, ← c64150, k2, ← c64151] at e_1035_19
  have hr := e_1035_19
  simp only [e_1035_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8645)) := by
  have c64152 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64153 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64154 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64155 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64156 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64157 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_0 := arithEq_of_rows h (row := 1036) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_0
  simp only [← c64152, ← c64153, ← c64154] at e_1036_0
  have e_1036_1 := arithEq_of_rows h (row := 1036) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_1
  simp only [← c64155, ← c64156, k2, ← c64157] at e_1036_1
  have hr := e_1036_1
  simp only [e_1036_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8646)) := by
  have c64158 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64159 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64160 := (publicBatchWrapper89_copies2005 a h).1
  have c64161 := (publicBatchWrapper89_copies2005 a h).2.1
  have c64162 := (publicBatchWrapper89_copies2005 a h).2.2.1
  have c64163 := (publicBatchWrapper89_copies2005 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_2 := arithEq_of_rows h (row := 1036) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_2
  simp only [← c64158, ← c64159, ← c64160] at e_1036_2
  have e_1036_3 := arithEq_of_rows h (row := 1036) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_3
  simp only [← c64161, ← c64162, k2, ← c64163] at e_1036_3
  have hr := e_1036_3
  simp only [e_1036_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8647)) := by
  have c64164 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.1
  have c64165 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.1
  have c64166 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.1
  have c64167 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.1
  have c64168 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.1
  have c64169 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_4 := arithEq_of_rows h (row := 1036) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_4
  simp only [← c64164, ← c64165, ← c64166] at e_1036_4
  have e_1036_5 := arithEq_of_rows h (row := 1036) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_5
  simp only [← c64167, ← c64168, k2, ← c64169] at e_1036_5
  have hr := e_1036_5
  simp only [e_1036_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8648)) := by
  have c64170 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64171 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64172 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64173 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64174 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64175 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_6 := arithEq_of_rows h (row := 1036) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_6
  simp only [← c64170, ← c64171, ← c64172] at e_1036_6
  have e_1036_7 := arithEq_of_rows h (row := 1036) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_7
  simp only [← c64173, ← c64174, k2, ← c64175] at e_1036_7
  have hr := e_1036_7
  simp only [e_1036_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8649)) := by
  have c64176 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64177 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64178 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64179 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64180 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64181 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_8 := arithEq_of_rows h (row := 1036) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_8
  simp only [← c64176, ← c64177, ← c64178] at e_1036_8
  have e_1036_9 := arithEq_of_rows h (row := 1036) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_9
  simp only [← c64179, ← c64180, k2, ← c64181] at e_1036_9
  have hr := e_1036_9
  simp only [e_1036_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8650)) := by
  have c64182 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64183 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64184 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64185 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64186 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64187 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_10 := arithEq_of_rows h (row := 1036) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_10
  simp only [← c64182, ← c64183, ← c64184] at e_1036_10
  have e_1036_11 := arithEq_of_rows h (row := 1036) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_11
  simp only [← c64185, ← c64186, k2, ← c64187] at e_1036_11
  have hr := e_1036_11
  simp only [e_1036_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8651)) := by
  have c64188 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64189 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64190 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64191 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64192 := (publicBatchWrapper89_copies2006 a h).1
  have c64193 := (publicBatchWrapper89_copies2006 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_12 := arithEq_of_rows h (row := 1036) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_12
  simp only [← c64188, ← c64189, ← c64190] at e_1036_12
  have e_1036_13 := arithEq_of_rows h (row := 1036) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_13
  simp only [← c64191, ← c64192, k2, ← c64193] at e_1036_13
  have hr := e_1036_13
  simp only [e_1036_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8652)) := by
  have c64194 := (publicBatchWrapper89_copies2006 a h).2.2.1
  have c64195 := (publicBatchWrapper89_copies2006 a h).2.2.2.1
  have c64196 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.1
  have c64197 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.1
  have c64198 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.1
  have c64199 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_14 := arithEq_of_rows h (row := 1036) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_14
  simp only [← c64194, ← c64195, ← c64196] at e_1036_14
  have e_1036_15 := arithEq_of_rows h (row := 1036) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_15
  simp only [← c64197, ← c64198, k2, ← c64199] at e_1036_15
  have hr := e_1036_15
  simp only [e_1036_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8653)) := by
  have c64200 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.1
  have c64201 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.1
  have c64202 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64203 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64204 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64205 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_16 := arithEq_of_rows h (row := 1036) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_16
  simp only [← c64200, ← c64201, ← c64202] at e_1036_16
  have e_1036_17 := arithEq_of_rows h (row := 1036) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_17
  simp only [← c64203, ← c64204, k2, ← c64205] at e_1036_17
  have hr := e_1036_17
  simp only [e_1036_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8654)) := by
  have c64206 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64207 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64208 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64209 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64210 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64211 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_18 := arithEq_of_rows h (row := 1036) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_18
  simp only [← c64206, ← c64207, ← c64208] at e_1036_18
  have e_1036_19 := arithEq_of_rows h (row := 1036) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_19
  simp only [← c64209, ← c64210, k2, ← c64211] at e_1036_19
  have hr := e_1036_19
  simp only [e_1036_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8655)) := by
  have c64212 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64213 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64214 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64215 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64216 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64217 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_0 := arithEq_of_rows h (row := 1037) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_0
  simp only [← c64212, ← c64213, ← c64214] at e_1037_0
  have e_1037_1 := arithEq_of_rows h (row := 1037) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_1
  simp only [← c64215, ← c64216, k2, ← c64217] at e_1037_1
  have hr := e_1037_1
  simp only [e_1037_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8656)) := by
  have c64218 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64219 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64220 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64221 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64222 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64223 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_2 := arithEq_of_rows h (row := 1037) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_2
  simp only [← c64218, ← c64219, ← c64220] at e_1037_2
  have e_1037_3 := arithEq_of_rows h (row := 1037) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_3
  simp only [← c64221, ← c64222, k2, ← c64223] at e_1037_3
  have hr := e_1037_3
  simp only [e_1037_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8657)) := by
  have c64224 := (publicBatchWrapper89_copies2007 a h).1
  have c64225 := (publicBatchWrapper89_copies2007 a h).2.1
  have c64226 := (publicBatchWrapper89_copies2007 a h).2.2.1
  have c64227 := (publicBatchWrapper89_copies2007 a h).2.2.2.1
  have c64228 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.1
  have c64229 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_4 := arithEq_of_rows h (row := 1037) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_4
  simp only [← c64224, ← c64225, ← c64226] at e_1037_4
  have e_1037_5 := arithEq_of_rows h (row := 1037) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_5
  simp only [← c64227, ← c64228, k2, ← c64229] at e_1037_5
  have hr := e_1037_5
  simp only [e_1037_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8658)) := by
  have c64230 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.1
  have c64231 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.1
  have c64232 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.1
  have c64233 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.1
  have c64234 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64235 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_6 := arithEq_of_rows h (row := 1037) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_6
  simp only [← c64230, ← c64231, ← c64232] at e_1037_6
  have e_1037_7 := arithEq_of_rows h (row := 1037) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_7
  simp only [← c64233, ← c64234, k2, ← c64235] at e_1037_7
  have hr := e_1037_7
  simp only [e_1037_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8659)) := by
  have c64236 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64237 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64238 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64239 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64240 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64241 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_8 := arithEq_of_rows h (row := 1037) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_8
  simp only [← c64236, ← c64237, ← c64238] at e_1037_8
  have e_1037_9 := arithEq_of_rows h (row := 1037) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_9
  simp only [← c64239, ← c64240, k2, ← c64241] at e_1037_9
  have hr := e_1037_9
  simp only [e_1037_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8660)) := by
  have c64242 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64243 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64244 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64245 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64246 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64247 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_10 := arithEq_of_rows h (row := 1037) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_10
  simp only [← c64242, ← c64243, ← c64244] at e_1037_10
  have e_1037_11 := arithEq_of_rows h (row := 1037) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_11
  simp only [← c64245, ← c64246, k2, ← c64247] at e_1037_11
  have hr := e_1037_11
  simp only [e_1037_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8661)) := by
  have c64248 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64249 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64250 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64251 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64252 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64253 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_12 := arithEq_of_rows h (row := 1037) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_12
  simp only [← c64248, ← c64249, ← c64250] at e_1037_12
  have e_1037_13 := arithEq_of_rows h (row := 1037) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_13
  simp only [← c64251, ← c64252, k2, ← c64253] at e_1037_13
  have hr := e_1037_13
  simp only [e_1037_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8662)) := by
  have c64254 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64255 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64256 := (publicBatchWrapper89_copies2008 a h).1
  have c64257 := (publicBatchWrapper89_copies2008 a h).2.1
  have c64258 := (publicBatchWrapper89_copies2008 a h).2.2.1
  have c64259 := (publicBatchWrapper89_copies2008 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_14 := arithEq_of_rows h (row := 1037) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_14
  simp only [← c64254, ← c64255, ← c64256] at e_1037_14
  have e_1037_15 := arithEq_of_rows h (row := 1037) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_15
  simp only [← c64257, ← c64258, k2, ← c64259] at e_1037_15
  have hr := e_1037_15
  simp only [e_1037_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8663)) := by
  have c64260 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.1
  have c64261 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.1
  have c64262 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.1
  have c64263 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.1
  have c64264 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.1
  have c64265 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_16 := arithEq_of_rows h (row := 1037) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_16
  simp only [← c64260, ← c64261, ← c64262] at e_1037_16
  have e_1037_17 := arithEq_of_rows h (row := 1037) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_17
  simp only [← c64263, ← c64264, k2, ← c64265] at e_1037_17
  have hr := e_1037_17
  simp only [e_1037_16] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
