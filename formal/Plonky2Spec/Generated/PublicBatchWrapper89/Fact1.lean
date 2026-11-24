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

theorem publicBatchWrapper89_f679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16807)) (a (.virt 200)) (a (.virt 18483)) (a (.virt 18484)) := by
  have c5577 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.1
  have c5578 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5579 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5580 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5581 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5582 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5583 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5584 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5585 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5586 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5587 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5588 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5589 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5590 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_80_16 := arithEq_of_rows h (row := 80) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_16
  simp only [← c5577, k3, ← c5578, k3, ← c5579] at e_80_16
  have e_80_17 := arithEq_of_rows h (row := 80) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_17
  simp only [← c5586, ← c5587, k3, ← c5588] at e_80_17
  have e_81_12 := arithEq_of_rows h (row := 81) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_12
  simp only [← c5580, ← c5581, ← c5582] at e_81_12
  have e_81_13 := arithEq_of_rows h (row := 81) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_13
  simp only [← c5583, ← c5584, ← c5585] at e_81_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_81_12
    linear_combination c5589.trans k2 - hc
  · have hc := e_80_17
    simp only [e_80_16, e_81_13] at hc
    linear_combination c5590.trans k2 - hc

theorem publicBatchWrapper89_f680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16808)) (a (.virt 200)) (a (.virt 18485)) (a (.virt 18486)) := by
  have c5591 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5592 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5593 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5594 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5595 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5596 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5597 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5598 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5599 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5600 := (publicBatchWrapper89_copies175 a h).1
  have c5601 := (publicBatchWrapper89_copies175 a h).2.1
  have c5602 := (publicBatchWrapper89_copies175 a h).2.2.1
  have c5603 := (publicBatchWrapper89_copies175 a h).2.2.2.1
  have c5604 := (publicBatchWrapper89_copies175 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_80_18 := arithEq_of_rows h (row := 80) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_18
  simp only [← c5591, k3, ← c5592, k3, ← c5593] at e_80_18
  have e_80_19 := arithEq_of_rows h (row := 80) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_19
  simp only [← c5600, ← c5601, k3, ← c5602] at e_80_19
  have e_81_14 := arithEq_of_rows h (row := 81) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_14
  simp only [← c5594, ← c5595, ← c5596] at e_81_14
  have e_81_15 := arithEq_of_rows h (row := 81) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_15
  simp only [← c5597, ← c5598, ← c5599] at e_81_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_81_14
    linear_combination c5603.trans k2 - hc
  · have hc := e_80_19
    simp only [e_80_18, e_81_15] at hc
    linear_combination c5604.trans k2 - hc

theorem publicBatchWrapper89_f681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 81 67) = band (a (.virt 18479)) (a (.virt 18481)) := by
  have c5605 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.1
  have c5606 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.1
  have c5607 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.1
  have e_81_16 := arithEq_of_rows h (row := 81) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_16
  simp only [← c5605, ← c5606, ← c5607] at e_81_16
  have hr := e_81_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 81 71) = band (a (.virt 18483)) (a (.virt 18485)) := by
  have c5608 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.1
  have c5609 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.1
  have c5610 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_81_17 := arithEq_of_rows h (row := 81) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_17
  simp only [← c5608, ← c5609, ← c5610] at e_81_17
  have hr := e_81_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 81 75) = band (a (.wire 81 67)) (a (.wire 81 71)) := by
  have c5611 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5612 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5613 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_81_18 := arithEq_of_rows h (row := 81) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_18
  simp only [← c5611, ← c5612, ← c5613] at e_81_18
  have hr := e_81_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17005)) (a (.virt 200)) (a (.virt 18487)) (a (.virt 18488)) := by
  have c5614 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5615 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5616 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5617 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5618 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5619 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5620 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5621 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5622 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5623 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5624 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5625 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5626 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5627 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_81_19 := arithEq_of_rows h (row := 81) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_19
  simp only [← c5617, ← c5618, ← c5619] at e_81_19
  have e_82_0 := arithEq_of_rows h (row := 82) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_0
  simp only [← c5614, k3, ← c5615, k3, ← c5616] at e_82_0
  have e_82_1 := arithEq_of_rows h (row := 82) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_1
  simp only [← c5623, ← c5624, k3, ← c5625] at e_82_1
  have e_83_0 := arithEq_of_rows h (row := 83) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_0
  simp only [← c5620, ← c5621, ← c5622] at e_83_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_81_19
    linear_combination c5626.trans k2 - hc
  · have hc := e_82_1
    simp only [e_82_0, e_83_0] at hc
    linear_combination c5627.trans k2 - hc

theorem publicBatchWrapper89_f685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17006)) (a (.virt 200)) (a (.virt 18489)) (a (.virt 18490)) := by
  have c5628 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5629 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5630 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5631 := (publicBatchWrapper89_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5632 := (publicBatchWrapper89_copies176 a h).1
  have c5633 := (publicBatchWrapper89_copies176 a h).2.1
  have c5634 := (publicBatchWrapper89_copies176 a h).2.2.1
  have c5635 := (publicBatchWrapper89_copies176 a h).2.2.2.1
  have c5636 := (publicBatchWrapper89_copies176 a h).2.2.2.2.1
  have c5637 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.1
  have c5638 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.1
  have c5639 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.1
  have c5640 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.1
  have c5641 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_82_2 := arithEq_of_rows h (row := 82) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_2
  simp only [← c5628, k3, ← c5629, k3, ← c5630] at e_82_2
  have e_82_3 := arithEq_of_rows h (row := 82) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_3
  simp only [← c5637, ← c5638, k3, ← c5639] at e_82_3
  have e_83_1 := arithEq_of_rows h (row := 83) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_1
  simp only [← c5631, ← c5632, ← c5633] at e_83_1
  have e_83_2 := arithEq_of_rows h (row := 83) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_2
  simp only [← c5634, ← c5635, ← c5636] at e_83_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_83_1
    linear_combination c5640.trans k2 - hc
  · have hc := e_82_3
    simp only [e_82_2, e_83_2] at hc
    linear_combination c5641.trans k2 - hc

theorem publicBatchWrapper89_f686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17007)) (a (.virt 200)) (a (.virt 18491)) (a (.virt 18492)) := by
  have c5642 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5643 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5644 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5645 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5646 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5647 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5648 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5649 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5650 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5651 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5652 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5653 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5654 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5655 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_82_4 := arithEq_of_rows h (row := 82) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_4
  simp only [← c5642, k3, ← c5643, k3, ← c5644] at e_82_4
  have e_82_5 := arithEq_of_rows h (row := 82) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_5
  simp only [← c5651, ← c5652, k3, ← c5653] at e_82_5
  have e_83_3 := arithEq_of_rows h (row := 83) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_3
  simp only [← c5645, ← c5646, ← c5647] at e_83_3
  have e_83_4 := arithEq_of_rows h (row := 83) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_4
  simp only [← c5648, ← c5649, ← c5650] at e_83_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_83_3
    linear_combination c5654.trans k2 - hc
  · have hc := e_82_5
    simp only [e_82_4, e_83_4] at hc
    linear_combination c5655.trans k2 - hc

theorem publicBatchWrapper89_f687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17008)) (a (.virt 200)) (a (.virt 18493)) (a (.virt 18494)) := by
  have c5656 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5657 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5658 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5659 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5660 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5661 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5662 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5663 := (publicBatchWrapper89_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5664 := (publicBatchWrapper89_copies177 a h).1
  have c5665 := (publicBatchWrapper89_copies177 a h).2.1
  have c5666 := (publicBatchWrapper89_copies177 a h).2.2.1
  have c5667 := (publicBatchWrapper89_copies177 a h).2.2.2.1
  have c5668 := (publicBatchWrapper89_copies177 a h).2.2.2.2.1
  have c5669 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_82_6 := arithEq_of_rows h (row := 82) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_6
  simp only [← c5656, k3, ← c5657, k3, ← c5658] at e_82_6
  have e_82_7 := arithEq_of_rows h (row := 82) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_7
  simp only [← c5665, ← c5666, k3, ← c5667] at e_82_7
  have e_83_5 := arithEq_of_rows h (row := 83) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_5
  simp only [← c5659, ← c5660, ← c5661] at e_83_5
  have e_83_6 := arithEq_of_rows h (row := 83) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_6
  simp only [← c5662, ← c5663, ← c5664] at e_83_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_83_5
    linear_combination c5668.trans k2 - hc
  · have hc := e_82_7
    simp only [e_82_6, e_83_6] at hc
    linear_combination c5669.trans k2 - hc

theorem publicBatchWrapper89_f688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 83 31) = band (a (.virt 18487)) (a (.virt 18489)) := by
  have c5670 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.1
  have c5671 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.1
  have c5672 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.1
  have e_83_7 := arithEq_of_rows h (row := 83) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_7
  simp only [← c5670, ← c5671, ← c5672] at e_83_7
  have hr := e_83_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 83 35) = band (a (.virt 18491)) (a (.virt 18493)) := by
  have c5673 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.1
  have c5674 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5675 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_83_8 := arithEq_of_rows h (row := 83) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_8
  simp only [← c5673, ← c5674, ← c5675] at e_83_8
  have hr := e_83_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 83 39) = band (a (.wire 83 31)) (a (.wire 83 35)) := by
  have c5676 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5677 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5678 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_83_9 := arithEq_of_rows h (row := 83) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_9
  simp only [← c5676, ← c5677, ← c5678] at e_83_9
  have hr := e_83_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17205)) (a (.virt 200)) (a (.virt 18495)) (a (.virt 18496)) := by
  have c5679 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5680 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5681 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5682 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5683 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5684 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5685 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5686 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5687 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5688 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5689 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5690 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5691 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5692 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_82_8 := arithEq_of_rows h (row := 82) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_8
  simp only [← c5679, k3, ← c5680, k3, ← c5681] at e_82_8
  have e_82_9 := arithEq_of_rows h (row := 82) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_9
  simp only [← c5688, ← c5689, k3, ← c5690] at e_82_9
  have e_83_10 := arithEq_of_rows h (row := 83) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_10
  simp only [← c5682, ← c5683, ← c5684] at e_83_10
  have e_83_11 := arithEq_of_rows h (row := 83) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_11
  simp only [← c5685, ← c5686, ← c5687] at e_83_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_83_10
    linear_combination c5691.trans k2 - hc
  · have hc := e_82_9
    simp only [e_82_8, e_83_11] at hc
    linear_combination c5692.trans k2 - hc

theorem publicBatchWrapper89_f692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17206)) (a (.virt 200)) (a (.virt 18497)) (a (.virt 18498)) := by
  have c5693 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5694 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5695 := (publicBatchWrapper89_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5696 := (publicBatchWrapper89_copies178 a h).1
  have c5697 := (publicBatchWrapper89_copies178 a h).2.1
  have c5698 := (publicBatchWrapper89_copies178 a h).2.2.1
  have c5699 := (publicBatchWrapper89_copies178 a h).2.2.2.1
  have c5700 := (publicBatchWrapper89_copies178 a h).2.2.2.2.1
  have c5701 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.1
  have c5702 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.1
  have c5703 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.1
  have c5704 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.1
  have c5705 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.1
  have c5706 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_82_10 := arithEq_of_rows h (row := 82) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_10
  simp only [← c5693, k3, ← c5694, k3, ← c5695] at e_82_10
  have e_82_11 := arithEq_of_rows h (row := 82) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_11
  simp only [← c5702, ← c5703, k3, ← c5704] at e_82_11
  have e_83_12 := arithEq_of_rows h (row := 83) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_12
  simp only [← c5696, ← c5697, ← c5698] at e_83_12
  have e_83_13 := arithEq_of_rows h (row := 83) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_13
  simp only [← c5699, ← c5700, ← c5701] at e_83_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_83_12
    linear_combination c5705.trans k2 - hc
  · have hc := e_82_11
    simp only [e_82_10, e_83_13] at hc
    linear_combination c5706.trans k2 - hc

theorem publicBatchWrapper89_f693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17207)) (a (.virt 200)) (a (.virt 18499)) (a (.virt 18500)) := by
  have c5707 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5708 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5709 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5710 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5711 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5712 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5713 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5714 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5715 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5716 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5717 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5718 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5719 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5720 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_82_12 := arithEq_of_rows h (row := 82) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_12
  simp only [← c5707, k3, ← c5708, k3, ← c5709] at e_82_12
  have e_82_13 := arithEq_of_rows h (row := 82) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_13
  simp only [← c5716, ← c5717, k3, ← c5718] at e_82_13
  have e_83_14 := arithEq_of_rows h (row := 83) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_14
  simp only [← c5710, ← c5711, ← c5712] at e_83_14
  have e_83_15 := arithEq_of_rows h (row := 83) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_15
  simp only [← c5713, ← c5714, ← c5715] at e_83_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_83_14
    linear_combination c5719.trans k2 - hc
  · have hc := e_82_13
    simp only [e_82_12, e_83_15] at hc
    linear_combination c5720.trans k2 - hc

theorem publicBatchWrapper89_f694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17208)) (a (.virt 200)) (a (.virt 18501)) (a (.virt 18502)) := by
  have c5721 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5722 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5723 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5724 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5725 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5726 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5727 := (publicBatchWrapper89_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5728 := (publicBatchWrapper89_copies179 a h).1
  have c5729 := (publicBatchWrapper89_copies179 a h).2.1
  have c5730 := (publicBatchWrapper89_copies179 a h).2.2.1
  have c5731 := (publicBatchWrapper89_copies179 a h).2.2.2.1
  have c5732 := (publicBatchWrapper89_copies179 a h).2.2.2.2.1
  have c5733 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.1
  have c5734 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_82_14 := arithEq_of_rows h (row := 82) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_14
  simp only [← c5721, k3, ← c5722, k3, ← c5723] at e_82_14
  have e_82_15 := arithEq_of_rows h (row := 82) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_15
  simp only [← c5730, ← c5731, k3, ← c5732] at e_82_15
  have e_83_16 := arithEq_of_rows h (row := 83) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_16
  simp only [← c5724, ← c5725, ← c5726] at e_83_16
  have e_83_17 := arithEq_of_rows h (row := 83) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_17
  simp only [← c5727, ← c5728, ← c5729] at e_83_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_83_16
    linear_combination c5733.trans k2 - hc
  · have hc := e_82_15
    simp only [e_82_14, e_83_17] at hc
    linear_combination c5734.trans k2 - hc

theorem publicBatchWrapper89_f695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 83 75) = band (a (.virt 18495)) (a (.virt 18497)) := by
  have c5735 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.1
  have c5736 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.1
  have c5737 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.1
  have e_83_18 := arithEq_of_rows h (row := 83) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_18
  simp only [← c5735, ← c5736, ← c5737] at e_83_18
  have hr := e_83_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 83 79) = band (a (.virt 18499)) (a (.virt 18501)) := by
  have c5738 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5739 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5740 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_83_19 := arithEq_of_rows h (row := 83) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_19
  simp only [← c5738, ← c5739, ← c5740] at e_83_19
  have hr := e_83_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 84 3) = band (a (.wire 83 75)) (a (.wire 83 79)) := by
  have c5741 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5742 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5743 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_0 := arithEq_of_rows h (row := 84) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_0
  simp only [← c5741, ← c5742, ← c5743] at e_84_0
  have hr := e_84_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17405)) (a (.virt 200)) (a (.virt 18503)) (a (.virt 18504)) := by
  have c5744 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5745 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5746 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5747 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5748 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5749 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5750 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5751 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5752 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5753 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5754 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5755 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5756 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5757 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_82_16 := arithEq_of_rows h (row := 82) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_16
  simp only [← c5744, k3, ← c5745, k3, ← c5746] at e_82_16
  have e_82_17 := arithEq_of_rows h (row := 82) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_17
  simp only [← c5753, ← c5754, k3, ← c5755] at e_82_17
  have e_84_1 := arithEq_of_rows h (row := 84) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_1
  simp only [← c5747, ← c5748, ← c5749] at e_84_1
  have e_84_2 := arithEq_of_rows h (row := 84) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_2
  simp only [← c5750, ← c5751, ← c5752] at e_84_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_84_1
    linear_combination c5756.trans k2 - hc
  · have hc := e_82_17
    simp only [e_82_16, e_84_2] at hc
    linear_combination c5757.trans k2 - hc

theorem publicBatchWrapper89_f699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17406)) (a (.virt 200)) (a (.virt 18505)) (a (.virt 18506)) := by
  have c5758 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5759 := (publicBatchWrapper89_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5760 := (publicBatchWrapper89_copies180 a h).1
  have c5761 := (publicBatchWrapper89_copies180 a h).2.1
  have c5762 := (publicBatchWrapper89_copies180 a h).2.2.1
  have c5763 := (publicBatchWrapper89_copies180 a h).2.2.2.1
  have c5764 := (publicBatchWrapper89_copies180 a h).2.2.2.2.1
  have c5765 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.1
  have c5766 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.1
  have c5767 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.1
  have c5768 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.1
  have c5769 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.1
  have c5770 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5771 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_82_18 := arithEq_of_rows h (row := 82) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_18
  simp only [← c5758, k3, ← c5759, k3, ← c5760] at e_82_18
  have e_82_19 := arithEq_of_rows h (row := 82) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_19
  simp only [← c5767, ← c5768, k3, ← c5769] at e_82_19
  have e_84_3 := arithEq_of_rows h (row := 84) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_3
  simp only [← c5761, ← c5762, ← c5763] at e_84_3
  have e_84_4 := arithEq_of_rows h (row := 84) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_4
  simp only [← c5764, ← c5765, ← c5766] at e_84_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_84_3
    linear_combination c5770.trans k2 - hc
  · have hc := e_82_19
    simp only [e_82_18, e_84_4] at hc
    linear_combination c5771.trans k2 - hc

theorem publicBatchWrapper89_f700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17407)) (a (.virt 200)) (a (.virt 18507)) (a (.virt 18508)) := by
  have c5772 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5773 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5774 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5775 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5776 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5777 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5778 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5779 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5780 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5781 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5782 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5783 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5784 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5785 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_84_5 := arithEq_of_rows h (row := 84) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_5
  simp only [← c5775, ← c5776, ← c5777] at e_84_5
  have e_84_6 := arithEq_of_rows h (row := 84) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_6
  simp only [← c5778, ← c5779, ← c5780] at e_84_6
  have e_85_0 := arithEq_of_rows h (row := 85) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_0
  simp only [← c5772, k3, ← c5773, k3, ← c5774] at e_85_0
  have e_85_1 := arithEq_of_rows h (row := 85) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_1
  simp only [← c5781, ← c5782, k3, ← c5783] at e_85_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_84_5
    linear_combination c5784.trans k2 - hc
  · have hc := e_85_1
    simp only [e_85_0, e_84_6] at hc
    linear_combination c5785.trans k2 - hc

theorem publicBatchWrapper89_f701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17408)) (a (.virt 200)) (a (.virt 18509)) (a (.virt 18510)) := by
  have c5786 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5787 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5788 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5789 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5790 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5791 := (publicBatchWrapper89_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5792 := (publicBatchWrapper89_copies181 a h).1
  have c5793 := (publicBatchWrapper89_copies181 a h).2.1
  have c5794 := (publicBatchWrapper89_copies181 a h).2.2.1
  have c5795 := (publicBatchWrapper89_copies181 a h).2.2.2.1
  have c5796 := (publicBatchWrapper89_copies181 a h).2.2.2.2.1
  have c5797 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.1
  have c5798 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.1
  have c5799 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_84_7 := arithEq_of_rows h (row := 84) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_7
  simp only [← c5789, ← c5790, ← c5791] at e_84_7
  have e_84_8 := arithEq_of_rows h (row := 84) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_8
  simp only [← c5792, ← c5793, ← c5794] at e_84_8
  have e_85_2 := arithEq_of_rows h (row := 85) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_2
  simp only [← c5786, k3, ← c5787, k3, ← c5788] at e_85_2
  have e_85_3 := arithEq_of_rows h (row := 85) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_3
  simp only [← c5795, ← c5796, k3, ← c5797] at e_85_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_84_7
    linear_combination c5798.trans k2 - hc
  · have hc := e_85_3
    simp only [e_85_2, e_84_8] at hc
    linear_combination c5799.trans k2 - hc

theorem publicBatchWrapper89_f702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 84 39) = band (a (.virt 18503)) (a (.virt 18505)) := by
  have c5800 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.1
  have c5801 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.1
  have c5802 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_84_9 := arithEq_of_rows h (row := 84) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_9
  simp only [← c5800, ← c5801, ← c5802] at e_84_9
  have hr := e_84_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 84 43) = band (a (.virt 18507)) (a (.virt 18509)) := by
  have c5803 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5804 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5805 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_10 := arithEq_of_rows h (row := 84) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_10
  simp only [← c5803, ← c5804, ← c5805] at e_84_10
  have hr := e_84_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 84 47) = band (a (.wire 84 39)) (a (.wire 84 43)) := by
  have c5806 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5807 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5808 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_11 := arithEq_of_rows h (row := 84) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_11
  simp only [← c5806, ← c5807, ← c5808] at e_84_11
  have hr := e_84_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17605)) (a (.virt 200)) (a (.virt 18511)) (a (.virt 18512)) := by
  have c5809 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5810 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5811 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5812 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5813 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5814 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5815 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5816 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5817 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5818 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5819 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5820 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5821 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5822 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_84_12 := arithEq_of_rows h (row := 84) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_12
  simp only [← c5812, ← c5813, ← c5814] at e_84_12
  have e_84_13 := arithEq_of_rows h (row := 84) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_13
  simp only [← c5815, ← c5816, ← c5817] at e_84_13
  have e_85_4 := arithEq_of_rows h (row := 85) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_4
  simp only [← c5809, k3, ← c5810, k3, ← c5811] at e_85_4
  have e_85_5 := arithEq_of_rows h (row := 85) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_5
  simp only [← c5818, ← c5819, k3, ← c5820] at e_85_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_84_12
    linear_combination c5821.trans k2 - hc
  · have hc := e_85_5
    simp only [e_85_4, e_84_13] at hc
    linear_combination c5822.trans k2 - hc

theorem publicBatchWrapper89_f706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17606)) (a (.virt 200)) (a (.virt 18513)) (a (.virt 18514)) := by
  have c5823 := (publicBatchWrapper89_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5824 := (publicBatchWrapper89_copies182 a h).1
  have c5825 := (publicBatchWrapper89_copies182 a h).2.1
  have c5826 := (publicBatchWrapper89_copies182 a h).2.2.1
  have c5827 := (publicBatchWrapper89_copies182 a h).2.2.2.1
  have c5828 := (publicBatchWrapper89_copies182 a h).2.2.2.2.1
  have c5829 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.1
  have c5830 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.1
  have c5831 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.1
  have c5832 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.1
  have c5833 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.1
  have c5834 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5835 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5836 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_84_14 := arithEq_of_rows h (row := 84) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_14
  simp only [← c5826, ← c5827, ← c5828] at e_84_14
  have e_84_15 := arithEq_of_rows h (row := 84) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_15
  simp only [← c5829, ← c5830, ← c5831] at e_84_15
  have e_85_6 := arithEq_of_rows h (row := 85) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_6
  simp only [← c5823, k3, ← c5824, k3, ← c5825] at e_85_6
  have e_85_7 := arithEq_of_rows h (row := 85) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_7
  simp only [← c5832, ← c5833, k3, ← c5834] at e_85_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_84_14
    linear_combination c5835.trans k2 - hc
  · have hc := e_85_7
    simp only [e_85_6, e_84_15] at hc
    linear_combination c5836.trans k2 - hc

theorem publicBatchWrapper89_f707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17607)) (a (.virt 200)) (a (.virt 18515)) (a (.virt 18516)) := by
  have c5837 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5838 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5839 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5840 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5841 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5842 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5843 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5844 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5845 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5846 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5847 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5848 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5849 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5850 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_84_16 := arithEq_of_rows h (row := 84) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_16
  simp only [← c5840, ← c5841, ← c5842] at e_84_16
  have e_84_17 := arithEq_of_rows h (row := 84) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_17
  simp only [← c5843, ← c5844, ← c5845] at e_84_17
  have e_85_8 := arithEq_of_rows h (row := 85) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_8
  simp only [← c5837, k3, ← c5838, k3, ← c5839] at e_85_8
  have e_85_9 := arithEq_of_rows h (row := 85) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_9
  simp only [← c5846, ← c5847, k3, ← c5848] at e_85_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_84_16
    linear_combination c5849.trans k2 - hc
  · have hc := e_85_9
    simp only [e_85_8, e_84_17] at hc
    linear_combination c5850.trans k2 - hc

theorem publicBatchWrapper89_f708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17608)) (a (.virt 200)) (a (.virt 18517)) (a (.virt 18518)) := by
  have c5851 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5852 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5853 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5854 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5855 := (publicBatchWrapper89_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5856 := (publicBatchWrapper89_copies183 a h).1
  have c5857 := (publicBatchWrapper89_copies183 a h).2.1
  have c5858 := (publicBatchWrapper89_copies183 a h).2.2.1
  have c5859 := (publicBatchWrapper89_copies183 a h).2.2.2.1
  have c5860 := (publicBatchWrapper89_copies183 a h).2.2.2.2.1
  have c5861 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.1
  have c5862 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.1
  have c5863 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.1
  have c5864 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_84_18 := arithEq_of_rows h (row := 84) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_18
  simp only [← c5854, ← c5855, ← c5856] at e_84_18
  have e_84_19 := arithEq_of_rows h (row := 84) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_19
  simp only [← c5857, ← c5858, ← c5859] at e_84_19
  have e_85_10 := arithEq_of_rows h (row := 85) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_10
  simp only [← c5851, k3, ← c5852, k3, ← c5853] at e_85_10
  have e_85_11 := arithEq_of_rows h (row := 85) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_11
  simp only [← c5860, ← c5861, k3, ← c5862] at e_85_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_84_18
    linear_combination c5863.trans k2 - hc
  · have hc := e_85_11
    simp only [e_85_10, e_84_19] at hc
    linear_combination c5864.trans k2 - hc

theorem publicBatchWrapper89_f709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 3) = band (a (.virt 18511)) (a (.virt 18513)) := by
  have c5865 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.1
  have c5866 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5867 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_0 := arithEq_of_rows h (row := 86) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_0
  simp only [← c5865, ← c5866, ← c5867] at e_86_0
  have hr := e_86_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 7) = band (a (.virt 18515)) (a (.virt 18517)) := by
  have c5868 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5869 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5870 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_1 := arithEq_of_rows h (row := 86) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_1
  simp only [← c5868, ← c5869, ← c5870] at e_86_1
  have hr := e_86_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 11) = band (a (.wire 86 3)) (a (.wire 86 7)) := by
  have c5871 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5872 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5873 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_2 := arithEq_of_rows h (row := 86) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_2
  simp only [← c5871, ← c5872, ← c5873] at e_86_2
  have hr := e_86_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 51) = bnot (a (.wire 1 59)) := by
  have c5874 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5875 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5876 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_85_12 := arithEq_of_rows h (row := 85) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_12
  simp only [← c5874, k3, ← c5875, k3, ← c5876] at e_85_12
  have hr := e_85_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 201) = bnot (a (.virt 200)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  simp only [bnot, k2, k3]
  ring

theorem publicBatchWrapper89_f714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 51) = band (a (.wire 85 51)) (a (.virt 201)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  simp only [band, k3]
  ring

theorem publicBatchWrapper89_f715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 55) = bselect (a (.wire 85 51)) (a (.virt 3)) (a (.virt 200)) := by
  have c5877 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5878 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5879 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_85_13 := arithEq_of_rows h (row := 85) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_13
  simp only [← c5877, ← c5878, ← c5879, k2] at e_85_13
  have hr := e_85_13
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 59) = bselect (a (.wire 85 51)) (a (.virt 4)) (a (.virt 200)) := by
  have c5880 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5881 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5882 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_85_14 := arithEq_of_rows h (row := 85) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_14
  simp only [← c5880, ← c5881, ← c5882, k2] at e_85_14
  have hr := e_85_14
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 63) = bselect (a (.wire 85 51)) (a (.virt 5)) (a (.virt 200)) := by
  have c5883 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5884 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5885 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_85_15 := arithEq_of_rows h (row := 85) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_15
  simp only [← c5883, ← c5884, ← c5885, k2] at e_85_15
  have hr := e_85_15
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 67) = bselect (a (.wire 85 51)) (a (.virt 6)) (a (.virt 200)) := by
  have c5886 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5887 := (publicBatchWrapper89_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5888 := (publicBatchWrapper89_copies184 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_85_16 := arithEq_of_rows h (row := 85) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_16
  simp only [← c5886, ← c5887, ← c5888, k2] at e_85_16
  have hr := e_85_16
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 71) = bselect (a (.wire 85 51)) (a (.virt 7)) (a (.virt 200)) := by
  have c5889 := (publicBatchWrapper89_copies184 a h).2.1
  have c5890 := (publicBatchWrapper89_copies184 a h).2.2.1
  have c5891 := (publicBatchWrapper89_copies184 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_85_17 := arithEq_of_rows h (row := 85) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_17
  simp only [← c5889, ← c5890, ← c5891, k2] at e_85_17
  have hr := e_85_17
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 75) = bselect (a (.wire 85 51)) (a (.virt 1)) (a (.virt 200)) := by
  have c5892 := (publicBatchWrapper89_copies184 a h).2.2.2.2.1
  have c5893 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.1
  have c5894 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_85_18 := arithEq_of_rows h (row := 85) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_18
  simp only [← c5892, ← c5893, ← c5894, k2] at e_85_18
  have hr := e_85_18
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 79) = bselect (a (.wire 85 51)) (a (.virt 2)) (a (.virt 200)) := by
  have c5895 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.1
  have c5896 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.1
  have c5897 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_85_19 := arithEq_of_rows h (row := 85) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_19
  simp only [← c5895, ← c5896, ← c5897, k2] at e_85_19
  have hr := e_85_19
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 85 51) = bor (a (.virt 200)) (a (.wire 85 51)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  simp only [bor, k2]
  ring

theorem publicBatchWrapper89_f723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 3) = bnot (a (.wire 3 23)) := by
  have c5898 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5899 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5900 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_87_0 := arithEq_of_rows h (row := 87) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_0
  simp only [← c5898, k3, ← c5899, k3, ← c5900] at e_87_0
  have hr := e_87_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 7) = bnot (a (.wire 85 51)) := by
  have c5901 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5902 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5903 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_87_1 := arithEq_of_rows h (row := 87) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_1
  simp only [← c5901, k3, ← c5902, k3, ← c5903] at e_87_1
  have hr := e_87_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 15) = band (a (.wire 87 3)) (a (.wire 87 7)) := by
  have c5904 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5905 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5906 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_3 := arithEq_of_rows h (row := 86) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_3
  simp only [← c5904, ← c5905, ← c5906] at e_86_3
  have hr := e_86_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 15) = bselect (a (.wire 86 15)) (a (.virt 205)) (a (.wire 85 55)) := by
  have c5907 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5908 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5909 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5910 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5911 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5912 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_2 := arithEq_of_rows h (row := 87) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_2
  simp only [← c5907, ← c5908, ← c5909] at e_87_2
  have e_87_3 := arithEq_of_rows h (row := 87) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_3
  simp only [← c5910, ← c5911, ← c5912] at e_87_3
  have hr := e_87_3
  simp only [e_87_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 23) = bselect (a (.wire 86 15)) (a (.virt 206)) (a (.wire 85 59)) := by
  have c5913 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5914 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5915 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5916 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5917 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5918 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_4 := arithEq_of_rows h (row := 87) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_4
  simp only [← c5913, ← c5914, ← c5915] at e_87_4
  have e_87_5 := arithEq_of_rows h (row := 87) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_5
  simp only [← c5916, ← c5917, ← c5918] at e_87_5
  have hr := e_87_5
  simp only [e_87_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 31) = bselect (a (.wire 86 15)) (a (.virt 207)) (a (.wire 85 63)) := by
  have c5919 := (publicBatchWrapper89_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5920 := (publicBatchWrapper89_copies185 a h).1
  have c5921 := (publicBatchWrapper89_copies185 a h).2.1
  have c5922 := (publicBatchWrapper89_copies185 a h).2.2.1
  have c5923 := (publicBatchWrapper89_copies185 a h).2.2.2.1
  have c5924 := (publicBatchWrapper89_copies185 a h).2.2.2.2.1
  have e_87_6 := arithEq_of_rows h (row := 87) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_6
  simp only [← c5919, ← c5920, ← c5921] at e_87_6
  have e_87_7 := arithEq_of_rows h (row := 87) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_7
  simp only [← c5922, ← c5923, ← c5924] at e_87_7
  have hr := e_87_7
  simp only [e_87_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 39) = bselect (a (.wire 86 15)) (a (.virt 208)) (a (.wire 85 67)) := by
  have c5925 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.1
  have c5926 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.1
  have c5927 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.1
  have c5928 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.1
  have c5929 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.1
  have c5930 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_87_8 := arithEq_of_rows h (row := 87) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_8
  simp only [← c5925, ← c5926, ← c5927] at e_87_8
  have e_87_9 := arithEq_of_rows h (row := 87) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_9
  simp only [← c5928, ← c5929, ← c5930] at e_87_9
  have hr := e_87_9
  simp only [e_87_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 47) = bselect (a (.wire 86 15)) (a (.virt 209)) (a (.wire 85 71)) := by
  have c5931 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5932 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5933 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5934 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5935 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5936 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_10 := arithEq_of_rows h (row := 87) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_10
  simp only [← c5931, ← c5932, ← c5933] at e_87_10
  have e_87_11 := arithEq_of_rows h (row := 87) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_11
  simp only [← c5934, ← c5935, ← c5936] at e_87_11
  have hr := e_87_11
  simp only [e_87_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 55) = bselect (a (.wire 86 15)) (a (.virt 203)) (a (.wire 85 75)) := by
  have c5937 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5938 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5939 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5940 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5941 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5942 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_12 := arithEq_of_rows h (row := 87) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_12
  simp only [← c5937, ← c5938, ← c5939] at e_87_12
  have e_87_13 := arithEq_of_rows h (row := 87) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_13
  simp only [← c5940, ← c5941, ← c5942] at e_87_13
  have hr := e_87_13
  simp only [e_87_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 63) = bselect (a (.wire 86 15)) (a (.virt 204)) (a (.wire 85 79)) := by
  have c5943 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5944 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5945 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5946 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5947 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5948 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_14 := arithEq_of_rows h (row := 87) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_14
  simp only [← c5943, ← c5944, ← c5945] at e_87_14
  have e_87_15 := arithEq_of_rows h (row := 87) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_15
  simp only [← c5946, ← c5947, ← c5948] at e_87_15
  have hr := e_87_15
  simp only [e_87_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 0 51) = bor (a (.wire 85 51)) (a (.wire 87 3)) := by
  have c5949 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5950 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5951 := (publicBatchWrapper89_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5952 := (publicBatchWrapper89_copies186 a h).1
  have c5953 := (publicBatchWrapper89_copies186 a h).2.1
  have c5954 := (publicBatchWrapper89_copies186 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_0_12 := arithEq_of_rows h (row := 0) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_12
  simp only [← c5952, ← c5953, k3, ← c5954] at e_0_12
  have e_88_0 := arithEq_of_rows h (row := 88) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_0
  simp only [← c5949, ← c5950, ← c5951] at e_88_0
  have hr := e_0_12
  simp only [e_88_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 67) = bnot (a (.wire 3 67)) := by
  have c5955 := (publicBatchWrapper89_copies186 a h).2.2.2.1
  have c5956 := (publicBatchWrapper89_copies186 a h).2.2.2.2.1
  have c5957 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_87_16 := arithEq_of_rows h (row := 87) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_16
  simp only [← c5955, k3, ← c5956, k3, ← c5957] at e_87_16
  have hr := e_87_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 71) = bnot (a (.wire 0 51)) := by
  have c5958 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.1
  have c5959 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.1
  have c5960 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_87_17 := arithEq_of_rows h (row := 87) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_17
  simp only [← c5958, k3, ← c5959, k3, ← c5960] at e_87_17
  have hr := e_87_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 19) = band (a (.wire 87 67)) (a (.wire 87 71)) := by
  have c5961 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.1
  have c5962 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5963 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_4 := arithEq_of_rows h (row := 86) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_4
  simp only [← c5961, ← c5962, ← c5963] at e_86_4
  have hr := e_86_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 87 79) = bselect (a (.wire 86 19)) (a (.virt 405)) (a (.wire 87 15)) := by
  have c5964 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5965 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5966 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5967 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5968 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5969 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_18 := arithEq_of_rows h (row := 87) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_18
  simp only [← c5964, ← c5965, ← c5966] at e_87_18
  have e_87_19 := arithEq_of_rows h (row := 87) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_19
  simp only [← c5967, ← c5968, ← c5969] at e_87_19
  have hr := e_87_19
  simp only [e_87_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 7) = bselect (a (.wire 86 19)) (a (.virt 406)) (a (.wire 87 23)) := by
  have c5970 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5971 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5972 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5973 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5974 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5975 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_0 := arithEq_of_rows h (row := 89) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_0
  simp only [← c5970, ← c5971, ← c5972] at e_89_0
  have e_89_1 := arithEq_of_rows h (row := 89) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_1
  simp only [← c5973, ← c5974, ← c5975] at e_89_1
  have hr := e_89_1
  simp only [e_89_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 15) = bselect (a (.wire 86 19)) (a (.virt 407)) (a (.wire 87 31)) := by
  have c5976 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5977 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5978 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5979 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5980 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5981 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_2 := arithEq_of_rows h (row := 89) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_2
  simp only [← c5976, ← c5977, ← c5978] at e_89_2
  have e_89_3 := arithEq_of_rows h (row := 89) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_3
  simp only [← c5979, ← c5980, ← c5981] at e_89_3
  have hr := e_89_3
  simp only [e_89_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 23) = bselect (a (.wire 86 19)) (a (.virt 408)) (a (.wire 87 39)) := by
  have c5982 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5983 := (publicBatchWrapper89_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5984 := (publicBatchWrapper89_copies187 a h).1
  have c5985 := (publicBatchWrapper89_copies187 a h).2.1
  have c5986 := (publicBatchWrapper89_copies187 a h).2.2.1
  have c5987 := (publicBatchWrapper89_copies187 a h).2.2.2.1
  have e_89_4 := arithEq_of_rows h (row := 89) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_4
  simp only [← c5982, ← c5983, ← c5984] at e_89_4
  have e_89_5 := arithEq_of_rows h (row := 89) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_5
  simp only [← c5985, ← c5986, ← c5987] at e_89_5
  have hr := e_89_5
  simp only [e_89_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 31) = bselect (a (.wire 86 19)) (a (.virt 409)) (a (.wire 87 47)) := by
  have c5988 := (publicBatchWrapper89_copies187 a h).2.2.2.2.1
  have c5989 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.1
  have c5990 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.1
  have c5991 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.1
  have c5992 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.1
  have c5993 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.1
  have e_89_6 := arithEq_of_rows h (row := 89) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_6
  simp only [← c5988, ← c5989, ← c5990] at e_89_6
  have e_89_7 := arithEq_of_rows h (row := 89) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_7
  simp only [← c5991, ← c5992, ← c5993] at e_89_7
  have hr := e_89_7
  simp only [e_89_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 39) = bselect (a (.wire 86 19)) (a (.virt 403)) (a (.wire 87 55)) := by
  have c5994 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5995 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5996 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5997 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5998 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5999 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_8 := arithEq_of_rows h (row := 89) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_8
  simp only [← c5994, ← c5995, ← c5996] at e_89_8
  have e_89_9 := arithEq_of_rows h (row := 89) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_9
  simp only [← c5997, ← c5998, ← c5999] at e_89_9
  have hr := e_89_9
  simp only [e_89_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 47) = bselect (a (.wire 86 19)) (a (.virt 404)) (a (.wire 87 63)) := by
  have c6000 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6001 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6002 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6003 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6004 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6005 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_10 := arithEq_of_rows h (row := 89) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_10
  simp only [← c6000, ← c6001, ← c6002] at e_89_10
  have e_89_11 := arithEq_of_rows h (row := 89) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_11
  simp only [← c6003, ← c6004, ← c6005] at e_89_11
  have hr := e_89_11
  simp only [e_89_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 0 55) = bor (a (.wire 0 51)) (a (.wire 87 67)) := by
  have c6006 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6007 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6008 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6009 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6010 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6011 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_0_13 := arithEq_of_rows h (row := 0) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_13
  simp only [← c6009, ← c6010, k3, ← c6011] at e_0_13
  have e_88_1 := arithEq_of_rows h (row := 88) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_1
  simp only [← c6006, ← c6007, ← c6008] at e_88_1
  have hr := e_0_13
  simp only [e_88_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 51) = bnot (a (.wire 5 31)) := by
  have c6012 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6013 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6014 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_89_12 := arithEq_of_rows h (row := 89) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_12
  simp only [← c6012, k3, ← c6013, k3, ← c6014] at e_89_12
  have hr := e_89_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 55) = bnot (a (.wire 0 55)) := by
  have c6015 := (publicBatchWrapper89_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6016 := (publicBatchWrapper89_copies188 a h).1
  have c6017 := (publicBatchWrapper89_copies188 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_89_13 := arithEq_of_rows h (row := 89) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_13
  simp only [← c6015, k3, ← c6016, k3, ← c6017] at e_89_13
  have hr := e_89_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 23) = band (a (.wire 89 51)) (a (.wire 89 55)) := by
  have c6018 := (publicBatchWrapper89_copies188 a h).2.2.1
  have c6019 := (publicBatchWrapper89_copies188 a h).2.2.2.1
  have c6020 := (publicBatchWrapper89_copies188 a h).2.2.2.2.1
  have e_86_5 := arithEq_of_rows h (row := 86) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_5
  simp only [← c6018, ← c6019, ← c6020] at e_86_5
  have hr := e_86_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 63) = bselect (a (.wire 86 23)) (a (.virt 605)) (a (.wire 87 79)) := by
  have c6021 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.1
  have c6022 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.1
  have c6023 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.1
  have c6024 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.1
  have c6025 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.1
  have c6026 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_89_14 := arithEq_of_rows h (row := 89) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_14
  simp only [← c6021, ← c6022, ← c6023] at e_89_14
  have e_89_15 := arithEq_of_rows h (row := 89) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_15
  simp only [← c6024, ← c6025, ← c6026] at e_89_15
  have hr := e_89_15
  simp only [e_89_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 71) = bselect (a (.wire 86 23)) (a (.virt 606)) (a (.wire 89 7)) := by
  have c6027 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6028 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6029 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6030 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6031 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6032 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_16 := arithEq_of_rows h (row := 89) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_16
  simp only [← c6027, ← c6028, ← c6029] at e_89_16
  have e_89_17 := arithEq_of_rows h (row := 89) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_17
  simp only [← c6030, ← c6031, ← c6032] at e_89_17
  have hr := e_89_17
  simp only [e_89_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 89 79) = bselect (a (.wire 86 23)) (a (.virt 607)) (a (.wire 89 15)) := by
  have c6033 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6034 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6035 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6036 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6037 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6038 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_18 := arithEq_of_rows h (row := 89) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_18
  simp only [← c6033, ← c6034, ← c6035] at e_89_18
  have e_89_19 := arithEq_of_rows h (row := 89) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_19
  simp only [← c6036, ← c6037, ← c6038] at e_89_19
  have hr := e_89_19
  simp only [e_89_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 7) = bselect (a (.wire 86 23)) (a (.virt 608)) (a (.wire 89 23)) := by
  have c6039 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6040 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6041 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6042 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6043 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6044 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_0 := arithEq_of_rows h (row := 90) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_0
  simp only [← c6039, ← c6040, ← c6041] at e_90_0
  have e_90_1 := arithEq_of_rows h (row := 90) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_1
  simp only [← c6042, ← c6043, ← c6044] at e_90_1
  have hr := e_90_1
  simp only [e_90_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 15) = bselect (a (.wire 86 23)) (a (.virt 609)) (a (.wire 89 31)) := by
  have c6045 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6046 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6047 := (publicBatchWrapper89_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6048 := (publicBatchWrapper89_copies189 a h).1
  have c6049 := (publicBatchWrapper89_copies189 a h).2.1
  have c6050 := (publicBatchWrapper89_copies189 a h).2.2.1
  have e_90_2 := arithEq_of_rows h (row := 90) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_2
  simp only [← c6045, ← c6046, ← c6047] at e_90_2
  have e_90_3 := arithEq_of_rows h (row := 90) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_3
  simp only [← c6048, ← c6049, ← c6050] at e_90_3
  have hr := e_90_3
  simp only [e_90_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 23) = bselect (a (.wire 86 23)) (a (.virt 603)) (a (.wire 89 39)) := by
  have c6051 := (publicBatchWrapper89_copies189 a h).2.2.2.1
  have c6052 := (publicBatchWrapper89_copies189 a h).2.2.2.2.1
  have c6053 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.1
  have c6054 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.1
  have c6055 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.1
  have c6056 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.1
  have e_90_4 := arithEq_of_rows h (row := 90) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_4
  simp only [← c6051, ← c6052, ← c6053] at e_90_4
  have e_90_5 := arithEq_of_rows h (row := 90) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_5
  simp only [← c6054, ← c6055, ← c6056] at e_90_5
  have hr := e_90_5
  simp only [e_90_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 31) = bselect (a (.wire 86 23)) (a (.virt 604)) (a (.wire 89 47)) := by
  have c6057 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.1
  have c6058 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6059 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6060 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6061 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6062 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_6 := arithEq_of_rows h (row := 90) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_6
  simp only [← c6057, ← c6058, ← c6059] at e_90_6
  have e_90_7 := arithEq_of_rows h (row := 90) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_7
  simp only [← c6060, ← c6061, ← c6062] at e_90_7
  have hr := e_90_7
  simp only [e_90_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 0 59) = bor (a (.wire 0 55)) (a (.wire 89 51)) := by
  have c6063 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6064 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6065 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6066 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6067 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6068 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_0_14 := arithEq_of_rows h (row := 0) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_14
  simp only [← c6066, ← c6067, k3, ← c6068] at e_0_14
  have e_88_2 := arithEq_of_rows h (row := 88) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_2
  simp only [← c6063, ← c6064, ← c6065] at e_88_2
  have hr := e_0_14
  simp only [e_88_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 35) = bnot (a (.wire 5 75)) := by
  have c6069 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6070 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6071 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_90_8 := arithEq_of_rows h (row := 90) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_8
  simp only [← c6069, k3, ← c6070, k3, ← c6071] at e_90_8
  have hr := e_90_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 39) = bnot (a (.wire 0 59)) := by
  have c6072 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6073 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6074 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_90_9 := arithEq_of_rows h (row := 90) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_9
  simp only [← c6072, k3, ← c6073, k3, ← c6074] at e_90_9
  have hr := e_90_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 27) = band (a (.wire 90 35)) (a (.wire 90 39)) := by
  have c6075 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6076 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6077 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_6 := arithEq_of_rows h (row := 86) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_6
  simp only [← c6075, ← c6076, ← c6077] at e_86_6
  have hr := e_86_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 47) = bselect (a (.wire 86 27)) (a (.virt 805)) (a (.wire 89 63)) := by
  have c6078 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6079 := (publicBatchWrapper89_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6080 := (publicBatchWrapper89_copies190 a h).1
  have c6081 := (publicBatchWrapper89_copies190 a h).2.1
  have c6082 := (publicBatchWrapper89_copies190 a h).2.2.1
  have c6083 := (publicBatchWrapper89_copies190 a h).2.2.2.1
  have e_90_10 := arithEq_of_rows h (row := 90) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_10
  simp only [← c6078, ← c6079, ← c6080] at e_90_10
  have e_90_11 := arithEq_of_rows h (row := 90) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_11
  simp only [← c6081, ← c6082, ← c6083] at e_90_11
  have hr := e_90_11
  simp only [e_90_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 55) = bselect (a (.wire 86 27)) (a (.virt 806)) (a (.wire 89 71)) := by
  have c6084 := (publicBatchWrapper89_copies190 a h).2.2.2.2.1
  have c6085 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.1
  have c6086 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.1
  have c6087 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.1
  have c6088 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.1
  have c6089 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.1
  have e_90_12 := arithEq_of_rows h (row := 90) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_12
  simp only [← c6084, ← c6085, ← c6086] at e_90_12
  have e_90_13 := arithEq_of_rows h (row := 90) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_13
  simp only [← c6087, ← c6088, ← c6089] at e_90_13
  have hr := e_90_13
  simp only [e_90_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 63) = bselect (a (.wire 86 27)) (a (.virt 807)) (a (.wire 89 79)) := by
  have c6090 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6091 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6092 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6093 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6094 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6095 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_14 := arithEq_of_rows h (row := 90) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_14
  simp only [← c6090, ← c6091, ← c6092] at e_90_14
  have e_90_15 := arithEq_of_rows h (row := 90) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_15
  simp only [← c6093, ← c6094, ← c6095] at e_90_15
  have hr := e_90_15
  simp only [e_90_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 71) = bselect (a (.wire 86 27)) (a (.virt 808)) (a (.wire 90 7)) := by
  have c6096 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6097 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6098 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6099 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6100 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6101 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_16 := arithEq_of_rows h (row := 90) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_16
  simp only [← c6096, ← c6097, ← c6098] at e_90_16
  have e_90_17 := arithEq_of_rows h (row := 90) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_17
  simp only [← c6099, ← c6100, ← c6101] at e_90_17
  have hr := e_90_17
  simp only [e_90_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 90 79) = bselect (a (.wire 86 27)) (a (.virt 809)) (a (.wire 90 15)) := by
  have c6102 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6103 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6104 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6105 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6106 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6107 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_18 := arithEq_of_rows h (row := 90) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_18
  simp only [← c6102, ← c6103, ← c6104] at e_90_18
  have e_90_19 := arithEq_of_rows h (row := 90) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_19
  simp only [← c6105, ← c6106, ← c6107] at e_90_19
  have hr := e_90_19
  simp only [e_90_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 7) = bselect (a (.wire 86 27)) (a (.virt 803)) (a (.wire 90 23)) := by
  have c6108 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6109 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6110 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6111 := (publicBatchWrapper89_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6112 := (publicBatchWrapper89_copies191 a h).1
  have c6113 := (publicBatchWrapper89_copies191 a h).2.1
  have e_91_0 := arithEq_of_rows h (row := 91) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_0
  simp only [← c6108, ← c6109, ← c6110] at e_91_0
  have e_91_1 := arithEq_of_rows h (row := 91) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_1
  simp only [← c6111, ← c6112, ← c6113] at e_91_1
  have hr := e_91_1
  simp only [e_91_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 15) = bselect (a (.wire 86 27)) (a (.virt 804)) (a (.wire 90 31)) := by
  have c6114 := (publicBatchWrapper89_copies191 a h).2.2.1
  have c6115 := (publicBatchWrapper89_copies191 a h).2.2.2.1
  have c6116 := (publicBatchWrapper89_copies191 a h).2.2.2.2.1
  have c6117 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.1
  have c6118 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.1
  have c6119 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.1
  have e_91_2 := arithEq_of_rows h (row := 91) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_2
  simp only [← c6114, ← c6115, ← c6116] at e_91_2
  have e_91_3 := arithEq_of_rows h (row := 91) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_3
  simp only [← c6117, ← c6118, ← c6119] at e_91_3
  have hr := e_91_3
  simp only [e_91_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 0 63) = bor (a (.wire 0 59)) (a (.wire 90 35)) := by
  have c6120 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.1
  have c6121 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.1
  have c6122 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6123 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6124 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6125 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_0_15 := arithEq_of_rows h (row := 0) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_15
  simp only [← c6123, ← c6124, k3, ← c6125] at e_0_15
  have e_88_3 := arithEq_of_rows h (row := 88) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_3
  simp only [← c6120, ← c6121, ← c6122] at e_88_3
  have hr := e_0_15
  simp only [e_88_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 19) = bnot (a (.wire 7 39)) := by
  have c6126 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6127 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6128 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_91_4 := arithEq_of_rows h (row := 91) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_4
  simp only [← c6126, k3, ← c6127, k3, ← c6128] at e_91_4
  have hr := e_91_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 23) = bnot (a (.wire 0 63)) := by
  have c6129 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6130 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6131 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_91_5 := arithEq_of_rows h (row := 91) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_5
  simp only [← c6129, k3, ← c6130, k3, ← c6131] at e_91_5
  have hr := e_91_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 31) = band (a (.wire 91 19)) (a (.wire 91 23)) := by
  have c6132 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6133 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6134 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_7 := arithEq_of_rows h (row := 86) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_7
  simp only [← c6132, ← c6133, ← c6134] at e_86_7
  have hr := e_86_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 31) = bselect (a (.wire 86 31)) (a (.virt 1005)) (a (.wire 90 47)) := by
  have c6135 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6136 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6137 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6138 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6139 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6140 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_6 := arithEq_of_rows h (row := 91) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_6
  simp only [← c6135, ← c6136, ← c6137] at e_91_6
  have e_91_7 := arithEq_of_rows h (row := 91) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_7
  simp only [← c6138, ← c6139, ← c6140] at e_91_7
  have hr := e_91_7
  simp only [e_91_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 39) = bselect (a (.wire 86 31)) (a (.virt 1006)) (a (.wire 90 55)) := by
  have c6141 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6142 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6143 := (publicBatchWrapper89_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6144 := (publicBatchWrapper89_copies192 a h).1
  have c6145 := (publicBatchWrapper89_copies192 a h).2.1
  have c6146 := (publicBatchWrapper89_copies192 a h).2.2.1
  have e_91_8 := arithEq_of_rows h (row := 91) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_8
  simp only [← c6141, ← c6142, ← c6143] at e_91_8
  have e_91_9 := arithEq_of_rows h (row := 91) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_9
  simp only [← c6144, ← c6145, ← c6146] at e_91_9
  have hr := e_91_9
  simp only [e_91_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 47) = bselect (a (.wire 86 31)) (a (.virt 1007)) (a (.wire 90 63)) := by
  have c6147 := (publicBatchWrapper89_copies192 a h).2.2.2.1
  have c6148 := (publicBatchWrapper89_copies192 a h).2.2.2.2.1
  have c6149 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.1
  have c6150 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.1
  have c6151 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.1
  have c6152 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.1
  have e_91_10 := arithEq_of_rows h (row := 91) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_10
  simp only [← c6147, ← c6148, ← c6149] at e_91_10
  have e_91_11 := arithEq_of_rows h (row := 91) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_11
  simp only [← c6150, ← c6151, ← c6152] at e_91_11
  have hr := e_91_11
  simp only [e_91_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 55) = bselect (a (.wire 86 31)) (a (.virt 1008)) (a (.wire 90 71)) := by
  have c6153 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.1
  have c6154 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6155 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6156 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6157 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6158 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_12 := arithEq_of_rows h (row := 91) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_12
  simp only [← c6153, ← c6154, ← c6155] at e_91_12
  have e_91_13 := arithEq_of_rows h (row := 91) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_13
  simp only [← c6156, ← c6157, ← c6158] at e_91_13
  have hr := e_91_13
  simp only [e_91_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 63) = bselect (a (.wire 86 31)) (a (.virt 1009)) (a (.wire 90 79)) := by
  have c6159 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6160 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6161 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6162 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6163 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6164 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_14 := arithEq_of_rows h (row := 91) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_14
  simp only [← c6159, ← c6160, ← c6161] at e_91_14
  have e_91_15 := arithEq_of_rows h (row := 91) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_15
  simp only [← c6162, ← c6163, ← c6164] at e_91_15
  have hr := e_91_15
  simp only [e_91_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 71) = bselect (a (.wire 86 31)) (a (.virt 1003)) (a (.wire 91 7)) := by
  have c6165 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6166 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6167 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6168 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6169 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6170 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_16 := arithEq_of_rows h (row := 91) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_16
  simp only [← c6165, ← c6166, ← c6167] at e_91_16
  have e_91_17 := arithEq_of_rows h (row := 91) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_17
  simp only [← c6168, ← c6169, ← c6170] at e_91_17
  have hr := e_91_17
  simp only [e_91_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 91 79) = bselect (a (.wire 86 31)) (a (.virt 1004)) (a (.wire 91 15)) := by
  have c6171 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6172 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6173 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6174 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6175 := (publicBatchWrapper89_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6176 := (publicBatchWrapper89_copies193 a h).1
  have e_91_18 := arithEq_of_rows h (row := 91) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_18
  simp only [← c6171, ← c6172, ← c6173] at e_91_18
  have e_91_19 := arithEq_of_rows h (row := 91) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_19
  simp only [← c6174, ← c6175, ← c6176] at e_91_19
  have hr := e_91_19
  simp only [e_91_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 0 67) = bor (a (.wire 0 63)) (a (.wire 91 19)) := by
  have c6177 := (publicBatchWrapper89_copies193 a h).2.1
  have c6178 := (publicBatchWrapper89_copies193 a h).2.2.1
  have c6179 := (publicBatchWrapper89_copies193 a h).2.2.2.1
  have c6180 := (publicBatchWrapper89_copies193 a h).2.2.2.2.1
  have c6181 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.1
  have c6182 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_0_16 := arithEq_of_rows h (row := 0) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_16
  simp only [← c6180, ← c6181, k3, ← c6182] at e_0_16
  have e_88_4 := arithEq_of_rows h (row := 88) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_4
  simp only [← c6177, ← c6178, ← c6179] at e_88_4
  have hr := e_0_16
  simp only [e_88_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 3) = bnot (a (.wire 8 3)) := by
  have c6183 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.1
  have c6184 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.1
  have c6185 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_92_0 := arithEq_of_rows h (row := 92) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_0
  simp only [← c6183, k3, ← c6184, k3, ← c6185] at e_92_0
  have hr := e_92_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 7) = bnot (a (.wire 0 67)) := by
  have c6186 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6187 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6188 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_92_1 := arithEq_of_rows h (row := 92) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_1
  simp only [← c6186, k3, ← c6187, k3, ← c6188] at e_92_1
  have hr := e_92_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 35) = band (a (.wire 92 3)) (a (.wire 92 7)) := by
  have c6189 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6190 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6191 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_8 := arithEq_of_rows h (row := 86) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_8
  simp only [← c6189, ← c6190, ← c6191] at e_86_8
  have hr := e_86_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 15) = bselect (a (.wire 86 35)) (a (.virt 1205)) (a (.wire 91 31)) := by
  have c6192 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6193 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6194 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6195 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6196 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6197 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_2 := arithEq_of_rows h (row := 92) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_2
  simp only [← c6192, ← c6193, ← c6194] at e_92_2
  have e_92_3 := arithEq_of_rows h (row := 92) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_3
  simp only [← c6195, ← c6196, ← c6197] at e_92_3
  have hr := e_92_3
  simp only [e_92_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 23) = bselect (a (.wire 86 35)) (a (.virt 1206)) (a (.wire 91 39)) := by
  have c6198 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6199 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6200 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6201 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6202 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6203 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_4 := arithEq_of_rows h (row := 92) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_4
  simp only [← c6198, ← c6199, ← c6200] at e_92_4
  have e_92_5 := arithEq_of_rows h (row := 92) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_5
  simp only [← c6201, ← c6202, ← c6203] at e_92_5
  have hr := e_92_5
  simp only [e_92_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 31) = bselect (a (.wire 86 35)) (a (.virt 1207)) (a (.wire 91 47)) := by
  have c6204 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6205 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6206 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6207 := (publicBatchWrapper89_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6208 := (publicBatchWrapper89_copies194 a h).1
  have c6209 := (publicBatchWrapper89_copies194 a h).2.1
  have e_92_6 := arithEq_of_rows h (row := 92) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_6
  simp only [← c6204, ← c6205, ← c6206] at e_92_6
  have e_92_7 := arithEq_of_rows h (row := 92) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_7
  simp only [← c6207, ← c6208, ← c6209] at e_92_7
  have hr := e_92_7
  simp only [e_92_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 39) = bselect (a (.wire 86 35)) (a (.virt 1208)) (a (.wire 91 55)) := by
  have c6210 := (publicBatchWrapper89_copies194 a h).2.2.1
  have c6211 := (publicBatchWrapper89_copies194 a h).2.2.2.1
  have c6212 := (publicBatchWrapper89_copies194 a h).2.2.2.2.1
  have c6213 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.1
  have c6214 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.1
  have c6215 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.1
  have e_92_8 := arithEq_of_rows h (row := 92) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_8
  simp only [← c6210, ← c6211, ← c6212] at e_92_8
  have e_92_9 := arithEq_of_rows h (row := 92) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_9
  simp only [← c6213, ← c6214, ← c6215] at e_92_9
  have hr := e_92_9
  simp only [e_92_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 47) = bselect (a (.wire 86 35)) (a (.virt 1209)) (a (.wire 91 63)) := by
  have c6216 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.1
  have c6217 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.1
  have c6218 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6219 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6220 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6221 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_10 := arithEq_of_rows h (row := 92) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_10
  simp only [← c6216, ← c6217, ← c6218] at e_92_10
  have e_92_11 := arithEq_of_rows h (row := 92) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_11
  simp only [← c6219, ← c6220, ← c6221] at e_92_11
  have hr := e_92_11
  simp only [e_92_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 55) = bselect (a (.wire 86 35)) (a (.virt 1203)) (a (.wire 91 71)) := by
  have c6222 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6223 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6224 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6225 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6226 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6227 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_12 := arithEq_of_rows h (row := 92) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_12
  simp only [← c6222, ← c6223, ← c6224] at e_92_12
  have e_92_13 := arithEq_of_rows h (row := 92) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_13
  simp only [← c6225, ← c6226, ← c6227] at e_92_13
  have hr := e_92_13
  simp only [e_92_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 63) = bselect (a (.wire 86 35)) (a (.virt 1204)) (a (.wire 91 79)) := by
  have c6228 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6229 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6230 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6231 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6232 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6233 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_14 := arithEq_of_rows h (row := 92) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_14
  simp only [← c6228, ← c6229, ← c6230] at e_92_14
  have e_92_15 := arithEq_of_rows h (row := 92) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_15
  simp only [← c6231, ← c6232, ← c6233] at e_92_15
  have hr := e_92_15
  simp only [e_92_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 0 71) = bor (a (.wire 0 67)) (a (.wire 92 3)) := by
  have c6234 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6235 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6236 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6237 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6238 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6239 := (publicBatchWrapper89_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_0_17 := arithEq_of_rows h (row := 0) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_17
  simp only [← c6237, ← c6238, k3, ← c6239] at e_0_17
  have e_88_5 := arithEq_of_rows h (row := 88) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_5
  simp only [← c6234, ← c6235, ← c6236] at e_88_5
  have hr := e_0_17
  simp only [e_88_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 67) = bnot (a (.wire 8 47)) := by
  have c6240 := (publicBatchWrapper89_copies195 a h).1
  have c6241 := (publicBatchWrapper89_copies195 a h).2.1
  have c6242 := (publicBatchWrapper89_copies195 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_92_16 := arithEq_of_rows h (row := 92) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_16
  simp only [← c6240, k3, ← c6241, k3, ← c6242] at e_92_16
  have hr := e_92_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 71) = bnot (a (.wire 0 71)) := by
  have c6243 := (publicBatchWrapper89_copies195 a h).2.2.2.1
  have c6244 := (publicBatchWrapper89_copies195 a h).2.2.2.2.1
  have c6245 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_92_17 := arithEq_of_rows h (row := 92) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_17
  simp only [← c6243, k3, ← c6244, k3, ← c6245] at e_92_17
  have hr := e_92_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 39) = band (a (.wire 92 67)) (a (.wire 92 71)) := by
  have c6246 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.1
  have c6247 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.1
  have c6248 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.1
  have e_86_9 := arithEq_of_rows h (row := 86) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_9
  simp only [← c6246, ← c6247, ← c6248] at e_86_9
  have hr := e_86_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 92 79) = bselect (a (.wire 86 39)) (a (.virt 1405)) (a (.wire 92 15)) := by
  have c6249 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.1
  have c6250 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6251 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6252 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6253 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6254 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_18 := arithEq_of_rows h (row := 92) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_18
  simp only [← c6249, ← c6250, ← c6251] at e_92_18
  have e_92_19 := arithEq_of_rows h (row := 92) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_19
  simp only [← c6252, ← c6253, ← c6254] at e_92_19
  have hr := e_92_19
  simp only [e_92_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 7) = bselect (a (.wire 86 39)) (a (.virt 1406)) (a (.wire 92 23)) := by
  have c6255 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6256 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6257 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6258 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6259 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6260 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_0 := arithEq_of_rows h (row := 93) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_0
  simp only [← c6255, ← c6256, ← c6257] at e_93_0
  have e_93_1 := arithEq_of_rows h (row := 93) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_1
  simp only [← c6258, ← c6259, ← c6260] at e_93_1
  have hr := e_93_1
  simp only [e_93_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 15) = bselect (a (.wire 86 39)) (a (.virt 1407)) (a (.wire 92 31)) := by
  have c6261 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6262 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6263 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6264 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6265 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6266 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_2 := arithEq_of_rows h (row := 93) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_2
  simp only [← c6261, ← c6262, ← c6263] at e_93_2
  have e_93_3 := arithEq_of_rows h (row := 93) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_3
  simp only [← c6264, ← c6265, ← c6266] at e_93_3
  have hr := e_93_3
  simp only [e_93_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 23) = bselect (a (.wire 86 39)) (a (.virt 1408)) (a (.wire 92 39)) := by
  have c6267 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6268 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6269 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6270 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6271 := (publicBatchWrapper89_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6272 := (publicBatchWrapper89_copies196 a h).1
  have e_93_4 := arithEq_of_rows h (row := 93) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_4
  simp only [← c6267, ← c6268, ← c6269] at e_93_4
  have e_93_5 := arithEq_of_rows h (row := 93) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_5
  simp only [← c6270, ← c6271, ← c6272] at e_93_5
  have hr := e_93_5
  simp only [e_93_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 31) = bselect (a (.wire 86 39)) (a (.virt 1409)) (a (.wire 92 47)) := by
  have c6273 := (publicBatchWrapper89_copies196 a h).2.1
  have c6274 := (publicBatchWrapper89_copies196 a h).2.2.1
  have c6275 := (publicBatchWrapper89_copies196 a h).2.2.2.1
  have c6276 := (publicBatchWrapper89_copies196 a h).2.2.2.2.1
  have c6277 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.1
  have c6278 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.1
  have e_93_6 := arithEq_of_rows h (row := 93) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_6
  simp only [← c6273, ← c6274, ← c6275] at e_93_6
  have e_93_7 := arithEq_of_rows h (row := 93) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_7
  simp only [← c6276, ← c6277, ← c6278] at e_93_7
  have hr := e_93_7
  simp only [e_93_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 39) = bselect (a (.wire 86 39)) (a (.virt 1403)) (a (.wire 92 55)) := by
  have c6279 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.1
  have c6280 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.1
  have c6281 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.1
  have c6282 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6283 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6284 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_8 := arithEq_of_rows h (row := 93) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_8
  simp only [← c6279, ← c6280, ← c6281] at e_93_8
  have e_93_9 := arithEq_of_rows h (row := 93) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_9
  simp only [← c6282, ← c6283, ← c6284] at e_93_9
  have hr := e_93_9
  simp only [e_93_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 47) = bselect (a (.wire 86 39)) (a (.virt 1404)) (a (.wire 92 63)) := by
  have c6285 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6286 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6287 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6288 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6289 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6290 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_10 := arithEq_of_rows h (row := 93) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_10
  simp only [← c6285, ← c6286, ← c6287] at e_93_10
  have e_93_11 := arithEq_of_rows h (row := 93) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_11
  simp only [← c6288, ← c6289, ← c6290] at e_93_11
  have hr := e_93_11
  simp only [e_93_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 0 75) = bor (a (.wire 0 71)) (a (.wire 92 67)) := by
  have c6291 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6292 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6293 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6294 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6295 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6296 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_0_18 := arithEq_of_rows h (row := 0) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_18
  simp only [← c6294, ← c6295, k3, ← c6296] at e_0_18
  have e_88_6 := arithEq_of_rows h (row := 88) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_6
  simp only [← c6291, ← c6292, ← c6293] at e_88_6
  have hr := e_0_18
  simp only [e_88_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 51) = bnot (a (.wire 10 11)) := by
  have c6297 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6298 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6299 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_93_12 := arithEq_of_rows h (row := 93) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_12
  simp only [← c6297, k3, ← c6298, k3, ← c6299] at e_93_12
  have hr := e_93_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 55) = bnot (a (.wire 0 75)) := by
  have c6300 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6301 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6302 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_93_13 := arithEq_of_rows h (row := 93) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_13
  simp only [← c6300, k3, ← c6301, k3, ← c6302] at e_93_13
  have hr := e_93_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 43) = band (a (.wire 93 51)) (a (.wire 93 55)) := by
  have c6303 := (publicBatchWrapper89_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6304 := (publicBatchWrapper89_copies197 a h).1
  have c6305 := (publicBatchWrapper89_copies197 a h).2.1
  have e_86_10 := arithEq_of_rows h (row := 86) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_10
  simp only [← c6303, ← c6304, ← c6305] at e_86_10
  have hr := e_86_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 63) = bselect (a (.wire 86 43)) (a (.virt 1605)) (a (.wire 92 79)) := by
  have c6306 := (publicBatchWrapper89_copies197 a h).2.2.1
  have c6307 := (publicBatchWrapper89_copies197 a h).2.2.2.1
  have c6308 := (publicBatchWrapper89_copies197 a h).2.2.2.2.1
  have c6309 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.1
  have c6310 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.1
  have c6311 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.1
  have e_93_14 := arithEq_of_rows h (row := 93) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_14
  simp only [← c6306, ← c6307, ← c6308] at e_93_14
  have e_93_15 := arithEq_of_rows h (row := 93) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_15
  simp only [← c6309, ← c6310, ← c6311] at e_93_15
  have hr := e_93_15
  simp only [e_93_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 71) = bselect (a (.wire 86 43)) (a (.virt 1606)) (a (.wire 93 7)) := by
  have c6312 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.1
  have c6313 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.1
  have c6314 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6315 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6316 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6317 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_16 := arithEq_of_rows h (row := 93) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_16
  simp only [← c6312, ← c6313, ← c6314] at e_93_16
  have e_93_17 := arithEq_of_rows h (row := 93) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_17
  simp only [← c6315, ← c6316, ← c6317] at e_93_17
  have hr := e_93_17
  simp only [e_93_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 93 79) = bselect (a (.wire 86 43)) (a (.virt 1607)) (a (.wire 93 15)) := by
  have c6318 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6319 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6320 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6321 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6322 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6323 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_18 := arithEq_of_rows h (row := 93) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_18
  simp only [← c6318, ← c6319, ← c6320] at e_93_18
  have e_93_19 := arithEq_of_rows h (row := 93) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_19
  simp only [← c6321, ← c6322, ← c6323] at e_93_19
  have hr := e_93_19
  simp only [e_93_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 7) = bselect (a (.wire 86 43)) (a (.virt 1608)) (a (.wire 93 23)) := by
  have c6324 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6325 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6326 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6327 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6328 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6329 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_0 := arithEq_of_rows h (row := 94) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_0
  simp only [← c6324, ← c6325, ← c6326] at e_94_0
  have e_94_1 := arithEq_of_rows h (row := 94) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_1
  simp only [← c6327, ← c6328, ← c6329] at e_94_1
  have hr := e_94_1
  simp only [e_94_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 15) = bselect (a (.wire 86 43)) (a (.virt 1609)) (a (.wire 93 31)) := by
  have c6330 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6331 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6332 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6333 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6334 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6335 := (publicBatchWrapper89_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_94_2 := arithEq_of_rows h (row := 94) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_2
  simp only [← c6330, ← c6331, ← c6332] at e_94_2
  have e_94_3 := arithEq_of_rows h (row := 94) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_3
  simp only [← c6333, ← c6334, ← c6335] at e_94_3
  have hr := e_94_3
  simp only [e_94_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 23) = bselect (a (.wire 86 43)) (a (.virt 1603)) (a (.wire 93 39)) := by
  have c6336 := (publicBatchWrapper89_copies198 a h).1
  have c6337 := (publicBatchWrapper89_copies198 a h).2.1
  have c6338 := (publicBatchWrapper89_copies198 a h).2.2.1
  have c6339 := (publicBatchWrapper89_copies198 a h).2.2.2.1
  have c6340 := (publicBatchWrapper89_copies198 a h).2.2.2.2.1
  have c6341 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.1
  have e_94_4 := arithEq_of_rows h (row := 94) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_4
  simp only [← c6336, ← c6337, ← c6338] at e_94_4
  have e_94_5 := arithEq_of_rows h (row := 94) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_5
  simp only [← c6339, ← c6340, ← c6341] at e_94_5
  have hr := e_94_5
  simp only [e_94_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 31) = bselect (a (.wire 86 43)) (a (.virt 1604)) (a (.wire 93 47)) := by
  have c6342 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.1
  have c6343 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.1
  have c6344 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.1
  have c6345 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.1
  have c6346 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6347 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_6 := arithEq_of_rows h (row := 94) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_6
  simp only [← c6342, ← c6343, ← c6344] at e_94_6
  have e_94_7 := arithEq_of_rows h (row := 94) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_7
  simp only [← c6345, ← c6346, ← c6347] at e_94_7
  have hr := e_94_7
  simp only [e_94_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 0 79) = bor (a (.wire 0 75)) (a (.wire 93 51)) := by
  have c6348 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6349 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6350 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6351 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6352 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6353 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_0_19 := arithEq_of_rows h (row := 0) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_19
  simp only [← c6351, ← c6352, k3, ← c6353] at e_0_19
  have e_88_7 := arithEq_of_rows h (row := 88) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_7
  simp only [← c6348, ← c6349, ← c6350] at e_88_7
  have hr := e_0_19
  simp only [e_88_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 35) = bnot (a (.wire 10 55)) := by
  have c6354 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6355 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6356 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_94_8 := arithEq_of_rows h (row := 94) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_8
  simp only [← c6354, k3, ← c6355, k3, ← c6356] at e_94_8
  have hr := e_94_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 39) = bnot (a (.wire 0 79)) := by
  have c6357 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6358 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6359 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_94_9 := arithEq_of_rows h (row := 94) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_9
  simp only [← c6357, k3, ← c6358, k3, ← c6359] at e_94_9
  have hr := e_94_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 47) = band (a (.wire 94 35)) (a (.wire 94 39)) := by
  have c6360 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6361 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6362 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_11 := arithEq_of_rows h (row := 86) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_11
  simp only [← c6360, ← c6361, ← c6362] at e_86_11
  have hr := e_86_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 47) = bselect (a (.wire 86 47)) (a (.virt 1805)) (a (.wire 93 63)) := by
  have c6363 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6364 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6365 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6366 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6367 := (publicBatchWrapper89_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6368 := (publicBatchWrapper89_copies199 a h).1
  have e_94_10 := arithEq_of_rows h (row := 94) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_10
  simp only [← c6363, ← c6364, ← c6365] at e_94_10
  have e_94_11 := arithEq_of_rows h (row := 94) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_11
  simp only [← c6366, ← c6367, ← c6368] at e_94_11
  have hr := e_94_11
  simp only [e_94_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 55) = bselect (a (.wire 86 47)) (a (.virt 1806)) (a (.wire 93 71)) := by
  have c6369 := (publicBatchWrapper89_copies199 a h).2.1
  have c6370 := (publicBatchWrapper89_copies199 a h).2.2.1
  have c6371 := (publicBatchWrapper89_copies199 a h).2.2.2.1
  have c6372 := (publicBatchWrapper89_copies199 a h).2.2.2.2.1
  have c6373 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.1
  have c6374 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.1
  have e_94_12 := arithEq_of_rows h (row := 94) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_12
  simp only [← c6369, ← c6370, ← c6371] at e_94_12
  have e_94_13 := arithEq_of_rows h (row := 94) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_13
  simp only [← c6372, ← c6373, ← c6374] at e_94_13
  have hr := e_94_13
  simp only [e_94_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 63) = bselect (a (.wire 86 47)) (a (.virt 1807)) (a (.wire 93 79)) := by
  have c6375 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.1
  have c6376 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.1
  have c6377 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.1
  have c6378 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6379 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6380 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_14 := arithEq_of_rows h (row := 94) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_14
  simp only [← c6375, ← c6376, ← c6377] at e_94_14
  have e_94_15 := arithEq_of_rows h (row := 94) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_15
  simp only [← c6378, ← c6379, ← c6380] at e_94_15
  have hr := e_94_15
  simp only [e_94_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 71) = bselect (a (.wire 86 47)) (a (.virt 1808)) (a (.wire 94 7)) := by
  have c6381 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6382 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6383 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6384 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6385 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6386 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_16 := arithEq_of_rows h (row := 94) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_16
  simp only [← c6381, ← c6382, ← c6383] at e_94_16
  have e_94_17 := arithEq_of_rows h (row := 94) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_17
  simp only [← c6384, ← c6385, ← c6386] at e_94_17
  have hr := e_94_17
  simp only [e_94_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 94 79) = bselect (a (.wire 86 47)) (a (.virt 1809)) (a (.wire 94 15)) := by
  have c6387 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6388 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6389 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6390 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6391 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6392 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_18 := arithEq_of_rows h (row := 94) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_18
  simp only [← c6387, ← c6388, ← c6389] at e_94_18
  have e_94_19 := arithEq_of_rows h (row := 94) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_19
  simp only [← c6390, ← c6391, ← c6392] at e_94_19
  have hr := e_94_19
  simp only [e_94_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 7) = bselect (a (.wire 86 47)) (a (.virt 1803)) (a (.wire 94 23)) := by
  have c6393 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6394 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6395 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6396 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6397 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6398 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_0 := arithEq_of_rows h (row := 95) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_0
  simp only [← c6393, ← c6394, ← c6395] at e_95_0
  have e_95_1 := arithEq_of_rows h (row := 95) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_1
  simp only [← c6396, ← c6397, ← c6398] at e_95_1
  have hr := e_95_1
  simp only [e_95_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 15) = bselect (a (.wire 86 47)) (a (.virt 1804)) (a (.wire 94 31)) := by
  have c6399 := (publicBatchWrapper89_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6400 := (publicBatchWrapper89_copies200 a h).1
  have c6401 := (publicBatchWrapper89_copies200 a h).2.1
  have c6402 := (publicBatchWrapper89_copies200 a h).2.2.1
  have c6403 := (publicBatchWrapper89_copies200 a h).2.2.2.1
  have c6404 := (publicBatchWrapper89_copies200 a h).2.2.2.2.1
  have e_95_2 := arithEq_of_rows h (row := 95) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_2
  simp only [← c6399, ← c6400, ← c6401] at e_95_2
  have e_95_3 := arithEq_of_rows h (row := 95) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_3
  simp only [← c6402, ← c6403, ← c6404] at e_95_3
  have hr := e_95_3
  simp only [e_95_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 3) = bor (a (.wire 0 79)) (a (.wire 94 35)) := by
  have c6405 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.1
  have c6406 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.1
  have c6407 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.1
  have c6408 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.1
  have c6409 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.1
  have c6410 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_8 := arithEq_of_rows h (row := 88) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_8
  simp only [← c6405, ← c6406, ← c6407] at e_88_8
  have e_96_0 := arithEq_of_rows h (row := 96) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_0
  simp only [← c6408, ← c6409, k3, ← c6410] at e_96_0
  have hr := e_96_0
  simp only [e_88_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 19) = bnot (a (.wire 12 19)) := by
  have c6411 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6412 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6413 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_95_4 := arithEq_of_rows h (row := 95) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_4
  simp only [← c6411, k3, ← c6412, k3, ← c6413] at e_95_4
  have hr := e_95_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 23) = bnot (a (.wire 96 3)) := by
  have c6414 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6415 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6416 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_95_5 := arithEq_of_rows h (row := 95) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_5
  simp only [← c6414, k3, ← c6415, k3, ← c6416] at e_95_5
  have hr := e_95_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 51) = band (a (.wire 95 19)) (a (.wire 95 23)) := by
  have c6417 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6418 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6419 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_12 := arithEq_of_rows h (row := 86) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_12
  simp only [← c6417, ← c6418, ← c6419] at e_86_12
  have hr := e_86_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 31) = bselect (a (.wire 86 51)) (a (.virt 2005)) (a (.wire 94 47)) := by
  have c6420 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6421 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6422 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6423 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6424 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6425 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_6 := arithEq_of_rows h (row := 95) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_6
  simp only [← c6420, ← c6421, ← c6422] at e_95_6
  have e_95_7 := arithEq_of_rows h (row := 95) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_7
  simp only [← c6423, ← c6424, ← c6425] at e_95_7
  have hr := e_95_7
  simp only [e_95_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 39) = bselect (a (.wire 86 51)) (a (.virt 2006)) (a (.wire 94 55)) := by
  have c6426 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6427 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6428 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6429 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6430 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6431 := (publicBatchWrapper89_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_95_8 := arithEq_of_rows h (row := 95) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_8
  simp only [← c6426, ← c6427, ← c6428] at e_95_8
  have e_95_9 := arithEq_of_rows h (row := 95) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_9
  simp only [← c6429, ← c6430, ← c6431] at e_95_9
  have hr := e_95_9
  simp only [e_95_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 47) = bselect (a (.wire 86 51)) (a (.virt 2007)) (a (.wire 94 63)) := by
  have c6432 := (publicBatchWrapper89_copies201 a h).1
  have c6433 := (publicBatchWrapper89_copies201 a h).2.1
  have c6434 := (publicBatchWrapper89_copies201 a h).2.2.1
  have c6435 := (publicBatchWrapper89_copies201 a h).2.2.2.1
  have c6436 := (publicBatchWrapper89_copies201 a h).2.2.2.2.1
  have c6437 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.1
  have e_95_10 := arithEq_of_rows h (row := 95) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_10
  simp only [← c6432, ← c6433, ← c6434] at e_95_10
  have e_95_11 := arithEq_of_rows h (row := 95) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_11
  simp only [← c6435, ← c6436, ← c6437] at e_95_11
  have hr := e_95_11
  simp only [e_95_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 55) = bselect (a (.wire 86 51)) (a (.virt 2008)) (a (.wire 94 71)) := by
  have c6438 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.1
  have c6439 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.1
  have c6440 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.1
  have c6441 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.1
  have c6442 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6443 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_12 := arithEq_of_rows h (row := 95) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_12
  simp only [← c6438, ← c6439, ← c6440] at e_95_12
  have e_95_13 := arithEq_of_rows h (row := 95) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_13
  simp only [← c6441, ← c6442, ← c6443] at e_95_13
  have hr := e_95_13
  simp only [e_95_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 63) = bselect (a (.wire 86 51)) (a (.virt 2009)) (a (.wire 94 79)) := by
  have c6444 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6445 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6446 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6447 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6448 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6449 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_14 := arithEq_of_rows h (row := 95) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_14
  simp only [← c6444, ← c6445, ← c6446] at e_95_14
  have e_95_15 := arithEq_of_rows h (row := 95) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_15
  simp only [← c6447, ← c6448, ← c6449] at e_95_15
  have hr := e_95_15
  simp only [e_95_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 71) = bselect (a (.wire 86 51)) (a (.virt 2003)) (a (.wire 95 7)) := by
  have c6450 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6451 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6452 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6453 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6454 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6455 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_16 := arithEq_of_rows h (row := 95) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_16
  simp only [← c6450, ← c6451, ← c6452] at e_95_16
  have e_95_17 := arithEq_of_rows h (row := 95) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_17
  simp only [← c6453, ← c6454, ← c6455] at e_95_17
  have hr := e_95_17
  simp only [e_95_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 95 79) = bselect (a (.wire 86 51)) (a (.virt 2004)) (a (.wire 95 15)) := by
  have c6456 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6457 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6458 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6459 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6460 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6461 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_18 := arithEq_of_rows h (row := 95) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_18
  simp only [← c6456, ← c6457, ← c6458] at e_95_18
  have e_95_19 := arithEq_of_rows h (row := 95) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_19
  simp only [← c6459, ← c6460, ← c6461] at e_95_19
  have hr := e_95_19
  simp only [e_95_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 7) = bor (a (.wire 96 3)) (a (.wire 95 19)) := by
  have c6462 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6463 := (publicBatchWrapper89_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6464 := (publicBatchWrapper89_copies202 a h).1
  have c6465 := (publicBatchWrapper89_copies202 a h).2.1
  have c6466 := (publicBatchWrapper89_copies202 a h).2.2.1
  have c6467 := (publicBatchWrapper89_copies202 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_9 := arithEq_of_rows h (row := 88) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_9
  simp only [← c6462, ← c6463, ← c6464] at e_88_9
  have e_96_1 := arithEq_of_rows h (row := 96) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_1
  simp only [← c6465, ← c6466, k3, ← c6467] at e_96_1
  have hr := e_96_1
  simp only [e_88_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 3) = bnot (a (.wire 12 63)) := by
  have c6468 := (publicBatchWrapper89_copies202 a h).2.2.2.2.1
  have c6469 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.1
  have c6470 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_97_0 := arithEq_of_rows h (row := 97) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_0
  simp only [← c6468, k3, ← c6469, k3, ← c6470] at e_97_0
  have hr := e_97_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 7) = bnot (a (.wire 96 7)) := by
  have c6471 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.1
  have c6472 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.1
  have c6473 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_97_1 := arithEq_of_rows h (row := 97) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_1
  simp only [← c6471, k3, ← c6472, k3, ← c6473] at e_97_1
  have hr := e_97_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 55) = band (a (.wire 97 3)) (a (.wire 97 7)) := by
  have c6474 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6475 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6476 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_13 := arithEq_of_rows h (row := 86) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_13
  simp only [← c6474, ← c6475, ← c6476] at e_86_13
  have hr := e_86_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 15) = bselect (a (.wire 86 55)) (a (.virt 2205)) (a (.wire 95 31)) := by
  have c6477 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6478 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6479 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6480 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6481 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6482 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_2 := arithEq_of_rows h (row := 97) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_2
  simp only [← c6477, ← c6478, ← c6479] at e_97_2
  have e_97_3 := arithEq_of_rows h (row := 97) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_3
  simp only [← c6480, ← c6481, ← c6482] at e_97_3
  have hr := e_97_3
  simp only [e_97_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 23) = bselect (a (.wire 86 55)) (a (.virt 2206)) (a (.wire 95 39)) := by
  have c6483 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6484 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6485 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6486 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6487 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6488 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_4 := arithEq_of_rows h (row := 97) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_4
  simp only [← c6483, ← c6484, ← c6485] at e_97_4
  have e_97_5 := arithEq_of_rows h (row := 97) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_5
  simp only [← c6486, ← c6487, ← c6488] at e_97_5
  have hr := e_97_5
  simp only [e_97_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 31) = bselect (a (.wire 86 55)) (a (.virt 2207)) (a (.wire 95 47)) := by
  have c6489 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6490 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6491 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6492 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6493 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6494 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_6 := arithEq_of_rows h (row := 97) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_6
  simp only [← c6489, ← c6490, ← c6491] at e_97_6
  have e_97_7 := arithEq_of_rows h (row := 97) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_7
  simp only [← c6492, ← c6493, ← c6494] at e_97_7
  have hr := e_97_7
  simp only [e_97_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 39) = bselect (a (.wire 86 55)) (a (.virt 2208)) (a (.wire 95 55)) := by
  have c6495 := (publicBatchWrapper89_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6496 := (publicBatchWrapper89_copies203 a h).1
  have c6497 := (publicBatchWrapper89_copies203 a h).2.1
  have c6498 := (publicBatchWrapper89_copies203 a h).2.2.1
  have c6499 := (publicBatchWrapper89_copies203 a h).2.2.2.1
  have c6500 := (publicBatchWrapper89_copies203 a h).2.2.2.2.1
  have e_97_8 := arithEq_of_rows h (row := 97) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_8
  simp only [← c6495, ← c6496, ← c6497] at e_97_8
  have e_97_9 := arithEq_of_rows h (row := 97) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_9
  simp only [← c6498, ← c6499, ← c6500] at e_97_9
  have hr := e_97_9
  simp only [e_97_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 47) = bselect (a (.wire 86 55)) (a (.virt 2209)) (a (.wire 95 63)) := by
  have c6501 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.1
  have c6502 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.1
  have c6503 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.1
  have c6504 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.1
  have c6505 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.1
  have c6506 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_97_10 := arithEq_of_rows h (row := 97) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_10
  simp only [← c6501, ← c6502, ← c6503] at e_97_10
  have e_97_11 := arithEq_of_rows h (row := 97) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_11
  simp only [← c6504, ← c6505, ← c6506] at e_97_11
  have hr := e_97_11
  simp only [e_97_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 55) = bselect (a (.wire 86 55)) (a (.virt 2203)) (a (.wire 95 71)) := by
  have c6507 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6508 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6509 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6510 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6511 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6512 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_12 := arithEq_of_rows h (row := 97) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_12
  simp only [← c6507, ← c6508, ← c6509] at e_97_12
  have e_97_13 := arithEq_of_rows h (row := 97) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_13
  simp only [← c6510, ← c6511, ← c6512] at e_97_13
  have hr := e_97_13
  simp only [e_97_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 63) = bselect (a (.wire 86 55)) (a (.virt 2204)) (a (.wire 95 79)) := by
  have c6513 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6514 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6515 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6516 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6517 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6518 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_14 := arithEq_of_rows h (row := 97) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_14
  simp only [← c6513, ← c6514, ← c6515] at e_97_14
  have e_97_15 := arithEq_of_rows h (row := 97) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_15
  simp only [← c6516, ← c6517, ← c6518] at e_97_15
  have hr := e_97_15
  simp only [e_97_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 11) = bor (a (.wire 96 7)) (a (.wire 97 3)) := by
  have c6519 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6520 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6521 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6522 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6523 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6524 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_10 := arithEq_of_rows h (row := 88) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_10
  simp only [← c6519, ← c6520, ← c6521] at e_88_10
  have e_96_2 := arithEq_of_rows h (row := 96) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_2
  simp only [← c6522, ← c6523, k3, ← c6524] at e_96_2
  have hr := e_96_2
  simp only [e_88_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 67) = bnot (a (.wire 14 27)) := by
  have c6525 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6526 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6527 := (publicBatchWrapper89_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_97_16 := arithEq_of_rows h (row := 97) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_16
  simp only [← c6525, k3, ← c6526, k3, ← c6527] at e_97_16
  have hr := e_97_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 71) = bnot (a (.wire 96 11)) := by
  have c6528 := (publicBatchWrapper89_copies204 a h).1
  have c6529 := (publicBatchWrapper89_copies204 a h).2.1
  have c6530 := (publicBatchWrapper89_copies204 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_97_17 := arithEq_of_rows h (row := 97) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_17
  simp only [← c6528, k3, ← c6529, k3, ← c6530] at e_97_17
  have hr := e_97_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 59) = band (a (.wire 97 67)) (a (.wire 97 71)) := by
  have c6531 := (publicBatchWrapper89_copies204 a h).2.2.2.1
  have c6532 := (publicBatchWrapper89_copies204 a h).2.2.2.2.1
  have c6533 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.1
  have e_86_14 := arithEq_of_rows h (row := 86) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_14
  simp only [← c6531, ← c6532, ← c6533] at e_86_14
  have hr := e_86_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 97 79) = bselect (a (.wire 86 59)) (a (.virt 2405)) (a (.wire 97 15)) := by
  have c6534 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.1
  have c6535 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.1
  have c6536 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.1
  have c6537 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.1
  have c6538 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6539 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_97_18 := arithEq_of_rows h (row := 97) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_18
  simp only [← c6534, ← c6535, ← c6536] at e_97_18
  have e_97_19 := arithEq_of_rows h (row := 97) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_19
  simp only [← c6537, ← c6538, ← c6539] at e_97_19
  have hr := e_97_19
  simp only [e_97_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 7) = bselect (a (.wire 86 59)) (a (.virt 2406)) (a (.wire 97 23)) := by
  have c6540 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6541 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6542 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6543 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6544 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6545 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_0 := arithEq_of_rows h (row := 98) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_0
  simp only [← c6540, ← c6541, ← c6542] at e_98_0
  have e_98_1 := arithEq_of_rows h (row := 98) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_1
  simp only [← c6543, ← c6544, ← c6545] at e_98_1
  have hr := e_98_1
  simp only [e_98_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 15) = bselect (a (.wire 86 59)) (a (.virt 2407)) (a (.wire 97 31)) := by
  have c6546 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6547 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6548 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6549 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6550 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6551 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_2 := arithEq_of_rows h (row := 98) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_2
  simp only [← c6546, ← c6547, ← c6548] at e_98_2
  have e_98_3 := arithEq_of_rows h (row := 98) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_3
  simp only [← c6549, ← c6550, ← c6551] at e_98_3
  have hr := e_98_3
  simp only [e_98_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 23) = bselect (a (.wire 86 59)) (a (.virt 2408)) (a (.wire 97 39)) := by
  have c6552 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6553 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6554 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6555 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6556 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6557 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_4 := arithEq_of_rows h (row := 98) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_4
  simp only [← c6552, ← c6553, ← c6554] at e_98_4
  have e_98_5 := arithEq_of_rows h (row := 98) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_5
  simp only [← c6555, ← c6556, ← c6557] at e_98_5
  have hr := e_98_5
  simp only [e_98_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 31) = bselect (a (.wire 86 59)) (a (.virt 2409)) (a (.wire 97 47)) := by
  have c6558 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6559 := (publicBatchWrapper89_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6560 := (publicBatchWrapper89_copies205 a h).1
  have c6561 := (publicBatchWrapper89_copies205 a h).2.1
  have c6562 := (publicBatchWrapper89_copies205 a h).2.2.1
  have c6563 := (publicBatchWrapper89_copies205 a h).2.2.2.1
  have e_98_6 := arithEq_of_rows h (row := 98) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_6
  simp only [← c6558, ← c6559, ← c6560] at e_98_6
  have e_98_7 := arithEq_of_rows h (row := 98) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_7
  simp only [← c6561, ← c6562, ← c6563] at e_98_7
  have hr := e_98_7
  simp only [e_98_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 39) = bselect (a (.wire 86 59)) (a (.virt 2403)) (a (.wire 97 55)) := by
  have c6564 := (publicBatchWrapper89_copies205 a h).2.2.2.2.1
  have c6565 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.1
  have c6566 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.1
  have c6567 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.1
  have c6568 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.1
  have c6569 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.1
  have e_98_8 := arithEq_of_rows h (row := 98) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_8
  simp only [← c6564, ← c6565, ← c6566] at e_98_8
  have e_98_9 := arithEq_of_rows h (row := 98) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_9
  simp only [← c6567, ← c6568, ← c6569] at e_98_9
  have hr := e_98_9
  simp only [e_98_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 47) = bselect (a (.wire 86 59)) (a (.virt 2404)) (a (.wire 97 63)) := by
  have c6570 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6571 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6572 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6573 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6574 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6575 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_10 := arithEq_of_rows h (row := 98) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_10
  simp only [← c6570, ← c6571, ← c6572] at e_98_10
  have e_98_11 := arithEq_of_rows h (row := 98) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_11
  simp only [← c6573, ← c6574, ← c6575] at e_98_11
  have hr := e_98_11
  simp only [e_98_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 15) = bor (a (.wire 96 11)) (a (.wire 97 67)) := by
  have c6576 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6577 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6578 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6579 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6580 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6581 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_11 := arithEq_of_rows h (row := 88) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_11
  simp only [← c6576, ← c6577, ← c6578] at e_88_11
  have e_96_3 := arithEq_of_rows h (row := 96) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_3
  simp only [← c6579, ← c6580, k3, ← c6581] at e_96_3
  have hr := e_96_3
  simp only [e_88_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 51) = bnot (a (.wire 14 71)) := by
  have c6582 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6583 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6584 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_98_12 := arithEq_of_rows h (row := 98) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_12
  simp only [← c6582, k3, ← c6583, k3, ← c6584] at e_98_12
  have hr := e_98_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 55) = bnot (a (.wire 96 15)) := by
  have c6585 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6586 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6587 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_98_13 := arithEq_of_rows h (row := 98) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_13
  simp only [← c6585, k3, ← c6586, k3, ← c6587] at e_98_13
  have hr := e_98_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 63) = band (a (.wire 98 51)) (a (.wire 98 55)) := by
  have c6588 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6589 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6590 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_15 := arithEq_of_rows h (row := 86) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_15
  simp only [← c6588, ← c6589, ← c6590] at e_86_15
  have hr := e_86_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 63) = bselect (a (.wire 86 63)) (a (.virt 2605)) (a (.wire 97 79)) := by
  have c6591 := (publicBatchWrapper89_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6592 := (publicBatchWrapper89_copies206 a h).1
  have c6593 := (publicBatchWrapper89_copies206 a h).2.1
  have c6594 := (publicBatchWrapper89_copies206 a h).2.2.1
  have c6595 := (publicBatchWrapper89_copies206 a h).2.2.2.1
  have c6596 := (publicBatchWrapper89_copies206 a h).2.2.2.2.1
  have e_98_14 := arithEq_of_rows h (row := 98) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_14
  simp only [← c6591, ← c6592, ← c6593] at e_98_14
  have e_98_15 := arithEq_of_rows h (row := 98) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_15
  simp only [← c6594, ← c6595, ← c6596] at e_98_15
  have hr := e_98_15
  simp only [e_98_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 71) = bselect (a (.wire 86 63)) (a (.virt 2606)) (a (.wire 98 7)) := by
  have c6597 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.1
  have c6598 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.1
  have c6599 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.1
  have c6600 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.1
  have c6601 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.1
  have c6602 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_98_16 := arithEq_of_rows h (row := 98) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_16
  simp only [← c6597, ← c6598, ← c6599] at e_98_16
  have e_98_17 := arithEq_of_rows h (row := 98) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_17
  simp only [← c6600, ← c6601, ← c6602] at e_98_17
  have hr := e_98_17
  simp only [e_98_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 98 79) = bselect (a (.wire 86 63)) (a (.virt 2607)) (a (.wire 98 15)) := by
  have c6603 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6604 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6605 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6606 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6607 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6608 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_18 := arithEq_of_rows h (row := 98) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_18
  simp only [← c6603, ← c6604, ← c6605] at e_98_18
  have e_98_19 := arithEq_of_rows h (row := 98) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_19
  simp only [← c6606, ← c6607, ← c6608] at e_98_19
  have hr := e_98_19
  simp only [e_98_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 7) = bselect (a (.wire 86 63)) (a (.virt 2608)) (a (.wire 98 23)) := by
  have c6609 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6610 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6611 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6612 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6613 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6614 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_0 := arithEq_of_rows h (row := 99) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_0
  simp only [← c6609, ← c6610, ← c6611] at e_99_0
  have e_99_1 := arithEq_of_rows h (row := 99) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_1
  simp only [← c6612, ← c6613, ← c6614] at e_99_1
  have hr := e_99_1
  simp only [e_99_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 15) = bselect (a (.wire 86 63)) (a (.virt 2609)) (a (.wire 98 31)) := by
  have c6615 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6616 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6617 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6618 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6619 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6620 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_2 := arithEq_of_rows h (row := 99) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_2
  simp only [← c6615, ← c6616, ← c6617] at e_99_2
  have e_99_3 := arithEq_of_rows h (row := 99) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_3
  simp only [← c6618, ← c6619, ← c6620] at e_99_3
  have hr := e_99_3
  simp only [e_99_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 23) = bselect (a (.wire 86 63)) (a (.virt 2603)) (a (.wire 98 39)) := by
  have c6621 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6622 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6623 := (publicBatchWrapper89_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6624 := (publicBatchWrapper89_copies207 a h).1
  have c6625 := (publicBatchWrapper89_copies207 a h).2.1
  have c6626 := (publicBatchWrapper89_copies207 a h).2.2.1
  have e_99_4 := arithEq_of_rows h (row := 99) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_4
  simp only [← c6621, ← c6622, ← c6623] at e_99_4
  have e_99_5 := arithEq_of_rows h (row := 99) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_5
  simp only [← c6624, ← c6625, ← c6626] at e_99_5
  have hr := e_99_5
  simp only [e_99_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 31) = bselect (a (.wire 86 63)) (a (.virt 2604)) (a (.wire 98 47)) := by
  have c6627 := (publicBatchWrapper89_copies207 a h).2.2.2.1
  have c6628 := (publicBatchWrapper89_copies207 a h).2.2.2.2.1
  have c6629 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.1
  have c6630 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.1
  have c6631 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.1
  have c6632 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.1
  have e_99_6 := arithEq_of_rows h (row := 99) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_6
  simp only [← c6627, ← c6628, ← c6629] at e_99_6
  have e_99_7 := arithEq_of_rows h (row := 99) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_7
  simp only [← c6630, ← c6631, ← c6632] at e_99_7
  have hr := e_99_7
  simp only [e_99_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 19) = bor (a (.wire 96 15)) (a (.wire 98 51)) := by
  have c6633 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.1
  have c6634 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6635 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6636 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6637 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6638 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_12 := arithEq_of_rows h (row := 88) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_12
  simp only [← c6633, ← c6634, ← c6635] at e_88_12
  have e_96_4 := arithEq_of_rows h (row := 96) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_4
  simp only [← c6636, ← c6637, k3, ← c6638] at e_96_4
  have hr := e_96_4
  simp only [e_88_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 35) = bnot (a (.wire 15 35)) := by
  have c6639 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6640 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6641 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_99_8 := arithEq_of_rows h (row := 99) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_8
  simp only [← c6639, k3, ← c6640, k3, ← c6641] at e_99_8
  have hr := e_99_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 39) = bnot (a (.wire 96 19)) := by
  have c6642 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6643 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6644 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_99_9 := arithEq_of_rows h (row := 99) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_9
  simp only [← c6642, k3, ← c6643, k3, ← c6644] at e_99_9
  have hr := e_99_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 67) = band (a (.wire 99 35)) (a (.wire 99 39)) := by
  have c6645 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6646 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6647 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_16 := arithEq_of_rows h (row := 86) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_16
  simp only [← c6645, ← c6646, ← c6647] at e_86_16
  have hr := e_86_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 47) = bselect (a (.wire 86 67)) (a (.virt 2805)) (a (.wire 98 63)) := by
  have c6648 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6649 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6650 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6651 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6652 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6653 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_10 := arithEq_of_rows h (row := 99) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_10
  simp only [← c6648, ← c6649, ← c6650] at e_99_10
  have e_99_11 := arithEq_of_rows h (row := 99) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_11
  simp only [← c6651, ← c6652, ← c6653] at e_99_11
  have hr := e_99_11
  simp only [e_99_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 55) = bselect (a (.wire 86 67)) (a (.virt 2806)) (a (.wire 98 71)) := by
  have c6654 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6655 := (publicBatchWrapper89_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6656 := (publicBatchWrapper89_copies208 a h).1
  have c6657 := (publicBatchWrapper89_copies208 a h).2.1
  have c6658 := (publicBatchWrapper89_copies208 a h).2.2.1
  have c6659 := (publicBatchWrapper89_copies208 a h).2.2.2.1
  have e_99_12 := arithEq_of_rows h (row := 99) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_12
  simp only [← c6654, ← c6655, ← c6656] at e_99_12
  have e_99_13 := arithEq_of_rows h (row := 99) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_13
  simp only [← c6657, ← c6658, ← c6659] at e_99_13
  have hr := e_99_13
  simp only [e_99_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 63) = bselect (a (.wire 86 67)) (a (.virt 2807)) (a (.wire 98 79)) := by
  have c6660 := (publicBatchWrapper89_copies208 a h).2.2.2.2.1
  have c6661 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.1
  have c6662 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.1
  have c6663 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.1
  have c6664 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.1
  have c6665 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.1
  have e_99_14 := arithEq_of_rows h (row := 99) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_14
  simp only [← c6660, ← c6661, ← c6662] at e_99_14
  have e_99_15 := arithEq_of_rows h (row := 99) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_15
  simp only [← c6663, ← c6664, ← c6665] at e_99_15
  have hr := e_99_15
  simp only [e_99_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 71) = bselect (a (.wire 86 67)) (a (.virt 2808)) (a (.wire 99 7)) := by
  have c6666 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6667 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6668 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6669 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6670 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6671 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_16 := arithEq_of_rows h (row := 99) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_16
  simp only [← c6666, ← c6667, ← c6668] at e_99_16
  have e_99_17 := arithEq_of_rows h (row := 99) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_17
  simp only [← c6669, ← c6670, ← c6671] at e_99_17
  have hr := e_99_17
  simp only [e_99_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 99 79) = bselect (a (.wire 86 67)) (a (.virt 2809)) (a (.wire 99 15)) := by
  have c6672 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6673 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6674 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6675 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6676 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6677 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_18 := arithEq_of_rows h (row := 99) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_18
  simp only [← c6672, ← c6673, ← c6674] at e_99_18
  have e_99_19 := arithEq_of_rows h (row := 99) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_19
  simp only [← c6675, ← c6676, ← c6677] at e_99_19
  have hr := e_99_19
  simp only [e_99_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 7) = bselect (a (.wire 86 67)) (a (.virt 2803)) (a (.wire 99 23)) := by
  have c6678 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6679 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6680 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6681 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6682 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6683 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_0 := arithEq_of_rows h (row := 100) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_0
  simp only [← c6678, ← c6679, ← c6680] at e_100_0
  have e_100_1 := arithEq_of_rows h (row := 100) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_1
  simp only [← c6681, ← c6682, ← c6683] at e_100_1
  have hr := e_100_1
  simp only [e_100_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 15) = bselect (a (.wire 86 67)) (a (.virt 2804)) (a (.wire 99 31)) := by
  have c6684 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6685 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6686 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6687 := (publicBatchWrapper89_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6688 := (publicBatchWrapper89_copies209 a h).1
  have c6689 := (publicBatchWrapper89_copies209 a h).2.1
  have e_100_2 := arithEq_of_rows h (row := 100) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_2
  simp only [← c6684, ← c6685, ← c6686] at e_100_2
  have e_100_3 := arithEq_of_rows h (row := 100) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_3
  simp only [← c6687, ← c6688, ← c6689] at e_100_3
  have hr := e_100_3
  simp only [e_100_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 23) = bor (a (.wire 96 19)) (a (.wire 99 35)) := by
  have c6690 := (publicBatchWrapper89_copies209 a h).2.2.1
  have c6691 := (publicBatchWrapper89_copies209 a h).2.2.2.1
  have c6692 := (publicBatchWrapper89_copies209 a h).2.2.2.2.1
  have c6693 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.1
  have c6694 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.1
  have c6695 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_13 := arithEq_of_rows h (row := 88) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_13
  simp only [← c6690, ← c6691, ← c6692] at e_88_13
  have e_96_5 := arithEq_of_rows h (row := 96) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_5
  simp only [← c6693, ← c6694, k3, ← c6695] at e_96_5
  have hr := e_96_5
  simp only [e_88_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 19) = bnot (a (.wire 15 79)) := by
  have c6696 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.1
  have c6697 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.1
  have c6698 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_100_4 := arithEq_of_rows h (row := 100) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_4
  simp only [← c6696, k3, ← c6697, k3, ← c6698] at e_100_4
  have hr := e_100_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 23) = bnot (a (.wire 96 23)) := by
  have c6699 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6700 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6701 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_100_5 := arithEq_of_rows h (row := 100) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_5
  simp only [← c6699, k3, ← c6700, k3, ← c6701] at e_100_5
  have hr := e_100_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 71) = band (a (.wire 100 19)) (a (.wire 100 23)) := by
  have c6702 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6703 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6704 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_86_17 := arithEq_of_rows h (row := 86) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_17
  simp only [← c6702, ← c6703, ← c6704] at e_86_17
  have hr := e_86_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 31) = bselect (a (.wire 86 71)) (a (.virt 3005)) (a (.wire 99 47)) := by
  have c6705 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6706 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6707 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6708 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6709 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6710 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_6 := arithEq_of_rows h (row := 100) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_6
  simp only [← c6705, ← c6706, ← c6707] at e_100_6
  have e_100_7 := arithEq_of_rows h (row := 100) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_7
  simp only [← c6708, ← c6709, ← c6710] at e_100_7
  have hr := e_100_7
  simp only [e_100_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 39) = bselect (a (.wire 86 71)) (a (.virt 3006)) (a (.wire 99 55)) := by
  have c6711 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6712 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6713 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6714 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6715 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6716 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_8 := arithEq_of_rows h (row := 100) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_8
  simp only [← c6711, ← c6712, ← c6713] at e_100_8
  have e_100_9 := arithEq_of_rows h (row := 100) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_9
  simp only [← c6714, ← c6715, ← c6716] at e_100_9
  have hr := e_100_9
  simp only [e_100_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 47) = bselect (a (.wire 86 71)) (a (.virt 3007)) (a (.wire 99 63)) := by
  have c6717 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6718 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6719 := (publicBatchWrapper89_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6720 := (publicBatchWrapper89_copies210 a h).1
  have c6721 := (publicBatchWrapper89_copies210 a h).2.1
  have c6722 := (publicBatchWrapper89_copies210 a h).2.2.1
  have e_100_10 := arithEq_of_rows h (row := 100) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_10
  simp only [← c6717, ← c6718, ← c6719] at e_100_10
  have e_100_11 := arithEq_of_rows h (row := 100) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_11
  simp only [← c6720, ← c6721, ← c6722] at e_100_11
  have hr := e_100_11
  simp only [e_100_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 55) = bselect (a (.wire 86 71)) (a (.virt 3008)) (a (.wire 99 71)) := by
  have c6723 := (publicBatchWrapper89_copies210 a h).2.2.2.1
  have c6724 := (publicBatchWrapper89_copies210 a h).2.2.2.2.1
  have c6725 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.1
  have c6726 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.1
  have c6727 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.1
  have c6728 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.1
  have e_100_12 := arithEq_of_rows h (row := 100) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_12
  simp only [← c6723, ← c6724, ← c6725] at e_100_12
  have e_100_13 := arithEq_of_rows h (row := 100) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_13
  simp only [← c6726, ← c6727, ← c6728] at e_100_13
  have hr := e_100_13
  simp only [e_100_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 63) = bselect (a (.wire 86 71)) (a (.virt 3009)) (a (.wire 99 79)) := by
  have c6729 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.1
  have c6730 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6731 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6732 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6733 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6734 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_14 := arithEq_of_rows h (row := 100) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_14
  simp only [← c6729, ← c6730, ← c6731] at e_100_14
  have e_100_15 := arithEq_of_rows h (row := 100) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_15
  simp only [← c6732, ← c6733, ← c6734] at e_100_15
  have hr := e_100_15
  simp only [e_100_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 71) = bselect (a (.wire 86 71)) (a (.virt 3003)) (a (.wire 100 7)) := by
  have c6735 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6736 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6737 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6738 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6739 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6740 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_16 := arithEq_of_rows h (row := 100) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_16
  simp only [← c6735, ← c6736, ← c6737] at e_100_16
  have e_100_17 := arithEq_of_rows h (row := 100) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_17
  simp only [← c6738, ← c6739, ← c6740] at e_100_17
  have hr := e_100_17
  simp only [e_100_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 100 79) = bselect (a (.wire 86 71)) (a (.virt 3004)) (a (.wire 100 15)) := by
  have c6741 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6742 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6743 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6744 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6745 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6746 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_18 := arithEq_of_rows h (row := 100) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_18
  simp only [← c6741, ← c6742, ← c6743] at e_100_18
  have e_100_19 := arithEq_of_rows h (row := 100) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_19
  simp only [← c6744, ← c6745, ← c6746] at e_100_19
  have hr := e_100_19
  simp only [e_100_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 27) = bor (a (.wire 96 23)) (a (.wire 100 19)) := by
  have c6747 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6748 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6749 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6750 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6751 := (publicBatchWrapper89_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6752 := (publicBatchWrapper89_copies211 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_14 := arithEq_of_rows h (row := 88) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_14
  simp only [← c6747, ← c6748, ← c6749] at e_88_14
  have e_96_6 := arithEq_of_rows h (row := 96) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_6
  simp only [← c6750, ← c6751, k3, ← c6752] at e_96_6
  have hr := e_96_6
  simp only [e_88_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 3) = bnot (a (.wire 17 43)) := by
  have c6753 := (publicBatchWrapper89_copies211 a h).2.1
  have c6754 := (publicBatchWrapper89_copies211 a h).2.2.1
  have c6755 := (publicBatchWrapper89_copies211 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_101_0 := arithEq_of_rows h (row := 101) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_0
  simp only [← c6753, k3, ← c6754, k3, ← c6755] at e_101_0
  have hr := e_101_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 7) = bnot (a (.wire 96 27)) := by
  have c6756 := (publicBatchWrapper89_copies211 a h).2.2.2.2.1
  have c6757 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.1
  have c6758 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_101_1 := arithEq_of_rows h (row := 101) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_1
  simp only [← c6756, k3, ← c6757, k3, ← c6758] at e_101_1
  have hr := e_101_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 75) = band (a (.wire 101 3)) (a (.wire 101 7)) := by
  have c6759 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.1
  have c6760 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.1
  have c6761 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.1
  have e_86_18 := arithEq_of_rows h (row := 86) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_18
  simp only [← c6759, ← c6760, ← c6761] at e_86_18
  have hr := e_86_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 15) = bselect (a (.wire 86 75)) (a (.virt 3205)) (a (.wire 100 31)) := by
  have c6762 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6763 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6764 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6765 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6766 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6767 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_2 := arithEq_of_rows h (row := 101) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_2
  simp only [← c6762, ← c6763, ← c6764] at e_101_2
  have e_101_3 := arithEq_of_rows h (row := 101) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_3
  simp only [← c6765, ← c6766, ← c6767] at e_101_3
  have hr := e_101_3
  simp only [e_101_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 23) = bselect (a (.wire 86 75)) (a (.virt 3206)) (a (.wire 100 39)) := by
  have c6768 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6769 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6770 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6771 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6772 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6773 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_4 := arithEq_of_rows h (row := 101) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_4
  simp only [← c6768, ← c6769, ← c6770] at e_101_4
  have e_101_5 := arithEq_of_rows h (row := 101) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_5
  simp only [← c6771, ← c6772, ← c6773] at e_101_5
  have hr := e_101_5
  simp only [e_101_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 31) = bselect (a (.wire 86 75)) (a (.virt 3207)) (a (.wire 100 47)) := by
  have c6774 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6775 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6776 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6777 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6778 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6779 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_6 := arithEq_of_rows h (row := 101) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_6
  simp only [← c6774, ← c6775, ← c6776] at e_101_6
  have e_101_7 := arithEq_of_rows h (row := 101) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_7
  simp only [← c6777, ← c6778, ← c6779] at e_101_7
  have hr := e_101_7
  simp only [e_101_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 39) = bselect (a (.wire 86 75)) (a (.virt 3208)) (a (.wire 100 55)) := by
  have c6780 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6781 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6782 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6783 := (publicBatchWrapper89_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6784 := (publicBatchWrapper89_copies212 a h).1
  have c6785 := (publicBatchWrapper89_copies212 a h).2.1
  have e_101_8 := arithEq_of_rows h (row := 101) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_8
  simp only [← c6780, ← c6781, ← c6782] at e_101_8
  have e_101_9 := arithEq_of_rows h (row := 101) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_9
  simp only [← c6783, ← c6784, ← c6785] at e_101_9
  have hr := e_101_9
  simp only [e_101_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 47) = bselect (a (.wire 86 75)) (a (.virt 3209)) (a (.wire 100 63)) := by
  have c6786 := (publicBatchWrapper89_copies212 a h).2.2.1
  have c6787 := (publicBatchWrapper89_copies212 a h).2.2.2.1
  have c6788 := (publicBatchWrapper89_copies212 a h).2.2.2.2.1
  have c6789 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.1
  have c6790 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.1
  have c6791 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.1
  have e_101_10 := arithEq_of_rows h (row := 101) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_10
  simp only [← c6786, ← c6787, ← c6788] at e_101_10
  have e_101_11 := arithEq_of_rows h (row := 101) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_11
  simp only [← c6789, ← c6790, ← c6791] at e_101_11
  have hr := e_101_11
  simp only [e_101_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 55) = bselect (a (.wire 86 75)) (a (.virt 3203)) (a (.wire 100 71)) := by
  have c6792 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.1
  have c6793 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.1
  have c6794 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6795 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6796 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6797 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_12 := arithEq_of_rows h (row := 101) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_12
  simp only [← c6792, ← c6793, ← c6794] at e_101_12
  have e_101_13 := arithEq_of_rows h (row := 101) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_13
  simp only [← c6795, ← c6796, ← c6797] at e_101_13
  have hr := e_101_13
  simp only [e_101_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 63) = bselect (a (.wire 86 75)) (a (.virt 3204)) (a (.wire 100 79)) := by
  have c6798 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6799 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6800 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6801 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6802 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6803 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_14 := arithEq_of_rows h (row := 101) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_14
  simp only [← c6798, ← c6799, ← c6800] at e_101_14
  have e_101_15 := arithEq_of_rows h (row := 101) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_15
  simp only [← c6801, ← c6802, ← c6803] at e_101_15
  have hr := e_101_15
  simp only [e_101_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 31) = bor (a (.wire 96 27)) (a (.wire 101 3)) := by
  have c6804 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6805 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6806 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6807 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6808 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6809 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_15 := arithEq_of_rows h (row := 88) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_15
  simp only [← c6804, ← c6805, ← c6806] at e_88_15
  have e_96_7 := arithEq_of_rows h (row := 96) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_7
  simp only [← c6807, ← c6808, k3, ← c6809] at e_96_7
  have hr := e_96_7
  simp only [e_88_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 67) = bnot (a (.wire 19 7)) := by
  have c6810 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6811 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6812 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_101_16 := arithEq_of_rows h (row := 101) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_16
  simp only [← c6810, k3, ← c6811, k3, ← c6812] at e_101_16
  have hr := e_101_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 71) = bnot (a (.wire 96 31)) := by
  have c6813 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6814 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6815 := (publicBatchWrapper89_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_101_17 := arithEq_of_rows h (row := 101) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_17
  simp only [← c6813, k3, ← c6814, k3, ← c6815] at e_101_17
  have hr := e_101_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 86 79) = band (a (.wire 101 67)) (a (.wire 101 71)) := by
  have c6816 := (publicBatchWrapper89_copies213 a h).1
  have c6817 := (publicBatchWrapper89_copies213 a h).2.1
  have c6818 := (publicBatchWrapper89_copies213 a h).2.2.1
  have e_86_19 := arithEq_of_rows h (row := 86) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_19
  simp only [← c6816, ← c6817, ← c6818] at e_86_19
  have hr := e_86_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 101 79) = bselect (a (.wire 86 79)) (a (.virt 3405)) (a (.wire 101 15)) := by
  have c6819 := (publicBatchWrapper89_copies213 a h).2.2.2.1
  have c6820 := (publicBatchWrapper89_copies213 a h).2.2.2.2.1
  have c6821 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.1
  have c6822 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.1
  have c6823 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.1
  have c6824 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.1
  have e_101_18 := arithEq_of_rows h (row := 101) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_18
  simp only [← c6819, ← c6820, ← c6821] at e_101_18
  have e_101_19 := arithEq_of_rows h (row := 101) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_19
  simp only [← c6822, ← c6823, ← c6824] at e_101_19
  have hr := e_101_19
  simp only [e_101_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 7) = bselect (a (.wire 86 79)) (a (.virt 3406)) (a (.wire 101 23)) := by
  have c6825 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.1
  have c6826 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6827 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6828 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6829 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6830 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_0 := arithEq_of_rows h (row := 102) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_0
  simp only [← c6825, ← c6826, ← c6827] at e_102_0
  have e_102_1 := arithEq_of_rows h (row := 102) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_1
  simp only [← c6828, ← c6829, ← c6830] at e_102_1
  have hr := e_102_1
  simp only [e_102_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 15) = bselect (a (.wire 86 79)) (a (.virt 3407)) (a (.wire 101 31)) := by
  have c6831 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6832 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6833 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6834 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6835 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6836 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_2 := arithEq_of_rows h (row := 102) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_2
  simp only [← c6831, ← c6832, ← c6833] at e_102_2
  have e_102_3 := arithEq_of_rows h (row := 102) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_3
  simp only [← c6834, ← c6835, ← c6836] at e_102_3
  have hr := e_102_3
  simp only [e_102_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 23) = bselect (a (.wire 86 79)) (a (.virt 3408)) (a (.wire 101 39)) := by
  have c6837 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6838 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6839 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6840 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6841 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6842 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_4 := arithEq_of_rows h (row := 102) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_4
  simp only [← c6837, ← c6838, ← c6839] at e_102_4
  have e_102_5 := arithEq_of_rows h (row := 102) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_5
  simp only [← c6840, ← c6841, ← c6842] at e_102_5
  have hr := e_102_5
  simp only [e_102_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 31) = bselect (a (.wire 86 79)) (a (.virt 3409)) (a (.wire 101 47)) := by
  have c6843 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6844 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6845 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6846 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6847 := (publicBatchWrapper89_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6848 := (publicBatchWrapper89_copies214 a h).1
  have e_102_6 := arithEq_of_rows h (row := 102) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_6
  simp only [← c6843, ← c6844, ← c6845] at e_102_6
  have e_102_7 := arithEq_of_rows h (row := 102) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_7
  simp only [← c6846, ← c6847, ← c6848] at e_102_7
  have hr := e_102_7
  simp only [e_102_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 39) = bselect (a (.wire 86 79)) (a (.virt 3403)) (a (.wire 101 55)) := by
  have c6849 := (publicBatchWrapper89_copies214 a h).2.1
  have c6850 := (publicBatchWrapper89_copies214 a h).2.2.1
  have c6851 := (publicBatchWrapper89_copies214 a h).2.2.2.1
  have c6852 := (publicBatchWrapper89_copies214 a h).2.2.2.2.1
  have c6853 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.1
  have c6854 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.1
  have e_102_8 := arithEq_of_rows h (row := 102) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_8
  simp only [← c6849, ← c6850, ← c6851] at e_102_8
  have e_102_9 := arithEq_of_rows h (row := 102) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_9
  simp only [← c6852, ← c6853, ← c6854] at e_102_9
  have hr := e_102_9
  simp only [e_102_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 47) = bselect (a (.wire 86 79)) (a (.virt 3404)) (a (.wire 101 63)) := by
  have c6855 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.1
  have c6856 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.1
  have c6857 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.1
  have c6858 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6859 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6860 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_10 := arithEq_of_rows h (row := 102) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_10
  simp only [← c6855, ← c6856, ← c6857] at e_102_10
  have e_102_11 := arithEq_of_rows h (row := 102) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_11
  simp only [← c6858, ← c6859, ← c6860] at e_102_11
  have hr := e_102_11
  simp only [e_102_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 35) = bor (a (.wire 96 31)) (a (.wire 101 67)) := by
  have c6861 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6862 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6863 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6864 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6865 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6866 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_16 := arithEq_of_rows h (row := 88) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_16
  simp only [← c6861, ← c6862, ← c6863] at e_88_16
  have e_96_8 := arithEq_of_rows h (row := 96) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_8
  simp only [← c6864, ← c6865, k3, ← c6866] at e_96_8
  have hr := e_96_8
  simp only [e_88_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 51) = bnot (a (.wire 19 51)) := by
  have c6867 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6868 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6869 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_102_12 := arithEq_of_rows h (row := 102) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_12
  simp only [← c6867, k3, ← c6868, k3, ← c6869] at e_102_12
  have hr := e_102_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 55) = bnot (a (.wire 96 35)) := by
  have c6870 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6871 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6872 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_102_13 := arithEq_of_rows h (row := 102) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_13
  simp only [← c6870, k3, ← c6871, k3, ← c6872] at e_102_13
  have hr := e_102_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 3) = band (a (.wire 102 51)) (a (.wire 102 55)) := by
  have c6873 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6874 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6875 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_0 := arithEq_of_rows h (row := 103) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_0
  simp only [← c6873, ← c6874, ← c6875] at e_103_0
  have hr := e_103_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 63) = bselect (a (.wire 103 3)) (a (.virt 3605)) (a (.wire 101 79)) := by
  have c6876 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6877 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6878 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6879 := (publicBatchWrapper89_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6880 := (publicBatchWrapper89_copies215 a h).1
  have c6881 := (publicBatchWrapper89_copies215 a h).2.1
  have e_102_14 := arithEq_of_rows h (row := 102) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_14
  simp only [← c6876, ← c6877, ← c6878] at e_102_14
  have e_102_15 := arithEq_of_rows h (row := 102) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_15
  simp only [← c6879, ← c6880, ← c6881] at e_102_15
  have hr := e_102_15
  simp only [e_102_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 71) = bselect (a (.wire 103 3)) (a (.virt 3606)) (a (.wire 102 7)) := by
  have c6882 := (publicBatchWrapper89_copies215 a h).2.2.1
  have c6883 := (publicBatchWrapper89_copies215 a h).2.2.2.1
  have c6884 := (publicBatchWrapper89_copies215 a h).2.2.2.2.1
  have c6885 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.1
  have c6886 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.1
  have c6887 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.1
  have e_102_16 := arithEq_of_rows h (row := 102) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_16
  simp only [← c6882, ← c6883, ← c6884] at e_102_16
  have e_102_17 := arithEq_of_rows h (row := 102) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_17
  simp only [← c6885, ← c6886, ← c6887] at e_102_17
  have hr := e_102_17
  simp only [e_102_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 102 79) = bselect (a (.wire 103 3)) (a (.virt 3607)) (a (.wire 102 15)) := by
  have c6888 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.1
  have c6889 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.1
  have c6890 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6891 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6892 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6893 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_18 := arithEq_of_rows h (row := 102) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_18
  simp only [← c6888, ← c6889, ← c6890] at e_102_18
  have e_102_19 := arithEq_of_rows h (row := 102) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_19
  simp only [← c6891, ← c6892, ← c6893] at e_102_19
  have hr := e_102_19
  simp only [e_102_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 7) = bselect (a (.wire 103 3)) (a (.virt 3608)) (a (.wire 102 23)) := by
  have c6894 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6895 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6896 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6897 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6898 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6899 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_0 := arithEq_of_rows h (row := 104) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_0
  simp only [← c6894, ← c6895, ← c6896] at e_104_0
  have e_104_1 := arithEq_of_rows h (row := 104) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_1
  simp only [← c6897, ← c6898, ← c6899] at e_104_1
  have hr := e_104_1
  simp only [e_104_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 15) = bselect (a (.wire 103 3)) (a (.virt 3609)) (a (.wire 102 31)) := by
  have c6900 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6901 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6902 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6903 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6904 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6905 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_2 := arithEq_of_rows h (row := 104) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_2
  simp only [← c6900, ← c6901, ← c6902] at e_104_2
  have e_104_3 := arithEq_of_rows h (row := 104) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_3
  simp only [← c6903, ← c6904, ← c6905] at e_104_3
  have hr := e_104_3
  simp only [e_104_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 23) = bselect (a (.wire 103 3)) (a (.virt 3603)) (a (.wire 102 39)) := by
  have c6906 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6907 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6908 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6909 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6910 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6911 := (publicBatchWrapper89_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_104_4 := arithEq_of_rows h (row := 104) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_4
  simp only [← c6906, ← c6907, ← c6908] at e_104_4
  have e_104_5 := arithEq_of_rows h (row := 104) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_5
  simp only [← c6909, ← c6910, ← c6911] at e_104_5
  have hr := e_104_5
  simp only [e_104_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 31) = bselect (a (.wire 103 3)) (a (.virt 3604)) (a (.wire 102 47)) := by
  have c6912 := (publicBatchWrapper89_copies216 a h).1
  have c6913 := (publicBatchWrapper89_copies216 a h).2.1
  have c6914 := (publicBatchWrapper89_copies216 a h).2.2.1
  have c6915 := (publicBatchWrapper89_copies216 a h).2.2.2.1
  have c6916 := (publicBatchWrapper89_copies216 a h).2.2.2.2.1
  have c6917 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.1
  have e_104_6 := arithEq_of_rows h (row := 104) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_6
  simp only [← c6912, ← c6913, ← c6914] at e_104_6
  have e_104_7 := arithEq_of_rows h (row := 104) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_7
  simp only [← c6915, ← c6916, ← c6917] at e_104_7
  have hr := e_104_7
  simp only [e_104_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 39) = bor (a (.wire 96 35)) (a (.wire 102 51)) := by
  have c6918 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.1
  have c6919 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.1
  have c6920 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.1
  have c6921 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.1
  have c6922 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6923 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_17 := arithEq_of_rows h (row := 88) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_17
  simp only [← c6918, ← c6919, ← c6920] at e_88_17
  have e_96_9 := arithEq_of_rows h (row := 96) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_9
  simp only [← c6921, ← c6922, k3, ← c6923] at e_96_9
  have hr := e_96_9
  simp only [e_88_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 35) = bnot (a (.wire 20 15)) := by
  have c6924 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6925 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6926 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_104_8 := arithEq_of_rows h (row := 104) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_8
  simp only [← c6924, k3, ← c6925, k3, ← c6926] at e_104_8
  have hr := e_104_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 39) = bnot (a (.wire 96 39)) := by
  have c6927 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6928 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6929 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_104_9 := arithEq_of_rows h (row := 104) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_9
  simp only [← c6927, k3, ← c6928, k3, ← c6929] at e_104_9
  have hr := e_104_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 7) = band (a (.wire 104 35)) (a (.wire 104 39)) := by
  have c6930 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6931 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6932 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_1 := arithEq_of_rows h (row := 103) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_1
  simp only [← c6930, ← c6931, ← c6932] at e_103_1
  have hr := e_103_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 47) = bselect (a (.wire 103 7)) (a (.virt 3805)) (a (.wire 102 63)) := by
  have c6933 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6934 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6935 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6936 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6937 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6938 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_10 := arithEq_of_rows h (row := 104) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_10
  simp only [← c6933, ← c6934, ← c6935] at e_104_10
  have e_104_11 := arithEq_of_rows h (row := 104) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_11
  simp only [← c6936, ← c6937, ← c6938] at e_104_11
  have hr := e_104_11
  simp only [e_104_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 55) = bselect (a (.wire 103 7)) (a (.virt 3806)) (a (.wire 102 71)) := by
  have c6939 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6940 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6941 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6942 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6943 := (publicBatchWrapper89_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6944 := (publicBatchWrapper89_copies217 a h).1
  have e_104_12 := arithEq_of_rows h (row := 104) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_12
  simp only [← c6939, ← c6940, ← c6941] at e_104_12
  have e_104_13 := arithEq_of_rows h (row := 104) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_13
  simp only [← c6942, ← c6943, ← c6944] at e_104_13
  have hr := e_104_13
  simp only [e_104_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 63) = bselect (a (.wire 103 7)) (a (.virt 3807)) (a (.wire 102 79)) := by
  have c6945 := (publicBatchWrapper89_copies217 a h).2.1
  have c6946 := (publicBatchWrapper89_copies217 a h).2.2.1
  have c6947 := (publicBatchWrapper89_copies217 a h).2.2.2.1
  have c6948 := (publicBatchWrapper89_copies217 a h).2.2.2.2.1
  have c6949 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.1
  have c6950 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.1
  have e_104_14 := arithEq_of_rows h (row := 104) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_14
  simp only [← c6945, ← c6946, ← c6947] at e_104_14
  have e_104_15 := arithEq_of_rows h (row := 104) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_15
  simp only [← c6948, ← c6949, ← c6950] at e_104_15
  have hr := e_104_15
  simp only [e_104_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 71) = bselect (a (.wire 103 7)) (a (.virt 3808)) (a (.wire 104 7)) := by
  have c6951 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.1
  have c6952 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.1
  have c6953 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.1
  have c6954 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6955 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6956 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_16 := arithEq_of_rows h (row := 104) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_16
  simp only [← c6951, ← c6952, ← c6953] at e_104_16
  have e_104_17 := arithEq_of_rows h (row := 104) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_17
  simp only [← c6954, ← c6955, ← c6956] at e_104_17
  have hr := e_104_17
  simp only [e_104_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 104 79) = bselect (a (.wire 103 7)) (a (.virt 3809)) (a (.wire 104 15)) := by
  have c6957 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6958 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6959 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6960 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6961 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6962 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_18 := arithEq_of_rows h (row := 104) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_18
  simp only [← c6957, ← c6958, ← c6959] at e_104_18
  have e_104_19 := arithEq_of_rows h (row := 104) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_19
  simp only [← c6960, ← c6961, ← c6962] at e_104_19
  have hr := e_104_19
  simp only [e_104_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 7) = bselect (a (.wire 103 7)) (a (.virt 3803)) (a (.wire 104 23)) := by
  have c6963 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6964 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6965 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6966 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6967 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6968 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_0 := arithEq_of_rows h (row := 105) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_0
  simp only [← c6963, ← c6964, ← c6965] at e_105_0
  have e_105_1 := arithEq_of_rows h (row := 105) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_1
  simp only [← c6966, ← c6967, ← c6968] at e_105_1
  have hr := e_105_1
  simp only [e_105_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 15) = bselect (a (.wire 103 7)) (a (.virt 3804)) (a (.wire 104 31)) := by
  have c6969 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6970 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6971 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6972 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6973 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6974 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_2 := arithEq_of_rows h (row := 105) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_2
  simp only [← c6969, ← c6970, ← c6971] at e_105_2
  have e_105_3 := arithEq_of_rows h (row := 105) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_3
  simp only [← c6972, ← c6973, ← c6974] at e_105_3
  have hr := e_105_3
  simp only [e_105_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 43) = bor (a (.wire 96 39)) (a (.wire 104 35)) := by
  have c6975 := (publicBatchWrapper89_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6976 := (publicBatchWrapper89_copies218 a h).1
  have c6977 := (publicBatchWrapper89_copies218 a h).2.1
  have c6978 := (publicBatchWrapper89_copies218 a h).2.2.1
  have c6979 := (publicBatchWrapper89_copies218 a h).2.2.2.1
  have c6980 := (publicBatchWrapper89_copies218 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_18 := arithEq_of_rows h (row := 88) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_18
  simp only [← c6975, ← c6976, ← c6977] at e_88_18
  have e_96_10 := arithEq_of_rows h (row := 96) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_10
  simp only [← c6978, ← c6979, k3, ← c6980] at e_96_10
  have hr := e_96_10
  simp only [e_88_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 19) = bnot (a (.wire 20 59)) := by
  have c6981 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.1
  have c6982 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.1
  have c6983 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_105_4 := arithEq_of_rows h (row := 105) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_4
  simp only [← c6981, k3, ← c6982, k3, ← c6983] at e_105_4
  have hr := e_105_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 23) = bnot (a (.wire 96 43)) := by
  have c6984 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.1
  have c6985 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.1
  have c6986 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_105_5 := arithEq_of_rows h (row := 105) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_5
  simp only [← c6984, k3, ← c6985, k3, ← c6986] at e_105_5
  have hr := e_105_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 11) = band (a (.wire 105 19)) (a (.wire 105 23)) := by
  have c6987 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6988 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6989 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_2 := arithEq_of_rows h (row := 103) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_2
  simp only [← c6987, ← c6988, ← c6989] at e_103_2
  have hr := e_103_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 31) = bselect (a (.wire 103 11)) (a (.virt 4005)) (a (.wire 104 47)) := by
  have c6990 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6991 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6992 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6993 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6994 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6995 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_6 := arithEq_of_rows h (row := 105) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_6
  simp only [← c6990, ← c6991, ← c6992] at e_105_6
  have e_105_7 := arithEq_of_rows h (row := 105) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_7
  simp only [← c6993, ← c6994, ← c6995] at e_105_7
  have hr := e_105_7
  simp only [e_105_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 39) = bselect (a (.wire 103 11)) (a (.virt 4006)) (a (.wire 104 55)) := by
  have c6996 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6997 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6998 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6999 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7000 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7001 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_8 := arithEq_of_rows h (row := 105) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_8
  simp only [← c6996, ← c6997, ← c6998] at e_105_8
  have e_105_9 := arithEq_of_rows h (row := 105) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_9
  simp only [← c6999, ← c7000, ← c7001] at e_105_9
  have hr := e_105_9
  simp only [e_105_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 47) = bselect (a (.wire 103 11)) (a (.virt 4007)) (a (.wire 104 63)) := by
  have c7002 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7003 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7004 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7005 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7006 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7007 := (publicBatchWrapper89_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_105_10 := arithEq_of_rows h (row := 105) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_10
  simp only [← c7002, ← c7003, ← c7004] at e_105_10
  have e_105_11 := arithEq_of_rows h (row := 105) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_11
  simp only [← c7005, ← c7006, ← c7007] at e_105_11
  have hr := e_105_11
  simp only [e_105_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 55) = bselect (a (.wire 103 11)) (a (.virt 4008)) (a (.wire 104 71)) := by
  have c7008 := (publicBatchWrapper89_copies219 a h).1
  have c7009 := (publicBatchWrapper89_copies219 a h).2.1
  have c7010 := (publicBatchWrapper89_copies219 a h).2.2.1
  have c7011 := (publicBatchWrapper89_copies219 a h).2.2.2.1
  have c7012 := (publicBatchWrapper89_copies219 a h).2.2.2.2.1
  have c7013 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.1
  have e_105_12 := arithEq_of_rows h (row := 105) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_12
  simp only [← c7008, ← c7009, ← c7010] at e_105_12
  have e_105_13 := arithEq_of_rows h (row := 105) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_13
  simp only [← c7011, ← c7012, ← c7013] at e_105_13
  have hr := e_105_13
  simp only [e_105_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 63) = bselect (a (.wire 103 11)) (a (.virt 4009)) (a (.wire 104 79)) := by
  have c7014 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.1
  have c7015 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.1
  have c7016 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.1
  have c7017 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.1
  have c7018 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7019 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_14 := arithEq_of_rows h (row := 105) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_14
  simp only [← c7014, ← c7015, ← c7016] at e_105_14
  have e_105_15 := arithEq_of_rows h (row := 105) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_15
  simp only [← c7017, ← c7018, ← c7019] at e_105_15
  have hr := e_105_15
  simp only [e_105_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 71) = bselect (a (.wire 103 11)) (a (.virt 4003)) (a (.wire 105 7)) := by
  have c7020 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7021 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7022 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7023 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7024 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7025 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_16 := arithEq_of_rows h (row := 105) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_16
  simp only [← c7020, ← c7021, ← c7022] at e_105_16
  have e_105_17 := arithEq_of_rows h (row := 105) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_17
  simp only [← c7023, ← c7024, ← c7025] at e_105_17
  have hr := e_105_17
  simp only [e_105_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 105 79) = bselect (a (.wire 103 11)) (a (.virt 4004)) (a (.wire 105 15)) := by
  have c7026 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7027 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7028 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7029 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7030 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7031 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_105_18 := arithEq_of_rows h (row := 105) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_18
  simp only [← c7026, ← c7027, ← c7028] at e_105_18
  have e_105_19 := arithEq_of_rows h (row := 105) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_19
  simp only [← c7029, ← c7030, ← c7031] at e_105_19
  have hr := e_105_19
  simp only [e_105_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 47) = bor (a (.wire 96 43)) (a (.wire 105 19)) := by
  have c7032 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7033 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7034 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7035 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7036 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7037 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_88_19 := arithEq_of_rows h (row := 88) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_19
  simp only [← c7032, ← c7033, ← c7034] at e_88_19
  have e_96_11 := arithEq_of_rows h (row := 96) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_11
  simp only [← c7035, ← c7036, k3, ← c7037] at e_96_11
  have hr := e_96_11
  simp only [e_88_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 3) = bnot (a (.wire 22 23)) := by
  have c7038 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7039 := (publicBatchWrapper89_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7040 := (publicBatchWrapper89_copies220 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_106_0 := arithEq_of_rows h (row := 106) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_0
  simp only [← c7038, k3, ← c7039, k3, ← c7040] at e_106_0
  have hr := e_106_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 7) = bnot (a (.wire 96 47)) := by
  have c7041 := (publicBatchWrapper89_copies220 a h).2.1
  have c7042 := (publicBatchWrapper89_copies220 a h).2.2.1
  have c7043 := (publicBatchWrapper89_copies220 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_106_1 := arithEq_of_rows h (row := 106) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_1
  simp only [← c7041, k3, ← c7042, k3, ← c7043] at e_106_1
  have hr := e_106_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 15) = band (a (.wire 106 3)) (a (.wire 106 7)) := by
  have c7044 := (publicBatchWrapper89_copies220 a h).2.2.2.2.1
  have c7045 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.1
  have c7046 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.1
  have e_103_3 := arithEq_of_rows h (row := 103) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_3
  simp only [← c7044, ← c7045, ← c7046] at e_103_3
  have hr := e_103_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 15) = bselect (a (.wire 103 15)) (a (.virt 4205)) (a (.wire 105 31)) := by
  have c7047 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.1
  have c7048 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.1
  have c7049 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.1
  have c7050 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7051 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7052 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_2 := arithEq_of_rows h (row := 106) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_2
  simp only [← c7047, ← c7048, ← c7049] at e_106_2
  have e_106_3 := arithEq_of_rows h (row := 106) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_3
  simp only [← c7050, ← c7051, ← c7052] at e_106_3
  have hr := e_106_3
  simp only [e_106_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 23) = bselect (a (.wire 103 15)) (a (.virt 4206)) (a (.wire 105 39)) := by
  have c7053 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7054 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7055 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7056 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7057 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7058 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_4 := arithEq_of_rows h (row := 106) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_4
  simp only [← c7053, ← c7054, ← c7055] at e_106_4
  have e_106_5 := arithEq_of_rows h (row := 106) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_5
  simp only [← c7056, ← c7057, ← c7058] at e_106_5
  have hr := e_106_5
  simp only [e_106_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 31) = bselect (a (.wire 103 15)) (a (.virt 4207)) (a (.wire 105 47)) := by
  have c7059 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7060 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7061 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7062 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7063 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7064 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_6 := arithEq_of_rows h (row := 106) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_6
  simp only [← c7059, ← c7060, ← c7061] at e_106_6
  have e_106_7 := arithEq_of_rows h (row := 106) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_7
  simp only [← c7062, ← c7063, ← c7064] at e_106_7
  have hr := e_106_7
  simp only [e_106_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 39) = bselect (a (.wire 103 15)) (a (.virt 4208)) (a (.wire 105 55)) := by
  have c7065 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7066 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7067 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7068 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7069 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7070 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_8 := arithEq_of_rows h (row := 106) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_8
  simp only [← c7065, ← c7066, ← c7067] at e_106_8
  have e_106_9 := arithEq_of_rows h (row := 106) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_9
  simp only [← c7068, ← c7069, ← c7070] at e_106_9
  have hr := e_106_9
  simp only [e_106_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 47) = bselect (a (.wire 103 15)) (a (.virt 4209)) (a (.wire 105 63)) := by
  have c7071 := (publicBatchWrapper89_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7072 := (publicBatchWrapper89_copies221 a h).1
  have c7073 := (publicBatchWrapper89_copies221 a h).2.1
  have c7074 := (publicBatchWrapper89_copies221 a h).2.2.1
  have c7075 := (publicBatchWrapper89_copies221 a h).2.2.2.1
  have c7076 := (publicBatchWrapper89_copies221 a h).2.2.2.2.1
  have e_106_10 := arithEq_of_rows h (row := 106) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_10
  simp only [← c7071, ← c7072, ← c7073] at e_106_10
  have e_106_11 := arithEq_of_rows h (row := 106) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_11
  simp only [← c7074, ← c7075, ← c7076] at e_106_11
  have hr := e_106_11
  simp only [e_106_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 55) = bselect (a (.wire 103 15)) (a (.virt 4203)) (a (.wire 105 71)) := by
  have c7077 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.1
  have c7078 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.1
  have c7079 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.1
  have c7080 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.1
  have c7081 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.1
  have c7082 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_106_12 := arithEq_of_rows h (row := 106) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_12
  simp only [← c7077, ← c7078, ← c7079] at e_106_12
  have e_106_13 := arithEq_of_rows h (row := 106) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_13
  simp only [← c7080, ← c7081, ← c7082] at e_106_13
  have hr := e_106_13
  simp only [e_106_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 63) = bselect (a (.wire 103 15)) (a (.virt 4204)) (a (.wire 105 79)) := by
  have c7083 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7084 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7085 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7086 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7087 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7088 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_14 := arithEq_of_rows h (row := 106) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_14
  simp only [← c7083, ← c7084, ← c7085] at e_106_14
  have e_106_15 := arithEq_of_rows h (row := 106) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_15
  simp only [← c7086, ← c7087, ← c7088] at e_106_15
  have hr := e_106_15
  simp only [e_106_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 51) = bor (a (.wire 96 47)) (a (.wire 106 3)) := by
  have c7089 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7090 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7091 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7092 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7093 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7094 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_96_12 := arithEq_of_rows h (row := 96) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_12
  simp only [← c7092, ← c7093, k3, ← c7094] at e_96_12
  have e_107_0 := arithEq_of_rows h (row := 107) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_0
  simp only [← c7089, ← c7090, ← c7091] at e_107_0
  have hr := e_96_12
  simp only [e_107_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 67) = bnot (a (.wire 22 67)) := by
  have c7095 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7096 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7097 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_106_16 := arithEq_of_rows h (row := 106) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_16
  simp only [← c7095, k3, ← c7096, k3, ← c7097] at e_106_16
  have hr := e_106_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 71) = bnot (a (.wire 96 51)) := by
  have c7098 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7099 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7100 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_106_17 := arithEq_of_rows h (row := 106) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_17
  simp only [← c7098, k3, ← c7099, k3, ← c7100] at e_106_17
  have hr := e_106_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 19) = band (a (.wire 106 67)) (a (.wire 106 71)) := by
  have c7101 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7102 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7103 := (publicBatchWrapper89_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_103_4 := arithEq_of_rows h (row := 103) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_4
  simp only [← c7101, ← c7102, ← c7103] at e_103_4
  have hr := e_103_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 106 79) = bselect (a (.wire 103 19)) (a (.virt 4405)) (a (.wire 106 15)) := by
  have c7104 := (publicBatchWrapper89_copies222 a h).1
  have c7105 := (publicBatchWrapper89_copies222 a h).2.1
  have c7106 := (publicBatchWrapper89_copies222 a h).2.2.1
  have c7107 := (publicBatchWrapper89_copies222 a h).2.2.2.1
  have c7108 := (publicBatchWrapper89_copies222 a h).2.2.2.2.1
  have c7109 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.1
  have e_106_18 := arithEq_of_rows h (row := 106) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_18
  simp only [← c7104, ← c7105, ← c7106] at e_106_18
  have e_106_19 := arithEq_of_rows h (row := 106) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_19
  simp only [← c7107, ← c7108, ← c7109] at e_106_19
  have hr := e_106_19
  simp only [e_106_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 7) = bselect (a (.wire 103 19)) (a (.virt 4406)) (a (.wire 106 23)) := by
  have c7110 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.1
  have c7111 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.1
  have c7112 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.1
  have c7113 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.1
  have c7114 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7115 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_108_0 := arithEq_of_rows h (row := 108) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_0
  simp only [← c7110, ← c7111, ← c7112] at e_108_0
  have e_108_1 := arithEq_of_rows h (row := 108) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_1
  simp only [← c7113, ← c7114, ← c7115] at e_108_1
  have hr := e_108_1
  simp only [e_108_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 15) = bselect (a (.wire 103 19)) (a (.virt 4407)) (a (.wire 106 31)) := by
  have c7116 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7117 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7118 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7119 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7120 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7121 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_108_2 := arithEq_of_rows h (row := 108) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_2
  simp only [← c7116, ← c7117, ← c7118] at e_108_2
  have e_108_3 := arithEq_of_rows h (row := 108) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_3
  simp only [← c7119, ← c7120, ← c7121] at e_108_3
  have hr := e_108_3
  simp only [e_108_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 23) = bselect (a (.wire 103 19)) (a (.virt 4408)) (a (.wire 106 39)) := by
  have c7122 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7123 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7124 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7125 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7126 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7127 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_108_4 := arithEq_of_rows h (row := 108) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_4
  simp only [← c7122, ← c7123, ← c7124] at e_108_4
  have e_108_5 := arithEq_of_rows h (row := 108) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_5
  simp only [← c7125, ← c7126, ← c7127] at e_108_5
  have hr := e_108_5
  simp only [e_108_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 31) = bselect (a (.wire 103 19)) (a (.virt 4409)) (a (.wire 106 47)) := by
  have c7128 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7129 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7130 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7131 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7132 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7133 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_108_6 := arithEq_of_rows h (row := 108) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_6
  simp only [← c7128, ← c7129, ← c7130] at e_108_6
  have e_108_7 := arithEq_of_rows h (row := 108) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_7
  simp only [← c7131, ← c7132, ← c7133] at e_108_7
  have hr := e_108_7
  simp only [e_108_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 39) = bselect (a (.wire 103 19)) (a (.virt 4403)) (a (.wire 106 55)) := by
  have c7134 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7135 := (publicBatchWrapper89_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7136 := (publicBatchWrapper89_copies223 a h).1
  have c7137 := (publicBatchWrapper89_copies223 a h).2.1
  have c7138 := (publicBatchWrapper89_copies223 a h).2.2.1
  have c7139 := (publicBatchWrapper89_copies223 a h).2.2.2.1
  have e_108_8 := arithEq_of_rows h (row := 108) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_8
  simp only [← c7134, ← c7135, ← c7136] at e_108_8
  have e_108_9 := arithEq_of_rows h (row := 108) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_9
  simp only [← c7137, ← c7138, ← c7139] at e_108_9
  have hr := e_108_9
  simp only [e_108_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 47) = bselect (a (.wire 103 19)) (a (.virt 4404)) (a (.wire 106 63)) := by
  have c7140 := (publicBatchWrapper89_copies223 a h).2.2.2.2.1
  have c7141 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.1
  have c7142 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.1
  have c7143 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.1
  have c7144 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.1
  have c7145 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.1
  have e_108_10 := arithEq_of_rows h (row := 108) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_10
  simp only [← c7140, ← c7141, ← c7142] at e_108_10
  have e_108_11 := arithEq_of_rows h (row := 108) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_11
  simp only [← c7143, ← c7144, ← c7145] at e_108_11
  have hr := e_108_11
  simp only [e_108_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 55) = bor (a (.wire 96 51)) (a (.wire 106 67)) := by
  have c7146 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7147 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7148 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7149 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7150 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7151 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_96_13 := arithEq_of_rows h (row := 96) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_13
  simp only [← c7149, ← c7150, k3, ← c7151] at e_96_13
  have e_107_1 := arithEq_of_rows h (row := 107) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_1
  simp only [← c7146, ← c7147, ← c7148] at e_107_1
  have hr := e_96_13
  simp only [e_107_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 51) = bnot (a (.wire 24 31)) := by
  have c7152 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7153 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7154 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_108_12 := arithEq_of_rows h (row := 108) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_12
  simp only [← c7152, k3, ← c7153, k3, ← c7154] at e_108_12
  have hr := e_108_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 55) = bnot (a (.wire 96 55)) := by
  have c7155 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7156 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7157 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_108_13 := arithEq_of_rows h (row := 108) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_13
  simp only [← c7155, k3, ← c7156, k3, ← c7157] at e_108_13
  have hr := e_108_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 23) = band (a (.wire 108 51)) (a (.wire 108 55)) := by
  have c7158 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7159 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7160 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_5 := arithEq_of_rows h (row := 103) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_5
  simp only [← c7158, ← c7159, ← c7160] at e_103_5
  have hr := e_103_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 63) = bselect (a (.wire 103 23)) (a (.virt 4605)) (a (.wire 106 79)) := by
  have c7161 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7162 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7163 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7164 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7165 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7166 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_108_14 := arithEq_of_rows h (row := 108) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_14
  simp only [← c7161, ← c7162, ← c7163] at e_108_14
  have e_108_15 := arithEq_of_rows h (row := 108) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_15
  simp only [← c7164, ← c7165, ← c7166] at e_108_15
  have hr := e_108_15
  simp only [e_108_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 71) = bselect (a (.wire 103 23)) (a (.virt 4606)) (a (.wire 108 7)) := by
  have c7167 := (publicBatchWrapper89_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7168 := (publicBatchWrapper89_copies224 a h).1
  have c7169 := (publicBatchWrapper89_copies224 a h).2.1
  have c7170 := (publicBatchWrapper89_copies224 a h).2.2.1
  have c7171 := (publicBatchWrapper89_copies224 a h).2.2.2.1
  have c7172 := (publicBatchWrapper89_copies224 a h).2.2.2.2.1
  have e_108_16 := arithEq_of_rows h (row := 108) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_16
  simp only [← c7167, ← c7168, ← c7169] at e_108_16
  have e_108_17 := arithEq_of_rows h (row := 108) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_17
  simp only [← c7170, ← c7171, ← c7172] at e_108_17
  have hr := e_108_17
  simp only [e_108_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 108 79) = bselect (a (.wire 103 23)) (a (.virt 4607)) (a (.wire 108 15)) := by
  have c7173 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.1
  have c7174 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.1
  have c7175 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.1
  have c7176 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.1
  have c7177 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.1
  have c7178 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_108_18 := arithEq_of_rows h (row := 108) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_18
  simp only [← c7173, ← c7174, ← c7175] at e_108_18
  have e_108_19 := arithEq_of_rows h (row := 108) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_19
  simp only [← c7176, ← c7177, ← c7178] at e_108_19
  have hr := e_108_19
  simp only [e_108_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 7) = bselect (a (.wire 103 23)) (a (.virt 4608)) (a (.wire 108 23)) := by
  have c7179 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7180 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7181 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7182 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7183 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7184 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_0 := arithEq_of_rows h (row := 109) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_0
  simp only [← c7179, ← c7180, ← c7181] at e_109_0
  have e_109_1 := arithEq_of_rows h (row := 109) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_1
  simp only [← c7182, ← c7183, ← c7184] at e_109_1
  have hr := e_109_1
  simp only [e_109_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 15) = bselect (a (.wire 103 23)) (a (.virt 4609)) (a (.wire 108 31)) := by
  have c7185 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7186 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7187 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7188 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7189 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7190 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_2 := arithEq_of_rows h (row := 109) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_2
  simp only [← c7185, ← c7186, ← c7187] at e_109_2
  have e_109_3 := arithEq_of_rows h (row := 109) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_3
  simp only [← c7188, ← c7189, ← c7190] at e_109_3
  have hr := e_109_3
  simp only [e_109_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 23) = bselect (a (.wire 103 23)) (a (.virt 4603)) (a (.wire 108 39)) := by
  have c7191 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7192 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7193 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7194 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7195 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7196 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_4 := arithEq_of_rows h (row := 109) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_4
  simp only [← c7191, ← c7192, ← c7193] at e_109_4
  have e_109_5 := arithEq_of_rows h (row := 109) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_5
  simp only [← c7194, ← c7195, ← c7196] at e_109_5
  have hr := e_109_5
  simp only [e_109_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 31) = bselect (a (.wire 103 23)) (a (.virt 4604)) (a (.wire 108 47)) := by
  have c7197 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7198 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7199 := (publicBatchWrapper89_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7200 := (publicBatchWrapper89_copies225 a h).1
  have c7201 := (publicBatchWrapper89_copies225 a h).2.1
  have c7202 := (publicBatchWrapper89_copies225 a h).2.2.1
  have e_109_6 := arithEq_of_rows h (row := 109) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_6
  simp only [← c7197, ← c7198, ← c7199] at e_109_6
  have e_109_7 := arithEq_of_rows h (row := 109) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_7
  simp only [← c7200, ← c7201, ← c7202] at e_109_7
  have hr := e_109_7
  simp only [e_109_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 59) = bor (a (.wire 96 55)) (a (.wire 108 51)) := by
  have c7203 := (publicBatchWrapper89_copies225 a h).2.2.2.1
  have c7204 := (publicBatchWrapper89_copies225 a h).2.2.2.2.1
  have c7205 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.1
  have c7206 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.1
  have c7207 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.1
  have c7208 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_96_14 := arithEq_of_rows h (row := 96) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_14
  simp only [← c7206, ← c7207, k3, ← c7208] at e_96_14
  have e_107_2 := arithEq_of_rows h (row := 107) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_2
  simp only [← c7203, ← c7204, ← c7205] at e_107_2
  have hr := e_96_14
  simp only [e_107_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 35) = bnot (a (.wire 24 75)) := by
  have c7209 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.1
  have c7210 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7211 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_109_8 := arithEq_of_rows h (row := 109) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_8
  simp only [← c7209, k3, ← c7210, k3, ← c7211] at e_109_8
  have hr := e_109_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 39) = bnot (a (.wire 96 59)) := by
  have c7212 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7213 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7214 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_109_9 := arithEq_of_rows h (row := 109) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_9
  simp only [← c7212, k3, ← c7213, k3, ← c7214] at e_109_9
  have hr := e_109_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 27) = band (a (.wire 109 35)) (a (.wire 109 39)) := by
  have c7215 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7216 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7217 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_6 := arithEq_of_rows h (row := 103) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_6
  simp only [← c7215, ← c7216, ← c7217] at e_103_6
  have hr := e_103_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 47) = bselect (a (.wire 103 27)) (a (.virt 4805)) (a (.wire 108 63)) := by
  have c7218 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7219 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7220 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7221 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7222 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7223 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_10 := arithEq_of_rows h (row := 109) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_10
  simp only [← c7218, ← c7219, ← c7220] at e_109_10
  have e_109_11 := arithEq_of_rows h (row := 109) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_11
  simp only [← c7221, ← c7222, ← c7223] at e_109_11
  have hr := e_109_11
  simp only [e_109_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 55) = bselect (a (.wire 103 27)) (a (.virt 4806)) (a (.wire 108 71)) := by
  have c7224 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7225 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7226 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7227 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7228 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7229 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_12 := arithEq_of_rows h (row := 109) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_12
  simp only [← c7224, ← c7225, ← c7226] at e_109_12
  have e_109_13 := arithEq_of_rows h (row := 109) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_13
  simp only [← c7227, ← c7228, ← c7229] at e_109_13
  have hr := e_109_13
  simp only [e_109_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 63) = bselect (a (.wire 103 27)) (a (.virt 4807)) (a (.wire 108 79)) := by
  have c7230 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7231 := (publicBatchWrapper89_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7232 := (publicBatchWrapper89_copies226 a h).1
  have c7233 := (publicBatchWrapper89_copies226 a h).2.1
  have c7234 := (publicBatchWrapper89_copies226 a h).2.2.1
  have c7235 := (publicBatchWrapper89_copies226 a h).2.2.2.1
  have e_109_14 := arithEq_of_rows h (row := 109) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_14
  simp only [← c7230, ← c7231, ← c7232] at e_109_14
  have e_109_15 := arithEq_of_rows h (row := 109) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_15
  simp only [← c7233, ← c7234, ← c7235] at e_109_15
  have hr := e_109_15
  simp only [e_109_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 71) = bselect (a (.wire 103 27)) (a (.virt 4808)) (a (.wire 109 7)) := by
  have c7236 := (publicBatchWrapper89_copies226 a h).2.2.2.2.1
  have c7237 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.1
  have c7238 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.1
  have c7239 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.1
  have c7240 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.1
  have c7241 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.1
  have e_109_16 := arithEq_of_rows h (row := 109) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_16
  simp only [← c7236, ← c7237, ← c7238] at e_109_16
  have e_109_17 := arithEq_of_rows h (row := 109) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_17
  simp only [← c7239, ← c7240, ← c7241] at e_109_17
  have hr := e_109_17
  simp only [e_109_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 109 79) = bselect (a (.wire 103 27)) (a (.virt 4809)) (a (.wire 109 15)) := by
  have c7242 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7243 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7244 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7245 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7246 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7247 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_18 := arithEq_of_rows h (row := 109) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_18
  simp only [← c7242, ← c7243, ← c7244] at e_109_18
  have e_109_19 := arithEq_of_rows h (row := 109) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_19
  simp only [← c7245, ← c7246, ← c7247] at e_109_19
  have hr := e_109_19
  simp only [e_109_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 7) = bselect (a (.wire 103 27)) (a (.virt 4803)) (a (.wire 109 23)) := by
  have c7248 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7249 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7250 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7251 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7252 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7253 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_0 := arithEq_of_rows h (row := 110) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_0
  simp only [← c7248, ← c7249, ← c7250] at e_110_0
  have e_110_1 := arithEq_of_rows h (row := 110) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_1
  simp only [← c7251, ← c7252, ← c7253] at e_110_1
  have hr := e_110_1
  simp only [e_110_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 15) = bselect (a (.wire 103 27)) (a (.virt 4804)) (a (.wire 109 31)) := by
  have c7254 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7255 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7256 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7257 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7258 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7259 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_2 := arithEq_of_rows h (row := 110) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_2
  simp only [← c7254, ← c7255, ← c7256] at e_110_2
  have e_110_3 := arithEq_of_rows h (row := 110) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_3
  simp only [← c7257, ← c7258, ← c7259] at e_110_3
  have hr := e_110_3
  simp only [e_110_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 63) = bor (a (.wire 96 59)) (a (.wire 109 35)) := by
  have c7260 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7261 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7262 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7263 := (publicBatchWrapper89_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7264 := (publicBatchWrapper89_copies227 a h).1
  have c7265 := (publicBatchWrapper89_copies227 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_96_15 := arithEq_of_rows h (row := 96) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_15
  simp only [← c7263, ← c7264, k3, ← c7265] at e_96_15
  have e_107_3 := arithEq_of_rows h (row := 107) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_3
  simp only [← c7260, ← c7261, ← c7262] at e_107_3
  have hr := e_96_15
  simp only [e_107_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 19) = bnot (a (.wire 26 39)) := by
  have c7266 := (publicBatchWrapper89_copies227 a h).2.2.1
  have c7267 := (publicBatchWrapper89_copies227 a h).2.2.2.1
  have c7268 := (publicBatchWrapper89_copies227 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_110_4 := arithEq_of_rows h (row := 110) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_4
  simp only [← c7266, k3, ← c7267, k3, ← c7268] at e_110_4
  have hr := e_110_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 23) = bnot (a (.wire 96 63)) := by
  have c7269 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.1
  have c7270 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.1
  have c7271 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_110_5 := arithEq_of_rows h (row := 110) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_5
  simp only [← c7269, k3, ← c7270, k3, ← c7271] at e_110_5
  have hr := e_110_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 31) = band (a (.wire 110 19)) (a (.wire 110 23)) := by
  have c7272 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.1
  have c7273 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.1
  have c7274 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_103_7 := arithEq_of_rows h (row := 103) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_7
  simp only [← c7272, ← c7273, ← c7274] at e_103_7
  have hr := e_103_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 31) = bselect (a (.wire 103 31)) (a (.virt 5005)) (a (.wire 109 47)) := by
  have c7275 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7276 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7277 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7278 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7279 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7280 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_6 := arithEq_of_rows h (row := 110) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_6
  simp only [← c7275, ← c7276, ← c7277] at e_110_6
  have e_110_7 := arithEq_of_rows h (row := 110) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_7
  simp only [← c7278, ← c7279, ← c7280] at e_110_7
  have hr := e_110_7
  simp only [e_110_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 39) = bselect (a (.wire 103 31)) (a (.virt 5006)) (a (.wire 109 55)) := by
  have c7281 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7282 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7283 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7284 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7285 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7286 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_8 := arithEq_of_rows h (row := 110) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_8
  simp only [← c7281, ← c7282, ← c7283] at e_110_8
  have e_110_9 := arithEq_of_rows h (row := 110) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_9
  simp only [← c7284, ← c7285, ← c7286] at e_110_9
  have hr := e_110_9
  simp only [e_110_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 47) = bselect (a (.wire 103 31)) (a (.virt 5007)) (a (.wire 109 63)) := by
  have c7287 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7288 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7289 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7290 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7291 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7292 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_10 := arithEq_of_rows h (row := 110) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_10
  simp only [← c7287, ← c7288, ← c7289] at e_110_10
  have e_110_11 := arithEq_of_rows h (row := 110) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_11
  simp only [← c7290, ← c7291, ← c7292] at e_110_11
  have hr := e_110_11
  simp only [e_110_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 55) = bselect (a (.wire 103 31)) (a (.virt 5008)) (a (.wire 109 71)) := by
  have c7293 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7294 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7295 := (publicBatchWrapper89_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7296 := (publicBatchWrapper89_copies228 a h).1
  have c7297 := (publicBatchWrapper89_copies228 a h).2.1
  have c7298 := (publicBatchWrapper89_copies228 a h).2.2.1
  have e_110_12 := arithEq_of_rows h (row := 110) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_12
  simp only [← c7293, ← c7294, ← c7295] at e_110_12
  have e_110_13 := arithEq_of_rows h (row := 110) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_13
  simp only [← c7296, ← c7297, ← c7298] at e_110_13
  have hr := e_110_13
  simp only [e_110_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 63) = bselect (a (.wire 103 31)) (a (.virt 5009)) (a (.wire 109 79)) := by
  have c7299 := (publicBatchWrapper89_copies228 a h).2.2.2.1
  have c7300 := (publicBatchWrapper89_copies228 a h).2.2.2.2.1
  have c7301 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.1
  have c7302 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.1
  have c7303 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.1
  have c7304 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.1
  have e_110_14 := arithEq_of_rows h (row := 110) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_14
  simp only [← c7299, ← c7300, ← c7301] at e_110_14
  have e_110_15 := arithEq_of_rows h (row := 110) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_15
  simp only [← c7302, ← c7303, ← c7304] at e_110_15
  have hr := e_110_15
  simp only [e_110_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 71) = bselect (a (.wire 103 31)) (a (.virt 5003)) (a (.wire 110 7)) := by
  have c7305 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.1
  have c7306 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7307 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7308 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7309 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7310 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_16 := arithEq_of_rows h (row := 110) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_16
  simp only [← c7305, ← c7306, ← c7307] at e_110_16
  have e_110_17 := arithEq_of_rows h (row := 110) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_17
  simp only [← c7308, ← c7309, ← c7310] at e_110_17
  have hr := e_110_17
  simp only [e_110_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 110 79) = bselect (a (.wire 103 31)) (a (.virt 5004)) (a (.wire 110 15)) := by
  have c7311 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7312 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7313 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7314 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7315 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7316 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_18 := arithEq_of_rows h (row := 110) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_18
  simp only [← c7311, ← c7312, ← c7313] at e_110_18
  have e_110_19 := arithEq_of_rows h (row := 110) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_19
  simp only [← c7314, ← c7315, ← c7316] at e_110_19
  have hr := e_110_19
  simp only [e_110_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 67) = bor (a (.wire 96 63)) (a (.wire 110 19)) := by
  have c7317 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7318 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7319 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7320 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7321 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7322 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_96_16 := arithEq_of_rows h (row := 96) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_16
  simp only [← c7320, ← c7321, k3, ← c7322] at e_96_16
  have e_107_4 := arithEq_of_rows h (row := 107) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_4
  simp only [← c7317, ← c7318, ← c7319] at e_107_4
  have hr := e_96_16
  simp only [e_107_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 3) = bnot (a (.wire 27 3)) := by
  have c7323 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7324 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7325 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_111_0 := arithEq_of_rows h (row := 111) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_0
  simp only [← c7323, k3, ← c7324, k3, ← c7325] at e_111_0
  have hr := e_111_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 7) = bnot (a (.wire 96 67)) := by
  have c7326 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7327 := (publicBatchWrapper89_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7328 := (publicBatchWrapper89_copies229 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_111_1 := arithEq_of_rows h (row := 111) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_1
  simp only [← c7326, k3, ← c7327, k3, ← c7328] at e_111_1
  have hr := e_111_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 35) = band (a (.wire 111 3)) (a (.wire 111 7)) := by
  have c7329 := (publicBatchWrapper89_copies229 a h).2.1
  have c7330 := (publicBatchWrapper89_copies229 a h).2.2.1
  have c7331 := (publicBatchWrapper89_copies229 a h).2.2.2.1
  have e_103_8 := arithEq_of_rows h (row := 103) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_8
  simp only [← c7329, ← c7330, ← c7331] at e_103_8
  have hr := e_103_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 15) = bselect (a (.wire 103 35)) (a (.virt 5205)) (a (.wire 110 31)) := by
  have c7332 := (publicBatchWrapper89_copies229 a h).2.2.2.2.1
  have c7333 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.1
  have c7334 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.1
  have c7335 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.1
  have c7336 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.1
  have c7337 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.1
  have e_111_2 := arithEq_of_rows h (row := 111) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_2
  simp only [← c7332, ← c7333, ← c7334] at e_111_2
  have e_111_3 := arithEq_of_rows h (row := 111) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_3
  simp only [← c7335, ← c7336, ← c7337] at e_111_3
  have hr := e_111_3
  simp only [e_111_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 23) = bselect (a (.wire 103 35)) (a (.virt 5206)) (a (.wire 110 39)) := by
  have c7338 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7339 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7340 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7341 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7342 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7343 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_4 := arithEq_of_rows h (row := 111) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_4
  simp only [← c7338, ← c7339, ← c7340] at e_111_4
  have e_111_5 := arithEq_of_rows h (row := 111) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_5
  simp only [← c7341, ← c7342, ← c7343] at e_111_5
  have hr := e_111_5
  simp only [e_111_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 31) = bselect (a (.wire 103 35)) (a (.virt 5207)) (a (.wire 110 47)) := by
  have c7344 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7345 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7346 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7347 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7348 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7349 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_6 := arithEq_of_rows h (row := 111) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_6
  simp only [← c7344, ← c7345, ← c7346] at e_111_6
  have e_111_7 := arithEq_of_rows h (row := 111) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_7
  simp only [← c7347, ← c7348, ← c7349] at e_111_7
  have hr := e_111_7
  simp only [e_111_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 39) = bselect (a (.wire 103 35)) (a (.virt 5208)) (a (.wire 110 55)) := by
  have c7350 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7351 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7352 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7353 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7354 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7355 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_8 := arithEq_of_rows h (row := 111) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_8
  simp only [← c7350, ← c7351, ← c7352] at e_111_8
  have e_111_9 := arithEq_of_rows h (row := 111) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_9
  simp only [← c7353, ← c7354, ← c7355] at e_111_9
  have hr := e_111_9
  simp only [e_111_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 47) = bselect (a (.wire 103 35)) (a (.virt 5209)) (a (.wire 110 63)) := by
  have c7356 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7357 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7358 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7359 := (publicBatchWrapper89_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7360 := (publicBatchWrapper89_copies230 a h).1
  have c7361 := (publicBatchWrapper89_copies230 a h).2.1
  have e_111_10 := arithEq_of_rows h (row := 111) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_10
  simp only [← c7356, ← c7357, ← c7358] at e_111_10
  have e_111_11 := arithEq_of_rows h (row := 111) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_11
  simp only [← c7359, ← c7360, ← c7361] at e_111_11
  have hr := e_111_11
  simp only [e_111_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 55) = bselect (a (.wire 103 35)) (a (.virt 5203)) (a (.wire 110 71)) := by
  have c7362 := (publicBatchWrapper89_copies230 a h).2.2.1
  have c7363 := (publicBatchWrapper89_copies230 a h).2.2.2.1
  have c7364 := (publicBatchWrapper89_copies230 a h).2.2.2.2.1
  have c7365 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.1
  have c7366 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.1
  have c7367 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.1
  have e_111_12 := arithEq_of_rows h (row := 111) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_12
  simp only [← c7362, ← c7363, ← c7364] at e_111_12
  have e_111_13 := arithEq_of_rows h (row := 111) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_13
  simp only [← c7365, ← c7366, ← c7367] at e_111_13
  have hr := e_111_13
  simp only [e_111_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 63) = bselect (a (.wire 103 35)) (a (.virt 5204)) (a (.wire 110 79)) := by
  have c7368 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.1
  have c7369 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.1
  have c7370 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7371 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7372 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7373 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_14 := arithEq_of_rows h (row := 111) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_14
  simp only [← c7368, ← c7369, ← c7370] at e_111_14
  have e_111_15 := arithEq_of_rows h (row := 111) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_15
  simp only [← c7371, ← c7372, ← c7373] at e_111_15
  have hr := e_111_15
  simp only [e_111_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 71) = bor (a (.wire 96 67)) (a (.wire 111 3)) := by
  have c7374 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7375 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7376 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7377 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7378 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7379 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_96_17 := arithEq_of_rows h (row := 96) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_17
  simp only [← c7377, ← c7378, k3, ← c7379] at e_96_17
  have e_107_5 := arithEq_of_rows h (row := 107) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_5
  simp only [← c7374, ← c7375, ← c7376] at e_107_5
  have hr := e_96_17
  simp only [e_107_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 67) = bnot (a (.wire 27 47)) := by
  have c7380 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7381 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7382 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_111_16 := arithEq_of_rows h (row := 111) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_16
  simp only [← c7380, k3, ← c7381, k3, ← c7382] at e_111_16
  have hr := e_111_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 71) = bnot (a (.wire 96 71)) := by
  have c7383 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7384 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7385 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_111_17 := arithEq_of_rows h (row := 111) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_17
  simp only [← c7383, k3, ← c7384, k3, ← c7385] at e_111_17
  have hr := e_111_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 39) = band (a (.wire 111 67)) (a (.wire 111 71)) := by
  have c7386 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7387 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7388 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_9 := arithEq_of_rows h (row := 103) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_9
  simp only [← c7386, ← c7387, ← c7388] at e_103_9
  have hr := e_103_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 111 79) = bselect (a (.wire 103 39)) (a (.virt 5405)) (a (.wire 111 15)) := by
  have c7389 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7390 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7391 := (publicBatchWrapper89_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7392 := (publicBatchWrapper89_copies231 a h).1
  have c7393 := (publicBatchWrapper89_copies231 a h).2.1
  have c7394 := (publicBatchWrapper89_copies231 a h).2.2.1
  have e_111_18 := arithEq_of_rows h (row := 111) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_18
  simp only [← c7389, ← c7390, ← c7391] at e_111_18
  have e_111_19 := arithEq_of_rows h (row := 111) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_19
  simp only [← c7392, ← c7393, ← c7394] at e_111_19
  have hr := e_111_19
  simp only [e_111_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 7) = bselect (a (.wire 103 39)) (a (.virt 5406)) (a (.wire 111 23)) := by
  have c7395 := (publicBatchWrapper89_copies231 a h).2.2.2.1
  have c7396 := (publicBatchWrapper89_copies231 a h).2.2.2.2.1
  have c7397 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.1
  have c7398 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.1
  have c7399 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.1
  have c7400 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.1
  have e_112_0 := arithEq_of_rows h (row := 112) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_0
  simp only [← c7395, ← c7396, ← c7397] at e_112_0
  have e_112_1 := arithEq_of_rows h (row := 112) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_1
  simp only [← c7398, ← c7399, ← c7400] at e_112_1
  have hr := e_112_1
  simp only [e_112_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 15) = bselect (a (.wire 103 39)) (a (.virt 5407)) (a (.wire 111 31)) := by
  have c7401 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.1
  have c7402 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7403 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7404 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7405 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7406 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_2 := arithEq_of_rows h (row := 112) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_2
  simp only [← c7401, ← c7402, ← c7403] at e_112_2
  have e_112_3 := arithEq_of_rows h (row := 112) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_3
  simp only [← c7404, ← c7405, ← c7406] at e_112_3
  have hr := e_112_3
  simp only [e_112_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 23) = bselect (a (.wire 103 39)) (a (.virt 5408)) (a (.wire 111 39)) := by
  have c7407 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7408 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7409 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7410 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7411 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7412 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_4 := arithEq_of_rows h (row := 112) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_4
  simp only [← c7407, ← c7408, ← c7409] at e_112_4
  have e_112_5 := arithEq_of_rows h (row := 112) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_5
  simp only [← c7410, ← c7411, ← c7412] at e_112_5
  have hr := e_112_5
  simp only [e_112_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 31) = bselect (a (.wire 103 39)) (a (.virt 5409)) (a (.wire 111 47)) := by
  have c7413 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7414 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7415 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7416 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7417 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7418 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_6 := arithEq_of_rows h (row := 112) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_6
  simp only [← c7413, ← c7414, ← c7415] at e_112_6
  have e_112_7 := arithEq_of_rows h (row := 112) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_7
  simp only [← c7416, ← c7417, ← c7418] at e_112_7
  have hr := e_112_7
  simp only [e_112_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 39) = bselect (a (.wire 103 39)) (a (.virt 5403)) (a (.wire 111 55)) := by
  have c7419 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7420 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7421 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7422 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7423 := (publicBatchWrapper89_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7424 := (publicBatchWrapper89_copies232 a h).1
  have e_112_8 := arithEq_of_rows h (row := 112) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_8
  simp only [← c7419, ← c7420, ← c7421] at e_112_8
  have e_112_9 := arithEq_of_rows h (row := 112) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_9
  simp only [← c7422, ← c7423, ← c7424] at e_112_9
  have hr := e_112_9
  simp only [e_112_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 47) = bselect (a (.wire 103 39)) (a (.virt 5404)) (a (.wire 111 63)) := by
  have c7425 := (publicBatchWrapper89_copies232 a h).2.1
  have c7426 := (publicBatchWrapper89_copies232 a h).2.2.1
  have c7427 := (publicBatchWrapper89_copies232 a h).2.2.2.1
  have c7428 := (publicBatchWrapper89_copies232 a h).2.2.2.2.1
  have c7429 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.1
  have c7430 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.1
  have e_112_10 := arithEq_of_rows h (row := 112) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_10
  simp only [← c7425, ← c7426, ← c7427] at e_112_10
  have e_112_11 := arithEq_of_rows h (row := 112) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_11
  simp only [← c7428, ← c7429, ← c7430] at e_112_11
  have hr := e_112_11
  simp only [e_112_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 75) = bor (a (.wire 96 71)) (a (.wire 111 67)) := by
  have c7431 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.1
  have c7432 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.1
  have c7433 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.1
  have c7434 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7435 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7436 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_96_18 := arithEq_of_rows h (row := 96) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_18
  simp only [← c7434, ← c7435, k3, ← c7436] at e_96_18
  have e_107_6 := arithEq_of_rows h (row := 107) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_6
  simp only [← c7431, ← c7432, ← c7433] at e_107_6
  have hr := e_96_18
  simp only [e_107_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 51) = bnot (a (.wire 29 11)) := by
  have c7437 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7438 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7439 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_112_12 := arithEq_of_rows h (row := 112) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_12
  simp only [← c7437, k3, ← c7438, k3, ← c7439] at e_112_12
  have hr := e_112_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 55) = bnot (a (.wire 96 75)) := by
  have c7440 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7441 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7442 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_112_13 := arithEq_of_rows h (row := 112) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_13
  simp only [← c7440, k3, ← c7441, k3, ← c7442] at e_112_13
  have hr := e_112_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 43) = band (a (.wire 112 51)) (a (.wire 112 55)) := by
  have c7443 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7444 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7445 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_10 := arithEq_of_rows h (row := 103) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_10
  simp only [← c7443, ← c7444, ← c7445] at e_103_10
  have hr := e_103_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 63) = bselect (a (.wire 103 43)) (a (.virt 5605)) (a (.wire 111 79)) := by
  have c7446 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7447 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7448 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7449 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7450 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7451 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_14 := arithEq_of_rows h (row := 112) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_14
  simp only [← c7446, ← c7447, ← c7448] at e_112_14
  have e_112_15 := arithEq_of_rows h (row := 112) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_15
  simp only [← c7449, ← c7450, ← c7451] at e_112_15
  have hr := e_112_15
  simp only [e_112_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 71) = bselect (a (.wire 103 43)) (a (.virt 5606)) (a (.wire 112 7)) := by
  have c7452 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7453 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7454 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7455 := (publicBatchWrapper89_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7456 := (publicBatchWrapper89_copies233 a h).1
  have c7457 := (publicBatchWrapper89_copies233 a h).2.1
  have e_112_16 := arithEq_of_rows h (row := 112) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_16
  simp only [← c7452, ← c7453, ← c7454] at e_112_16
  have e_112_17 := arithEq_of_rows h (row := 112) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_17
  simp only [← c7455, ← c7456, ← c7457] at e_112_17
  have hr := e_112_17
  simp only [e_112_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 112 79) = bselect (a (.wire 103 43)) (a (.virt 5607)) (a (.wire 112 15)) := by
  have c7458 := (publicBatchWrapper89_copies233 a h).2.2.1
  have c7459 := (publicBatchWrapper89_copies233 a h).2.2.2.1
  have c7460 := (publicBatchWrapper89_copies233 a h).2.2.2.2.1
  have c7461 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.1
  have c7462 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.1
  have c7463 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.1
  have e_112_18 := arithEq_of_rows h (row := 112) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_18
  simp only [← c7458, ← c7459, ← c7460] at e_112_18
  have e_112_19 := arithEq_of_rows h (row := 112) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_19
  simp only [← c7461, ← c7462, ← c7463] at e_112_19
  have hr := e_112_19
  simp only [e_112_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 7) = bselect (a (.wire 103 43)) (a (.virt 5608)) (a (.wire 112 23)) := by
  have c7464 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.1
  have c7465 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.1
  have c7466 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7467 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7468 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7469 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_0 := arithEq_of_rows h (row := 113) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_0
  simp only [← c7464, ← c7465, ← c7466] at e_113_0
  have e_113_1 := arithEq_of_rows h (row := 113) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_1
  simp only [← c7467, ← c7468, ← c7469] at e_113_1
  have hr := e_113_1
  simp only [e_113_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 15) = bselect (a (.wire 103 43)) (a (.virt 5609)) (a (.wire 112 31)) := by
  have c7470 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7471 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7472 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7473 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7474 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7475 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_2 := arithEq_of_rows h (row := 113) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_2
  simp only [← c7470, ← c7471, ← c7472] at e_113_2
  have e_113_3 := arithEq_of_rows h (row := 113) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_3
  simp only [← c7473, ← c7474, ← c7475] at e_113_3
  have hr := e_113_3
  simp only [e_113_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 23) = bselect (a (.wire 103 43)) (a (.virt 5603)) (a (.wire 112 39)) := by
  have c7476 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7477 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7478 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7479 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7480 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7481 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_4 := arithEq_of_rows h (row := 113) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_4
  simp only [← c7476, ← c7477, ← c7478] at e_113_4
  have e_113_5 := arithEq_of_rows h (row := 113) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_5
  simp only [← c7479, ← c7480, ← c7481] at e_113_5
  have hr := e_113_5
  simp only [e_113_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 31) = bselect (a (.wire 103 43)) (a (.virt 5604)) (a (.wire 112 47)) := by
  have c7482 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7483 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7484 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7485 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7486 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7487 := (publicBatchWrapper89_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_113_6 := arithEq_of_rows h (row := 113) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_6
  simp only [← c7482, ← c7483, ← c7484] at e_113_6
  have e_113_7 := arithEq_of_rows h (row := 113) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_7
  simp only [← c7485, ← c7486, ← c7487] at e_113_7
  have hr := e_113_7
  simp only [e_113_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 96 79) = bor (a (.wire 96 75)) (a (.wire 112 51)) := by
  have c7488 := (publicBatchWrapper89_copies234 a h).1
  have c7489 := (publicBatchWrapper89_copies234 a h).2.1
  have c7490 := (publicBatchWrapper89_copies234 a h).2.2.1
  have c7491 := (publicBatchWrapper89_copies234 a h).2.2.2.1
  have c7492 := (publicBatchWrapper89_copies234 a h).2.2.2.2.1
  have c7493 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_96_19 := arithEq_of_rows h (row := 96) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_19
  simp only [← c7491, ← c7492, k3, ← c7493] at e_96_19
  have e_107_7 := arithEq_of_rows h (row := 107) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_7
  simp only [← c7488, ← c7489, ← c7490] at e_107_7
  have hr := e_96_19
  simp only [e_107_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 35) = bnot (a (.wire 29 55)) := by
  have c7494 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.1
  have c7495 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.1
  have c7496 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_113_8 := arithEq_of_rows h (row := 113) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_8
  simp only [← c7494, k3, ← c7495, k3, ← c7496] at e_113_8
  have hr := e_113_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 39) = bnot (a (.wire 96 79)) := by
  have c7497 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.1
  have c7498 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7499 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_113_9 := arithEq_of_rows h (row := 113) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_9
  simp only [← c7497, k3, ← c7498, k3, ← c7499] at e_113_9
  have hr := e_113_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 47) = band (a (.wire 113 35)) (a (.wire 113 39)) := by
  have c7500 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7501 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7502 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_11 := arithEq_of_rows h (row := 103) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_11
  simp only [← c7500, ← c7501, ← c7502] at e_103_11
  have hr := e_103_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 47) = bselect (a (.wire 103 47)) (a (.virt 5805)) (a (.wire 112 63)) := by
  have c7503 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7504 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7505 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7506 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7507 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7508 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_10 := arithEq_of_rows h (row := 113) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_10
  simp only [← c7503, ← c7504, ← c7505] at e_113_10
  have e_113_11 := arithEq_of_rows h (row := 113) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_11
  simp only [← c7506, ← c7507, ← c7508] at e_113_11
  have hr := e_113_11
  simp only [e_113_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 55) = bselect (a (.wire 103 47)) (a (.virt 5806)) (a (.wire 112 71)) := by
  have c7509 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7510 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7511 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7512 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7513 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7514 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_12 := arithEq_of_rows h (row := 113) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_12
  simp only [← c7509, ← c7510, ← c7511] at e_113_12
  have e_113_13 := arithEq_of_rows h (row := 113) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_13
  simp only [← c7512, ← c7513, ← c7514] at e_113_13
  have hr := e_113_13
  simp only [e_113_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 63) = bselect (a (.wire 103 47)) (a (.virt 5807)) (a (.wire 112 79)) := by
  have c7515 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7516 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7517 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7518 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7519 := (publicBatchWrapper89_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7520 := (publicBatchWrapper89_copies235 a h).1
  have e_113_14 := arithEq_of_rows h (row := 113) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_14
  simp only [← c7515, ← c7516, ← c7517] at e_113_14
  have e_113_15 := arithEq_of_rows h (row := 113) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_15
  simp only [← c7518, ← c7519, ← c7520] at e_113_15
  have hr := e_113_15
  simp only [e_113_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 71) = bselect (a (.wire 103 47)) (a (.virt 5808)) (a (.wire 113 7)) := by
  have c7521 := (publicBatchWrapper89_copies235 a h).2.1
  have c7522 := (publicBatchWrapper89_copies235 a h).2.2.1
  have c7523 := (publicBatchWrapper89_copies235 a h).2.2.2.1
  have c7524 := (publicBatchWrapper89_copies235 a h).2.2.2.2.1
  have c7525 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.1
  have c7526 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.1
  have e_113_16 := arithEq_of_rows h (row := 113) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_16
  simp only [← c7521, ← c7522, ← c7523] at e_113_16
  have e_113_17 := arithEq_of_rows h (row := 113) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_17
  simp only [← c7524, ← c7525, ← c7526] at e_113_17
  have hr := e_113_17
  simp only [e_113_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 113 79) = bselect (a (.wire 103 47)) (a (.virt 5809)) (a (.wire 113 15)) := by
  have c7527 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.1
  have c7528 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.1
  have c7529 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.1
  have c7530 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7531 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7532 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_18 := arithEq_of_rows h (row := 113) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_18
  simp only [← c7527, ← c7528, ← c7529] at e_113_18
  have e_113_19 := arithEq_of_rows h (row := 113) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_19
  simp only [← c7530, ← c7531, ← c7532] at e_113_19
  have hr := e_113_19
  simp only [e_113_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 7) = bselect (a (.wire 103 47)) (a (.virt 5803)) (a (.wire 113 23)) := by
  have c7533 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7534 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7535 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7536 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7537 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7538 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_0 := arithEq_of_rows h (row := 114) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_0
  simp only [← c7533, ← c7534, ← c7535] at e_114_0
  have e_114_1 := arithEq_of_rows h (row := 114) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_1
  simp only [← c7536, ← c7537, ← c7538] at e_114_1
  have hr := e_114_1
  simp only [e_114_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 15) = bselect (a (.wire 103 47)) (a (.virt 5804)) (a (.wire 113 31)) := by
  have c7539 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7540 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7541 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7542 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7543 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7544 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_2 := arithEq_of_rows h (row := 114) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_2
  simp only [← c7539, ← c7540, ← c7541] at e_114_2
  have e_114_3 := arithEq_of_rows h (row := 114) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_3
  simp only [← c7542, ← c7543, ← c7544] at e_114_3
  have hr := e_114_3
  simp only [e_114_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 3) = bor (a (.wire 96 79)) (a (.wire 113 35)) := by
  have c7545 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7546 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7547 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7548 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7549 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7550 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_8 := arithEq_of_rows h (row := 107) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_8
  simp only [← c7545, ← c7546, ← c7547] at e_107_8
  have e_115_0 := arithEq_of_rows h (row := 115) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_0
  simp only [← c7548, ← c7549, k3, ← c7550] at e_115_0
  have hr := e_115_0
  simp only [e_107_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 19) = bnot (a (.wire 31 19)) := by
  have c7551 := (publicBatchWrapper89_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7552 := (publicBatchWrapper89_copies236 a h).1
  have c7553 := (publicBatchWrapper89_copies236 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_114_4 := arithEq_of_rows h (row := 114) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_4
  simp only [← c7551, k3, ← c7552, k3, ← c7553] at e_114_4
  have hr := e_114_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 23) = bnot (a (.wire 115 3)) := by
  have c7554 := (publicBatchWrapper89_copies236 a h).2.2.1
  have c7555 := (publicBatchWrapper89_copies236 a h).2.2.2.1
  have c7556 := (publicBatchWrapper89_copies236 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_114_5 := arithEq_of_rows h (row := 114) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_5
  simp only [← c7554, k3, ← c7555, k3, ← c7556] at e_114_5
  have hr := e_114_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 51) = band (a (.wire 114 19)) (a (.wire 114 23)) := by
  have c7557 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.1
  have c7558 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.1
  have c7559 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.1
  have e_103_12 := arithEq_of_rows h (row := 103) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_12
  simp only [← c7557, ← c7558, ← c7559] at e_103_12
  have hr := e_103_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 31) = bselect (a (.wire 103 51)) (a (.virt 6005)) (a (.wire 113 47)) := by
  have c7560 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.1
  have c7561 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.1
  have c7562 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7563 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7564 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7565 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_6 := arithEq_of_rows h (row := 114) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_6
  simp only [← c7560, ← c7561, ← c7562] at e_114_6
  have e_114_7 := arithEq_of_rows h (row := 114) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_7
  simp only [← c7563, ← c7564, ← c7565] at e_114_7
  have hr := e_114_7
  simp only [e_114_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 39) = bselect (a (.wire 103 51)) (a (.virt 6006)) (a (.wire 113 55)) := by
  have c7566 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7567 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7568 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7569 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7570 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7571 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_8 := arithEq_of_rows h (row := 114) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_8
  simp only [← c7566, ← c7567, ← c7568] at e_114_8
  have e_114_9 := arithEq_of_rows h (row := 114) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_9
  simp only [← c7569, ← c7570, ← c7571] at e_114_9
  have hr := e_114_9
  simp only [e_114_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 47) = bselect (a (.wire 103 51)) (a (.virt 6007)) (a (.wire 113 63)) := by
  have c7572 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7573 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7574 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7575 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7576 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7577 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_10 := arithEq_of_rows h (row := 114) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_10
  simp only [← c7572, ← c7573, ← c7574] at e_114_10
  have e_114_11 := arithEq_of_rows h (row := 114) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_11
  simp only [← c7575, ← c7576, ← c7577] at e_114_11
  have hr := e_114_11
  simp only [e_114_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 55) = bselect (a (.wire 103 51)) (a (.virt 6008)) (a (.wire 113 71)) := by
  have c7578 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7579 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7580 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7581 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7582 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7583 := (publicBatchWrapper89_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_114_12 := arithEq_of_rows h (row := 114) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_12
  simp only [← c7578, ← c7579, ← c7580] at e_114_12
  have e_114_13 := arithEq_of_rows h (row := 114) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_13
  simp only [← c7581, ← c7582, ← c7583] at e_114_13
  have hr := e_114_13
  simp only [e_114_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 63) = bselect (a (.wire 103 51)) (a (.virt 6009)) (a (.wire 113 79)) := by
  have c7584 := (publicBatchWrapper89_copies237 a h).1
  have c7585 := (publicBatchWrapper89_copies237 a h).2.1
  have c7586 := (publicBatchWrapper89_copies237 a h).2.2.1
  have c7587 := (publicBatchWrapper89_copies237 a h).2.2.2.1
  have c7588 := (publicBatchWrapper89_copies237 a h).2.2.2.2.1
  have c7589 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.1
  have e_114_14 := arithEq_of_rows h (row := 114) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_14
  simp only [← c7584, ← c7585, ← c7586] at e_114_14
  have e_114_15 := arithEq_of_rows h (row := 114) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_15
  simp only [← c7587, ← c7588, ← c7589] at e_114_15
  have hr := e_114_15
  simp only [e_114_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 71) = bselect (a (.wire 103 51)) (a (.virt 6003)) (a (.wire 114 7)) := by
  have c7590 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.1
  have c7591 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.1
  have c7592 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.1
  have c7593 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.1
  have c7594 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7595 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_16 := arithEq_of_rows h (row := 114) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_16
  simp only [← c7590, ← c7591, ← c7592] at e_114_16
  have e_114_17 := arithEq_of_rows h (row := 114) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_17
  simp only [← c7593, ← c7594, ← c7595] at e_114_17
  have hr := e_114_17
  simp only [e_114_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 114 79) = bselect (a (.wire 103 51)) (a (.virt 6004)) (a (.wire 114 15)) := by
  have c7596 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7597 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7598 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7599 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7600 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7601 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_18 := arithEq_of_rows h (row := 114) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_18
  simp only [← c7596, ← c7597, ← c7598] at e_114_18
  have e_114_19 := arithEq_of_rows h (row := 114) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_19
  simp only [← c7599, ← c7600, ← c7601] at e_114_19
  have hr := e_114_19
  simp only [e_114_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 7) = bor (a (.wire 115 3)) (a (.wire 114 19)) := by
  have c7602 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7603 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7604 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7605 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7606 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7607 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_9 := arithEq_of_rows h (row := 107) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_9
  simp only [← c7602, ← c7603, ← c7604] at e_107_9
  have e_115_1 := arithEq_of_rows h (row := 115) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_1
  simp only [← c7605, ← c7606, k3, ← c7607] at e_115_1
  have hr := e_115_1
  simp only [e_107_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 3) = bnot (a (.wire 31 63)) := by
  have c7608 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7609 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7610 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_116_0 := arithEq_of_rows h (row := 116) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_0
  simp only [← c7608, k3, ← c7609, k3, ← c7610] at e_116_0
  have hr := e_116_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 7) = bnot (a (.wire 115 7)) := by
  have c7611 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7612 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7613 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_116_1 := arithEq_of_rows h (row := 116) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_1
  simp only [← c7611, k3, ← c7612, k3, ← c7613] at e_116_1
  have hr := e_116_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 55) = band (a (.wire 116 3)) (a (.wire 116 7)) := by
  have c7614 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7615 := (publicBatchWrapper89_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7616 := (publicBatchWrapper89_copies238 a h).1
  have e_103_13 := arithEq_of_rows h (row := 103) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_13
  simp only [← c7614, ← c7615, ← c7616] at e_103_13
  have hr := e_103_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 15) = bselect (a (.wire 103 55)) (a (.virt 6205)) (a (.wire 114 31)) := by
  have c7617 := (publicBatchWrapper89_copies238 a h).2.1
  have c7618 := (publicBatchWrapper89_copies238 a h).2.2.1
  have c7619 := (publicBatchWrapper89_copies238 a h).2.2.2.1
  have c7620 := (publicBatchWrapper89_copies238 a h).2.2.2.2.1
  have c7621 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.1
  have c7622 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.1
  have e_116_2 := arithEq_of_rows h (row := 116) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_2
  simp only [← c7617, ← c7618, ← c7619] at e_116_2
  have e_116_3 := arithEq_of_rows h (row := 116) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_3
  simp only [← c7620, ← c7621, ← c7622] at e_116_3
  have hr := e_116_3
  simp only [e_116_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 23) = bselect (a (.wire 103 55)) (a (.virt 6206)) (a (.wire 114 39)) := by
  have c7623 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.1
  have c7624 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.1
  have c7625 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.1
  have c7626 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7627 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7628 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_116_4 := arithEq_of_rows h (row := 116) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_4
  simp only [← c7623, ← c7624, ← c7625] at e_116_4
  have e_116_5 := arithEq_of_rows h (row := 116) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_5
  simp only [← c7626, ← c7627, ← c7628] at e_116_5
  have hr := e_116_5
  simp only [e_116_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 31) = bselect (a (.wire 103 55)) (a (.virt 6207)) (a (.wire 114 47)) := by
  have c7629 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7630 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7631 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7632 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7633 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7634 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_116_6 := arithEq_of_rows h (row := 116) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_6
  simp only [← c7629, ← c7630, ← c7631] at e_116_6
  have e_116_7 := arithEq_of_rows h (row := 116) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_7
  simp only [← c7632, ← c7633, ← c7634] at e_116_7
  have hr := e_116_7
  simp only [e_116_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 39) = bselect (a (.wire 103 55)) (a (.virt 6208)) (a (.wire 114 55)) := by
  have c7635 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7636 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7637 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7638 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7639 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7640 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_116_8 := arithEq_of_rows h (row := 116) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_8
  simp only [← c7635, ← c7636, ← c7637] at e_116_8
  have e_116_9 := arithEq_of_rows h (row := 116) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_9
  simp only [← c7638, ← c7639, ← c7640] at e_116_9
  have hr := e_116_9
  simp only [e_116_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 47) = bselect (a (.wire 103 55)) (a (.virt 6209)) (a (.wire 114 63)) := by
  have c7641 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7642 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7643 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7644 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7645 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7646 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_116_10 := arithEq_of_rows h (row := 116) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_10
  simp only [← c7641, ← c7642, ← c7643] at e_116_10
  have e_116_11 := arithEq_of_rows h (row := 116) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_11
  simp only [← c7644, ← c7645, ← c7646] at e_116_11
  have hr := e_116_11
  simp only [e_116_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 55) = bselect (a (.wire 103 55)) (a (.virt 6203)) (a (.wire 114 71)) := by
  have c7647 := (publicBatchWrapper89_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7648 := (publicBatchWrapper89_copies239 a h).1
  have c7649 := (publicBatchWrapper89_copies239 a h).2.1
  have c7650 := (publicBatchWrapper89_copies239 a h).2.2.1
  have c7651 := (publicBatchWrapper89_copies239 a h).2.2.2.1
  have c7652 := (publicBatchWrapper89_copies239 a h).2.2.2.2.1
  have e_116_12 := arithEq_of_rows h (row := 116) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_12
  simp only [← c7647, ← c7648, ← c7649] at e_116_12
  have e_116_13 := arithEq_of_rows h (row := 116) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_13
  simp only [← c7650, ← c7651, ← c7652] at e_116_13
  have hr := e_116_13
  simp only [e_116_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 63) = bselect (a (.wire 103 55)) (a (.virt 6204)) (a (.wire 114 79)) := by
  have c7653 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.1
  have c7654 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.1
  have c7655 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.1
  have c7656 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.1
  have c7657 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.1
  have c7658 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_116_14 := arithEq_of_rows h (row := 116) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_14
  simp only [← c7653, ← c7654, ← c7655] at e_116_14
  have e_116_15 := arithEq_of_rows h (row := 116) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_15
  simp only [← c7656, ← c7657, ← c7658] at e_116_15
  have hr := e_116_15
  simp only [e_116_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 11) = bor (a (.wire 115 7)) (a (.wire 116 3)) := by
  have c7659 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7660 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7661 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7662 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7663 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7664 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_10 := arithEq_of_rows h (row := 107) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_10
  simp only [← c7659, ← c7660, ← c7661] at e_107_10
  have e_115_2 := arithEq_of_rows h (row := 115) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_2
  simp only [← c7662, ← c7663, k3, ← c7664] at e_115_2
  have hr := e_115_2
  simp only [e_107_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 67) = bnot (a (.wire 33 27)) := by
  have c7665 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7666 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7667 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_116_16 := arithEq_of_rows h (row := 116) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_16
  simp only [← c7665, k3, ← c7666, k3, ← c7667] at e_116_16
  have hr := e_116_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 71) = bnot (a (.wire 115 11)) := by
  have c7668 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7669 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7670 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_116_17 := arithEq_of_rows h (row := 116) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_17
  simp only [← c7668, k3, ← c7669, k3, ← c7670] at e_116_17
  have hr := e_116_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 59) = band (a (.wire 116 67)) (a (.wire 116 71)) := by
  have c7671 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7672 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7673 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_14 := arithEq_of_rows h (row := 103) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_14
  simp only [← c7671, ← c7672, ← c7673] at e_103_14
  have hr := e_103_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 116 79) = bselect (a (.wire 103 59)) (a (.virt 6405)) (a (.wire 116 15)) := by
  have c7674 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7675 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7676 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7677 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7678 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7679 := (publicBatchWrapper89_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_116_18 := arithEq_of_rows h (row := 116) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_18
  simp only [← c7674, ← c7675, ← c7676] at e_116_18
  have e_116_19 := arithEq_of_rows h (row := 116) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_19
  simp only [← c7677, ← c7678, ← c7679] at e_116_19
  have hr := e_116_19
  simp only [e_116_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 7) = bselect (a (.wire 103 59)) (a (.virt 6406)) (a (.wire 116 23)) := by
  have c7680 := (publicBatchWrapper89_copies240 a h).1
  have c7681 := (publicBatchWrapper89_copies240 a h).2.1
  have c7682 := (publicBatchWrapper89_copies240 a h).2.2.1
  have c7683 := (publicBatchWrapper89_copies240 a h).2.2.2.1
  have c7684 := (publicBatchWrapper89_copies240 a h).2.2.2.2.1
  have c7685 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.1
  have e_117_0 := arithEq_of_rows h (row := 117) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_0
  simp only [← c7680, ← c7681, ← c7682] at e_117_0
  have e_117_1 := arithEq_of_rows h (row := 117) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_1
  simp only [← c7683, ← c7684, ← c7685] at e_117_1
  have hr := e_117_1
  simp only [e_117_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 15) = bselect (a (.wire 103 59)) (a (.virt 6407)) (a (.wire 116 31)) := by
  have c7686 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.1
  have c7687 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.1
  have c7688 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.1
  have c7689 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.1
  have c7690 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7691 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_2 := arithEq_of_rows h (row := 117) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_2
  simp only [← c7686, ← c7687, ← c7688] at e_117_2
  have e_117_3 := arithEq_of_rows h (row := 117) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_3
  simp only [← c7689, ← c7690, ← c7691] at e_117_3
  have hr := e_117_3
  simp only [e_117_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 23) = bselect (a (.wire 103 59)) (a (.virt 6408)) (a (.wire 116 39)) := by
  have c7692 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7693 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7694 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7695 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7696 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7697 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_4 := arithEq_of_rows h (row := 117) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_4
  simp only [← c7692, ← c7693, ← c7694] at e_117_4
  have e_117_5 := arithEq_of_rows h (row := 117) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_5
  simp only [← c7695, ← c7696, ← c7697] at e_117_5
  have hr := e_117_5
  simp only [e_117_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 31) = bselect (a (.wire 103 59)) (a (.virt 6409)) (a (.wire 116 47)) := by
  have c7698 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7699 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7700 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7701 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7702 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7703 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_6 := arithEq_of_rows h (row := 117) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_6
  simp only [← c7698, ← c7699, ← c7700] at e_117_6
  have e_117_7 := arithEq_of_rows h (row := 117) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_7
  simp only [← c7701, ← c7702, ← c7703] at e_117_7
  have hr := e_117_7
  simp only [e_117_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 39) = bselect (a (.wire 103 59)) (a (.virt 6403)) (a (.wire 116 55)) := by
  have c7704 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7705 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7706 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7707 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7708 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7709 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_8 := arithEq_of_rows h (row := 117) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_8
  simp only [← c7704, ← c7705, ← c7706] at e_117_8
  have e_117_9 := arithEq_of_rows h (row := 117) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_9
  simp only [← c7707, ← c7708, ← c7709] at e_117_9
  have hr := e_117_9
  simp only [e_117_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 47) = bselect (a (.wire 103 59)) (a (.virt 6404)) (a (.wire 116 63)) := by
  have c7710 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7711 := (publicBatchWrapper89_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7712 := (publicBatchWrapper89_copies241 a h).1
  have c7713 := (publicBatchWrapper89_copies241 a h).2.1
  have c7714 := (publicBatchWrapper89_copies241 a h).2.2.1
  have c7715 := (publicBatchWrapper89_copies241 a h).2.2.2.1
  have e_117_10 := arithEq_of_rows h (row := 117) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_10
  simp only [← c7710, ← c7711, ← c7712] at e_117_10
  have e_117_11 := arithEq_of_rows h (row := 117) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_11
  simp only [← c7713, ← c7714, ← c7715] at e_117_11
  have hr := e_117_11
  simp only [e_117_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 15) = bor (a (.wire 115 11)) (a (.wire 116 67)) := by
  have c7716 := (publicBatchWrapper89_copies241 a h).2.2.2.2.1
  have c7717 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.1
  have c7718 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.1
  have c7719 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.1
  have c7720 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.1
  have c7721 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_11 := arithEq_of_rows h (row := 107) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_11
  simp only [← c7716, ← c7717, ← c7718] at e_107_11
  have e_115_3 := arithEq_of_rows h (row := 115) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_3
  simp only [← c7719, ← c7720, k3, ← c7721] at e_115_3
  have hr := e_115_3
  simp only [e_107_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 51) = bnot (a (.wire 33 71)) := by
  have c7722 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7723 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7724 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_117_12 := arithEq_of_rows h (row := 117) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_12
  simp only [← c7722, k3, ← c7723, k3, ← c7724] at e_117_12
  have hr := e_117_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 55) = bnot (a (.wire 115 15)) := by
  have c7725 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7726 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7727 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_117_13 := arithEq_of_rows h (row := 117) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_13
  simp only [← c7725, k3, ← c7726, k3, ← c7727] at e_117_13
  have hr := e_117_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 63) = band (a (.wire 117 51)) (a (.wire 117 55)) := by
  have c7728 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7729 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7730 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_15 := arithEq_of_rows h (row := 103) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_15
  simp only [← c7728, ← c7729, ← c7730] at e_103_15
  have hr := e_103_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 63) = bselect (a (.wire 103 63)) (a (.virt 6605)) (a (.wire 116 79)) := by
  have c7731 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7732 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7733 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7734 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7735 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7736 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_14 := arithEq_of_rows h (row := 117) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_14
  simp only [← c7731, ← c7732, ← c7733] at e_117_14
  have e_117_15 := arithEq_of_rows h (row := 117) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_15
  simp only [← c7734, ← c7735, ← c7736] at e_117_15
  have hr := e_117_15
  simp only [e_117_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 71) = bselect (a (.wire 103 63)) (a (.virt 6606)) (a (.wire 117 7)) := by
  have c7737 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7738 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7739 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7740 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7741 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7742 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_16 := arithEq_of_rows h (row := 117) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_16
  simp only [← c7737, ← c7738, ← c7739] at e_117_16
  have e_117_17 := arithEq_of_rows h (row := 117) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_17
  simp only [← c7740, ← c7741, ← c7742] at e_117_17
  have hr := e_117_17
  simp only [e_117_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 117 79) = bselect (a (.wire 103 63)) (a (.virt 6607)) (a (.wire 117 15)) := by
  have c7743 := (publicBatchWrapper89_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7744 := (publicBatchWrapper89_copies242 a h).1
  have c7745 := (publicBatchWrapper89_copies242 a h).2.1
  have c7746 := (publicBatchWrapper89_copies242 a h).2.2.1
  have c7747 := (publicBatchWrapper89_copies242 a h).2.2.2.1
  have c7748 := (publicBatchWrapper89_copies242 a h).2.2.2.2.1
  have e_117_18 := arithEq_of_rows h (row := 117) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_18
  simp only [← c7743, ← c7744, ← c7745] at e_117_18
  have e_117_19 := arithEq_of_rows h (row := 117) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_19
  simp only [← c7746, ← c7747, ← c7748] at e_117_19
  have hr := e_117_19
  simp only [e_117_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 7) = bselect (a (.wire 103 63)) (a (.virt 6608)) (a (.wire 117 23)) := by
  have c7749 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.1
  have c7750 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.1
  have c7751 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.1
  have c7752 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.1
  have c7753 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.1
  have c7754 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_118_0 := arithEq_of_rows h (row := 118) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_0
  simp only [← c7749, ← c7750, ← c7751] at e_118_0
  have e_118_1 := arithEq_of_rows h (row := 118) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_1
  simp only [← c7752, ← c7753, ← c7754] at e_118_1
  have hr := e_118_1
  simp only [e_118_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 15) = bselect (a (.wire 103 63)) (a (.virt 6609)) (a (.wire 117 31)) := by
  have c7755 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7756 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7757 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7758 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7759 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7760 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_2 := arithEq_of_rows h (row := 118) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_2
  simp only [← c7755, ← c7756, ← c7757] at e_118_2
  have e_118_3 := arithEq_of_rows h (row := 118) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_3
  simp only [← c7758, ← c7759, ← c7760] at e_118_3
  have hr := e_118_3
  simp only [e_118_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 23) = bselect (a (.wire 103 63)) (a (.virt 6603)) (a (.wire 117 39)) := by
  have c7761 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7762 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7763 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7764 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7765 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7766 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_4 := arithEq_of_rows h (row := 118) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_4
  simp only [← c7761, ← c7762, ← c7763] at e_118_4
  have e_118_5 := arithEq_of_rows h (row := 118) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_5
  simp only [← c7764, ← c7765, ← c7766] at e_118_5
  have hr := e_118_5
  simp only [e_118_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 31) = bselect (a (.wire 103 63)) (a (.virt 6604)) (a (.wire 117 47)) := by
  have c7767 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7768 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7769 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7770 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7771 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7772 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_6 := arithEq_of_rows h (row := 118) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_6
  simp only [← c7767, ← c7768, ← c7769] at e_118_6
  have e_118_7 := arithEq_of_rows h (row := 118) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_7
  simp only [← c7770, ← c7771, ← c7772] at e_118_7
  have hr := e_118_7
  simp only [e_118_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 19) = bor (a (.wire 115 15)) (a (.wire 117 51)) := by
  have c7773 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7774 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7775 := (publicBatchWrapper89_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7776 := (publicBatchWrapper89_copies243 a h).1
  have c7777 := (publicBatchWrapper89_copies243 a h).2.1
  have c7778 := (publicBatchWrapper89_copies243 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_12 := arithEq_of_rows h (row := 107) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_12
  simp only [← c7773, ← c7774, ← c7775] at e_107_12
  have e_115_4 := arithEq_of_rows h (row := 115) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_4
  simp only [← c7776, ← c7777, k3, ← c7778] at e_115_4
  have hr := e_115_4
  simp only [e_107_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 35) = bnot (a (.wire 34 35)) := by
  have c7779 := (publicBatchWrapper89_copies243 a h).2.2.2.1
  have c7780 := (publicBatchWrapper89_copies243 a h).2.2.2.2.1
  have c7781 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_118_8 := arithEq_of_rows h (row := 118) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_8
  simp only [← c7779, k3, ← c7780, k3, ← c7781] at e_118_8
  have hr := e_118_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 39) = bnot (a (.wire 115 19)) := by
  have c7782 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.1
  have c7783 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.1
  have c7784 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_118_9 := arithEq_of_rows h (row := 118) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_9
  simp only [← c7782, k3, ← c7783, k3, ← c7784] at e_118_9
  have hr := e_118_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 67) = band (a (.wire 118 35)) (a (.wire 118 39)) := by
  have c7785 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.1
  have c7786 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7787 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_16 := arithEq_of_rows h (row := 103) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_16
  simp only [← c7785, ← c7786, ← c7787] at e_103_16
  have hr := e_103_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 47) = bselect (a (.wire 103 67)) (a (.virt 6805)) (a (.wire 117 63)) := by
  have c7788 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7789 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7790 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7791 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7792 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7793 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_10 := arithEq_of_rows h (row := 118) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_10
  simp only [← c7788, ← c7789, ← c7790] at e_118_10
  have e_118_11 := arithEq_of_rows h (row := 118) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_11
  simp only [← c7791, ← c7792, ← c7793] at e_118_11
  have hr := e_118_11
  simp only [e_118_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 55) = bselect (a (.wire 103 67)) (a (.virt 6806)) (a (.wire 117 71)) := by
  have c7794 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7795 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7796 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7797 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7798 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7799 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_12 := arithEq_of_rows h (row := 118) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_12
  simp only [← c7794, ← c7795, ← c7796] at e_118_12
  have e_118_13 := arithEq_of_rows h (row := 118) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_13
  simp only [← c7797, ← c7798, ← c7799] at e_118_13
  have hr := e_118_13
  simp only [e_118_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 63) = bselect (a (.wire 103 67)) (a (.virt 6807)) (a (.wire 117 79)) := by
  have c7800 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7801 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7802 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7803 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7804 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7805 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_14 := arithEq_of_rows h (row := 118) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_14
  simp only [← c7800, ← c7801, ← c7802] at e_118_14
  have e_118_15 := arithEq_of_rows h (row := 118) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_15
  simp only [← c7803, ← c7804, ← c7805] at e_118_15
  have hr := e_118_15
  simp only [e_118_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 71) = bselect (a (.wire 103 67)) (a (.virt 6808)) (a (.wire 118 7)) := by
  have c7806 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7807 := (publicBatchWrapper89_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7808 := (publicBatchWrapper89_copies244 a h).1
  have c7809 := (publicBatchWrapper89_copies244 a h).2.1
  have c7810 := (publicBatchWrapper89_copies244 a h).2.2.1
  have c7811 := (publicBatchWrapper89_copies244 a h).2.2.2.1
  have e_118_16 := arithEq_of_rows h (row := 118) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_16
  simp only [← c7806, ← c7807, ← c7808] at e_118_16
  have e_118_17 := arithEq_of_rows h (row := 118) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_17
  simp only [← c7809, ← c7810, ← c7811] at e_118_17
  have hr := e_118_17
  simp only [e_118_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 118 79) = bselect (a (.wire 103 67)) (a (.virt 6809)) (a (.wire 118 15)) := by
  have c7812 := (publicBatchWrapper89_copies244 a h).2.2.2.2.1
  have c7813 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.1
  have c7814 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.1
  have c7815 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.1
  have c7816 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.1
  have c7817 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.1
  have e_118_18 := arithEq_of_rows h (row := 118) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_18
  simp only [← c7812, ← c7813, ← c7814] at e_118_18
  have e_118_19 := arithEq_of_rows h (row := 118) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_19
  simp only [← c7815, ← c7816, ← c7817] at e_118_19
  have hr := e_118_19
  simp only [e_118_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 7) = bselect (a (.wire 103 67)) (a (.virt 6803)) (a (.wire 118 23)) := by
  have c7818 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7819 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7820 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7821 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7822 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7823 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_0 := arithEq_of_rows h (row := 119) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_0
  simp only [← c7818, ← c7819, ← c7820] at e_119_0
  have e_119_1 := arithEq_of_rows h (row := 119) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_1
  simp only [← c7821, ← c7822, ← c7823] at e_119_1
  have hr := e_119_1
  simp only [e_119_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 15) = bselect (a (.wire 103 67)) (a (.virt 6804)) (a (.wire 118 31)) := by
  have c7824 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7825 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7826 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7827 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7828 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7829 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_2 := arithEq_of_rows h (row := 119) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_2
  simp only [← c7824, ← c7825, ← c7826] at e_119_2
  have e_119_3 := arithEq_of_rows h (row := 119) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_3
  simp only [← c7827, ← c7828, ← c7829] at e_119_3
  have hr := e_119_3
  simp only [e_119_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 23) = bor (a (.wire 115 19)) (a (.wire 118 35)) := by
  have c7830 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7831 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7832 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7833 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7834 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7835 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_13 := arithEq_of_rows h (row := 107) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_13
  simp only [← c7830, ← c7831, ← c7832] at e_107_13
  have e_115_5 := arithEq_of_rows h (row := 115) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_5
  simp only [← c7833, ← c7834, k3, ← c7835] at e_115_5
  have hr := e_115_5
  simp only [e_107_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 19) = bnot (a (.wire 34 79)) := by
  have c7836 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7837 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7838 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_119_4 := arithEq_of_rows h (row := 119) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_4
  simp only [← c7836, k3, ← c7837, k3, ← c7838] at e_119_4
  have hr := e_119_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 23) = bnot (a (.wire 115 23)) := by
  have c7839 := (publicBatchWrapper89_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7840 := (publicBatchWrapper89_copies245 a h).1
  have c7841 := (publicBatchWrapper89_copies245 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_119_5 := arithEq_of_rows h (row := 119) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_5
  simp only [← c7839, k3, ← c7840, k3, ← c7841] at e_119_5
  have hr := e_119_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 71) = band (a (.wire 119 19)) (a (.wire 119 23)) := by
  have c7842 := (publicBatchWrapper89_copies245 a h).2.2.1
  have c7843 := (publicBatchWrapper89_copies245 a h).2.2.2.1
  have c7844 := (publicBatchWrapper89_copies245 a h).2.2.2.2.1
  have e_103_17 := arithEq_of_rows h (row := 103) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_17
  simp only [← c7842, ← c7843, ← c7844] at e_103_17
  have hr := e_103_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 31) = bselect (a (.wire 103 71)) (a (.virt 7005)) (a (.wire 118 47)) := by
  have c7845 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.1
  have c7846 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.1
  have c7847 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.1
  have c7848 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.1
  have c7849 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.1
  have c7850 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_119_6 := arithEq_of_rows h (row := 119) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_6
  simp only [← c7845, ← c7846, ← c7847] at e_119_6
  have e_119_7 := arithEq_of_rows h (row := 119) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_7
  simp only [← c7848, ← c7849, ← c7850] at e_119_7
  have hr := e_119_7
  simp only [e_119_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 39) = bselect (a (.wire 103 71)) (a (.virt 7006)) (a (.wire 118 55)) := by
  have c7851 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7852 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7853 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7854 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7855 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7856 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_8 := arithEq_of_rows h (row := 119) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_8
  simp only [← c7851, ← c7852, ← c7853] at e_119_8
  have e_119_9 := arithEq_of_rows h (row := 119) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_9
  simp only [← c7854, ← c7855, ← c7856] at e_119_9
  have hr := e_119_9
  simp only [e_119_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 47) = bselect (a (.wire 103 71)) (a (.virt 7007)) (a (.wire 118 63)) := by
  have c7857 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7858 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7859 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7860 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7861 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7862 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_10 := arithEq_of_rows h (row := 119) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_10
  simp only [← c7857, ← c7858, ← c7859] at e_119_10
  have e_119_11 := arithEq_of_rows h (row := 119) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_11
  simp only [← c7860, ← c7861, ← c7862] at e_119_11
  have hr := e_119_11
  simp only [e_119_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 55) = bselect (a (.wire 103 71)) (a (.virt 7008)) (a (.wire 118 71)) := by
  have c7863 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7864 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7865 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7866 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7867 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7868 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_12 := arithEq_of_rows h (row := 119) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_12
  simp only [← c7863, ← c7864, ← c7865] at e_119_12
  have e_119_13 := arithEq_of_rows h (row := 119) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_13
  simp only [← c7866, ← c7867, ← c7868] at e_119_13
  have hr := e_119_13
  simp only [e_119_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 63) = bselect (a (.wire 103 71)) (a (.virt 7009)) (a (.wire 118 79)) := by
  have c7869 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7870 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7871 := (publicBatchWrapper89_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7872 := (publicBatchWrapper89_copies246 a h).1
  have c7873 := (publicBatchWrapper89_copies246 a h).2.1
  have c7874 := (publicBatchWrapper89_copies246 a h).2.2.1
  have e_119_14 := arithEq_of_rows h (row := 119) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_14
  simp only [← c7869, ← c7870, ← c7871] at e_119_14
  have e_119_15 := arithEq_of_rows h (row := 119) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_15
  simp only [← c7872, ← c7873, ← c7874] at e_119_15
  have hr := e_119_15
  simp only [e_119_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 71) = bselect (a (.wire 103 71)) (a (.virt 7003)) (a (.wire 119 7)) := by
  have c7875 := (publicBatchWrapper89_copies246 a h).2.2.2.1
  have c7876 := (publicBatchWrapper89_copies246 a h).2.2.2.2.1
  have c7877 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.1
  have c7878 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.1
  have c7879 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.1
  have c7880 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.1
  have e_119_16 := arithEq_of_rows h (row := 119) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_16
  simp only [← c7875, ← c7876, ← c7877] at e_119_16
  have e_119_17 := arithEq_of_rows h (row := 119) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_17
  simp only [← c7878, ← c7879, ← c7880] at e_119_17
  have hr := e_119_17
  simp only [e_119_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 119 79) = bselect (a (.wire 103 71)) (a (.virt 7004)) (a (.wire 119 15)) := by
  have c7881 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.1
  have c7882 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7883 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7884 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7885 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7886 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_18 := arithEq_of_rows h (row := 119) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_18
  simp only [← c7881, ← c7882, ← c7883] at e_119_18
  have e_119_19 := arithEq_of_rows h (row := 119) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_19
  simp only [← c7884, ← c7885, ← c7886] at e_119_19
  have hr := e_119_19
  simp only [e_119_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 27) = bor (a (.wire 115 23)) (a (.wire 119 19)) := by
  have c7887 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7888 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7889 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7890 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7891 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7892 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_14 := arithEq_of_rows h (row := 107) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_14
  simp only [← c7887, ← c7888, ← c7889] at e_107_14
  have e_115_6 := arithEq_of_rows h (row := 115) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_6
  simp only [← c7890, ← c7891, k3, ← c7892] at e_115_6
  have hr := e_115_6
  simp only [e_107_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 3) = bnot (a (.wire 36 43)) := by
  have c7893 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7894 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7895 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_120_0 := arithEq_of_rows h (row := 120) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_0
  simp only [← c7893, k3, ← c7894, k3, ← c7895] at e_120_0
  have hr := e_120_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 7) = bnot (a (.wire 115 27)) := by
  have c7896 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7897 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7898 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_120_1 := arithEq_of_rows h (row := 120) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_1
  simp only [← c7896, k3, ← c7897, k3, ← c7898] at e_120_1
  have hr := e_120_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 75) = band (a (.wire 120 3)) (a (.wire 120 7)) := by
  have c7899 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7900 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7901 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_18 := arithEq_of_rows h (row := 103) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_18
  simp only [← c7899, ← c7900, ← c7901] at e_103_18
  have hr := e_103_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 15) = bselect (a (.wire 103 75)) (a (.virt 7205)) (a (.wire 119 31)) := by
  have c7902 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7903 := (publicBatchWrapper89_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7904 := (publicBatchWrapper89_copies247 a h).1
  have c7905 := (publicBatchWrapper89_copies247 a h).2.1
  have c7906 := (publicBatchWrapper89_copies247 a h).2.2.1
  have c7907 := (publicBatchWrapper89_copies247 a h).2.2.2.1
  have e_120_2 := arithEq_of_rows h (row := 120) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_2
  simp only [← c7902, ← c7903, ← c7904] at e_120_2
  have e_120_3 := arithEq_of_rows h (row := 120) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_3
  simp only [← c7905, ← c7906, ← c7907] at e_120_3
  have hr := e_120_3
  simp only [e_120_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 23) = bselect (a (.wire 103 75)) (a (.virt 7206)) (a (.wire 119 39)) := by
  have c7908 := (publicBatchWrapper89_copies247 a h).2.2.2.2.1
  have c7909 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.1
  have c7910 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.1
  have c7911 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.1
  have c7912 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.1
  have c7913 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.1
  have e_120_4 := arithEq_of_rows h (row := 120) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_4
  simp only [← c7908, ← c7909, ← c7910] at e_120_4
  have e_120_5 := arithEq_of_rows h (row := 120) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_5
  simp only [← c7911, ← c7912, ← c7913] at e_120_5
  have hr := e_120_5
  simp only [e_120_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 31) = bselect (a (.wire 103 75)) (a (.virt 7207)) (a (.wire 119 47)) := by
  have c7914 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7915 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7916 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7917 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7918 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7919 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_6 := arithEq_of_rows h (row := 120) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_6
  simp only [← c7914, ← c7915, ← c7916] at e_120_6
  have e_120_7 := arithEq_of_rows h (row := 120) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_7
  simp only [← c7917, ← c7918, ← c7919] at e_120_7
  have hr := e_120_7
  simp only [e_120_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 39) = bselect (a (.wire 103 75)) (a (.virt 7208)) (a (.wire 119 55)) := by
  have c7920 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7921 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7922 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7923 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7924 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7925 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_8 := arithEq_of_rows h (row := 120) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_8
  simp only [← c7920, ← c7921, ← c7922] at e_120_8
  have e_120_9 := arithEq_of_rows h (row := 120) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_9
  simp only [← c7923, ← c7924, ← c7925] at e_120_9
  have hr := e_120_9
  simp only [e_120_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 47) = bselect (a (.wire 103 75)) (a (.virt 7209)) (a (.wire 119 63)) := by
  have c7926 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7927 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7928 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7929 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7930 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7931 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_10 := arithEq_of_rows h (row := 120) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_10
  simp only [← c7926, ← c7927, ← c7928] at e_120_10
  have e_120_11 := arithEq_of_rows h (row := 120) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_11
  simp only [← c7929, ← c7930, ← c7931] at e_120_11
  have hr := e_120_11
  simp only [e_120_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 55) = bselect (a (.wire 103 75)) (a (.virt 7203)) (a (.wire 119 71)) := by
  have c7932 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7933 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7934 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7935 := (publicBatchWrapper89_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7936 := (publicBatchWrapper89_copies248 a h).1
  have c7937 := (publicBatchWrapper89_copies248 a h).2.1
  have e_120_12 := arithEq_of_rows h (row := 120) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_12
  simp only [← c7932, ← c7933, ← c7934] at e_120_12
  have e_120_13 := arithEq_of_rows h (row := 120) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_13
  simp only [← c7935, ← c7936, ← c7937] at e_120_13
  have hr := e_120_13
  simp only [e_120_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 63) = bselect (a (.wire 103 75)) (a (.virt 7204)) (a (.wire 119 79)) := by
  have c7938 := (publicBatchWrapper89_copies248 a h).2.2.1
  have c7939 := (publicBatchWrapper89_copies248 a h).2.2.2.1
  have c7940 := (publicBatchWrapper89_copies248 a h).2.2.2.2.1
  have c7941 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.1
  have c7942 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.1
  have c7943 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.1
  have e_120_14 := arithEq_of_rows h (row := 120) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_14
  simp only [← c7938, ← c7939, ← c7940] at e_120_14
  have e_120_15 := arithEq_of_rows h (row := 120) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_15
  simp only [← c7941, ← c7942, ← c7943] at e_120_15
  have hr := e_120_15
  simp only [e_120_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 31) = bor (a (.wire 115 27)) (a (.wire 120 3)) := by
  have c7944 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.1
  have c7945 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.1
  have c7946 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7947 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7948 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7949 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_15 := arithEq_of_rows h (row := 107) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_15
  simp only [← c7944, ← c7945, ← c7946] at e_107_15
  have e_115_7 := arithEq_of_rows h (row := 115) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_7
  simp only [← c7947, ← c7948, k3, ← c7949] at e_115_7
  have hr := e_115_7
  simp only [e_107_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 67) = bnot (a (.wire 38 7)) := by
  have c7950 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7951 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7952 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_120_16 := arithEq_of_rows h (row := 120) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_16
  simp only [← c7950, k3, ← c7951, k3, ← c7952] at e_120_16
  have hr := e_120_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 71) = bnot (a (.wire 115 31)) := by
  have c7953 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7954 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7955 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_120_17 := arithEq_of_rows h (row := 120) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_17
  simp only [← c7953, k3, ← c7954, k3, ← c7955] at e_120_17
  have hr := e_120_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 103 79) = band (a (.wire 120 67)) (a (.wire 120 71)) := by
  have c7956 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7957 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7958 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_19 := arithEq_of_rows h (row := 103) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_19
  simp only [← c7956, ← c7957, ← c7958] at e_103_19
  have hr := e_103_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 120 79) = bselect (a (.wire 103 79)) (a (.virt 7405)) (a (.wire 120 15)) := by
  have c7959 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7960 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7961 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7962 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7963 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7964 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_18 := arithEq_of_rows h (row := 120) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_18
  simp only [← c7959, ← c7960, ← c7961] at e_120_18
  have e_120_19 := arithEq_of_rows h (row := 120) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_19
  simp only [← c7962, ← c7963, ← c7964] at e_120_19
  have hr := e_120_19
  simp only [e_120_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 7) = bselect (a (.wire 103 79)) (a (.virt 7406)) (a (.wire 120 23)) := by
  have c7965 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7966 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7967 := (publicBatchWrapper89_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7968 := (publicBatchWrapper89_copies249 a h).1
  have c7969 := (publicBatchWrapper89_copies249 a h).2.1
  have c7970 := (publicBatchWrapper89_copies249 a h).2.2.1
  have e_121_0 := arithEq_of_rows h (row := 121) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_0
  simp only [← c7965, ← c7966, ← c7967] at e_121_0
  have e_121_1 := arithEq_of_rows h (row := 121) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_1
  simp only [← c7968, ← c7969, ← c7970] at e_121_1
  have hr := e_121_1
  simp only [e_121_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 15) = bselect (a (.wire 103 79)) (a (.virt 7407)) (a (.wire 120 31)) := by
  have c7971 := (publicBatchWrapper89_copies249 a h).2.2.2.1
  have c7972 := (publicBatchWrapper89_copies249 a h).2.2.2.2.1
  have c7973 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.1
  have c7974 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.1
  have c7975 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.1
  have c7976 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.1
  have e_121_2 := arithEq_of_rows h (row := 121) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_2
  simp only [← c7971, ← c7972, ← c7973] at e_121_2
  have e_121_3 := arithEq_of_rows h (row := 121) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_3
  simp only [← c7974, ← c7975, ← c7976] at e_121_3
  have hr := e_121_3
  simp only [e_121_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 23) = bselect (a (.wire 103 79)) (a (.virt 7408)) (a (.wire 120 39)) := by
  have c7977 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.1
  have c7978 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7979 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7980 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7981 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7982 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_4 := arithEq_of_rows h (row := 121) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_4
  simp only [← c7977, ← c7978, ← c7979] at e_121_4
  have e_121_5 := arithEq_of_rows h (row := 121) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_5
  simp only [← c7980, ← c7981, ← c7982] at e_121_5
  have hr := e_121_5
  simp only [e_121_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 31) = bselect (a (.wire 103 79)) (a (.virt 7409)) (a (.wire 120 47)) := by
  have c7983 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7984 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7985 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7986 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7987 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7988 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_6 := arithEq_of_rows h (row := 121) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_6
  simp only [← c7983, ← c7984, ← c7985] at e_121_6
  have e_121_7 := arithEq_of_rows h (row := 121) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_7
  simp only [← c7986, ← c7987, ← c7988] at e_121_7
  have hr := e_121_7
  simp only [e_121_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 39) = bselect (a (.wire 103 79)) (a (.virt 7403)) (a (.wire 120 55)) := by
  have c7989 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7990 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7991 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7992 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7993 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7994 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_8 := arithEq_of_rows h (row := 121) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_8
  simp only [← c7989, ← c7990, ← c7991] at e_121_8
  have e_121_9 := arithEq_of_rows h (row := 121) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_9
  simp only [← c7992, ← c7993, ← c7994] at e_121_9
  have hr := e_121_9
  simp only [e_121_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 47) = bselect (a (.wire 103 79)) (a (.virt 7404)) (a (.wire 120 63)) := by
  have c7995 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7996 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7997 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7998 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7999 := (publicBatchWrapper89_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8000 := (publicBatchWrapper89_copies250 a h).1
  have e_121_10 := arithEq_of_rows h (row := 121) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_10
  simp only [← c7995, ← c7996, ← c7997] at e_121_10
  have e_121_11 := arithEq_of_rows h (row := 121) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_11
  simp only [← c7998, ← c7999, ← c8000] at e_121_11
  have hr := e_121_11
  simp only [e_121_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 35) = bor (a (.wire 115 31)) (a (.wire 120 67)) := by
  have c8001 := (publicBatchWrapper89_copies250 a h).2.1
  have c8002 := (publicBatchWrapper89_copies250 a h).2.2.1
  have c8003 := (publicBatchWrapper89_copies250 a h).2.2.2.1
  have c8004 := (publicBatchWrapper89_copies250 a h).2.2.2.2.1
  have c8005 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.1
  have c8006 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_16 := arithEq_of_rows h (row := 107) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_16
  simp only [← c8001, ← c8002, ← c8003] at e_107_16
  have e_115_8 := arithEq_of_rows h (row := 115) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_8
  simp only [← c8004, ← c8005, k3, ← c8006] at e_115_8
  have hr := e_115_8
  simp only [e_107_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 51) = bnot (a (.wire 38 51)) := by
  have c8007 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.1
  have c8008 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.1
  have c8009 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_121_12 := arithEq_of_rows h (row := 121) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_12
  simp only [← c8007, k3, ← c8008, k3, ← c8009] at e_121_12
  have hr := e_121_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 55) = bnot (a (.wire 115 35)) := by
  have c8010 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8011 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8012 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_121_13 := arithEq_of_rows h (row := 121) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_13
  simp only [← c8010, k3, ← c8011, k3, ← c8012] at e_121_13
  have hr := e_121_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 3) = band (a (.wire 121 51)) (a (.wire 121 55)) := by
  have c8013 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8014 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8015 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_0 := arithEq_of_rows h (row := 122) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_0
  simp only [← c8013, ← c8014, ← c8015] at e_122_0
  have hr := e_122_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 63) = bselect (a (.wire 122 3)) (a (.virt 7605)) (a (.wire 120 79)) := by
  have c8016 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8017 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8018 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8019 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8020 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8021 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_14 := arithEq_of_rows h (row := 121) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_14
  simp only [← c8016, ← c8017, ← c8018] at e_121_14
  have e_121_15 := arithEq_of_rows h (row := 121) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_15
  simp only [← c8019, ← c8020, ← c8021] at e_121_15
  have hr := e_121_15
  simp only [e_121_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 71) = bselect (a (.wire 122 3)) (a (.virt 7606)) (a (.wire 121 7)) := by
  have c8022 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8023 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8024 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8025 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8026 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8027 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_16 := arithEq_of_rows h (row := 121) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_16
  simp only [← c8022, ← c8023, ← c8024] at e_121_16
  have e_121_17 := arithEq_of_rows h (row := 121) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_17
  simp only [← c8025, ← c8026, ← c8027] at e_121_17
  have hr := e_121_17
  simp only [e_121_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 121 79) = bselect (a (.wire 122 3)) (a (.virt 7607)) (a (.wire 121 15)) := by
  have c8028 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8029 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8030 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8031 := (publicBatchWrapper89_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8032 := (publicBatchWrapper89_copies251 a h).1
  have c8033 := (publicBatchWrapper89_copies251 a h).2.1
  have e_121_18 := arithEq_of_rows h (row := 121) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_18
  simp only [← c8028, ← c8029, ← c8030] at e_121_18
  have e_121_19 := arithEq_of_rows h (row := 121) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_19
  simp only [← c8031, ← c8032, ← c8033] at e_121_19
  have hr := e_121_19
  simp only [e_121_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 7) = bselect (a (.wire 122 3)) (a (.virt 7608)) (a (.wire 121 23)) := by
  have c8034 := (publicBatchWrapper89_copies251 a h).2.2.1
  have c8035 := (publicBatchWrapper89_copies251 a h).2.2.2.1
  have c8036 := (publicBatchWrapper89_copies251 a h).2.2.2.2.1
  have c8037 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.1
  have c8038 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.1
  have c8039 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.1
  have e_123_0 := arithEq_of_rows h (row := 123) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_0
  simp only [← c8034, ← c8035, ← c8036] at e_123_0
  have e_123_1 := arithEq_of_rows h (row := 123) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_1
  simp only [← c8037, ← c8038, ← c8039] at e_123_1
  have hr := e_123_1
  simp only [e_123_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 15) = bselect (a (.wire 122 3)) (a (.virt 7609)) (a (.wire 121 31)) := by
  have c8040 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.1
  have c8041 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.1
  have c8042 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8043 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8044 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8045 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_2 := arithEq_of_rows h (row := 123) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_2
  simp only [← c8040, ← c8041, ← c8042] at e_123_2
  have e_123_3 := arithEq_of_rows h (row := 123) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_3
  simp only [← c8043, ← c8044, ← c8045] at e_123_3
  have hr := e_123_3
  simp only [e_123_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 23) = bselect (a (.wire 122 3)) (a (.virt 7603)) (a (.wire 121 39)) := by
  have c8046 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8047 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8048 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8049 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8050 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8051 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_4 := arithEq_of_rows h (row := 123) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_4
  simp only [← c8046, ← c8047, ← c8048] at e_123_4
  have e_123_5 := arithEq_of_rows h (row := 123) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_5
  simp only [← c8049, ← c8050, ← c8051] at e_123_5
  have hr := e_123_5
  simp only [e_123_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 31) = bselect (a (.wire 122 3)) (a (.virt 7604)) (a (.wire 121 47)) := by
  have c8052 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8053 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8054 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8055 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8056 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8057 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_6 := arithEq_of_rows h (row := 123) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_6
  simp only [← c8052, ← c8053, ← c8054] at e_123_6
  have e_123_7 := arithEq_of_rows h (row := 123) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_7
  simp only [← c8055, ← c8056, ← c8057] at e_123_7
  have hr := e_123_7
  simp only [e_123_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 39) = bor (a (.wire 115 35)) (a (.wire 121 51)) := by
  have c8058 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8059 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8060 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8061 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8062 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8063 := (publicBatchWrapper89_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_17 := arithEq_of_rows h (row := 107) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_17
  simp only [← c8058, ← c8059, ← c8060] at e_107_17
  have e_115_9 := arithEq_of_rows h (row := 115) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_9
  simp only [← c8061, ← c8062, k3, ← c8063] at e_115_9
  have hr := e_115_9
  simp only [e_107_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 35) = bnot (a (.wire 39 15)) := by
  have c8064 := (publicBatchWrapper89_copies252 a h).1
  have c8065 := (publicBatchWrapper89_copies252 a h).2.1
  have c8066 := (publicBatchWrapper89_copies252 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_123_8 := arithEq_of_rows h (row := 123) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_8
  simp only [← c8064, k3, ← c8065, k3, ← c8066] at e_123_8
  have hr := e_123_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 39) = bnot (a (.wire 115 39)) := by
  have c8067 := (publicBatchWrapper89_copies252 a h).2.2.2.1
  have c8068 := (publicBatchWrapper89_copies252 a h).2.2.2.2.1
  have c8069 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_123_9 := arithEq_of_rows h (row := 123) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_9
  simp only [← c8067, k3, ← c8068, k3, ← c8069] at e_123_9
  have hr := e_123_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 7) = band (a (.wire 123 35)) (a (.wire 123 39)) := by
  have c8070 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.1
  have c8071 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.1
  have c8072 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.1
  have e_122_1 := arithEq_of_rows h (row := 122) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_1
  simp only [← c8070, ← c8071, ← c8072] at e_122_1
  have hr := e_122_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 47) = bselect (a (.wire 122 7)) (a (.virt 7805)) (a (.wire 121 63)) := by
  have c8073 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.1
  have c8074 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8075 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8076 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8077 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8078 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_10 := arithEq_of_rows h (row := 123) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_10
  simp only [← c8073, ← c8074, ← c8075] at e_123_10
  have e_123_11 := arithEq_of_rows h (row := 123) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_11
  simp only [← c8076, ← c8077, ← c8078] at e_123_11
  have hr := e_123_11
  simp only [e_123_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 55) = bselect (a (.wire 122 7)) (a (.virt 7806)) (a (.wire 121 71)) := by
  have c8079 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8080 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8081 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8082 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8083 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8084 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_12 := arithEq_of_rows h (row := 123) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_12
  simp only [← c8079, ← c8080, ← c8081] at e_123_12
  have e_123_13 := arithEq_of_rows h (row := 123) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_13
  simp only [← c8082, ← c8083, ← c8084] at e_123_13
  have hr := e_123_13
  simp only [e_123_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 63) = bselect (a (.wire 122 7)) (a (.virt 7807)) (a (.wire 121 79)) := by
  have c8085 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8086 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8087 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8088 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8089 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8090 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_14 := arithEq_of_rows h (row := 123) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_14
  simp only [← c8085, ← c8086, ← c8087] at e_123_14
  have e_123_15 := arithEq_of_rows h (row := 123) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_15
  simp only [← c8088, ← c8089, ← c8090] at e_123_15
  have hr := e_123_15
  simp only [e_123_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 71) = bselect (a (.wire 122 7)) (a (.virt 7808)) (a (.wire 123 7)) := by
  have c8091 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8092 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8093 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8094 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8095 := (publicBatchWrapper89_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8096 := (publicBatchWrapper89_copies253 a h).1
  have e_123_16 := arithEq_of_rows h (row := 123) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_16
  simp only [← c8091, ← c8092, ← c8093] at e_123_16
  have e_123_17 := arithEq_of_rows h (row := 123) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_17
  simp only [← c8094, ← c8095, ← c8096] at e_123_17
  have hr := e_123_17
  simp only [e_123_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 123 79) = bselect (a (.wire 122 7)) (a (.virt 7809)) (a (.wire 123 15)) := by
  have c8097 := (publicBatchWrapper89_copies253 a h).2.1
  have c8098 := (publicBatchWrapper89_copies253 a h).2.2.1
  have c8099 := (publicBatchWrapper89_copies253 a h).2.2.2.1
  have c8100 := (publicBatchWrapper89_copies253 a h).2.2.2.2.1
  have c8101 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.1
  have c8102 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.1
  have e_123_18 := arithEq_of_rows h (row := 123) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_18
  simp only [← c8097, ← c8098, ← c8099] at e_123_18
  have e_123_19 := arithEq_of_rows h (row := 123) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_19
  simp only [← c8100, ← c8101, ← c8102] at e_123_19
  have hr := e_123_19
  simp only [e_123_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 7) = bselect (a (.wire 122 7)) (a (.virt 7803)) (a (.wire 123 23)) := by
  have c8103 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.1
  have c8104 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.1
  have c8105 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.1
  have c8106 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8107 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8108 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_0 := arithEq_of_rows h (row := 124) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_0
  simp only [← c8103, ← c8104, ← c8105] at e_124_0
  have e_124_1 := arithEq_of_rows h (row := 124) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_1
  simp only [← c8106, ← c8107, ← c8108] at e_124_1
  have hr := e_124_1
  simp only [e_124_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 15) = bselect (a (.wire 122 7)) (a (.virt 7804)) (a (.wire 123 31)) := by
  have c8109 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8110 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8111 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8112 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8113 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8114 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_2 := arithEq_of_rows h (row := 124) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_2
  simp only [← c8109, ← c8110, ← c8111] at e_124_2
  have e_124_3 := arithEq_of_rows h (row := 124) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_3
  simp only [← c8112, ← c8113, ← c8114] at e_124_3
  have hr := e_124_3
  simp only [e_124_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 43) = bor (a (.wire 115 39)) (a (.wire 123 35)) := by
  have c8115 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8116 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8117 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8118 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8119 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8120 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_18 := arithEq_of_rows h (row := 107) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_18
  simp only [← c8115, ← c8116, ← c8117] at e_107_18
  have e_115_10 := arithEq_of_rows h (row := 115) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_10
  simp only [← c8118, ← c8119, k3, ← c8120] at e_115_10
  have hr := e_115_10
  simp only [e_107_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 19) = bnot (a (.wire 39 59)) := by
  have c8121 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8122 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8123 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_124_4 := arithEq_of_rows h (row := 124) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_4
  simp only [← c8121, k3, ← c8122, k3, ← c8123] at e_124_4
  have hr := e_124_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 23) = bnot (a (.wire 115 43)) := by
  have c8124 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8125 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8126 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_124_5 := arithEq_of_rows h (row := 124) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_5
  simp only [← c8124, k3, ← c8125, k3, ← c8126] at e_124_5
  have hr := e_124_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 11) = band (a (.wire 124 19)) (a (.wire 124 23)) := by
  have c8127 := (publicBatchWrapper89_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8128 := (publicBatchWrapper89_copies254 a h).1
  have c8129 := (publicBatchWrapper89_copies254 a h).2.1
  have e_122_2 := arithEq_of_rows h (row := 122) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_2
  simp only [← c8127, ← c8128, ← c8129] at e_122_2
  have hr := e_122_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 31) = bselect (a (.wire 122 11)) (a (.virt 8005)) (a (.wire 123 47)) := by
  have c8130 := (publicBatchWrapper89_copies254 a h).2.2.1
  have c8131 := (publicBatchWrapper89_copies254 a h).2.2.2.1
  have c8132 := (publicBatchWrapper89_copies254 a h).2.2.2.2.1
  have c8133 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.1
  have c8134 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.1
  have c8135 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.1
  have e_124_6 := arithEq_of_rows h (row := 124) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_6
  simp only [← c8130, ← c8131, ← c8132] at e_124_6
  have e_124_7 := arithEq_of_rows h (row := 124) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_7
  simp only [← c8133, ← c8134, ← c8135] at e_124_7
  have hr := e_124_7
  simp only [e_124_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 39) = bselect (a (.wire 122 11)) (a (.virt 8006)) (a (.wire 123 55)) := by
  have c8136 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.1
  have c8137 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.1
  have c8138 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8139 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8140 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8141 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_8 := arithEq_of_rows h (row := 124) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_8
  simp only [← c8136, ← c8137, ← c8138] at e_124_8
  have e_124_9 := arithEq_of_rows h (row := 124) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_9
  simp only [← c8139, ← c8140, ← c8141] at e_124_9
  have hr := e_124_9
  simp only [e_124_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 47) = bselect (a (.wire 122 11)) (a (.virt 8007)) (a (.wire 123 63)) := by
  have c8142 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8143 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8144 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8145 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8146 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8147 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_10 := arithEq_of_rows h (row := 124) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_10
  simp only [← c8142, ← c8143, ← c8144] at e_124_10
  have e_124_11 := arithEq_of_rows h (row := 124) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_11
  simp only [← c8145, ← c8146, ← c8147] at e_124_11
  have hr := e_124_11
  simp only [e_124_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 55) = bselect (a (.wire 122 11)) (a (.virt 8008)) (a (.wire 123 71)) := by
  have c8148 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8149 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8150 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8151 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8152 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8153 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_12 := arithEq_of_rows h (row := 124) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_12
  simp only [← c8148, ← c8149, ← c8150] at e_124_12
  have e_124_13 := arithEq_of_rows h (row := 124) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_13
  simp only [← c8151, ← c8152, ← c8153] at e_124_13
  have hr := e_124_13
  simp only [e_124_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 63) = bselect (a (.wire 122 11)) (a (.virt 8009)) (a (.wire 123 79)) := by
  have c8154 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8155 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8156 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8157 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8158 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8159 := (publicBatchWrapper89_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_124_14 := arithEq_of_rows h (row := 124) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_14
  simp only [← c8154, ← c8155, ← c8156] at e_124_14
  have e_124_15 := arithEq_of_rows h (row := 124) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_15
  simp only [← c8157, ← c8158, ← c8159] at e_124_15
  have hr := e_124_15
  simp only [e_124_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 71) = bselect (a (.wire 122 11)) (a (.virt 8003)) (a (.wire 124 7)) := by
  have c8160 := (publicBatchWrapper89_copies255 a h).1
  have c8161 := (publicBatchWrapper89_copies255 a h).2.1
  have c8162 := (publicBatchWrapper89_copies255 a h).2.2.1
  have c8163 := (publicBatchWrapper89_copies255 a h).2.2.2.1
  have c8164 := (publicBatchWrapper89_copies255 a h).2.2.2.2.1
  have c8165 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.1
  have e_124_16 := arithEq_of_rows h (row := 124) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_16
  simp only [← c8160, ← c8161, ← c8162] at e_124_16
  have e_124_17 := arithEq_of_rows h (row := 124) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_17
  simp only [← c8163, ← c8164, ← c8165] at e_124_17
  have hr := e_124_17
  simp only [e_124_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 124 79) = bselect (a (.wire 122 11)) (a (.virt 8004)) (a (.wire 124 15)) := by
  have c8166 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.1
  have c8167 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.1
  have c8168 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.1
  have c8169 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.1
  have c8170 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8171 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_124_18 := arithEq_of_rows h (row := 124) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_18
  simp only [← c8166, ← c8167, ← c8168] at e_124_18
  have e_124_19 := arithEq_of_rows h (row := 124) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_19
  simp only [← c8169, ← c8170, ← c8171] at e_124_19
  have hr := e_124_19
  simp only [e_124_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 47) = bor (a (.wire 115 43)) (a (.wire 124 19)) := by
  have c8172 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8173 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8174 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8175 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8176 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8177 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_107_19 := arithEq_of_rows h (row := 107) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_19
  simp only [← c8172, ← c8173, ← c8174] at e_107_19
  have e_115_11 := arithEq_of_rows h (row := 115) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_11
  simp only [← c8175, ← c8176, k3, ← c8177] at e_115_11
  have hr := e_115_11
  simp only [e_107_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 3) = bnot (a (.wire 41 23)) := by
  have c8178 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8179 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8180 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_125_0 := arithEq_of_rows h (row := 125) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_0
  simp only [← c8178, k3, ← c8179, k3, ← c8180] at e_125_0
  have hr := e_125_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 7) = bnot (a (.wire 115 47)) := by
  have c8181 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8182 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8183 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_125_1 := arithEq_of_rows h (row := 125) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_1
  simp only [← c8181, k3, ← c8182, k3, ← c8183] at e_125_1
  have hr := e_125_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 15) = band (a (.wire 125 3)) (a (.wire 125 7)) := by
  have c8184 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8185 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8186 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_3 := arithEq_of_rows h (row := 122) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_3
  simp only [← c8184, ← c8185, ← c8186] at e_122_3
  have hr := e_122_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 15) = bselect (a (.wire 122 15)) (a (.virt 8205)) (a (.wire 124 31)) := by
  have c8187 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8188 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8189 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8190 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8191 := (publicBatchWrapper89_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8192 := (publicBatchWrapper89_copies256 a h).1
  have e_125_2 := arithEq_of_rows h (row := 125) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_2
  simp only [← c8187, ← c8188, ← c8189] at e_125_2
  have e_125_3 := arithEq_of_rows h (row := 125) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_3
  simp only [← c8190, ← c8191, ← c8192] at e_125_3
  have hr := e_125_3
  simp only [e_125_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 23) = bselect (a (.wire 122 15)) (a (.virt 8206)) (a (.wire 124 39)) := by
  have c8193 := (publicBatchWrapper89_copies256 a h).2.1
  have c8194 := (publicBatchWrapper89_copies256 a h).2.2.1
  have c8195 := (publicBatchWrapper89_copies256 a h).2.2.2.1
  have c8196 := (publicBatchWrapper89_copies256 a h).2.2.2.2.1
  have c8197 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.1
  have c8198 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.1
  have e_125_4 := arithEq_of_rows h (row := 125) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_4
  simp only [← c8193, ← c8194, ← c8195] at e_125_4
  have e_125_5 := arithEq_of_rows h (row := 125) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_5
  simp only [← c8196, ← c8197, ← c8198] at e_125_5
  have hr := e_125_5
  simp only [e_125_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 31) = bselect (a (.wire 122 15)) (a (.virt 8207)) (a (.wire 124 47)) := by
  have c8199 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.1
  have c8200 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.1
  have c8201 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.1
  have c8202 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8203 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8204 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_6 := arithEq_of_rows h (row := 125) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_6
  simp only [← c8199, ← c8200, ← c8201] at e_125_6
  have e_125_7 := arithEq_of_rows h (row := 125) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_7
  simp only [← c8202, ← c8203, ← c8204] at e_125_7
  have hr := e_125_7
  simp only [e_125_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 39) = bselect (a (.wire 122 15)) (a (.virt 8208)) (a (.wire 124 55)) := by
  have c8205 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8206 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8207 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8208 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8209 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8210 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_8 := arithEq_of_rows h (row := 125) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_8
  simp only [← c8205, ← c8206, ← c8207] at e_125_8
  have e_125_9 := arithEq_of_rows h (row := 125) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_9
  simp only [← c8208, ← c8209, ← c8210] at e_125_9
  have hr := e_125_9
  simp only [e_125_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 47) = bselect (a (.wire 122 15)) (a (.virt 8209)) (a (.wire 124 63)) := by
  have c8211 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8212 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8213 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8214 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8215 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8216 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_10 := arithEq_of_rows h (row := 125) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_10
  simp only [← c8211, ← c8212, ← c8213] at e_125_10
  have e_125_11 := arithEq_of_rows h (row := 125) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_11
  simp only [← c8214, ← c8215, ← c8216] at e_125_11
  have hr := e_125_11
  simp only [e_125_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 55) = bselect (a (.wire 122 15)) (a (.virt 8203)) (a (.wire 124 71)) := by
  have c8217 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8218 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8219 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8220 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8221 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8222 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_12 := arithEq_of_rows h (row := 125) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_12
  simp only [← c8217, ← c8218, ← c8219] at e_125_12
  have e_125_13 := arithEq_of_rows h (row := 125) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_13
  simp only [← c8220, ← c8221, ← c8222] at e_125_13
  have hr := e_125_13
  simp only [e_125_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 63) = bselect (a (.wire 122 15)) (a (.virt 8204)) (a (.wire 124 79)) := by
  have c8223 := (publicBatchWrapper89_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8224 := (publicBatchWrapper89_copies257 a h).1
  have c8225 := (publicBatchWrapper89_copies257 a h).2.1
  have c8226 := (publicBatchWrapper89_copies257 a h).2.2.1
  have c8227 := (publicBatchWrapper89_copies257 a h).2.2.2.1
  have c8228 := (publicBatchWrapper89_copies257 a h).2.2.2.2.1
  have e_125_14 := arithEq_of_rows h (row := 125) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_14
  simp only [← c8223, ← c8224, ← c8225] at e_125_14
  have e_125_15 := arithEq_of_rows h (row := 125) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_15
  simp only [← c8226, ← c8227, ← c8228] at e_125_15
  have hr := e_125_15
  simp only [e_125_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 51) = bor (a (.wire 115 47)) (a (.wire 125 3)) := by
  have c8229 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.1
  have c8230 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.1
  have c8231 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.1
  have c8232 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.1
  have c8233 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.1
  have c8234 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_115_12 := arithEq_of_rows h (row := 115) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_12
  simp only [← c8232, ← c8233, k3, ← c8234] at e_115_12
  have e_126_0 := arithEq_of_rows h (row := 126) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_0
  simp only [← c8229, ← c8230, ← c8231] at e_126_0
  have hr := e_115_12
  simp only [e_126_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 67) = bnot (a (.wire 41 67)) := by
  have c8235 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8236 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8237 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_125_16 := arithEq_of_rows h (row := 125) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_16
  simp only [← c8235, k3, ← c8236, k3, ← c8237] at e_125_16
  have hr := e_125_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 71) = bnot (a (.wire 115 51)) := by
  have c8238 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8239 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8240 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_125_17 := arithEq_of_rows h (row := 125) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_17
  simp only [← c8238, k3, ← c8239, k3, ← c8240] at e_125_17
  have hr := e_125_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 19) = band (a (.wire 125 67)) (a (.wire 125 71)) := by
  have c8241 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8242 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8243 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_4 := arithEq_of_rows h (row := 122) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_4
  simp only [← c8241, ← c8242, ← c8243] at e_122_4
  have hr := e_122_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 125 79) = bselect (a (.wire 122 19)) (a (.virt 8405)) (a (.wire 125 15)) := by
  have c8244 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8245 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8246 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8247 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8248 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8249 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_18 := arithEq_of_rows h (row := 125) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_18
  simp only [← c8244, ← c8245, ← c8246] at e_125_18
  have e_125_19 := arithEq_of_rows h (row := 125) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_19
  simp only [← c8247, ← c8248, ← c8249] at e_125_19
  have hr := e_125_19
  simp only [e_125_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 7) = bselect (a (.wire 122 19)) (a (.virt 8406)) (a (.wire 125 23)) := by
  have c8250 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8251 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8252 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8253 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8254 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8255 := (publicBatchWrapper89_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_127_0 := arithEq_of_rows h (row := 127) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_0
  simp only [← c8250, ← c8251, ← c8252] at e_127_0
  have e_127_1 := arithEq_of_rows h (row := 127) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_1
  simp only [← c8253, ← c8254, ← c8255] at e_127_1
  have hr := e_127_1
  simp only [e_127_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 15) = bselect (a (.wire 122 19)) (a (.virt 8407)) (a (.wire 125 31)) := by
  have c8256 := (publicBatchWrapper89_copies258 a h).1
  have c8257 := (publicBatchWrapper89_copies258 a h).2.1
  have c8258 := (publicBatchWrapper89_copies258 a h).2.2.1
  have c8259 := (publicBatchWrapper89_copies258 a h).2.2.2.1
  have c8260 := (publicBatchWrapper89_copies258 a h).2.2.2.2.1
  have c8261 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.1
  have e_127_2 := arithEq_of_rows h (row := 127) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_2
  simp only [← c8256, ← c8257, ← c8258] at e_127_2
  have e_127_3 := arithEq_of_rows h (row := 127) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_3
  simp only [← c8259, ← c8260, ← c8261] at e_127_3
  have hr := e_127_3
  simp only [e_127_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 23) = bselect (a (.wire 122 19)) (a (.virt 8408)) (a (.wire 125 39)) := by
  have c8262 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.1
  have c8263 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.1
  have c8264 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.1
  have c8265 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.1
  have c8266 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8267 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_4 := arithEq_of_rows h (row := 127) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_4
  simp only [← c8262, ← c8263, ← c8264] at e_127_4
  have e_127_5 := arithEq_of_rows h (row := 127) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_5
  simp only [← c8265, ← c8266, ← c8267] at e_127_5
  have hr := e_127_5
  simp only [e_127_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 31) = bselect (a (.wire 122 19)) (a (.virt 8409)) (a (.wire 125 47)) := by
  have c8268 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8269 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8270 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8271 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8272 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8273 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_6 := arithEq_of_rows h (row := 127) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_6
  simp only [← c8268, ← c8269, ← c8270] at e_127_6
  have e_127_7 := arithEq_of_rows h (row := 127) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_7
  simp only [← c8271, ← c8272, ← c8273] at e_127_7
  have hr := e_127_7
  simp only [e_127_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 39) = bselect (a (.wire 122 19)) (a (.virt 8403)) (a (.wire 125 55)) := by
  have c8274 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8275 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8276 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8277 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8278 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8279 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_8 := arithEq_of_rows h (row := 127) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_8
  simp only [← c8274, ← c8275, ← c8276] at e_127_8
  have e_127_9 := arithEq_of_rows h (row := 127) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_9
  simp only [← c8277, ← c8278, ← c8279] at e_127_9
  have hr := e_127_9
  simp only [e_127_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 47) = bselect (a (.wire 122 19)) (a (.virt 8404)) (a (.wire 125 63)) := by
  have c8280 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8281 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8282 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8283 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8284 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8285 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_10 := arithEq_of_rows h (row := 127) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_10
  simp only [← c8280, ← c8281, ← c8282] at e_127_10
  have e_127_11 := arithEq_of_rows h (row := 127) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_11
  simp only [← c8283, ← c8284, ← c8285] at e_127_11
  have hr := e_127_11
  simp only [e_127_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 55) = bor (a (.wire 115 51)) (a (.wire 125 67)) := by
  have c8286 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8287 := (publicBatchWrapper89_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8288 := (publicBatchWrapper89_copies259 a h).1
  have c8289 := (publicBatchWrapper89_copies259 a h).2.1
  have c8290 := (publicBatchWrapper89_copies259 a h).2.2.1
  have c8291 := (publicBatchWrapper89_copies259 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_115_13 := arithEq_of_rows h (row := 115) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_13
  simp only [← c8289, ← c8290, k3, ← c8291] at e_115_13
  have e_126_1 := arithEq_of_rows h (row := 126) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_1
  simp only [← c8286, ← c8287, ← c8288] at e_126_1
  have hr := e_115_13
  simp only [e_126_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 51) = bnot (a (.wire 43 31)) := by
  have c8292 := (publicBatchWrapper89_copies259 a h).2.2.2.2.1
  have c8293 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.1
  have c8294 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_127_12 := arithEq_of_rows h (row := 127) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_12
  simp only [← c8292, k3, ← c8293, k3, ← c8294] at e_127_12
  have hr := e_127_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 55) = bnot (a (.wire 115 55)) := by
  have c8295 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.1
  have c8296 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.1
  have c8297 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_127_13 := arithEq_of_rows h (row := 127) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_13
  simp only [← c8295, k3, ← c8296, k3, ← c8297] at e_127_13
  have hr := e_127_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 23) = band (a (.wire 127 51)) (a (.wire 127 55)) := by
  have c8298 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8299 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8300 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_5 := arithEq_of_rows h (row := 122) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_5
  simp only [← c8298, ← c8299, ← c8300] at e_122_5
  have hr := e_122_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 63) = bselect (a (.wire 122 23)) (a (.virt 8605)) (a (.wire 125 79)) := by
  have c8301 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8302 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8303 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8304 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8305 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8306 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_14 := arithEq_of_rows h (row := 127) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_14
  simp only [← c8301, ← c8302, ← c8303] at e_127_14
  have e_127_15 := arithEq_of_rows h (row := 127) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_15
  simp only [← c8304, ← c8305, ← c8306] at e_127_15
  have hr := e_127_15
  simp only [e_127_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 71) = bselect (a (.wire 122 23)) (a (.virt 8606)) (a (.wire 127 7)) := by
  have c8307 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8308 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8309 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8310 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8311 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8312 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_16 := arithEq_of_rows h (row := 127) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_16
  simp only [← c8307, ← c8308, ← c8309] at e_127_16
  have e_127_17 := arithEq_of_rows h (row := 127) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_17
  simp only [← c8310, ← c8311, ← c8312] at e_127_17
  have hr := e_127_17
  simp only [e_127_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 127 79) = bselect (a (.wire 122 23)) (a (.virt 8607)) (a (.wire 127 15)) := by
  have c8313 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8314 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8315 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8316 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8317 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8318 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_18 := arithEq_of_rows h (row := 127) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_18
  simp only [← c8313, ← c8314, ← c8315] at e_127_18
  have e_127_19 := arithEq_of_rows h (row := 127) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_19
  simp only [← c8316, ← c8317, ← c8318] at e_127_19
  have hr := e_127_19
  simp only [e_127_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 7) = bselect (a (.wire 122 23)) (a (.virt 8608)) (a (.wire 127 23)) := by
  have c8319 := (publicBatchWrapper89_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8320 := (publicBatchWrapper89_copies260 a h).1
  have c8321 := (publicBatchWrapper89_copies260 a h).2.1
  have c8322 := (publicBatchWrapper89_copies260 a h).2.2.1
  have c8323 := (publicBatchWrapper89_copies260 a h).2.2.2.1
  have c8324 := (publicBatchWrapper89_copies260 a h).2.2.2.2.1
  have e_128_0 := arithEq_of_rows h (row := 128) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_0
  simp only [← c8319, ← c8320, ← c8321] at e_128_0
  have e_128_1 := arithEq_of_rows h (row := 128) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_1
  simp only [← c8322, ← c8323, ← c8324] at e_128_1
  have hr := e_128_1
  simp only [e_128_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 15) = bselect (a (.wire 122 23)) (a (.virt 8609)) (a (.wire 127 31)) := by
  have c8325 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.1
  have c8326 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.1
  have c8327 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.1
  have c8328 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.1
  have c8329 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.1
  have c8330 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_128_2 := arithEq_of_rows h (row := 128) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_2
  simp only [← c8325, ← c8326, ← c8327] at e_128_2
  have e_128_3 := arithEq_of_rows h (row := 128) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_3
  simp only [← c8328, ← c8329, ← c8330] at e_128_3
  have hr := e_128_3
  simp only [e_128_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 23) = bselect (a (.wire 122 23)) (a (.virt 8603)) (a (.wire 127 39)) := by
  have c8331 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8332 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8333 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8334 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8335 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8336 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_4 := arithEq_of_rows h (row := 128) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_4
  simp only [← c8331, ← c8332, ← c8333] at e_128_4
  have e_128_5 := arithEq_of_rows h (row := 128) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_5
  simp only [← c8334, ← c8335, ← c8336] at e_128_5
  have hr := e_128_5
  simp only [e_128_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 31) = bselect (a (.wire 122 23)) (a (.virt 8604)) (a (.wire 127 47)) := by
  have c8337 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8338 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8339 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8340 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8341 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8342 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_6 := arithEq_of_rows h (row := 128) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_6
  simp only [← c8337, ← c8338, ← c8339] at e_128_6
  have e_128_7 := arithEq_of_rows h (row := 128) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_7
  simp only [← c8340, ← c8341, ← c8342] at e_128_7
  have hr := e_128_7
  simp only [e_128_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 59) = bor (a (.wire 115 55)) (a (.wire 127 51)) := by
  have c8343 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8344 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8345 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8346 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8347 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8348 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_115_14 := arithEq_of_rows h (row := 115) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_14
  simp only [← c8346, ← c8347, k3, ← c8348] at e_115_14
  have e_126_2 := arithEq_of_rows h (row := 126) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_2
  simp only [← c8343, ← c8344, ← c8345] at e_126_2
  have hr := e_115_14
  simp only [e_126_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 35) = bnot (a (.wire 43 75)) := by
  have c8349 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8350 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8351 := (publicBatchWrapper89_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_128_8 := arithEq_of_rows h (row := 128) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_8
  simp only [← c8349, k3, ← c8350, k3, ← c8351] at e_128_8
  have hr := e_128_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 39) = bnot (a (.wire 115 59)) := by
  have c8352 := (publicBatchWrapper89_copies261 a h).1
  have c8353 := (publicBatchWrapper89_copies261 a h).2.1
  have c8354 := (publicBatchWrapper89_copies261 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_128_9 := arithEq_of_rows h (row := 128) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_9
  simp only [← c8352, k3, ← c8353, k3, ← c8354] at e_128_9
  have hr := e_128_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 27) = band (a (.wire 128 35)) (a (.wire 128 39)) := by
  have c8355 := (publicBatchWrapper89_copies261 a h).2.2.2.1
  have c8356 := (publicBatchWrapper89_copies261 a h).2.2.2.2.1
  have c8357 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.1
  have e_122_6 := arithEq_of_rows h (row := 122) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_6
  simp only [← c8355, ← c8356, ← c8357] at e_122_6
  have hr := e_122_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 47) = bselect (a (.wire 122 27)) (a (.virt 8805)) (a (.wire 127 63)) := by
  have c8358 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.1
  have c8359 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.1
  have c8360 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.1
  have c8361 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.1
  have c8362 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8363 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_10 := arithEq_of_rows h (row := 128) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_10
  simp only [← c8358, ← c8359, ← c8360] at e_128_10
  have e_128_11 := arithEq_of_rows h (row := 128) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_11
  simp only [← c8361, ← c8362, ← c8363] at e_128_11
  have hr := e_128_11
  simp only [e_128_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 55) = bselect (a (.wire 122 27)) (a (.virt 8806)) (a (.wire 127 71)) := by
  have c8364 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8365 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8366 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8367 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8368 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8369 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_12 := arithEq_of_rows h (row := 128) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_12
  simp only [← c8364, ← c8365, ← c8366] at e_128_12
  have e_128_13 := arithEq_of_rows h (row := 128) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_13
  simp only [← c8367, ← c8368, ← c8369] at e_128_13
  have hr := e_128_13
  simp only [e_128_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 63) = bselect (a (.wire 122 27)) (a (.virt 8807)) (a (.wire 127 79)) := by
  have c8370 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8371 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8372 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8373 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8374 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8375 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_14 := arithEq_of_rows h (row := 128) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_14
  simp only [← c8370, ← c8371, ← c8372] at e_128_14
  have e_128_15 := arithEq_of_rows h (row := 128) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_15
  simp only [← c8373, ← c8374, ← c8375] at e_128_15
  have hr := e_128_15
  simp only [e_128_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 71) = bselect (a (.wire 122 27)) (a (.virt 8808)) (a (.wire 128 7)) := by
  have c8376 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8377 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8378 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8379 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8380 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8381 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_16 := arithEq_of_rows h (row := 128) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_16
  simp only [← c8376, ← c8377, ← c8378] at e_128_16
  have e_128_17 := arithEq_of_rows h (row := 128) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_17
  simp only [← c8379, ← c8380, ← c8381] at e_128_17
  have hr := e_128_17
  simp only [e_128_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 128 79) = bselect (a (.wire 122 27)) (a (.virt 8809)) (a (.wire 128 15)) := by
  have c8382 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8383 := (publicBatchWrapper89_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8384 := (publicBatchWrapper89_copies262 a h).1
  have c8385 := (publicBatchWrapper89_copies262 a h).2.1
  have c8386 := (publicBatchWrapper89_copies262 a h).2.2.1
  have c8387 := (publicBatchWrapper89_copies262 a h).2.2.2.1
  have e_128_18 := arithEq_of_rows h (row := 128) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_18
  simp only [← c8382, ← c8383, ← c8384] at e_128_18
  have e_128_19 := arithEq_of_rows h (row := 128) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_19
  simp only [← c8385, ← c8386, ← c8387] at e_128_19
  have hr := e_128_19
  simp only [e_128_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 7) = bselect (a (.wire 122 27)) (a (.virt 8803)) (a (.wire 128 23)) := by
  have c8388 := (publicBatchWrapper89_copies262 a h).2.2.2.2.1
  have c8389 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.1
  have c8390 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.1
  have c8391 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.1
  have c8392 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.1
  have c8393 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.1
  have e_129_0 := arithEq_of_rows h (row := 129) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_0
  simp only [← c8388, ← c8389, ← c8390] at e_129_0
  have e_129_1 := arithEq_of_rows h (row := 129) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_1
  simp only [← c8391, ← c8392, ← c8393] at e_129_1
  have hr := e_129_1
  simp only [e_129_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 15) = bselect (a (.wire 122 27)) (a (.virt 8804)) (a (.wire 128 31)) := by
  have c8394 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8395 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8396 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8397 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8398 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8399 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_2 := arithEq_of_rows h (row := 129) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_2
  simp only [← c8394, ← c8395, ← c8396] at e_129_2
  have e_129_3 := arithEq_of_rows h (row := 129) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_3
  simp only [← c8397, ← c8398, ← c8399] at e_129_3
  have hr := e_129_3
  simp only [e_129_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 63) = bor (a (.wire 115 59)) (a (.wire 128 35)) := by
  have c8400 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8401 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8402 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8403 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8404 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8405 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_115_15 := arithEq_of_rows h (row := 115) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_15
  simp only [← c8403, ← c8404, k3, ← c8405] at e_115_15
  have e_126_3 := arithEq_of_rows h (row := 126) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_3
  simp only [← c8400, ← c8401, ← c8402] at e_126_3
  have hr := e_115_15
  simp only [e_126_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 19) = bnot (a (.wire 45 39)) := by
  have c8406 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8407 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8408 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_129_4 := arithEq_of_rows h (row := 129) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_4
  simp only [← c8406, k3, ← c8407, k3, ← c8408] at e_129_4
  have hr := e_129_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 23) = bnot (a (.wire 115 63)) := by
  have c8409 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8410 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8411 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_129_5 := arithEq_of_rows h (row := 129) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_5
  simp only [← c8409, k3, ← c8410, k3, ← c8411] at e_129_5
  have hr := e_129_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 31) = band (a (.wire 129 19)) (a (.wire 129 23)) := by
  have c8412 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8413 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8414 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_7 := arithEq_of_rows h (row := 122) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_7
  simp only [← c8412, ← c8413, ← c8414] at e_122_7
  have hr := e_122_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 31) = bselect (a (.wire 122 31)) (a (.virt 9005)) (a (.wire 128 47)) := by
  have c8415 := (publicBatchWrapper89_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8416 := (publicBatchWrapper89_copies263 a h).1
  have c8417 := (publicBatchWrapper89_copies263 a h).2.1
  have c8418 := (publicBatchWrapper89_copies263 a h).2.2.1
  have c8419 := (publicBatchWrapper89_copies263 a h).2.2.2.1
  have c8420 := (publicBatchWrapper89_copies263 a h).2.2.2.2.1
  have e_129_6 := arithEq_of_rows h (row := 129) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_6
  simp only [← c8415, ← c8416, ← c8417] at e_129_6
  have e_129_7 := arithEq_of_rows h (row := 129) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_7
  simp only [← c8418, ← c8419, ← c8420] at e_129_7
  have hr := e_129_7
  simp only [e_129_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 39) = bselect (a (.wire 122 31)) (a (.virt 9006)) (a (.wire 128 55)) := by
  have c8421 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.1
  have c8422 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.1
  have c8423 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.1
  have c8424 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.1
  have c8425 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.1
  have c8426 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_129_8 := arithEq_of_rows h (row := 129) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_8
  simp only [← c8421, ← c8422, ← c8423] at e_129_8
  have e_129_9 := arithEq_of_rows h (row := 129) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_9
  simp only [← c8424, ← c8425, ← c8426] at e_129_9
  have hr := e_129_9
  simp only [e_129_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 47) = bselect (a (.wire 122 31)) (a (.virt 9007)) (a (.wire 128 63)) := by
  have c8427 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8428 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8429 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8430 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8431 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8432 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_10 := arithEq_of_rows h (row := 129) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_10
  simp only [← c8427, ← c8428, ← c8429] at e_129_10
  have e_129_11 := arithEq_of_rows h (row := 129) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_11
  simp only [← c8430, ← c8431, ← c8432] at e_129_11
  have hr := e_129_11
  simp only [e_129_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 55) = bselect (a (.wire 122 31)) (a (.virt 9008)) (a (.wire 128 71)) := by
  have c8433 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8434 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8435 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8436 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8437 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8438 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_12 := arithEq_of_rows h (row := 129) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_12
  simp only [← c8433, ← c8434, ← c8435] at e_129_12
  have e_129_13 := arithEq_of_rows h (row := 129) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_13
  simp only [← c8436, ← c8437, ← c8438] at e_129_13
  have hr := e_129_13
  simp only [e_129_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 63) = bselect (a (.wire 122 31)) (a (.virt 9009)) (a (.wire 128 79)) := by
  have c8439 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8440 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8441 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8442 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8443 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8444 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_14 := arithEq_of_rows h (row := 129) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_14
  simp only [← c8439, ← c8440, ← c8441] at e_129_14
  have e_129_15 := arithEq_of_rows h (row := 129) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_15
  simp only [← c8442, ← c8443, ← c8444] at e_129_15
  have hr := e_129_15
  simp only [e_129_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 71) = bselect (a (.wire 122 31)) (a (.virt 9003)) (a (.wire 129 7)) := by
  have c8445 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8446 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8447 := (publicBatchWrapper89_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8448 := (publicBatchWrapper89_copies264 a h).1
  have c8449 := (publicBatchWrapper89_copies264 a h).2.1
  have c8450 := (publicBatchWrapper89_copies264 a h).2.2.1
  have e_129_16 := arithEq_of_rows h (row := 129) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_16
  simp only [← c8445, ← c8446, ← c8447] at e_129_16
  have e_129_17 := arithEq_of_rows h (row := 129) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_17
  simp only [← c8448, ← c8449, ← c8450] at e_129_17
  have hr := e_129_17
  simp only [e_129_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 129 79) = bselect (a (.wire 122 31)) (a (.virt 9004)) (a (.wire 129 15)) := by
  have c8451 := (publicBatchWrapper89_copies264 a h).2.2.2.1
  have c8452 := (publicBatchWrapper89_copies264 a h).2.2.2.2.1
  have c8453 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.1
  have c8454 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.1
  have c8455 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.1
  have c8456 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.1
  have e_129_18 := arithEq_of_rows h (row := 129) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_18
  simp only [← c8451, ← c8452, ← c8453] at e_129_18
  have e_129_19 := arithEq_of_rows h (row := 129) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_19
  simp only [← c8454, ← c8455, ← c8456] at e_129_19
  have hr := e_129_19
  simp only [e_129_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 67) = bor (a (.wire 115 63)) (a (.wire 129 19)) := by
  have c8457 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.1
  have c8458 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8459 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8460 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8461 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8462 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_115_16 := arithEq_of_rows h (row := 115) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_16
  simp only [← c8460, ← c8461, k3, ← c8462] at e_115_16
  have e_126_4 := arithEq_of_rows h (row := 126) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_4
  simp only [← c8457, ← c8458, ← c8459] at e_126_4
  have hr := e_115_16
  simp only [e_126_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 3) = bnot (a (.wire 46 3)) := by
  have c8463 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8464 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8465 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_130_0 := arithEq_of_rows h (row := 130) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_0
  simp only [← c8463, k3, ← c8464, k3, ← c8465] at e_130_0
  have hr := e_130_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 7) = bnot (a (.wire 115 67)) := by
  have c8466 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8467 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8468 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_130_1 := arithEq_of_rows h (row := 130) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_1
  simp only [← c8466, k3, ← c8467, k3, ← c8468] at e_130_1
  have hr := e_130_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 35) = band (a (.wire 130 3)) (a (.wire 130 7)) := by
  have c8469 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8470 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8471 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_8 := arithEq_of_rows h (row := 122) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_8
  simp only [← c8469, ← c8470, ← c8471] at e_122_8
  have hr := e_122_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 15) = bselect (a (.wire 122 35)) (a (.virt 9205)) (a (.wire 129 31)) := by
  have c8472 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8473 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8474 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8475 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8476 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8477 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_130_2 := arithEq_of_rows h (row := 130) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_2
  simp only [← c8472, ← c8473, ← c8474] at e_130_2
  have e_130_3 := arithEq_of_rows h (row := 130) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_3
  simp only [← c8475, ← c8476, ← c8477] at e_130_3
  have hr := e_130_3
  simp only [e_130_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 23) = bselect (a (.wire 122 35)) (a (.virt 9206)) (a (.wire 129 39)) := by
  have c8478 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8479 := (publicBatchWrapper89_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8480 := (publicBatchWrapper89_copies265 a h).1
  have c8481 := (publicBatchWrapper89_copies265 a h).2.1
  have c8482 := (publicBatchWrapper89_copies265 a h).2.2.1
  have c8483 := (publicBatchWrapper89_copies265 a h).2.2.2.1
  have e_130_4 := arithEq_of_rows h (row := 130) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_4
  simp only [← c8478, ← c8479, ← c8480] at e_130_4
  have e_130_5 := arithEq_of_rows h (row := 130) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_5
  simp only [← c8481, ← c8482, ← c8483] at e_130_5
  have hr := e_130_5
  simp only [e_130_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 31) = bselect (a (.wire 122 35)) (a (.virt 9207)) (a (.wire 129 47)) := by
  have c8484 := (publicBatchWrapper89_copies265 a h).2.2.2.2.1
  have c8485 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.1
  have c8486 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.1
  have c8487 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.1
  have c8488 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.1
  have c8489 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.1
  have e_130_6 := arithEq_of_rows h (row := 130) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_6
  simp only [← c8484, ← c8485, ← c8486] at e_130_6
  have e_130_7 := arithEq_of_rows h (row := 130) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_7
  simp only [← c8487, ← c8488, ← c8489] at e_130_7
  have hr := e_130_7
  simp only [e_130_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 39) = bselect (a (.wire 122 35)) (a (.virt 9208)) (a (.wire 129 55)) := by
  have c8490 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8491 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8492 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8493 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8494 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8495 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_130_8 := arithEq_of_rows h (row := 130) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_8
  simp only [← c8490, ← c8491, ← c8492] at e_130_8
  have e_130_9 := arithEq_of_rows h (row := 130) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_9
  simp only [← c8493, ← c8494, ← c8495] at e_130_9
  have hr := e_130_9
  simp only [e_130_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 47) = bselect (a (.wire 122 35)) (a (.virt 9209)) (a (.wire 129 63)) := by
  have c8496 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8497 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8498 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8499 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8500 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8501 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_130_10 := arithEq_of_rows h (row := 130) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_10
  simp only [← c8496, ← c8497, ← c8498] at e_130_10
  have e_130_11 := arithEq_of_rows h (row := 130) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_11
  simp only [← c8499, ← c8500, ← c8501] at e_130_11
  have hr := e_130_11
  simp only [e_130_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 55) = bselect (a (.wire 122 35)) (a (.virt 9203)) (a (.wire 129 71)) := by
  have c8502 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8503 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8504 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8505 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8506 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8507 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_130_12 := arithEq_of_rows h (row := 130) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_12
  simp only [← c8502, ← c8503, ← c8504] at e_130_12
  have e_130_13 := arithEq_of_rows h (row := 130) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_13
  simp only [← c8505, ← c8506, ← c8507] at e_130_13
  have hr := e_130_13
  simp only [e_130_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 63) = bselect (a (.wire 122 35)) (a (.virt 9204)) (a (.wire 129 79)) := by
  have c8508 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8509 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8510 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8511 := (publicBatchWrapper89_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8512 := (publicBatchWrapper89_copies266 a h).1
  have c8513 := (publicBatchWrapper89_copies266 a h).2.1
  have e_130_14 := arithEq_of_rows h (row := 130) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_14
  simp only [← c8508, ← c8509, ← c8510] at e_130_14
  have e_130_15 := arithEq_of_rows h (row := 130) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_15
  simp only [← c8511, ← c8512, ← c8513] at e_130_15
  have hr := e_130_15
  simp only [e_130_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 71) = bor (a (.wire 115 67)) (a (.wire 130 3)) := by
  have c8514 := (publicBatchWrapper89_copies266 a h).2.2.1
  have c8515 := (publicBatchWrapper89_copies266 a h).2.2.2.1
  have c8516 := (publicBatchWrapper89_copies266 a h).2.2.2.2.1
  have c8517 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.1
  have c8518 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.1
  have c8519 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_115_17 := arithEq_of_rows h (row := 115) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_17
  simp only [← c8517, ← c8518, k3, ← c8519] at e_115_17
  have e_126_5 := arithEq_of_rows h (row := 126) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_5
  simp only [← c8514, ← c8515, ← c8516] at e_126_5
  have hr := e_115_17
  simp only [e_126_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 67) = bnot (a (.wire 46 47)) := by
  have c8520 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.1
  have c8521 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.1
  have c8522 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_130_16 := arithEq_of_rows h (row := 130) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_16
  simp only [← c8520, k3, ← c8521, k3, ← c8522] at e_130_16
  have hr := e_130_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 71) = bnot (a (.wire 115 71)) := by
  have c8523 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8524 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8525 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_130_17 := arithEq_of_rows h (row := 130) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_17
  simp only [← c8523, k3, ← c8524, k3, ← c8525] at e_130_17
  have hr := e_130_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 39) = band (a (.wire 130 67)) (a (.wire 130 71)) := by
  have c8526 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8527 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8528 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_9 := arithEq_of_rows h (row := 122) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_9
  simp only [← c8526, ← c8527, ← c8528] at e_122_9
  have hr := e_122_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 130 79) = bselect (a (.wire 122 39)) (a (.virt 9405)) (a (.wire 130 15)) := by
  have c8529 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8530 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8531 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8532 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8533 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8534 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_130_18 := arithEq_of_rows h (row := 130) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_18
  simp only [← c8529, ← c8530, ← c8531] at e_130_18
  have e_130_19 := arithEq_of_rows h (row := 130) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_19
  simp only [← c8532, ← c8533, ← c8534] at e_130_19
  have hr := e_130_19
  simp only [e_130_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 7) = bselect (a (.wire 122 39)) (a (.virt 9406)) (a (.wire 130 23)) := by
  have c8535 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8536 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8537 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8538 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8539 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8540 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_0 := arithEq_of_rows h (row := 131) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_0
  simp only [← c8535, ← c8536, ← c8537] at e_131_0
  have e_131_1 := arithEq_of_rows h (row := 131) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_1
  simp only [← c8538, ← c8539, ← c8540] at e_131_1
  have hr := e_131_1
  simp only [e_131_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 15) = bselect (a (.wire 122 39)) (a (.virt 9407)) (a (.wire 130 31)) := by
  have c8541 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8542 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8543 := (publicBatchWrapper89_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8544 := (publicBatchWrapper89_copies267 a h).1
  have c8545 := (publicBatchWrapper89_copies267 a h).2.1
  have c8546 := (publicBatchWrapper89_copies267 a h).2.2.1
  have e_131_2 := arithEq_of_rows h (row := 131) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_2
  simp only [← c8541, ← c8542, ← c8543] at e_131_2
  have e_131_3 := arithEq_of_rows h (row := 131) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_3
  simp only [← c8544, ← c8545, ← c8546] at e_131_3
  have hr := e_131_3
  simp only [e_131_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 23) = bselect (a (.wire 122 39)) (a (.virt 9408)) (a (.wire 130 39)) := by
  have c8547 := (publicBatchWrapper89_copies267 a h).2.2.2.1
  have c8548 := (publicBatchWrapper89_copies267 a h).2.2.2.2.1
  have c8549 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.1
  have c8550 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.1
  have c8551 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.1
  have c8552 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.1
  have e_131_4 := arithEq_of_rows h (row := 131) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_4
  simp only [← c8547, ← c8548, ← c8549] at e_131_4
  have e_131_5 := arithEq_of_rows h (row := 131) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_5
  simp only [← c8550, ← c8551, ← c8552] at e_131_5
  have hr := e_131_5
  simp only [e_131_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 31) = bselect (a (.wire 122 39)) (a (.virt 9409)) (a (.wire 130 47)) := by
  have c8553 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.1
  have c8554 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8555 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8556 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8557 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8558 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_6 := arithEq_of_rows h (row := 131) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_6
  simp only [← c8553, ← c8554, ← c8555] at e_131_6
  have e_131_7 := arithEq_of_rows h (row := 131) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_7
  simp only [← c8556, ← c8557, ← c8558] at e_131_7
  have hr := e_131_7
  simp only [e_131_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 39) = bselect (a (.wire 122 39)) (a (.virt 9403)) (a (.wire 130 55)) := by
  have c8559 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8560 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8561 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8562 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8563 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8564 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_8 := arithEq_of_rows h (row := 131) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_8
  simp only [← c8559, ← c8560, ← c8561] at e_131_8
  have e_131_9 := arithEq_of_rows h (row := 131) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_9
  simp only [← c8562, ← c8563, ← c8564] at e_131_9
  have hr := e_131_9
  simp only [e_131_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 47) = bselect (a (.wire 122 39)) (a (.virt 9404)) (a (.wire 130 63)) := by
  have c8565 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8566 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8567 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8568 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8569 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8570 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_10 := arithEq_of_rows h (row := 131) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_10
  simp only [← c8565, ← c8566, ← c8567] at e_131_10
  have e_131_11 := arithEq_of_rows h (row := 131) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_11
  simp only [← c8568, ← c8569, ← c8570] at e_131_11
  have hr := e_131_11
  simp only [e_131_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 75) = bor (a (.wire 115 71)) (a (.wire 130 67)) := by
  have c8571 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8572 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8573 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8574 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8575 := (publicBatchWrapper89_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8576 := (publicBatchWrapper89_copies268 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_115_18 := arithEq_of_rows h (row := 115) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_18
  simp only [← c8574, ← c8575, k3, ← c8576] at e_115_18
  have e_126_6 := arithEq_of_rows h (row := 126) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_6
  simp only [← c8571, ← c8572, ← c8573] at e_126_6
  have hr := e_115_18
  simp only [e_126_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 51) = bnot (a (.wire 48 11)) := by
  have c8577 := (publicBatchWrapper89_copies268 a h).2.1
  have c8578 := (publicBatchWrapper89_copies268 a h).2.2.1
  have c8579 := (publicBatchWrapper89_copies268 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_131_12 := arithEq_of_rows h (row := 131) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_12
  simp only [← c8577, k3, ← c8578, k3, ← c8579] at e_131_12
  have hr := e_131_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 55) = bnot (a (.wire 115 75)) := by
  have c8580 := (publicBatchWrapper89_copies268 a h).2.2.2.2.1
  have c8581 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.1
  have c8582 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_131_13 := arithEq_of_rows h (row := 131) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_13
  simp only [← c8580, k3, ← c8581, k3, ← c8582] at e_131_13
  have hr := e_131_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 43) = band (a (.wire 131 51)) (a (.wire 131 55)) := by
  have c8583 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.1
  have c8584 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.1
  have c8585 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.1
  have e_122_10 := arithEq_of_rows h (row := 122) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_10
  simp only [← c8583, ← c8584, ← c8585] at e_122_10
  have hr := e_122_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 63) = bselect (a (.wire 122 43)) (a (.virt 9605)) (a (.wire 130 79)) := by
  have c8586 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8587 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8588 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8589 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8590 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8591 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_14 := arithEq_of_rows h (row := 131) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_14
  simp only [← c8586, ← c8587, ← c8588] at e_131_14
  have e_131_15 := arithEq_of_rows h (row := 131) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_15
  simp only [← c8589, ← c8590, ← c8591] at e_131_15
  have hr := e_131_15
  simp only [e_131_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 71) = bselect (a (.wire 122 43)) (a (.virt 9606)) (a (.wire 131 7)) := by
  have c8592 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8593 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8594 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8595 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8596 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8597 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_16 := arithEq_of_rows h (row := 131) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_16
  simp only [← c8592, ← c8593, ← c8594] at e_131_16
  have e_131_17 := arithEq_of_rows h (row := 131) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_17
  simp only [← c8595, ← c8596, ← c8597] at e_131_17
  have hr := e_131_17
  simp only [e_131_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 131 79) = bselect (a (.wire 122 43)) (a (.virt 9607)) (a (.wire 131 15)) := by
  have c8598 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8599 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8600 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8601 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8602 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8603 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_18 := arithEq_of_rows h (row := 131) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_18
  simp only [← c8598, ← c8599, ← c8600] at e_131_18
  have e_131_19 := arithEq_of_rows h (row := 131) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_19
  simp only [← c8601, ← c8602, ← c8603] at e_131_19
  have hr := e_131_19
  simp only [e_131_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 7) = bselect (a (.wire 122 43)) (a (.virt 9608)) (a (.wire 131 23)) := by
  have c8604 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8605 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8606 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8607 := (publicBatchWrapper89_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8608 := (publicBatchWrapper89_copies269 a h).1
  have c8609 := (publicBatchWrapper89_copies269 a h).2.1
  have e_132_0 := arithEq_of_rows h (row := 132) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_0
  simp only [← c8604, ← c8605, ← c8606] at e_132_0
  have e_132_1 := arithEq_of_rows h (row := 132) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_1
  simp only [← c8607, ← c8608, ← c8609] at e_132_1
  have hr := e_132_1
  simp only [e_132_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 15) = bselect (a (.wire 122 43)) (a (.virt 9609)) (a (.wire 131 31)) := by
  have c8610 := (publicBatchWrapper89_copies269 a h).2.2.1
  have c8611 := (publicBatchWrapper89_copies269 a h).2.2.2.1
  have c8612 := (publicBatchWrapper89_copies269 a h).2.2.2.2.1
  have c8613 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.1
  have c8614 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.1
  have c8615 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.1
  have e_132_2 := arithEq_of_rows h (row := 132) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_2
  simp only [← c8610, ← c8611, ← c8612] at e_132_2
  have e_132_3 := arithEq_of_rows h (row := 132) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_3
  simp only [← c8613, ← c8614, ← c8615] at e_132_3
  have hr := e_132_3
  simp only [e_132_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 23) = bselect (a (.wire 122 43)) (a (.virt 9603)) (a (.wire 131 39)) := by
  have c8616 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.1
  have c8617 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.1
  have c8618 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8619 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8620 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8621 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_4 := arithEq_of_rows h (row := 132) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_4
  simp only [← c8616, ← c8617, ← c8618] at e_132_4
  have e_132_5 := arithEq_of_rows h (row := 132) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_5
  simp only [← c8619, ← c8620, ← c8621] at e_132_5
  have hr := e_132_5
  simp only [e_132_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 31) = bselect (a (.wire 122 43)) (a (.virt 9604)) (a (.wire 131 47)) := by
  have c8622 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8623 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8624 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8625 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8626 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8627 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_6 := arithEq_of_rows h (row := 132) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_6
  simp only [← c8622, ← c8623, ← c8624] at e_132_6
  have e_132_7 := arithEq_of_rows h (row := 132) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_7
  simp only [← c8625, ← c8626, ← c8627] at e_132_7
  have hr := e_132_7
  simp only [e_132_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 115 79) = bor (a (.wire 115 75)) (a (.wire 131 51)) := by
  have c8628 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8629 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8630 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8631 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8632 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8633 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_115_19 := arithEq_of_rows h (row := 115) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_19
  simp only [← c8631, ← c8632, k3, ← c8633] at e_115_19
  have e_126_7 := arithEq_of_rows h (row := 126) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_7
  simp only [← c8628, ← c8629, ← c8630] at e_126_7
  have hr := e_115_19
  simp only [e_126_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 35) = bnot (a (.wire 48 55)) := by
  have c8634 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8635 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8636 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_132_8 := arithEq_of_rows h (row := 132) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_8
  simp only [← c8634, k3, ← c8635, k3, ← c8636] at e_132_8
  have hr := e_132_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 39) = bnot (a (.wire 115 79)) := by
  have c8637 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8638 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8639 := (publicBatchWrapper89_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_132_9 := arithEq_of_rows h (row := 132) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_9
  simp only [← c8637, k3, ← c8638, k3, ← c8639] at e_132_9
  have hr := e_132_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 47) = band (a (.wire 132 35)) (a (.wire 132 39)) := by
  have c8640 := (publicBatchWrapper89_copies270 a h).1
  have c8641 := (publicBatchWrapper89_copies270 a h).2.1
  have c8642 := (publicBatchWrapper89_copies270 a h).2.2.1
  have e_122_11 := arithEq_of_rows h (row := 122) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_11
  simp only [← c8640, ← c8641, ← c8642] at e_122_11
  have hr := e_122_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 47) = bselect (a (.wire 122 47)) (a (.virt 9805)) (a (.wire 131 63)) := by
  have c8643 := (publicBatchWrapper89_copies270 a h).2.2.2.1
  have c8644 := (publicBatchWrapper89_copies270 a h).2.2.2.2.1
  have c8645 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.1
  have c8646 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.1
  have c8647 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.1
  have c8648 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.1
  have e_132_10 := arithEq_of_rows h (row := 132) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_10
  simp only [← c8643, ← c8644, ← c8645] at e_132_10
  have e_132_11 := arithEq_of_rows h (row := 132) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_11
  simp only [← c8646, ← c8647, ← c8648] at e_132_11
  have hr := e_132_11
  simp only [e_132_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 55) = bselect (a (.wire 122 47)) (a (.virt 9806)) (a (.wire 131 71)) := by
  have c8649 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.1
  have c8650 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8651 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8652 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8653 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8654 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_12 := arithEq_of_rows h (row := 132) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_12
  simp only [← c8649, ← c8650, ← c8651] at e_132_12
  have e_132_13 := arithEq_of_rows h (row := 132) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_13
  simp only [← c8652, ← c8653, ← c8654] at e_132_13
  have hr := e_132_13
  simp only [e_132_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 63) = bselect (a (.wire 122 47)) (a (.virt 9807)) (a (.wire 131 79)) := by
  have c8655 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8656 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8657 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8658 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8659 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8660 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_14 := arithEq_of_rows h (row := 132) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_14
  simp only [← c8655, ← c8656, ← c8657] at e_132_14
  have e_132_15 := arithEq_of_rows h (row := 132) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_15
  simp only [← c8658, ← c8659, ← c8660] at e_132_15
  have hr := e_132_15
  simp only [e_132_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 71) = bselect (a (.wire 122 47)) (a (.virt 9808)) (a (.wire 132 7)) := by
  have c8661 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8662 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8663 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8664 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8665 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8666 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_16 := arithEq_of_rows h (row := 132) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_16
  simp only [← c8661, ← c8662, ← c8663] at e_132_16
  have e_132_17 := arithEq_of_rows h (row := 132) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_17
  simp only [← c8664, ← c8665, ← c8666] at e_132_17
  have hr := e_132_17
  simp only [e_132_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 132 79) = bselect (a (.wire 122 47)) (a (.virt 9809)) (a (.wire 132 15)) := by
  have c8667 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8668 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8669 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8670 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8671 := (publicBatchWrapper89_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8672 := (publicBatchWrapper89_copies271 a h).1
  have e_132_18 := arithEq_of_rows h (row := 132) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_18
  simp only [← c8667, ← c8668, ← c8669] at e_132_18
  have e_132_19 := arithEq_of_rows h (row := 132) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_19
  simp only [← c8670, ← c8671, ← c8672] at e_132_19
  have hr := e_132_19
  simp only [e_132_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 7) = bselect (a (.wire 122 47)) (a (.virt 9803)) (a (.wire 132 23)) := by
  have c8673 := (publicBatchWrapper89_copies271 a h).2.1
  have c8674 := (publicBatchWrapper89_copies271 a h).2.2.1
  have c8675 := (publicBatchWrapper89_copies271 a h).2.2.2.1
  have c8676 := (publicBatchWrapper89_copies271 a h).2.2.2.2.1
  have c8677 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.1
  have c8678 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.1
  have e_133_0 := arithEq_of_rows h (row := 133) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_0
  simp only [← c8673, ← c8674, ← c8675] at e_133_0
  have e_133_1 := arithEq_of_rows h (row := 133) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_1
  simp only [← c8676, ← c8677, ← c8678] at e_133_1
  have hr := e_133_1
  simp only [e_133_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 15) = bselect (a (.wire 122 47)) (a (.virt 9804)) (a (.wire 132 31)) := by
  have c8679 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.1
  have c8680 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.1
  have c8681 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.1
  have c8682 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8683 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8684 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_2 := arithEq_of_rows h (row := 133) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_2
  simp only [← c8679, ← c8680, ← c8681] at e_133_2
  have e_133_3 := arithEq_of_rows h (row := 133) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_3
  simp only [← c8682, ← c8683, ← c8684] at e_133_3
  have hr := e_133_3
  simp only [e_133_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 3) = bor (a (.wire 115 79)) (a (.wire 132 35)) := by
  have c8685 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8686 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8687 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8688 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8689 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8690 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_8 := arithEq_of_rows h (row := 126) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_8
  simp only [← c8685, ← c8686, ← c8687] at e_126_8
  have e_134_0 := arithEq_of_rows h (row := 134) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_0
  simp only [← c8688, ← c8689, k3, ← c8690] at e_134_0
  have hr := e_134_0
  simp only [e_126_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 19) = bnot (a (.wire 50 19)) := by
  have c8691 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8692 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8693 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_133_4 := arithEq_of_rows h (row := 133) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_4
  simp only [← c8691, k3, ← c8692, k3, ← c8693] at e_133_4
  have hr := e_133_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 23) = bnot (a (.wire 134 3)) := by
  have c8694 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8695 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8696 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_133_5 := arithEq_of_rows h (row := 133) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_5
  simp only [← c8694, k3, ← c8695, k3, ← c8696] at e_133_5
  have hr := e_133_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 51) = band (a (.wire 133 19)) (a (.wire 133 23)) := by
  have c8697 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8698 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8699 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_12 := arithEq_of_rows h (row := 122) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_12
  simp only [← c8697, ← c8698, ← c8699] at e_122_12
  have hr := e_122_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 31) = bselect (a (.wire 122 51)) (a (.virt 10005)) (a (.wire 132 47)) := by
  have c8700 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8701 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8702 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8703 := (publicBatchWrapper89_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8704 := (publicBatchWrapper89_copies272 a h).1
  have c8705 := (publicBatchWrapper89_copies272 a h).2.1
  have e_133_6 := arithEq_of_rows h (row := 133) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_6
  simp only [← c8700, ← c8701, ← c8702] at e_133_6
  have e_133_7 := arithEq_of_rows h (row := 133) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_7
  simp only [← c8703, ← c8704, ← c8705] at e_133_7
  have hr := e_133_7
  simp only [e_133_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 39) = bselect (a (.wire 122 51)) (a (.virt 10006)) (a (.wire 132 55)) := by
  have c8706 := (publicBatchWrapper89_copies272 a h).2.2.1
  have c8707 := (publicBatchWrapper89_copies272 a h).2.2.2.1
  have c8708 := (publicBatchWrapper89_copies272 a h).2.2.2.2.1
  have c8709 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.1
  have c8710 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.1
  have c8711 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.1
  have e_133_8 := arithEq_of_rows h (row := 133) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_8
  simp only [← c8706, ← c8707, ← c8708] at e_133_8
  have e_133_9 := arithEq_of_rows h (row := 133) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_9
  simp only [← c8709, ← c8710, ← c8711] at e_133_9
  have hr := e_133_9
  simp only [e_133_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 47) = bselect (a (.wire 122 51)) (a (.virt 10007)) (a (.wire 132 63)) := by
  have c8712 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.1
  have c8713 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.1
  have c8714 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8715 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8716 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8717 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_10 := arithEq_of_rows h (row := 133) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_10
  simp only [← c8712, ← c8713, ← c8714] at e_133_10
  have e_133_11 := arithEq_of_rows h (row := 133) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_11
  simp only [← c8715, ← c8716, ← c8717] at e_133_11
  have hr := e_133_11
  simp only [e_133_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 55) = bselect (a (.wire 122 51)) (a (.virt 10008)) (a (.wire 132 71)) := by
  have c8718 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8719 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8720 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8721 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8722 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8723 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_12 := arithEq_of_rows h (row := 133) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_12
  simp only [← c8718, ← c8719, ← c8720] at e_133_12
  have e_133_13 := arithEq_of_rows h (row := 133) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_13
  simp only [← c8721, ← c8722, ← c8723] at e_133_13
  have hr := e_133_13
  simp only [e_133_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 63) = bselect (a (.wire 122 51)) (a (.virt 10009)) (a (.wire 132 79)) := by
  have c8724 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8725 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8726 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8727 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8728 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8729 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_14 := arithEq_of_rows h (row := 133) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_14
  simp only [← c8724, ← c8725, ← c8726] at e_133_14
  have e_133_15 := arithEq_of_rows h (row := 133) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_15
  simp only [← c8727, ← c8728, ← c8729] at e_133_15
  have hr := e_133_15
  simp only [e_133_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 71) = bselect (a (.wire 122 51)) (a (.virt 10003)) (a (.wire 133 7)) := by
  have c8730 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8731 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8732 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8733 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8734 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8735 := (publicBatchWrapper89_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_133_16 := arithEq_of_rows h (row := 133) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_16
  simp only [← c8730, ← c8731, ← c8732] at e_133_16
  have e_133_17 := arithEq_of_rows h (row := 133) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_17
  simp only [← c8733, ← c8734, ← c8735] at e_133_17
  have hr := e_133_17
  simp only [e_133_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 133 79) = bselect (a (.wire 122 51)) (a (.virt 10004)) (a (.wire 133 15)) := by
  have c8736 := (publicBatchWrapper89_copies273 a h).1
  have c8737 := (publicBatchWrapper89_copies273 a h).2.1
  have c8738 := (publicBatchWrapper89_copies273 a h).2.2.1
  have c8739 := (publicBatchWrapper89_copies273 a h).2.2.2.1
  have c8740 := (publicBatchWrapper89_copies273 a h).2.2.2.2.1
  have c8741 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.1
  have e_133_18 := arithEq_of_rows h (row := 133) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_18
  simp only [← c8736, ← c8737, ← c8738] at e_133_18
  have e_133_19 := arithEq_of_rows h (row := 133) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_19
  simp only [← c8739, ← c8740, ← c8741] at e_133_19
  have hr := e_133_19
  simp only [e_133_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 7) = bor (a (.wire 134 3)) (a (.wire 133 19)) := by
  have c8742 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.1
  have c8743 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.1
  have c8744 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.1
  have c8745 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.1
  have c8746 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8747 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_9 := arithEq_of_rows h (row := 126) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_9
  simp only [← c8742, ← c8743, ← c8744] at e_126_9
  have e_134_1 := arithEq_of_rows h (row := 134) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_1
  simp only [← c8745, ← c8746, k3, ← c8747] at e_134_1
  have hr := e_134_1
  simp only [e_126_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 3) = bnot (a (.wire 50 63)) := by
  have c8748 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8749 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8750 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_135_0 := arithEq_of_rows h (row := 135) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_0
  simp only [← c8748, k3, ← c8749, k3, ← c8750] at e_135_0
  have hr := e_135_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 7) = bnot (a (.wire 134 7)) := by
  have c8751 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8752 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8753 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_135_1 := arithEq_of_rows h (row := 135) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_1
  simp only [← c8751, k3, ← c8752, k3, ← c8753] at e_135_1
  have hr := e_135_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 55) = band (a (.wire 135 3)) (a (.wire 135 7)) := by
  have c8754 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8755 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8756 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_13 := arithEq_of_rows h (row := 122) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_13
  simp only [← c8754, ← c8755, ← c8756] at e_122_13
  have hr := e_122_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 15) = bselect (a (.wire 122 55)) (a (.virt 10205)) (a (.wire 133 31)) := by
  have c8757 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8758 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8759 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8760 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8761 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8762 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_135_2 := arithEq_of_rows h (row := 135) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_2
  simp only [← c8757, ← c8758, ← c8759] at e_135_2
  have e_135_3 := arithEq_of_rows h (row := 135) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_3
  simp only [← c8760, ← c8761, ← c8762] at e_135_3
  have hr := e_135_3
  simp only [e_135_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 23) = bselect (a (.wire 122 55)) (a (.virt 10206)) (a (.wire 133 39)) := by
  have c8763 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8764 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8765 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8766 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8767 := (publicBatchWrapper89_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8768 := (publicBatchWrapper89_copies274 a h).1
  have e_135_4 := arithEq_of_rows h (row := 135) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_4
  simp only [← c8763, ← c8764, ← c8765] at e_135_4
  have e_135_5 := arithEq_of_rows h (row := 135) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_5
  simp only [← c8766, ← c8767, ← c8768] at e_135_5
  have hr := e_135_5
  simp only [e_135_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 31) = bselect (a (.wire 122 55)) (a (.virt 10207)) (a (.wire 133 47)) := by
  have c8769 := (publicBatchWrapper89_copies274 a h).2.1
  have c8770 := (publicBatchWrapper89_copies274 a h).2.2.1
  have c8771 := (publicBatchWrapper89_copies274 a h).2.2.2.1
  have c8772 := (publicBatchWrapper89_copies274 a h).2.2.2.2.1
  have c8773 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.1
  have c8774 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.1
  have e_135_6 := arithEq_of_rows h (row := 135) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_6
  simp only [← c8769, ← c8770, ← c8771] at e_135_6
  have e_135_7 := arithEq_of_rows h (row := 135) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_7
  simp only [← c8772, ← c8773, ← c8774] at e_135_7
  have hr := e_135_7
  simp only [e_135_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 39) = bselect (a (.wire 122 55)) (a (.virt 10208)) (a (.wire 133 55)) := by
  have c8775 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.1
  have c8776 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.1
  have c8777 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.1
  have c8778 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8779 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8780 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_135_8 := arithEq_of_rows h (row := 135) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_8
  simp only [← c8775, ← c8776, ← c8777] at e_135_8
  have e_135_9 := arithEq_of_rows h (row := 135) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_9
  simp only [← c8778, ← c8779, ← c8780] at e_135_9
  have hr := e_135_9
  simp only [e_135_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 47) = bselect (a (.wire 122 55)) (a (.virt 10209)) (a (.wire 133 63)) := by
  have c8781 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8782 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8783 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8784 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8785 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8786 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_135_10 := arithEq_of_rows h (row := 135) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_10
  simp only [← c8781, ← c8782, ← c8783] at e_135_10
  have e_135_11 := arithEq_of_rows h (row := 135) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_11
  simp only [← c8784, ← c8785, ← c8786] at e_135_11
  have hr := e_135_11
  simp only [e_135_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 55) = bselect (a (.wire 122 55)) (a (.virt 10203)) (a (.wire 133 71)) := by
  have c8787 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8788 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8789 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8790 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8791 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8792 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_135_12 := arithEq_of_rows h (row := 135) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_12
  simp only [← c8787, ← c8788, ← c8789] at e_135_12
  have e_135_13 := arithEq_of_rows h (row := 135) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_13
  simp only [← c8790, ← c8791, ← c8792] at e_135_13
  have hr := e_135_13
  simp only [e_135_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 63) = bselect (a (.wire 122 55)) (a (.virt 10204)) (a (.wire 133 79)) := by
  have c8793 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8794 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8795 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8796 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8797 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8798 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_135_14 := arithEq_of_rows h (row := 135) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_14
  simp only [← c8793, ← c8794, ← c8795] at e_135_14
  have e_135_15 := arithEq_of_rows h (row := 135) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_15
  simp only [← c8796, ← c8797, ← c8798] at e_135_15
  have hr := e_135_15
  simp only [e_135_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 11) = bor (a (.wire 134 7)) (a (.wire 135 3)) := by
  have c8799 := (publicBatchWrapper89_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8800 := (publicBatchWrapper89_copies275 a h).1
  have c8801 := (publicBatchWrapper89_copies275 a h).2.1
  have c8802 := (publicBatchWrapper89_copies275 a h).2.2.1
  have c8803 := (publicBatchWrapper89_copies275 a h).2.2.2.1
  have c8804 := (publicBatchWrapper89_copies275 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_10 := arithEq_of_rows h (row := 126) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_10
  simp only [← c8799, ← c8800, ← c8801] at e_126_10
  have e_134_2 := arithEq_of_rows h (row := 134) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_2
  simp only [← c8802, ← c8803, k3, ← c8804] at e_134_2
  have hr := e_134_2
  simp only [e_126_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 67) = bnot (a (.wire 52 27)) := by
  have c8805 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.1
  have c8806 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.1
  have c8807 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_135_16 := arithEq_of_rows h (row := 135) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_16
  simp only [← c8805, k3, ← c8806, k3, ← c8807] at e_135_16
  have hr := e_135_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 71) = bnot (a (.wire 134 11)) := by
  have c8808 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.1
  have c8809 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.1
  have c8810 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_135_17 := arithEq_of_rows h (row := 135) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_17
  simp only [← c8808, k3, ← c8809, k3, ← c8810] at e_135_17
  have hr := e_135_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 59) = band (a (.wire 135 67)) (a (.wire 135 71)) := by
  have c8811 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8812 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8813 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_14 := arithEq_of_rows h (row := 122) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_14
  simp only [← c8811, ← c8812, ← c8813] at e_122_14
  have hr := e_122_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 135 79) = bselect (a (.wire 122 59)) (a (.virt 10405)) (a (.wire 135 15)) := by
  have c8814 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8815 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8816 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8817 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8818 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8819 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_135_18 := arithEq_of_rows h (row := 135) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_18
  simp only [← c8814, ← c8815, ← c8816] at e_135_18
  have e_135_19 := arithEq_of_rows h (row := 135) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_19
  simp only [← c8817, ← c8818, ← c8819] at e_135_19
  have hr := e_135_19
  simp only [e_135_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 7) = bselect (a (.wire 122 59)) (a (.virt 10406)) (a (.wire 135 23)) := by
  have c8820 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8821 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8822 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8823 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8824 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8825 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_0 := arithEq_of_rows h (row := 136) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_0
  simp only [← c8820, ← c8821, ← c8822] at e_136_0
  have e_136_1 := arithEq_of_rows h (row := 136) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_1
  simp only [← c8823, ← c8824, ← c8825] at e_136_1
  have hr := e_136_1
  simp only [e_136_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 15) = bselect (a (.wire 122 59)) (a (.virt 10407)) (a (.wire 135 31)) := by
  have c8826 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8827 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8828 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8829 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8830 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8831 := (publicBatchWrapper89_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_136_2 := arithEq_of_rows h (row := 136) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_2
  simp only [← c8826, ← c8827, ← c8828] at e_136_2
  have e_136_3 := arithEq_of_rows h (row := 136) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_3
  simp only [← c8829, ← c8830, ← c8831] at e_136_3
  have hr := e_136_3
  simp only [e_136_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 23) = bselect (a (.wire 122 59)) (a (.virt 10408)) (a (.wire 135 39)) := by
  have c8832 := (publicBatchWrapper89_copies276 a h).1
  have c8833 := (publicBatchWrapper89_copies276 a h).2.1
  have c8834 := (publicBatchWrapper89_copies276 a h).2.2.1
  have c8835 := (publicBatchWrapper89_copies276 a h).2.2.2.1
  have c8836 := (publicBatchWrapper89_copies276 a h).2.2.2.2.1
  have c8837 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.1
  have e_136_4 := arithEq_of_rows h (row := 136) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_4
  simp only [← c8832, ← c8833, ← c8834] at e_136_4
  have e_136_5 := arithEq_of_rows h (row := 136) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_5
  simp only [← c8835, ← c8836, ← c8837] at e_136_5
  have hr := e_136_5
  simp only [e_136_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 31) = bselect (a (.wire 122 59)) (a (.virt 10409)) (a (.wire 135 47)) := by
  have c8838 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.1
  have c8839 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.1
  have c8840 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.1
  have c8841 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.1
  have c8842 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8843 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_6 := arithEq_of_rows h (row := 136) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_6
  simp only [← c8838, ← c8839, ← c8840] at e_136_6
  have e_136_7 := arithEq_of_rows h (row := 136) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_7
  simp only [← c8841, ← c8842, ← c8843] at e_136_7
  have hr := e_136_7
  simp only [e_136_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 39) = bselect (a (.wire 122 59)) (a (.virt 10403)) (a (.wire 135 55)) := by
  have c8844 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8845 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8846 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8847 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8848 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8849 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_8 := arithEq_of_rows h (row := 136) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_8
  simp only [← c8844, ← c8845, ← c8846] at e_136_8
  have e_136_9 := arithEq_of_rows h (row := 136) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_9
  simp only [← c8847, ← c8848, ← c8849] at e_136_9
  have hr := e_136_9
  simp only [e_136_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 47) = bselect (a (.wire 122 59)) (a (.virt 10404)) (a (.wire 135 63)) := by
  have c8850 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8851 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8852 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8853 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8854 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8855 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_10 := arithEq_of_rows h (row := 136) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_10
  simp only [← c8850, ← c8851, ← c8852] at e_136_10
  have e_136_11 := arithEq_of_rows h (row := 136) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_11
  simp only [← c8853, ← c8854, ← c8855] at e_136_11
  have hr := e_136_11
  simp only [e_136_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 15) = bor (a (.wire 134 11)) (a (.wire 135 67)) := by
  have c8856 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8857 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8858 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8859 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8860 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8861 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_11 := arithEq_of_rows h (row := 126) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_11
  simp only [← c8856, ← c8857, ← c8858] at e_126_11
  have e_134_3 := arithEq_of_rows h (row := 134) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_3
  simp only [← c8859, ← c8860, k3, ← c8861] at e_134_3
  have hr := e_134_3
  simp only [e_126_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 51) = bnot (a (.wire 52 71)) := by
  have c8862 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8863 := (publicBatchWrapper89_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8864 := (publicBatchWrapper89_copies277 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_136_12 := arithEq_of_rows h (row := 136) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_12
  simp only [← c8862, k3, ← c8863, k3, ← c8864] at e_136_12
  have hr := e_136_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 55) = bnot (a (.wire 134 15)) := by
  have c8865 := (publicBatchWrapper89_copies277 a h).2.1
  have c8866 := (publicBatchWrapper89_copies277 a h).2.2.1
  have c8867 := (publicBatchWrapper89_copies277 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_136_13 := arithEq_of_rows h (row := 136) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_13
  simp only [← c8865, k3, ← c8866, k3, ← c8867] at e_136_13
  have hr := e_136_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 63) = band (a (.wire 136 51)) (a (.wire 136 55)) := by
  have c8868 := (publicBatchWrapper89_copies277 a h).2.2.2.2.1
  have c8869 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.1
  have c8870 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.1
  have e_122_15 := arithEq_of_rows h (row := 122) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_15
  simp only [← c8868, ← c8869, ← c8870] at e_122_15
  have hr := e_122_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 63) = bselect (a (.wire 122 63)) (a (.virt 10605)) (a (.wire 135 79)) := by
  have c8871 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.1
  have c8872 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.1
  have c8873 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.1
  have c8874 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8875 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8876 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_14 := arithEq_of_rows h (row := 136) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_14
  simp only [← c8871, ← c8872, ← c8873] at e_136_14
  have e_136_15 := arithEq_of_rows h (row := 136) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_15
  simp only [← c8874, ← c8875, ← c8876] at e_136_15
  have hr := e_136_15
  simp only [e_136_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 71) = bselect (a (.wire 122 63)) (a (.virt 10606)) (a (.wire 136 7)) := by
  have c8877 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8878 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8879 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8880 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8881 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8882 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_16 := arithEq_of_rows h (row := 136) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_16
  simp only [← c8877, ← c8878, ← c8879] at e_136_16
  have e_136_17 := arithEq_of_rows h (row := 136) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_17
  simp only [← c8880, ← c8881, ← c8882] at e_136_17
  have hr := e_136_17
  simp only [e_136_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 136 79) = bselect (a (.wire 122 63)) (a (.virt 10607)) (a (.wire 136 15)) := by
  have c8883 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8884 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8885 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8886 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8887 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8888 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_18 := arithEq_of_rows h (row := 136) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_18
  simp only [← c8883, ← c8884, ← c8885] at e_136_18
  have e_136_19 := arithEq_of_rows h (row := 136) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_19
  simp only [← c8886, ← c8887, ← c8888] at e_136_19
  have hr := e_136_19
  simp only [e_136_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 7) = bselect (a (.wire 122 63)) (a (.virt 10608)) (a (.wire 136 23)) := by
  have c8889 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8890 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8891 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8892 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8893 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8894 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_0 := arithEq_of_rows h (row := 137) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_0
  simp only [← c8889, ← c8890, ← c8891] at e_137_0
  have e_137_1 := arithEq_of_rows h (row := 137) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_1
  simp only [← c8892, ← c8893, ← c8894] at e_137_1
  have hr := e_137_1
  simp only [e_137_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 15) = bselect (a (.wire 122 63)) (a (.virt 10609)) (a (.wire 136 31)) := by
  have c8895 := (publicBatchWrapper89_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8896 := (publicBatchWrapper89_copies278 a h).1
  have c8897 := (publicBatchWrapper89_copies278 a h).2.1
  have c8898 := (publicBatchWrapper89_copies278 a h).2.2.1
  have c8899 := (publicBatchWrapper89_copies278 a h).2.2.2.1
  have c8900 := (publicBatchWrapper89_copies278 a h).2.2.2.2.1
  have e_137_2 := arithEq_of_rows h (row := 137) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_2
  simp only [← c8895, ← c8896, ← c8897] at e_137_2
  have e_137_3 := arithEq_of_rows h (row := 137) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_3
  simp only [← c8898, ← c8899, ← c8900] at e_137_3
  have hr := e_137_3
  simp only [e_137_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 23) = bselect (a (.wire 122 63)) (a (.virt 10603)) (a (.wire 136 39)) := by
  have c8901 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.1
  have c8902 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.1
  have c8903 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.1
  have c8904 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.1
  have c8905 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.1
  have c8906 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_137_4 := arithEq_of_rows h (row := 137) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_4
  simp only [← c8901, ← c8902, ← c8903] at e_137_4
  have e_137_5 := arithEq_of_rows h (row := 137) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_5
  simp only [← c8904, ← c8905, ← c8906] at e_137_5
  have hr := e_137_5
  simp only [e_137_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 31) = bselect (a (.wire 122 63)) (a (.virt 10604)) (a (.wire 136 47)) := by
  have c8907 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8908 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8909 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8910 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8911 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8912 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_6 := arithEq_of_rows h (row := 137) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_6
  simp only [← c8907, ← c8908, ← c8909] at e_137_6
  have e_137_7 := arithEq_of_rows h (row := 137) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_7
  simp only [← c8910, ← c8911, ← c8912] at e_137_7
  have hr := e_137_7
  simp only [e_137_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 19) = bor (a (.wire 134 15)) (a (.wire 136 51)) := by
  have c8913 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8914 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8915 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8916 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8917 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8918 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_12 := arithEq_of_rows h (row := 126) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_12
  simp only [← c8913, ← c8914, ← c8915] at e_126_12
  have e_134_4 := arithEq_of_rows h (row := 134) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_4
  simp only [← c8916, ← c8917, k3, ← c8918] at e_134_4
  have hr := e_134_4
  simp only [e_126_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 35) = bnot (a (.wire 53 35)) := by
  have c8919 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8920 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8921 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_137_8 := arithEq_of_rows h (row := 137) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_8
  simp only [← c8919, k3, ← c8920, k3, ← c8921] at e_137_8
  have hr := e_137_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 39) = bnot (a (.wire 134 19)) := by
  have c8922 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8923 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8924 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_137_9 := arithEq_of_rows h (row := 137) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_9
  simp only [← c8922, k3, ← c8923, k3, ← c8924] at e_137_9
  have hr := e_137_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 67) = band (a (.wire 137 35)) (a (.wire 137 39)) := by
  have c8925 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8926 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8927 := (publicBatchWrapper89_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_122_16 := arithEq_of_rows h (row := 122) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_16
  simp only [← c8925, ← c8926, ← c8927] at e_122_16
  have hr := e_122_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 47) = bselect (a (.wire 122 67)) (a (.virt 10805)) (a (.wire 136 63)) := by
  have c8928 := (publicBatchWrapper89_copies279 a h).1
  have c8929 := (publicBatchWrapper89_copies279 a h).2.1
  have c8930 := (publicBatchWrapper89_copies279 a h).2.2.1
  have c8931 := (publicBatchWrapper89_copies279 a h).2.2.2.1
  have c8932 := (publicBatchWrapper89_copies279 a h).2.2.2.2.1
  have c8933 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.1
  have e_137_10 := arithEq_of_rows h (row := 137) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_10
  simp only [← c8928, ← c8929, ← c8930] at e_137_10
  have e_137_11 := arithEq_of_rows h (row := 137) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_11
  simp only [← c8931, ← c8932, ← c8933] at e_137_11
  have hr := e_137_11
  simp only [e_137_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 55) = bselect (a (.wire 122 67)) (a (.virt 10806)) (a (.wire 136 71)) := by
  have c8934 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.1
  have c8935 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.1
  have c8936 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.1
  have c8937 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.1
  have c8938 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8939 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_12 := arithEq_of_rows h (row := 137) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_12
  simp only [← c8934, ← c8935, ← c8936] at e_137_12
  have e_137_13 := arithEq_of_rows h (row := 137) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_13
  simp only [← c8937, ← c8938, ← c8939] at e_137_13
  have hr := e_137_13
  simp only [e_137_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 63) = bselect (a (.wire 122 67)) (a (.virt 10807)) (a (.wire 136 79)) := by
  have c8940 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8941 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8942 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8943 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8944 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8945 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_14 := arithEq_of_rows h (row := 137) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_14
  simp only [← c8940, ← c8941, ← c8942] at e_137_14
  have e_137_15 := arithEq_of_rows h (row := 137) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_15
  simp only [← c8943, ← c8944, ← c8945] at e_137_15
  have hr := e_137_15
  simp only [e_137_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 71) = bselect (a (.wire 122 67)) (a (.virt 10808)) (a (.wire 137 7)) := by
  have c8946 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8947 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8948 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8949 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8950 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8951 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_16 := arithEq_of_rows h (row := 137) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_16
  simp only [← c8946, ← c8947, ← c8948] at e_137_16
  have e_137_17 := arithEq_of_rows h (row := 137) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_17
  simp only [← c8949, ← c8950, ← c8951] at e_137_17
  have hr := e_137_17
  simp only [e_137_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 137 79) = bselect (a (.wire 122 67)) (a (.virt 10809)) (a (.wire 137 15)) := by
  have c8952 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8953 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8954 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8955 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8956 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8957 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_18 := arithEq_of_rows h (row := 137) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_18
  simp only [← c8952, ← c8953, ← c8954] at e_137_18
  have e_137_19 := arithEq_of_rows h (row := 137) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_19
  simp only [← c8955, ← c8956, ← c8957] at e_137_19
  have hr := e_137_19
  simp only [e_137_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 7) = bselect (a (.wire 122 67)) (a (.virt 10803)) (a (.wire 137 23)) := by
  have c8958 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8959 := (publicBatchWrapper89_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8960 := (publicBatchWrapper89_copies280 a h).1
  have c8961 := (publicBatchWrapper89_copies280 a h).2.1
  have c8962 := (publicBatchWrapper89_copies280 a h).2.2.1
  have c8963 := (publicBatchWrapper89_copies280 a h).2.2.2.1
  have e_138_0 := arithEq_of_rows h (row := 138) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_0
  simp only [← c8958, ← c8959, ← c8960] at e_138_0
  have e_138_1 := arithEq_of_rows h (row := 138) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_1
  simp only [← c8961, ← c8962, ← c8963] at e_138_1
  have hr := e_138_1
  simp only [e_138_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 15) = bselect (a (.wire 122 67)) (a (.virt 10804)) (a (.wire 137 31)) := by
  have c8964 := (publicBatchWrapper89_copies280 a h).2.2.2.2.1
  have c8965 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.1
  have c8966 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.1
  have c8967 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.1
  have c8968 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.1
  have c8969 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.1
  have e_138_2 := arithEq_of_rows h (row := 138) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_2
  simp only [← c8964, ← c8965, ← c8966] at e_138_2
  have e_138_3 := arithEq_of_rows h (row := 138) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_3
  simp only [← c8967, ← c8968, ← c8969] at e_138_3
  have hr := e_138_3
  simp only [e_138_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 23) = bor (a (.wire 134 19)) (a (.wire 137 35)) := by
  have c8970 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8971 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8972 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8973 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8974 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8975 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_13 := arithEq_of_rows h (row := 126) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_13
  simp only [← c8970, ← c8971, ← c8972] at e_126_13
  have e_134_5 := arithEq_of_rows h (row := 134) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_5
  simp only [← c8973, ← c8974, k3, ← c8975] at e_134_5
  have hr := e_134_5
  simp only [e_126_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 19) = bnot (a (.wire 53 79)) := by
  have c8976 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8977 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8978 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_138_4 := arithEq_of_rows h (row := 138) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_4
  simp only [← c8976, k3, ← c8977, k3, ← c8978] at e_138_4
  have hr := e_138_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 23) = bnot (a (.wire 134 23)) := by
  have c8979 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8980 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8981 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_138_5 := arithEq_of_rows h (row := 138) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_5
  simp only [← c8979, k3, ← c8980, k3, ← c8981] at e_138_5
  have hr := e_138_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 71) = band (a (.wire 138 19)) (a (.wire 138 23)) := by
  have c8982 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8983 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8984 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_17 := arithEq_of_rows h (row := 122) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_17
  simp only [← c8982, ← c8983, ← c8984] at e_122_17
  have hr := e_122_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 31) = bselect (a (.wire 122 71)) (a (.virt 11005)) (a (.wire 137 47)) := by
  have c8985 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8986 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8987 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8988 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8989 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8990 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_6 := arithEq_of_rows h (row := 138) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_6
  simp only [← c8985, ← c8986, ← c8987] at e_138_6
  have e_138_7 := arithEq_of_rows h (row := 138) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_7
  simp only [← c8988, ← c8989, ← c8990] at e_138_7
  have hr := e_138_7
  simp only [e_138_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 39) = bselect (a (.wire 122 71)) (a (.virt 11006)) (a (.wire 137 55)) := by
  have c8991 := (publicBatchWrapper89_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8992 := (publicBatchWrapper89_copies281 a h).1
  have c8993 := (publicBatchWrapper89_copies281 a h).2.1
  have c8994 := (publicBatchWrapper89_copies281 a h).2.2.1
  have c8995 := (publicBatchWrapper89_copies281 a h).2.2.2.1
  have c8996 := (publicBatchWrapper89_copies281 a h).2.2.2.2.1
  have e_138_8 := arithEq_of_rows h (row := 138) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_8
  simp only [← c8991, ← c8992, ← c8993] at e_138_8
  have e_138_9 := arithEq_of_rows h (row := 138) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_9
  simp only [← c8994, ← c8995, ← c8996] at e_138_9
  have hr := e_138_9
  simp only [e_138_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 47) = bselect (a (.wire 122 71)) (a (.virt 11007)) (a (.wire 137 63)) := by
  have c8997 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.1
  have c8998 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.1
  have c8999 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.1
  have c9000 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.1
  have c9001 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.1
  have c9002 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_138_10 := arithEq_of_rows h (row := 138) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_10
  simp only [← c8997, ← c8998, ← c8999] at e_138_10
  have e_138_11 := arithEq_of_rows h (row := 138) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_11
  simp only [← c9000, ← c9001, ← c9002] at e_138_11
  have hr := e_138_11
  simp only [e_138_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 55) = bselect (a (.wire 122 71)) (a (.virt 11008)) (a (.wire 137 71)) := by
  have c9003 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9004 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9005 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9006 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9007 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9008 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_12 := arithEq_of_rows h (row := 138) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_12
  simp only [← c9003, ← c9004, ← c9005] at e_138_12
  have e_138_13 := arithEq_of_rows h (row := 138) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_13
  simp only [← c9006, ← c9007, ← c9008] at e_138_13
  have hr := e_138_13
  simp only [e_138_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 63) = bselect (a (.wire 122 71)) (a (.virt 11009)) (a (.wire 137 79)) := by
  have c9009 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9010 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9011 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9012 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9013 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9014 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_14 := arithEq_of_rows h (row := 138) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_14
  simp only [← c9009, ← c9010, ← c9011] at e_138_14
  have e_138_15 := arithEq_of_rows h (row := 138) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_15
  simp only [← c9012, ← c9013, ← c9014] at e_138_15
  have hr := e_138_15
  simp only [e_138_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 71) = bselect (a (.wire 122 71)) (a (.virt 11003)) (a (.wire 138 7)) := by
  have c9015 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9016 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9017 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9018 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9019 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9020 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_16 := arithEq_of_rows h (row := 138) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_16
  simp only [← c9015, ← c9016, ← c9017] at e_138_16
  have e_138_17 := arithEq_of_rows h (row := 138) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_17
  simp only [← c9018, ← c9019, ← c9020] at e_138_17
  have hr := e_138_17
  simp only [e_138_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 138 79) = bselect (a (.wire 122 71)) (a (.virt 11004)) (a (.wire 138 15)) := by
  have c9021 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9022 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9023 := (publicBatchWrapper89_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9024 := (publicBatchWrapper89_copies282 a h).1
  have c9025 := (publicBatchWrapper89_copies282 a h).2.1
  have c9026 := (publicBatchWrapper89_copies282 a h).2.2.1
  have e_138_18 := arithEq_of_rows h (row := 138) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_18
  simp only [← c9021, ← c9022, ← c9023] at e_138_18
  have e_138_19 := arithEq_of_rows h (row := 138) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_19
  simp only [← c9024, ← c9025, ← c9026] at e_138_19
  have hr := e_138_19
  simp only [e_138_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 27) = bor (a (.wire 134 23)) (a (.wire 138 19)) := by
  have c9027 := (publicBatchWrapper89_copies282 a h).2.2.2.1
  have c9028 := (publicBatchWrapper89_copies282 a h).2.2.2.2.1
  have c9029 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.1
  have c9030 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.1
  have c9031 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.1
  have c9032 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_14 := arithEq_of_rows h (row := 126) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_14
  simp only [← c9027, ← c9028, ← c9029] at e_126_14
  have e_134_6 := arithEq_of_rows h (row := 134) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_6
  simp only [← c9030, ← c9031, k3, ← c9032] at e_134_6
  have hr := e_134_6
  simp only [e_126_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 3) = bnot (a (.wire 55 43)) := by
  have c9033 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.1
  have c9034 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9035 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_139_0 := arithEq_of_rows h (row := 139) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_0
  simp only [← c9033, k3, ← c9034, k3, ← c9035] at e_139_0
  have hr := e_139_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 7) = bnot (a (.wire 134 27)) := by
  have c9036 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9037 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9038 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_139_1 := arithEq_of_rows h (row := 139) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_1
  simp only [← c9036, k3, ← c9037, k3, ← c9038] at e_139_1
  have hr := e_139_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 75) = band (a (.wire 139 3)) (a (.wire 139 7)) := by
  have c9039 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9040 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9041 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_18 := arithEq_of_rows h (row := 122) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_18
  simp only [← c9039, ← c9040, ← c9041] at e_122_18
  have hr := e_122_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 15) = bselect (a (.wire 122 75)) (a (.virt 11205)) (a (.wire 138 31)) := by
  have c9042 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9043 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9044 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9045 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9046 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9047 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_2 := arithEq_of_rows h (row := 139) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_2
  simp only [← c9042, ← c9043, ← c9044] at e_139_2
  have e_139_3 := arithEq_of_rows h (row := 139) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_3
  simp only [← c9045, ← c9046, ← c9047] at e_139_3
  have hr := e_139_3
  simp only [e_139_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 23) = bselect (a (.wire 122 75)) (a (.virt 11206)) (a (.wire 138 39)) := by
  have c9048 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9049 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9050 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9051 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9052 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9053 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_4 := arithEq_of_rows h (row := 139) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_4
  simp only [← c9048, ← c9049, ← c9050] at e_139_4
  have e_139_5 := arithEq_of_rows h (row := 139) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_5
  simp only [← c9051, ← c9052, ← c9053] at e_139_5
  have hr := e_139_5
  simp only [e_139_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 31) = bselect (a (.wire 122 75)) (a (.virt 11207)) (a (.wire 138 47)) := by
  have c9054 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9055 := (publicBatchWrapper89_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9056 := (publicBatchWrapper89_copies283 a h).1
  have c9057 := (publicBatchWrapper89_copies283 a h).2.1
  have c9058 := (publicBatchWrapper89_copies283 a h).2.2.1
  have c9059 := (publicBatchWrapper89_copies283 a h).2.2.2.1
  have e_139_6 := arithEq_of_rows h (row := 139) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_6
  simp only [← c9054, ← c9055, ← c9056] at e_139_6
  have e_139_7 := arithEq_of_rows h (row := 139) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_7
  simp only [← c9057, ← c9058, ← c9059] at e_139_7
  have hr := e_139_7
  simp only [e_139_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 39) = bselect (a (.wire 122 75)) (a (.virt 11208)) (a (.wire 138 55)) := by
  have c9060 := (publicBatchWrapper89_copies283 a h).2.2.2.2.1
  have c9061 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.1
  have c9062 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.1
  have c9063 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.1
  have c9064 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.1
  have c9065 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.1
  have e_139_8 := arithEq_of_rows h (row := 139) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_8
  simp only [← c9060, ← c9061, ← c9062] at e_139_8
  have e_139_9 := arithEq_of_rows h (row := 139) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_9
  simp only [← c9063, ← c9064, ← c9065] at e_139_9
  have hr := e_139_9
  simp only [e_139_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 47) = bselect (a (.wire 122 75)) (a (.virt 11209)) (a (.wire 138 63)) := by
  have c9066 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9067 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9068 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9069 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9070 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9071 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_10 := arithEq_of_rows h (row := 139) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_10
  simp only [← c9066, ← c9067, ← c9068] at e_139_10
  have e_139_11 := arithEq_of_rows h (row := 139) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_11
  simp only [← c9069, ← c9070, ← c9071] at e_139_11
  have hr := e_139_11
  simp only [e_139_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 55) = bselect (a (.wire 122 75)) (a (.virt 11203)) (a (.wire 138 71)) := by
  have c9072 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9073 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9074 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9075 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9076 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9077 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_12 := arithEq_of_rows h (row := 139) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_12
  simp only [← c9072, ← c9073, ← c9074] at e_139_12
  have e_139_13 := arithEq_of_rows h (row := 139) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_13
  simp only [← c9075, ← c9076, ← c9077] at e_139_13
  have hr := e_139_13
  simp only [e_139_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 63) = bselect (a (.wire 122 75)) (a (.virt 11204)) (a (.wire 138 79)) := by
  have c9078 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9079 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9080 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9081 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9082 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9083 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_14 := arithEq_of_rows h (row := 139) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_14
  simp only [← c9078, ← c9079, ← c9080] at e_139_14
  have e_139_15 := arithEq_of_rows h (row := 139) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_15
  simp only [← c9081, ← c9082, ← c9083] at e_139_15
  have hr := e_139_15
  simp only [e_139_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 31) = bor (a (.wire 134 27)) (a (.wire 139 3)) := by
  have c9084 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9085 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9086 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9087 := (publicBatchWrapper89_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9088 := (publicBatchWrapper89_copies284 a h).1
  have c9089 := (publicBatchWrapper89_copies284 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_15 := arithEq_of_rows h (row := 126) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_15
  simp only [← c9084, ← c9085, ← c9086] at e_126_15
  have e_134_7 := arithEq_of_rows h (row := 134) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_7
  simp only [← c9087, ← c9088, k3, ← c9089] at e_134_7
  have hr := e_134_7
  simp only [e_126_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 67) = bnot (a (.wire 57 7)) := by
  have c9090 := (publicBatchWrapper89_copies284 a h).2.2.1
  have c9091 := (publicBatchWrapper89_copies284 a h).2.2.2.1
  have c9092 := (publicBatchWrapper89_copies284 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_139_16 := arithEq_of_rows h (row := 139) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_16
  simp only [← c9090, k3, ← c9091, k3, ← c9092] at e_139_16
  have hr := e_139_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 71) = bnot (a (.wire 134 31)) := by
  have c9093 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.1
  have c9094 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.1
  have c9095 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_139_17 := arithEq_of_rows h (row := 139) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_17
  simp only [← c9093, k3, ← c9094, k3, ← c9095] at e_139_17
  have hr := e_139_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 122 79) = band (a (.wire 139 67)) (a (.wire 139 71)) := by
  have c9096 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.1
  have c9097 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.1
  have c9098 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_122_19 := arithEq_of_rows h (row := 122) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_19
  simp only [← c9096, ← c9097, ← c9098] at e_122_19
  have hr := e_122_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 139 79) = bselect (a (.wire 122 79)) (a (.virt 11405)) (a (.wire 139 15)) := by
  have c9099 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9100 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9101 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9102 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9103 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9104 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_18 := arithEq_of_rows h (row := 139) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_18
  simp only [← c9099, ← c9100, ← c9101] at e_139_18
  have e_139_19 := arithEq_of_rows h (row := 139) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_19
  simp only [← c9102, ← c9103, ← c9104] at e_139_19
  have hr := e_139_19
  simp only [e_139_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 7) = bselect (a (.wire 122 79)) (a (.virt 11406)) (a (.wire 139 23)) := by
  have c9105 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9106 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9107 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9108 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9109 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9110 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_0 := arithEq_of_rows h (row := 140) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_0
  simp only [← c9105, ← c9106, ← c9107] at e_140_0
  have e_140_1 := arithEq_of_rows h (row := 140) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_1
  simp only [← c9108, ← c9109, ← c9110] at e_140_1
  have hr := e_140_1
  simp only [e_140_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 15) = bselect (a (.wire 122 79)) (a (.virt 11407)) (a (.wire 139 31)) := by
  have c9111 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9112 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9113 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9114 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9115 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9116 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_2 := arithEq_of_rows h (row := 140) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_2
  simp only [← c9111, ← c9112, ← c9113] at e_140_2
  have e_140_3 := arithEq_of_rows h (row := 140) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_3
  simp only [← c9114, ← c9115, ← c9116] at e_140_3
  have hr := e_140_3
  simp only [e_140_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 23) = bselect (a (.wire 122 79)) (a (.virt 11408)) (a (.wire 139 39)) := by
  have c9117 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9118 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9119 := (publicBatchWrapper89_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9120 := (publicBatchWrapper89_copies285 a h).1
  have c9121 := (publicBatchWrapper89_copies285 a h).2.1
  have c9122 := (publicBatchWrapper89_copies285 a h).2.2.1
  have e_140_4 := arithEq_of_rows h (row := 140) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_4
  simp only [← c9117, ← c9118, ← c9119] at e_140_4
  have e_140_5 := arithEq_of_rows h (row := 140) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_5
  simp only [← c9120, ← c9121, ← c9122] at e_140_5
  have hr := e_140_5
  simp only [e_140_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 31) = bselect (a (.wire 122 79)) (a (.virt 11409)) (a (.wire 139 47)) := by
  have c9123 := (publicBatchWrapper89_copies285 a h).2.2.2.1
  have c9124 := (publicBatchWrapper89_copies285 a h).2.2.2.2.1
  have c9125 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.1
  have c9126 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.1
  have c9127 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.1
  have c9128 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.1
  have e_140_6 := arithEq_of_rows h (row := 140) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_6
  simp only [← c9123, ← c9124, ← c9125] at e_140_6
  have e_140_7 := arithEq_of_rows h (row := 140) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_7
  simp only [← c9126, ← c9127, ← c9128] at e_140_7
  have hr := e_140_7
  simp only [e_140_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 39) = bselect (a (.wire 122 79)) (a (.virt 11403)) (a (.wire 139 55)) := by
  have c9129 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.1
  have c9130 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9131 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9132 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9133 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9134 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_8 := arithEq_of_rows h (row := 140) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_8
  simp only [← c9129, ← c9130, ← c9131] at e_140_8
  have e_140_9 := arithEq_of_rows h (row := 140) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_9
  simp only [← c9132, ← c9133, ← c9134] at e_140_9
  have hr := e_140_9
  simp only [e_140_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 47) = bselect (a (.wire 122 79)) (a (.virt 11404)) (a (.wire 139 63)) := by
  have c9135 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9136 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9137 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9138 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9139 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9140 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_10 := arithEq_of_rows h (row := 140) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_10
  simp only [← c9135, ← c9136, ← c9137] at e_140_10
  have e_140_11 := arithEq_of_rows h (row := 140) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_11
  simp only [← c9138, ← c9139, ← c9140] at e_140_11
  have hr := e_140_11
  simp only [e_140_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 35) = bor (a (.wire 134 31)) (a (.wire 139 67)) := by
  have c9141 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9142 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9143 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9144 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9145 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9146 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_16 := arithEq_of_rows h (row := 126) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_16
  simp only [← c9141, ← c9142, ← c9143] at e_126_16
  have e_134_8 := arithEq_of_rows h (row := 134) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_8
  simp only [← c9144, ← c9145, k3, ← c9146] at e_134_8
  have hr := e_134_8
  simp only [e_126_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 51) = bnot (a (.wire 57 51)) := by
  have c9147 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9148 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9149 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_140_12 := arithEq_of_rows h (row := 140) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_12
  simp only [← c9147, k3, ← c9148, k3, ← c9149] at e_140_12
  have hr := e_140_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 55) = bnot (a (.wire 134 35)) := by
  have c9150 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9151 := (publicBatchWrapper89_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9152 := (publicBatchWrapper89_copies286 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_140_13 := arithEq_of_rows h (row := 140) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_13
  simp only [← c9150, k3, ← c9151, k3, ← c9152] at e_140_13
  have hr := e_140_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 3) = band (a (.wire 140 51)) (a (.wire 140 55)) := by
  have c9153 := (publicBatchWrapper89_copies286 a h).2.1
  have c9154 := (publicBatchWrapper89_copies286 a h).2.2.1
  have c9155 := (publicBatchWrapper89_copies286 a h).2.2.2.1
  have e_141_0 := arithEq_of_rows h (row := 141) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_0
  simp only [← c9153, ← c9154, ← c9155] at e_141_0
  have hr := e_141_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 63) = bselect (a (.wire 141 3)) (a (.virt 11605)) (a (.wire 139 79)) := by
  have c9156 := (publicBatchWrapper89_copies286 a h).2.2.2.2.1
  have c9157 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.1
  have c9158 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.1
  have c9159 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.1
  have c9160 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.1
  have c9161 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.1
  have e_140_14 := arithEq_of_rows h (row := 140) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_14
  simp only [← c9156, ← c9157, ← c9158] at e_140_14
  have e_140_15 := arithEq_of_rows h (row := 140) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_15
  simp only [← c9159, ← c9160, ← c9161] at e_140_15
  have hr := e_140_15
  simp only [e_140_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 71) = bselect (a (.wire 141 3)) (a (.virt 11606)) (a (.wire 140 7)) := by
  have c9162 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9163 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9164 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9165 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9166 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9167 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_16 := arithEq_of_rows h (row := 140) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_16
  simp only [← c9162, ← c9163, ← c9164] at e_140_16
  have e_140_17 := arithEq_of_rows h (row := 140) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_17
  simp only [← c9165, ← c9166, ← c9167] at e_140_17
  have hr := e_140_17
  simp only [e_140_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 140 79) = bselect (a (.wire 141 3)) (a (.virt 11607)) (a (.wire 140 15)) := by
  have c9168 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9169 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9170 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9171 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9172 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9173 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_18 := arithEq_of_rows h (row := 140) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_18
  simp only [← c9168, ← c9169, ← c9170] at e_140_18
  have e_140_19 := arithEq_of_rows h (row := 140) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_19
  simp only [← c9171, ← c9172, ← c9173] at e_140_19
  have hr := e_140_19
  simp only [e_140_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 7) = bselect (a (.wire 141 3)) (a (.virt 11608)) (a (.wire 140 23)) := by
  have c9174 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9175 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9176 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9177 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9178 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9179 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_0 := arithEq_of_rows h (row := 142) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_0
  simp only [← c9174, ← c9175, ← c9176] at e_142_0
  have e_142_1 := arithEq_of_rows h (row := 142) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_1
  simp only [← c9177, ← c9178, ← c9179] at e_142_1
  have hr := e_142_1
  simp only [e_142_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 15) = bselect (a (.wire 141 3)) (a (.virt 11609)) (a (.wire 140 31)) := by
  have c9180 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9181 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9182 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9183 := (publicBatchWrapper89_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9184 := (publicBatchWrapper89_copies287 a h).1
  have c9185 := (publicBatchWrapper89_copies287 a h).2.1
  have e_142_2 := arithEq_of_rows h (row := 142) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_2
  simp only [← c9180, ← c9181, ← c9182] at e_142_2
  have e_142_3 := arithEq_of_rows h (row := 142) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_3
  simp only [← c9183, ← c9184, ← c9185] at e_142_3
  have hr := e_142_3
  simp only [e_142_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 23) = bselect (a (.wire 141 3)) (a (.virt 11603)) (a (.wire 140 39)) := by
  have c9186 := (publicBatchWrapper89_copies287 a h).2.2.1
  have c9187 := (publicBatchWrapper89_copies287 a h).2.2.2.1
  have c9188 := (publicBatchWrapper89_copies287 a h).2.2.2.2.1
  have c9189 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.1
  have c9190 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.1
  have c9191 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.1
  have e_142_4 := arithEq_of_rows h (row := 142) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_4
  simp only [← c9186, ← c9187, ← c9188] at e_142_4
  have e_142_5 := arithEq_of_rows h (row := 142) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_5
  simp only [← c9189, ← c9190, ← c9191] at e_142_5
  have hr := e_142_5
  simp only [e_142_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 31) = bselect (a (.wire 141 3)) (a (.virt 11604)) (a (.wire 140 47)) := by
  have c9192 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.1
  have c9193 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.1
  have c9194 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9195 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9196 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9197 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_6 := arithEq_of_rows h (row := 142) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_6
  simp only [← c9192, ← c9193, ← c9194] at e_142_6
  have e_142_7 := arithEq_of_rows h (row := 142) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_7
  simp only [← c9195, ← c9196, ← c9197] at e_142_7
  have hr := e_142_7
  simp only [e_142_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 39) = bor (a (.wire 134 35)) (a (.wire 140 51)) := by
  have c9198 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9199 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9200 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9201 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9202 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9203 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_17 := arithEq_of_rows h (row := 126) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_17
  simp only [← c9198, ← c9199, ← c9200] at e_126_17
  have e_134_9 := arithEq_of_rows h (row := 134) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_9
  simp only [← c9201, ← c9202, k3, ← c9203] at e_134_9
  have hr := e_134_9
  simp only [e_126_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 35) = bnot (a (.wire 58 15)) := by
  have c9204 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9205 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9206 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_142_8 := arithEq_of_rows h (row := 142) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_8
  simp only [← c9204, k3, ← c9205, k3, ← c9206] at e_142_8
  have hr := e_142_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 39) = bnot (a (.wire 134 39)) := by
  have c9207 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9208 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9209 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_142_9 := arithEq_of_rows h (row := 142) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_9
  simp only [← c9207, k3, ← c9208, k3, ← c9209] at e_142_9
  have hr := e_142_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 7) = band (a (.wire 142 35)) (a (.wire 142 39)) := by
  have c9210 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9211 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9212 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_1 := arithEq_of_rows h (row := 141) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_1
  simp only [← c9210, ← c9211, ← c9212] at e_141_1
  have hr := e_141_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 47) = bselect (a (.wire 141 7)) (a (.virt 11805)) (a (.wire 140 63)) := by
  have c9213 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9214 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9215 := (publicBatchWrapper89_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9216 := (publicBatchWrapper89_copies288 a h).1
  have c9217 := (publicBatchWrapper89_copies288 a h).2.1
  have c9218 := (publicBatchWrapper89_copies288 a h).2.2.1
  have e_142_10 := arithEq_of_rows h (row := 142) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_10
  simp only [← c9213, ← c9214, ← c9215] at e_142_10
  have e_142_11 := arithEq_of_rows h (row := 142) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_11
  simp only [← c9216, ← c9217, ← c9218] at e_142_11
  have hr := e_142_11
  simp only [e_142_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 55) = bselect (a (.wire 141 7)) (a (.virt 11806)) (a (.wire 140 71)) := by
  have c9219 := (publicBatchWrapper89_copies288 a h).2.2.2.1
  have c9220 := (publicBatchWrapper89_copies288 a h).2.2.2.2.1
  have c9221 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.1
  have c9222 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.1
  have c9223 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.1
  have c9224 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.1
  have e_142_12 := arithEq_of_rows h (row := 142) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_12
  simp only [← c9219, ← c9220, ← c9221] at e_142_12
  have e_142_13 := arithEq_of_rows h (row := 142) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_13
  simp only [← c9222, ← c9223, ← c9224] at e_142_13
  have hr := e_142_13
  simp only [e_142_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 63) = bselect (a (.wire 141 7)) (a (.virt 11807)) (a (.wire 140 79)) := by
  have c9225 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.1
  have c9226 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9227 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9228 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9229 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9230 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_14 := arithEq_of_rows h (row := 142) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_14
  simp only [← c9225, ← c9226, ← c9227] at e_142_14
  have e_142_15 := arithEq_of_rows h (row := 142) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_15
  simp only [← c9228, ← c9229, ← c9230] at e_142_15
  have hr := e_142_15
  simp only [e_142_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 71) = bselect (a (.wire 141 7)) (a (.virt 11808)) (a (.wire 142 7)) := by
  have c9231 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9232 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9233 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9234 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9235 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9236 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_16 := arithEq_of_rows h (row := 142) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_16
  simp only [← c9231, ← c9232, ← c9233] at e_142_16
  have e_142_17 := arithEq_of_rows h (row := 142) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_17
  simp only [← c9234, ← c9235, ← c9236] at e_142_17
  have hr := e_142_17
  simp only [e_142_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 142 79) = bselect (a (.wire 141 7)) (a (.virt 11809)) (a (.wire 142 15)) := by
  have c9237 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9238 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9239 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9240 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9241 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9242 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_18 := arithEq_of_rows h (row := 142) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_18
  simp only [← c9237, ← c9238, ← c9239] at e_142_18
  have e_142_19 := arithEq_of_rows h (row := 142) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_19
  simp only [← c9240, ← c9241, ← c9242] at e_142_19
  have hr := e_142_19
  simp only [e_142_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 7) = bselect (a (.wire 141 7)) (a (.virt 11803)) (a (.wire 142 23)) := by
  have c9243 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9244 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9245 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9246 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9247 := (publicBatchWrapper89_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9248 := (publicBatchWrapper89_copies289 a h).1
  have e_143_0 := arithEq_of_rows h (row := 143) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_0
  simp only [← c9243, ← c9244, ← c9245] at e_143_0
  have e_143_1 := arithEq_of_rows h (row := 143) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_1
  simp only [← c9246, ← c9247, ← c9248] at e_143_1
  have hr := e_143_1
  simp only [e_143_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 15) = bselect (a (.wire 141 7)) (a (.virt 11804)) (a (.wire 142 31)) := by
  have c9249 := (publicBatchWrapper89_copies289 a h).2.1
  have c9250 := (publicBatchWrapper89_copies289 a h).2.2.1
  have c9251 := (publicBatchWrapper89_copies289 a h).2.2.2.1
  have c9252 := (publicBatchWrapper89_copies289 a h).2.2.2.2.1
  have c9253 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.1
  have c9254 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.1
  have e_143_2 := arithEq_of_rows h (row := 143) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_2
  simp only [← c9249, ← c9250, ← c9251] at e_143_2
  have e_143_3 := arithEq_of_rows h (row := 143) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_3
  simp only [← c9252, ← c9253, ← c9254] at e_143_3
  have hr := e_143_3
  simp only [e_143_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 43) = bor (a (.wire 134 39)) (a (.wire 142 35)) := by
  have c9255 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.1
  have c9256 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.1
  have c9257 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.1
  have c9258 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9259 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9260 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_18 := arithEq_of_rows h (row := 126) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_18
  simp only [← c9255, ← c9256, ← c9257] at e_126_18
  have e_134_10 := arithEq_of_rows h (row := 134) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_10
  simp only [← c9258, ← c9259, k3, ← c9260] at e_134_10
  have hr := e_134_10
  simp only [e_126_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 19) = bnot (a (.wire 58 59)) := by
  have c9261 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9262 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9263 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_143_4 := arithEq_of_rows h (row := 143) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_4
  simp only [← c9261, k3, ← c9262, k3, ← c9263] at e_143_4
  have hr := e_143_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 23) = bnot (a (.wire 134 43)) := by
  have c9264 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9265 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9266 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_143_5 := arithEq_of_rows h (row := 143) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_5
  simp only [← c9264, k3, ← c9265, k3, ← c9266] at e_143_5
  have hr := e_143_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 11) = band (a (.wire 143 19)) (a (.wire 143 23)) := by
  have c9267 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9268 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9269 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_2 := arithEq_of_rows h (row := 141) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_2
  simp only [← c9267, ← c9268, ← c9269] at e_141_2
  have hr := e_141_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 31) = bselect (a (.wire 141 11)) (a (.virt 12005)) (a (.wire 142 47)) := by
  have c9270 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9271 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9272 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9273 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9274 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9275 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_143_6 := arithEq_of_rows h (row := 143) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_6
  simp only [← c9270, ← c9271, ← c9272] at e_143_6
  have e_143_7 := arithEq_of_rows h (row := 143) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_7
  simp only [← c9273, ← c9274, ← c9275] at e_143_7
  have hr := e_143_7
  simp only [e_143_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 39) = bselect (a (.wire 141 11)) (a (.virt 12006)) (a (.wire 142 55)) := by
  have c9276 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9277 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9278 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9279 := (publicBatchWrapper89_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9280 := (publicBatchWrapper89_copies290 a h).1
  have c9281 := (publicBatchWrapper89_copies290 a h).2.1
  have e_143_8 := arithEq_of_rows h (row := 143) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_8
  simp only [← c9276, ← c9277, ← c9278] at e_143_8
  have e_143_9 := arithEq_of_rows h (row := 143) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_9
  simp only [← c9279, ← c9280, ← c9281] at e_143_9
  have hr := e_143_9
  simp only [e_143_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 47) = bselect (a (.wire 141 11)) (a (.virt 12007)) (a (.wire 142 63)) := by
  have c9282 := (publicBatchWrapper89_copies290 a h).2.2.1
  have c9283 := (publicBatchWrapper89_copies290 a h).2.2.2.1
  have c9284 := (publicBatchWrapper89_copies290 a h).2.2.2.2.1
  have c9285 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.1
  have c9286 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.1
  have c9287 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.1
  have e_143_10 := arithEq_of_rows h (row := 143) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_10
  simp only [← c9282, ← c9283, ← c9284] at e_143_10
  have e_143_11 := arithEq_of_rows h (row := 143) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_11
  simp only [← c9285, ← c9286, ← c9287] at e_143_11
  have hr := e_143_11
  simp only [e_143_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 55) = bselect (a (.wire 141 11)) (a (.virt 12008)) (a (.wire 142 71)) := by
  have c9288 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.1
  have c9289 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.1
  have c9290 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9291 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9292 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9293 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_143_12 := arithEq_of_rows h (row := 143) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_12
  simp only [← c9288, ← c9289, ← c9290] at e_143_12
  have e_143_13 := arithEq_of_rows h (row := 143) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_13
  simp only [← c9291, ← c9292, ← c9293] at e_143_13
  have hr := e_143_13
  simp only [e_143_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 63) = bselect (a (.wire 141 11)) (a (.virt 12009)) (a (.wire 142 79)) := by
  have c9294 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9295 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9296 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9297 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9298 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9299 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_143_14 := arithEq_of_rows h (row := 143) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_14
  simp only [← c9294, ← c9295, ← c9296] at e_143_14
  have e_143_15 := arithEq_of_rows h (row := 143) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_15
  simp only [← c9297, ← c9298, ← c9299] at e_143_15
  have hr := e_143_15
  simp only [e_143_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 71) = bselect (a (.wire 141 11)) (a (.virt 12003)) (a (.wire 143 7)) := by
  have c9300 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9301 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9302 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9303 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9304 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9305 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_143_16 := arithEq_of_rows h (row := 143) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_16
  simp only [← c9300, ← c9301, ← c9302] at e_143_16
  have e_143_17 := arithEq_of_rows h (row := 143) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_17
  simp only [← c9303, ← c9304, ← c9305] at e_143_17
  have hr := e_143_17
  simp only [e_143_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 143 79) = bselect (a (.wire 141 11)) (a (.virt 12004)) (a (.wire 143 15)) := by
  have c9306 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9307 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9308 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9309 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9310 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9311 := (publicBatchWrapper89_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_143_18 := arithEq_of_rows h (row := 143) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_18
  simp only [← c9306, ← c9307, ← c9308] at e_143_18
  have e_143_19 := arithEq_of_rows h (row := 143) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_19
  simp only [← c9309, ← c9310, ← c9311] at e_143_19
  have hr := e_143_19
  simp only [e_143_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 47) = bor (a (.wire 134 43)) (a (.wire 143 19)) := by
  have c9312 := (publicBatchWrapper89_copies291 a h).1
  have c9313 := (publicBatchWrapper89_copies291 a h).2.1
  have c9314 := (publicBatchWrapper89_copies291 a h).2.2.1
  have c9315 := (publicBatchWrapper89_copies291 a h).2.2.2.1
  have c9316 := (publicBatchWrapper89_copies291 a h).2.2.2.2.1
  have c9317 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_126_19 := arithEq_of_rows h (row := 126) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_19
  simp only [← c9312, ← c9313, ← c9314] at e_126_19
  have e_134_11 := arithEq_of_rows h (row := 134) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_11
  simp only [← c9315, ← c9316, k3, ← c9317] at e_134_11
  have hr := e_134_11
  simp only [e_126_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 3) = bnot (a (.wire 60 23)) := by
  have c9318 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.1
  have c9319 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.1
  have c9320 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_144_0 := arithEq_of_rows h (row := 144) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_0
  simp only [← c9318, k3, ← c9319, k3, ← c9320] at e_144_0
  have hr := e_144_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 7) = bnot (a (.wire 134 47)) := by
  have c9321 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.1
  have c9322 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9323 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_144_1 := arithEq_of_rows h (row := 144) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_1
  simp only [← c9321, k3, ← c9322, k3, ← c9323] at e_144_1
  have hr := e_144_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 15) = band (a (.wire 144 3)) (a (.wire 144 7)) := by
  have c9324 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9325 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9326 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_3 := arithEq_of_rows h (row := 141) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_3
  simp only [← c9324, ← c9325, ← c9326] at e_141_3
  have hr := e_141_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 15) = bselect (a (.wire 141 15)) (a (.virt 12205)) (a (.wire 143 31)) := by
  have c9327 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9328 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9329 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9330 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9331 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9332 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_2 := arithEq_of_rows h (row := 144) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_2
  simp only [← c9327, ← c9328, ← c9329] at e_144_2
  have e_144_3 := arithEq_of_rows h (row := 144) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_3
  simp only [← c9330, ← c9331, ← c9332] at e_144_3
  have hr := e_144_3
  simp only [e_144_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 23) = bselect (a (.wire 141 15)) (a (.virt 12206)) (a (.wire 143 39)) := by
  have c9333 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9334 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9335 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9336 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9337 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9338 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_4 := arithEq_of_rows h (row := 144) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_4
  simp only [← c9333, ← c9334, ← c9335] at e_144_4
  have e_144_5 := arithEq_of_rows h (row := 144) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_5
  simp only [← c9336, ← c9337, ← c9338] at e_144_5
  have hr := e_144_5
  simp only [e_144_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 31) = bselect (a (.wire 141 15)) (a (.virt 12207)) (a (.wire 143 47)) := by
  have c9339 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9340 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9341 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9342 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9343 := (publicBatchWrapper89_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9344 := (publicBatchWrapper89_copies292 a h).1
  have e_144_6 := arithEq_of_rows h (row := 144) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_6
  simp only [← c9339, ← c9340, ← c9341] at e_144_6
  have e_144_7 := arithEq_of_rows h (row := 144) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_7
  simp only [← c9342, ← c9343, ← c9344] at e_144_7
  have hr := e_144_7
  simp only [e_144_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 39) = bselect (a (.wire 141 15)) (a (.virt 12208)) (a (.wire 143 55)) := by
  have c9345 := (publicBatchWrapper89_copies292 a h).2.1
  have c9346 := (publicBatchWrapper89_copies292 a h).2.2.1
  have c9347 := (publicBatchWrapper89_copies292 a h).2.2.2.1
  have c9348 := (publicBatchWrapper89_copies292 a h).2.2.2.2.1
  have c9349 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.1
  have c9350 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.1
  have e_144_8 := arithEq_of_rows h (row := 144) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_8
  simp only [← c9345, ← c9346, ← c9347] at e_144_8
  have e_144_9 := arithEq_of_rows h (row := 144) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_9
  simp only [← c9348, ← c9349, ← c9350] at e_144_9
  have hr := e_144_9
  simp only [e_144_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 47) = bselect (a (.wire 141 15)) (a (.virt 12209)) (a (.wire 143 63)) := by
  have c9351 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.1
  have c9352 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.1
  have c9353 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.1
  have c9354 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9355 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9356 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_10 := arithEq_of_rows h (row := 144) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_10
  simp only [← c9351, ← c9352, ← c9353] at e_144_10
  have e_144_11 := arithEq_of_rows h (row := 144) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_11
  simp only [← c9354, ← c9355, ← c9356] at e_144_11
  have hr := e_144_11
  simp only [e_144_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 55) = bselect (a (.wire 141 15)) (a (.virt 12203)) (a (.wire 143 71)) := by
  have c9357 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9358 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9359 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9360 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9361 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9362 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_12 := arithEq_of_rows h (row := 144) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_12
  simp only [← c9357, ← c9358, ← c9359] at e_144_12
  have e_144_13 := arithEq_of_rows h (row := 144) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_13
  simp only [← c9360, ← c9361, ← c9362] at e_144_13
  have hr := e_144_13
  simp only [e_144_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 63) = bselect (a (.wire 141 15)) (a (.virt 12204)) (a (.wire 143 79)) := by
  have c9363 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9364 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9365 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9366 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9367 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9368 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_14 := arithEq_of_rows h (row := 144) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_14
  simp only [← c9363, ← c9364, ← c9365] at e_144_14
  have e_144_15 := arithEq_of_rows h (row := 144) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_15
  simp only [← c9366, ← c9367, ← c9368] at e_144_15
  have hr := e_144_15
  simp only [e_144_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 51) = bor (a (.wire 134 47)) (a (.wire 144 3)) := by
  have c9369 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9370 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9371 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9372 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9373 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9374 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_134_12 := arithEq_of_rows h (row := 134) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_12
  simp only [← c9372, ← c9373, k3, ← c9374] at e_134_12
  have e_145_0 := arithEq_of_rows h (row := 145) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_0
  simp only [← c9369, ← c9370, ← c9371] at e_145_0
  have hr := e_134_12
  simp only [e_145_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 67) = bnot (a (.wire 60 67)) := by
  have c9375 := (publicBatchWrapper89_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9376 := (publicBatchWrapper89_copies293 a h).1
  have c9377 := (publicBatchWrapper89_copies293 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_144_16 := arithEq_of_rows h (row := 144) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_16
  simp only [← c9375, k3, ← c9376, k3, ← c9377] at e_144_16
  have hr := e_144_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 71) = bnot (a (.wire 134 51)) := by
  have c9378 := (publicBatchWrapper89_copies293 a h).2.2.1
  have c9379 := (publicBatchWrapper89_copies293 a h).2.2.2.1
  have c9380 := (publicBatchWrapper89_copies293 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_144_17 := arithEq_of_rows h (row := 144) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_17
  simp only [← c9378, k3, ← c9379, k3, ← c9380] at e_144_17
  have hr := e_144_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 19) = band (a (.wire 144 67)) (a (.wire 144 71)) := by
  have c9381 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.1
  have c9382 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.1
  have c9383 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.1
  have e_141_4 := arithEq_of_rows h (row := 141) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_4
  simp only [← c9381, ← c9382, ← c9383] at e_141_4
  have hr := e_141_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 144 79) = bselect (a (.wire 141 19)) (a (.virt 12405)) (a (.wire 144 15)) := by
  have c9384 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.1
  have c9385 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.1
  have c9386 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9387 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9388 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9389 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_18 := arithEq_of_rows h (row := 144) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_18
  simp only [← c9384, ← c9385, ← c9386] at e_144_18
  have e_144_19 := arithEq_of_rows h (row := 144) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_19
  simp only [← c9387, ← c9388, ← c9389] at e_144_19
  have hr := e_144_19
  simp only [e_144_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 7) = bselect (a (.wire 141 19)) (a (.virt 12406)) (a (.wire 144 23)) := by
  have c9390 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9391 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9392 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9393 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9394 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9395 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_0 := arithEq_of_rows h (row := 146) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_0
  simp only [← c9390, ← c9391, ← c9392] at e_146_0
  have e_146_1 := arithEq_of_rows h (row := 146) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_1
  simp only [← c9393, ← c9394, ← c9395] at e_146_1
  have hr := e_146_1
  simp only [e_146_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 15) = bselect (a (.wire 141 19)) (a (.virt 12407)) (a (.wire 144 31)) := by
  have c9396 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9397 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9398 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9399 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9400 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9401 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_2 := arithEq_of_rows h (row := 146) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_2
  simp only [← c9396, ← c9397, ← c9398] at e_146_2
  have e_146_3 := arithEq_of_rows h (row := 146) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_3
  simp only [← c9399, ← c9400, ← c9401] at e_146_3
  have hr := e_146_3
  simp only [e_146_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 23) = bselect (a (.wire 141 19)) (a (.virt 12408)) (a (.wire 144 39)) := by
  have c9402 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9403 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9404 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9405 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9406 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9407 := (publicBatchWrapper89_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_146_4 := arithEq_of_rows h (row := 146) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_4
  simp only [← c9402, ← c9403, ← c9404] at e_146_4
  have e_146_5 := arithEq_of_rows h (row := 146) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_5
  simp only [← c9405, ← c9406, ← c9407] at e_146_5
  have hr := e_146_5
  simp only [e_146_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 31) = bselect (a (.wire 141 19)) (a (.virt 12409)) (a (.wire 144 47)) := by
  have c9408 := (publicBatchWrapper89_copies294 a h).1
  have c9409 := (publicBatchWrapper89_copies294 a h).2.1
  have c9410 := (publicBatchWrapper89_copies294 a h).2.2.1
  have c9411 := (publicBatchWrapper89_copies294 a h).2.2.2.1
  have c9412 := (publicBatchWrapper89_copies294 a h).2.2.2.2.1
  have c9413 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.1
  have e_146_6 := arithEq_of_rows h (row := 146) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_6
  simp only [← c9408, ← c9409, ← c9410] at e_146_6
  have e_146_7 := arithEq_of_rows h (row := 146) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_7
  simp only [← c9411, ← c9412, ← c9413] at e_146_7
  have hr := e_146_7
  simp only [e_146_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 39) = bselect (a (.wire 141 19)) (a (.virt 12403)) (a (.wire 144 55)) := by
  have c9414 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.1
  have c9415 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.1
  have c9416 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.1
  have c9417 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.1
  have c9418 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9419 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_8 := arithEq_of_rows h (row := 146) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_8
  simp only [← c9414, ← c9415, ← c9416] at e_146_8
  have e_146_9 := arithEq_of_rows h (row := 146) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_9
  simp only [← c9417, ← c9418, ← c9419] at e_146_9
  have hr := e_146_9
  simp only [e_146_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 47) = bselect (a (.wire 141 19)) (a (.virt 12404)) (a (.wire 144 63)) := by
  have c9420 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9421 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9422 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9423 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9424 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9425 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_10 := arithEq_of_rows h (row := 146) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_10
  simp only [← c9420, ← c9421, ← c9422] at e_146_10
  have e_146_11 := arithEq_of_rows h (row := 146) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_11
  simp only [← c9423, ← c9424, ← c9425] at e_146_11
  have hr := e_146_11
  simp only [e_146_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 55) = bor (a (.wire 134 51)) (a (.wire 144 67)) := by
  have c9426 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9427 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9428 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9429 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9430 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9431 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_134_13 := arithEq_of_rows h (row := 134) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_13
  simp only [← c9429, ← c9430, k3, ← c9431] at e_134_13
  have e_145_1 := arithEq_of_rows h (row := 145) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_1
  simp only [← c9426, ← c9427, ← c9428] at e_145_1
  have hr := e_134_13
  simp only [e_145_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 51) = bnot (a (.wire 62 31)) := by
  have c9432 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9433 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9434 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_146_12 := arithEq_of_rows h (row := 146) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_12
  simp only [← c9432, k3, ← c9433, k3, ← c9434] at e_146_12
  have hr := e_146_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 55) = bnot (a (.wire 134 55)) := by
  have c9435 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9436 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9437 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_146_13 := arithEq_of_rows h (row := 146) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_13
  simp only [← c9435, k3, ← c9436, k3, ← c9437] at e_146_13
  have hr := e_146_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 23) = band (a (.wire 146 51)) (a (.wire 146 55)) := by
  have c9438 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9439 := (publicBatchWrapper89_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9440 := (publicBatchWrapper89_copies295 a h).1
  have e_141_5 := arithEq_of_rows h (row := 141) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_5
  simp only [← c9438, ← c9439, ← c9440] at e_141_5
  have hr := e_141_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 63) = bselect (a (.wire 141 23)) (a (.virt 12605)) (a (.wire 144 79)) := by
  have c9441 := (publicBatchWrapper89_copies295 a h).2.1
  have c9442 := (publicBatchWrapper89_copies295 a h).2.2.1
  have c9443 := (publicBatchWrapper89_copies295 a h).2.2.2.1
  have c9444 := (publicBatchWrapper89_copies295 a h).2.2.2.2.1
  have c9445 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.1
  have c9446 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.1
  have e_146_14 := arithEq_of_rows h (row := 146) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_14
  simp only [← c9441, ← c9442, ← c9443] at e_146_14
  have e_146_15 := arithEq_of_rows h (row := 146) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_15
  simp only [← c9444, ← c9445, ← c9446] at e_146_15
  have hr := e_146_15
  simp only [e_146_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 71) = bselect (a (.wire 141 23)) (a (.virt 12606)) (a (.wire 146 7)) := by
  have c9447 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.1
  have c9448 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.1
  have c9449 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.1
  have c9450 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9451 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9452 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_16 := arithEq_of_rows h (row := 146) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_16
  simp only [← c9447, ← c9448, ← c9449] at e_146_16
  have e_146_17 := arithEq_of_rows h (row := 146) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_17
  simp only [← c9450, ← c9451, ← c9452] at e_146_17
  have hr := e_146_17
  simp only [e_146_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 146 79) = bselect (a (.wire 141 23)) (a (.virt 12607)) (a (.wire 146 15)) := by
  have c9453 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9454 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9455 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9456 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9457 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9458 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_18 := arithEq_of_rows h (row := 146) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_18
  simp only [← c9453, ← c9454, ← c9455] at e_146_18
  have e_146_19 := arithEq_of_rows h (row := 146) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_19
  simp only [← c9456, ← c9457, ← c9458] at e_146_19
  have hr := e_146_19
  simp only [e_146_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 7) = bselect (a (.wire 141 23)) (a (.virt 12608)) (a (.wire 146 23)) := by
  have c9459 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9460 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9461 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9462 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9463 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9464 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_0 := arithEq_of_rows h (row := 147) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_0
  simp only [← c9459, ← c9460, ← c9461] at e_147_0
  have e_147_1 := arithEq_of_rows h (row := 147) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_1
  simp only [← c9462, ← c9463, ← c9464] at e_147_1
  have hr := e_147_1
  simp only [e_147_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 15) = bselect (a (.wire 141 23)) (a (.virt 12609)) (a (.wire 146 31)) := by
  have c9465 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9466 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9467 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9468 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9469 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9470 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_2 := arithEq_of_rows h (row := 147) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_2
  simp only [← c9465, ← c9466, ← c9467] at e_147_2
  have e_147_3 := arithEq_of_rows h (row := 147) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_3
  simp only [← c9468, ← c9469, ← c9470] at e_147_3
  have hr := e_147_3
  simp only [e_147_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 23) = bselect (a (.wire 141 23)) (a (.virt 12603)) (a (.wire 146 39)) := by
  have c9471 := (publicBatchWrapper89_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9472 := (publicBatchWrapper89_copies296 a h).1
  have c9473 := (publicBatchWrapper89_copies296 a h).2.1
  have c9474 := (publicBatchWrapper89_copies296 a h).2.2.1
  have c9475 := (publicBatchWrapper89_copies296 a h).2.2.2.1
  have c9476 := (publicBatchWrapper89_copies296 a h).2.2.2.2.1
  have e_147_4 := arithEq_of_rows h (row := 147) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_4
  simp only [← c9471, ← c9472, ← c9473] at e_147_4
  have e_147_5 := arithEq_of_rows h (row := 147) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_5
  simp only [← c9474, ← c9475, ← c9476] at e_147_5
  have hr := e_147_5
  simp only [e_147_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 31) = bselect (a (.wire 141 23)) (a (.virt 12604)) (a (.wire 146 47)) := by
  have c9477 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.1
  have c9478 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.1
  have c9479 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.1
  have c9480 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.1
  have c9481 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.1
  have c9482 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_147_6 := arithEq_of_rows h (row := 147) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_6
  simp only [← c9477, ← c9478, ← c9479] at e_147_6
  have e_147_7 := arithEq_of_rows h (row := 147) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_7
  simp only [← c9480, ← c9481, ← c9482] at e_147_7
  have hr := e_147_7
  simp only [e_147_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 59) = bor (a (.wire 134 55)) (a (.wire 146 51)) := by
  have c9483 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9484 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9485 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9486 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9487 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9488 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_134_14 := arithEq_of_rows h (row := 134) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_14
  simp only [← c9486, ← c9487, k3, ← c9488] at e_134_14
  have e_145_2 := arithEq_of_rows h (row := 145) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_2
  simp only [← c9483, ← c9484, ← c9485] at e_145_2
  have hr := e_134_14
  simp only [e_145_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 35) = bnot (a (.wire 62 75)) := by
  have c9489 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9490 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9491 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_147_8 := arithEq_of_rows h (row := 147) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_8
  simp only [← c9489, k3, ← c9490, k3, ← c9491] at e_147_8
  have hr := e_147_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 39) = bnot (a (.wire 134 59)) := by
  have c9492 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9493 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9494 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_147_9 := arithEq_of_rows h (row := 147) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_9
  simp only [← c9492, k3, ← c9493, k3, ← c9494] at e_147_9
  have hr := e_147_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 27) = band (a (.wire 147 35)) (a (.wire 147 39)) := by
  have c9495 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9496 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9497 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_6 := arithEq_of_rows h (row := 141) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_6
  simp only [← c9495, ← c9496, ← c9497] at e_141_6
  have hr := e_141_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 47) = bselect (a (.wire 141 27)) (a (.virt 12805)) (a (.wire 146 63)) := by
  have c9498 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9499 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9500 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9501 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9502 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9503 := (publicBatchWrapper89_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_147_10 := arithEq_of_rows h (row := 147) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_10
  simp only [← c9498, ← c9499, ← c9500] at e_147_10
  have e_147_11 := arithEq_of_rows h (row := 147) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_11
  simp only [← c9501, ← c9502, ← c9503] at e_147_11
  have hr := e_147_11
  simp only [e_147_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 55) = bselect (a (.wire 141 27)) (a (.virt 12806)) (a (.wire 146 71)) := by
  have c9504 := (publicBatchWrapper89_copies297 a h).1
  have c9505 := (publicBatchWrapper89_copies297 a h).2.1
  have c9506 := (publicBatchWrapper89_copies297 a h).2.2.1
  have c9507 := (publicBatchWrapper89_copies297 a h).2.2.2.1
  have c9508 := (publicBatchWrapper89_copies297 a h).2.2.2.2.1
  have c9509 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.1
  have e_147_12 := arithEq_of_rows h (row := 147) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_12
  simp only [← c9504, ← c9505, ← c9506] at e_147_12
  have e_147_13 := arithEq_of_rows h (row := 147) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_13
  simp only [← c9507, ← c9508, ← c9509] at e_147_13
  have hr := e_147_13
  simp only [e_147_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 63) = bselect (a (.wire 141 27)) (a (.virt 12807)) (a (.wire 146 79)) := by
  have c9510 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.1
  have c9511 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.1
  have c9512 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.1
  have c9513 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.1
  have c9514 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9515 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_14 := arithEq_of_rows h (row := 147) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_14
  simp only [← c9510, ← c9511, ← c9512] at e_147_14
  have e_147_15 := arithEq_of_rows h (row := 147) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_15
  simp only [← c9513, ← c9514, ← c9515] at e_147_15
  have hr := e_147_15
  simp only [e_147_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 71) = bselect (a (.wire 141 27)) (a (.virt 12808)) (a (.wire 147 7)) := by
  have c9516 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9517 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9518 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9519 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9520 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9521 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_16 := arithEq_of_rows h (row := 147) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_16
  simp only [← c9516, ← c9517, ← c9518] at e_147_16
  have e_147_17 := arithEq_of_rows h (row := 147) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_17
  simp only [← c9519, ← c9520, ← c9521] at e_147_17
  have hr := e_147_17
  simp only [e_147_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 147 79) = bselect (a (.wire 141 27)) (a (.virt 12809)) (a (.wire 147 15)) := by
  have c9522 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9523 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9524 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9525 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9526 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9527 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_18 := arithEq_of_rows h (row := 147) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_18
  simp only [← c9522, ← c9523, ← c9524] at e_147_18
  have e_147_19 := arithEq_of_rows h (row := 147) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_19
  simp only [← c9525, ← c9526, ← c9527] at e_147_19
  have hr := e_147_19
  simp only [e_147_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 7) = bselect (a (.wire 141 27)) (a (.virt 12803)) (a (.wire 147 23)) := by
  have c9528 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9529 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9530 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9531 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9532 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9533 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_0 := arithEq_of_rows h (row := 148) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_0
  simp only [← c9528, ← c9529, ← c9530] at e_148_0
  have e_148_1 := arithEq_of_rows h (row := 148) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_1
  simp only [← c9531, ← c9532, ← c9533] at e_148_1
  have hr := e_148_1
  simp only [e_148_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 15) = bselect (a (.wire 141 27)) (a (.virt 12804)) (a (.wire 147 31)) := by
  have c9534 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9535 := (publicBatchWrapper89_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9536 := (publicBatchWrapper89_copies298 a h).1
  have c9537 := (publicBatchWrapper89_copies298 a h).2.1
  have c9538 := (publicBatchWrapper89_copies298 a h).2.2.1
  have c9539 := (publicBatchWrapper89_copies298 a h).2.2.2.1
  have e_148_2 := arithEq_of_rows h (row := 148) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_2
  simp only [← c9534, ← c9535, ← c9536] at e_148_2
  have e_148_3 := arithEq_of_rows h (row := 148) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_3
  simp only [← c9537, ← c9538, ← c9539] at e_148_3
  have hr := e_148_3
  simp only [e_148_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 63) = bor (a (.wire 134 59)) (a (.wire 147 35)) := by
  have c9540 := (publicBatchWrapper89_copies298 a h).2.2.2.2.1
  have c9541 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.1
  have c9542 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.1
  have c9543 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.1
  have c9544 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.1
  have c9545 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_134_15 := arithEq_of_rows h (row := 134) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_15
  simp only [← c9543, ← c9544, k3, ← c9545] at e_134_15
  have e_145_3 := arithEq_of_rows h (row := 145) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_3
  simp only [← c9540, ← c9541, ← c9542] at e_145_3
  have hr := e_134_15
  simp only [e_145_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 19) = bnot (a (.wire 64 39)) := by
  have c9546 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9547 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9548 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_148_4 := arithEq_of_rows h (row := 148) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_4
  simp only [← c9546, k3, ← c9547, k3, ← c9548] at e_148_4
  have hr := e_148_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 23) = bnot (a (.wire 134 63)) := by
  have c9549 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9550 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9551 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_148_5 := arithEq_of_rows h (row := 148) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_5
  simp only [← c9549, k3, ← c9550, k3, ← c9551] at e_148_5
  have hr := e_148_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 31) = band (a (.wire 148 19)) (a (.wire 148 23)) := by
  have c9552 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9553 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9554 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_7 := arithEq_of_rows h (row := 141) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_7
  simp only [← c9552, ← c9553, ← c9554] at e_141_7
  have hr := e_141_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 31) = bselect (a (.wire 141 31)) (a (.virt 13005)) (a (.wire 147 47)) := by
  have c9555 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9556 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9557 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9558 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9559 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9560 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_6 := arithEq_of_rows h (row := 148) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_6
  simp only [← c9555, ← c9556, ← c9557] at e_148_6
  have e_148_7 := arithEq_of_rows h (row := 148) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_7
  simp only [← c9558, ← c9559, ← c9560] at e_148_7
  have hr := e_148_7
  simp only [e_148_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 39) = bselect (a (.wire 141 31)) (a (.virt 13006)) (a (.wire 147 55)) := by
  have c9561 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9562 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9563 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9564 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9565 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9566 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_8 := arithEq_of_rows h (row := 148) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_8
  simp only [← c9561, ← c9562, ← c9563] at e_148_8
  have e_148_9 := arithEq_of_rows h (row := 148) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_9
  simp only [← c9564, ← c9565, ← c9566] at e_148_9
  have hr := e_148_9
  simp only [e_148_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 47) = bselect (a (.wire 141 31)) (a (.virt 13007)) (a (.wire 147 63)) := by
  have c9567 := (publicBatchWrapper89_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9568 := (publicBatchWrapper89_copies299 a h).1
  have c9569 := (publicBatchWrapper89_copies299 a h).2.1
  have c9570 := (publicBatchWrapper89_copies299 a h).2.2.1
  have c9571 := (publicBatchWrapper89_copies299 a h).2.2.2.1
  have c9572 := (publicBatchWrapper89_copies299 a h).2.2.2.2.1
  have e_148_10 := arithEq_of_rows h (row := 148) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_10
  simp only [← c9567, ← c9568, ← c9569] at e_148_10
  have e_148_11 := arithEq_of_rows h (row := 148) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_11
  simp only [← c9570, ← c9571, ← c9572] at e_148_11
  have hr := e_148_11
  simp only [e_148_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 55) = bselect (a (.wire 141 31)) (a (.virt 13008)) (a (.wire 147 71)) := by
  have c9573 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.1
  have c9574 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.1
  have c9575 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.1
  have c9576 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.1
  have c9577 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.1
  have c9578 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_148_12 := arithEq_of_rows h (row := 148) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_12
  simp only [← c9573, ← c9574, ← c9575] at e_148_12
  have e_148_13 := arithEq_of_rows h (row := 148) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_13
  simp only [← c9576, ← c9577, ← c9578] at e_148_13
  have hr := e_148_13
  simp only [e_148_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 63) = bselect (a (.wire 141 31)) (a (.virt 13009)) (a (.wire 147 79)) := by
  have c9579 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9580 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9581 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9582 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9583 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9584 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_14 := arithEq_of_rows h (row := 148) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_14
  simp only [← c9579, ← c9580, ← c9581] at e_148_14
  have e_148_15 := arithEq_of_rows h (row := 148) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_15
  simp only [← c9582, ← c9583, ← c9584] at e_148_15
  have hr := e_148_15
  simp only [e_148_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 71) = bselect (a (.wire 141 31)) (a (.virt 13003)) (a (.wire 148 7)) := by
  have c9585 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9586 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9587 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9588 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9589 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9590 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_16 := arithEq_of_rows h (row := 148) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_16
  simp only [← c9585, ← c9586, ← c9587] at e_148_16
  have e_148_17 := arithEq_of_rows h (row := 148) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_17
  simp only [← c9588, ← c9589, ← c9590] at e_148_17
  have hr := e_148_17
  simp only [e_148_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 148 79) = bselect (a (.wire 141 31)) (a (.virt 13004)) (a (.wire 148 15)) := by
  have c9591 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9592 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9593 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9594 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9595 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9596 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_18 := arithEq_of_rows h (row := 148) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_18
  simp only [← c9591, ← c9592, ← c9593] at e_148_18
  have e_148_19 := arithEq_of_rows h (row := 148) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_19
  simp only [← c9594, ← c9595, ← c9596] at e_148_19
  have hr := e_148_19
  simp only [e_148_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 67) = bor (a (.wire 134 63)) (a (.wire 148 19)) := by
  have c9597 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9598 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9599 := (publicBatchWrapper89_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9600 := (publicBatchWrapper89_copies300 a h).1
  have c9601 := (publicBatchWrapper89_copies300 a h).2.1
  have c9602 := (publicBatchWrapper89_copies300 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_134_16 := arithEq_of_rows h (row := 134) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_16
  simp only [← c9600, ← c9601, k3, ← c9602] at e_134_16
  have e_145_4 := arithEq_of_rows h (row := 145) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_4
  simp only [← c9597, ← c9598, ← c9599] at e_145_4
  have hr := e_134_16
  simp only [e_145_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 3) = bnot (a (.wire 65 3)) := by
  have c9603 := (publicBatchWrapper89_copies300 a h).2.2.2.1
  have c9604 := (publicBatchWrapper89_copies300 a h).2.2.2.2.1
  have c9605 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_149_0 := arithEq_of_rows h (row := 149) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_0
  simp only [← c9603, k3, ← c9604, k3, ← c9605] at e_149_0
  have hr := e_149_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 7) = bnot (a (.wire 134 67)) := by
  have c9606 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.1
  have c9607 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.1
  have c9608 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_149_1 := arithEq_of_rows h (row := 149) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_1
  simp only [← c9606, k3, ← c9607, k3, ← c9608] at e_149_1
  have hr := e_149_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 35) = band (a (.wire 149 3)) (a (.wire 149 7)) := by
  have c9609 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.1
  have c9610 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9611 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_8 := arithEq_of_rows h (row := 141) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_8
  simp only [← c9609, ← c9610, ← c9611] at e_141_8
  have hr := e_141_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 15) = bselect (a (.wire 141 35)) (a (.virt 13205)) (a (.wire 148 31)) := by
  have c9612 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9613 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9614 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9615 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9616 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9617 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_2 := arithEq_of_rows h (row := 149) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_2
  simp only [← c9612, ← c9613, ← c9614] at e_149_2
  have e_149_3 := arithEq_of_rows h (row := 149) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_3
  simp only [← c9615, ← c9616, ← c9617] at e_149_3
  have hr := e_149_3
  simp only [e_149_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 23) = bselect (a (.wire 141 35)) (a (.virt 13206)) (a (.wire 148 39)) := by
  have c9618 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9619 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9620 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9621 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9622 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9623 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_4 := arithEq_of_rows h (row := 149) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_4
  simp only [← c9618, ← c9619, ← c9620] at e_149_4
  have e_149_5 := arithEq_of_rows h (row := 149) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_5
  simp only [← c9621, ← c9622, ← c9623] at e_149_5
  have hr := e_149_5
  simp only [e_149_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 31) = bselect (a (.wire 141 35)) (a (.virt 13207)) (a (.wire 148 47)) := by
  have c9624 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9625 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9626 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9627 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9628 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9629 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_6 := arithEq_of_rows h (row := 149) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_6
  simp only [← c9624, ← c9625, ← c9626] at e_149_6
  have e_149_7 := arithEq_of_rows h (row := 149) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_7
  simp only [← c9627, ← c9628, ← c9629] at e_149_7
  have hr := e_149_7
  simp only [e_149_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 39) = bselect (a (.wire 141 35)) (a (.virt 13208)) (a (.wire 148 55)) := by
  have c9630 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9631 := (publicBatchWrapper89_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9632 := (publicBatchWrapper89_copies301 a h).1
  have c9633 := (publicBatchWrapper89_copies301 a h).2.1
  have c9634 := (publicBatchWrapper89_copies301 a h).2.2.1
  have c9635 := (publicBatchWrapper89_copies301 a h).2.2.2.1
  have e_149_8 := arithEq_of_rows h (row := 149) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_8
  simp only [← c9630, ← c9631, ← c9632] at e_149_8
  have e_149_9 := arithEq_of_rows h (row := 149) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_9
  simp only [← c9633, ← c9634, ← c9635] at e_149_9
  have hr := e_149_9
  simp only [e_149_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 47) = bselect (a (.wire 141 35)) (a (.virt 13209)) (a (.wire 148 63)) := by
  have c9636 := (publicBatchWrapper89_copies301 a h).2.2.2.2.1
  have c9637 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.1
  have c9638 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.1
  have c9639 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.1
  have c9640 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.1
  have c9641 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.1
  have e_149_10 := arithEq_of_rows h (row := 149) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_10
  simp only [← c9636, ← c9637, ← c9638] at e_149_10
  have e_149_11 := arithEq_of_rows h (row := 149) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_11
  simp only [← c9639, ← c9640, ← c9641] at e_149_11
  have hr := e_149_11
  simp only [e_149_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 55) = bselect (a (.wire 141 35)) (a (.virt 13203)) (a (.wire 148 71)) := by
  have c9642 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9643 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9644 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9645 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9646 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9647 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_12 := arithEq_of_rows h (row := 149) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_12
  simp only [← c9642, ← c9643, ← c9644] at e_149_12
  have e_149_13 := arithEq_of_rows h (row := 149) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_13
  simp only [← c9645, ← c9646, ← c9647] at e_149_13
  have hr := e_149_13
  simp only [e_149_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 63) = bselect (a (.wire 141 35)) (a (.virt 13204)) (a (.wire 148 79)) := by
  have c9648 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9649 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9650 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9651 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9652 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9653 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_14 := arithEq_of_rows h (row := 149) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_14
  simp only [← c9648, ← c9649, ← c9650] at e_149_14
  have e_149_15 := arithEq_of_rows h (row := 149) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_15
  simp only [← c9651, ← c9652, ← c9653] at e_149_15
  have hr := e_149_15
  simp only [e_149_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 71) = bor (a (.wire 134 67)) (a (.wire 149 3)) := by
  have c9654 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9655 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9656 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9657 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9658 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9659 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_134_17 := arithEq_of_rows h (row := 134) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_17
  simp only [← c9657, ← c9658, k3, ← c9659] at e_134_17
  have e_145_5 := arithEq_of_rows h (row := 145) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_5
  simp only [← c9654, ← c9655, ← c9656] at e_145_5
  have hr := e_134_17
  simp only [e_145_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 67) = bnot (a (.wire 65 47)) := by
  have c9660 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9661 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9662 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_149_16 := arithEq_of_rows h (row := 149) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_16
  simp only [← c9660, k3, ← c9661, k3, ← c9662] at e_149_16
  have hr := e_149_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 71) = bnot (a (.wire 134 71)) := by
  have c9663 := (publicBatchWrapper89_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9664 := (publicBatchWrapper89_copies302 a h).1
  have c9665 := (publicBatchWrapper89_copies302 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_149_17 := arithEq_of_rows h (row := 149) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_17
  simp only [← c9663, k3, ← c9664, k3, ← c9665] at e_149_17
  have hr := e_149_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 39) = band (a (.wire 149 67)) (a (.wire 149 71)) := by
  have c9666 := (publicBatchWrapper89_copies302 a h).2.2.1
  have c9667 := (publicBatchWrapper89_copies302 a h).2.2.2.1
  have c9668 := (publicBatchWrapper89_copies302 a h).2.2.2.2.1
  have e_141_9 := arithEq_of_rows h (row := 141) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_9
  simp only [← c9666, ← c9667, ← c9668] at e_141_9
  have hr := e_141_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 149 79) = bselect (a (.wire 141 39)) (a (.virt 13405)) (a (.wire 149 15)) := by
  have c9669 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.1
  have c9670 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.1
  have c9671 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.1
  have c9672 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.1
  have c9673 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.1
  have c9674 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_149_18 := arithEq_of_rows h (row := 149) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_18
  simp only [← c9669, ← c9670, ← c9671] at e_149_18
  have e_149_19 := arithEq_of_rows h (row := 149) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_19
  simp only [← c9672, ← c9673, ← c9674] at e_149_19
  have hr := e_149_19
  simp only [e_149_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 7) = bselect (a (.wire 141 39)) (a (.virt 13406)) (a (.wire 149 23)) := by
  have c9675 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9676 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9677 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9678 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9679 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9680 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_0 := arithEq_of_rows h (row := 150) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_0
  simp only [← c9675, ← c9676, ← c9677] at e_150_0
  have e_150_1 := arithEq_of_rows h (row := 150) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_1
  simp only [← c9678, ← c9679, ← c9680] at e_150_1
  have hr := e_150_1
  simp only [e_150_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 15) = bselect (a (.wire 141 39)) (a (.virt 13407)) (a (.wire 149 31)) := by
  have c9681 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9682 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9683 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9684 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9685 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9686 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_2 := arithEq_of_rows h (row := 150) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_2
  simp only [← c9681, ← c9682, ← c9683] at e_150_2
  have e_150_3 := arithEq_of_rows h (row := 150) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_3
  simp only [← c9684, ← c9685, ← c9686] at e_150_3
  have hr := e_150_3
  simp only [e_150_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 23) = bselect (a (.wire 141 39)) (a (.virt 13408)) (a (.wire 149 39)) := by
  have c9687 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9688 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9689 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9690 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9691 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9692 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_4 := arithEq_of_rows h (row := 150) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_4
  simp only [← c9687, ← c9688, ← c9689] at e_150_4
  have e_150_5 := arithEq_of_rows h (row := 150) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_5
  simp only [← c9690, ← c9691, ← c9692] at e_150_5
  have hr := e_150_5
  simp only [e_150_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 31) = bselect (a (.wire 141 39)) (a (.virt 13409)) (a (.wire 149 47)) := by
  have c9693 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9694 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9695 := (publicBatchWrapper89_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9696 := (publicBatchWrapper89_copies303 a h).1
  have c9697 := (publicBatchWrapper89_copies303 a h).2.1
  have c9698 := (publicBatchWrapper89_copies303 a h).2.2.1
  have e_150_6 := arithEq_of_rows h (row := 150) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_6
  simp only [← c9693, ← c9694, ← c9695] at e_150_6
  have e_150_7 := arithEq_of_rows h (row := 150) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_7
  simp only [← c9696, ← c9697, ← c9698] at e_150_7
  have hr := e_150_7
  simp only [e_150_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 39) = bselect (a (.wire 141 39)) (a (.virt 13403)) (a (.wire 149 55)) := by
  have c9699 := (publicBatchWrapper89_copies303 a h).2.2.2.1
  have c9700 := (publicBatchWrapper89_copies303 a h).2.2.2.2.1
  have c9701 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.1
  have c9702 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.1
  have c9703 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.1
  have c9704 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.1
  have e_150_8 := arithEq_of_rows h (row := 150) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_8
  simp only [← c9699, ← c9700, ← c9701] at e_150_8
  have e_150_9 := arithEq_of_rows h (row := 150) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_9
  simp only [← c9702, ← c9703, ← c9704] at e_150_9
  have hr := e_150_9
  simp only [e_150_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 47) = bselect (a (.wire 141 39)) (a (.virt 13404)) (a (.wire 149 63)) := by
  have c9705 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.1
  have c9706 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9707 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9708 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9709 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9710 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_10 := arithEq_of_rows h (row := 150) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_10
  simp only [← c9705, ← c9706, ← c9707] at e_150_10
  have e_150_11 := arithEq_of_rows h (row := 150) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_11
  simp only [← c9708, ← c9709, ← c9710] at e_150_11
  have hr := e_150_11
  simp only [e_150_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 75) = bor (a (.wire 134 71)) (a (.wire 149 67)) := by
  have c9711 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9712 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9713 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9714 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9715 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9716 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_134_18 := arithEq_of_rows h (row := 134) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_18
  simp only [← c9714, ← c9715, k3, ← c9716] at e_134_18
  have e_145_6 := arithEq_of_rows h (row := 145) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_6
  simp only [← c9711, ← c9712, ← c9713] at e_145_6
  have hr := e_134_18
  simp only [e_145_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 51) = bnot (a (.wire 67 11)) := by
  have c9717 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9718 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9719 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_150_12 := arithEq_of_rows h (row := 150) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_12
  simp only [← c9717, k3, ← c9718, k3, ← c9719] at e_150_12
  have hr := e_150_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 55) = bnot (a (.wire 134 75)) := by
  have c9720 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9721 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9722 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_150_13 := arithEq_of_rows h (row := 150) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_13
  simp only [← c9720, k3, ← c9721, k3, ← c9722] at e_150_13
  have hr := e_150_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 43) = band (a (.wire 150 51)) (a (.wire 150 55)) := by
  have c9723 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9724 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9725 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_10 := arithEq_of_rows h (row := 141) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_10
  simp only [← c9723, ← c9724, ← c9725] at e_141_10
  have hr := e_141_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 63) = bselect (a (.wire 141 43)) (a (.virt 13605)) (a (.wire 149 79)) := by
  have c9726 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9727 := (publicBatchWrapper89_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9728 := (publicBatchWrapper89_copies304 a h).1
  have c9729 := (publicBatchWrapper89_copies304 a h).2.1
  have c9730 := (publicBatchWrapper89_copies304 a h).2.2.1
  have c9731 := (publicBatchWrapper89_copies304 a h).2.2.2.1
  have e_150_14 := arithEq_of_rows h (row := 150) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_14
  simp only [← c9726, ← c9727, ← c9728] at e_150_14
  have e_150_15 := arithEq_of_rows h (row := 150) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_15
  simp only [← c9729, ← c9730, ← c9731] at e_150_15
  have hr := e_150_15
  simp only [e_150_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 71) = bselect (a (.wire 141 43)) (a (.virt 13606)) (a (.wire 150 7)) := by
  have c9732 := (publicBatchWrapper89_copies304 a h).2.2.2.2.1
  have c9733 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.1
  have c9734 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.1
  have c9735 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.1
  have c9736 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.1
  have c9737 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.1
  have e_150_16 := arithEq_of_rows h (row := 150) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_16
  simp only [← c9732, ← c9733, ← c9734] at e_150_16
  have e_150_17 := arithEq_of_rows h (row := 150) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_17
  simp only [← c9735, ← c9736, ← c9737] at e_150_17
  have hr := e_150_17
  simp only [e_150_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 150 79) = bselect (a (.wire 141 43)) (a (.virt 13607)) (a (.wire 150 15)) := by
  have c9738 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9739 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9740 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9741 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9742 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9743 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_18 := arithEq_of_rows h (row := 150) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_18
  simp only [← c9738, ← c9739, ← c9740] at e_150_18
  have e_150_19 := arithEq_of_rows h (row := 150) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_19
  simp only [← c9741, ← c9742, ← c9743] at e_150_19
  have hr := e_150_19
  simp only [e_150_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 7) = bselect (a (.wire 141 43)) (a (.virt 13608)) (a (.wire 150 23)) := by
  have c9744 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9745 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9746 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9747 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9748 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9749 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_0 := arithEq_of_rows h (row := 151) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_0
  simp only [← c9744, ← c9745, ← c9746] at e_151_0
  have e_151_1 := arithEq_of_rows h (row := 151) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_1
  simp only [← c9747, ← c9748, ← c9749] at e_151_1
  have hr := e_151_1
  simp only [e_151_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 15) = bselect (a (.wire 141 43)) (a (.virt 13609)) (a (.wire 150 31)) := by
  have c9750 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9751 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9752 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9753 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9754 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9755 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_2 := arithEq_of_rows h (row := 151) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_2
  simp only [← c9750, ← c9751, ← c9752] at e_151_2
  have e_151_3 := arithEq_of_rows h (row := 151) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_3
  simp only [← c9753, ← c9754, ← c9755] at e_151_3
  have hr := e_151_3
  simp only [e_151_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 23) = bselect (a (.wire 141 43)) (a (.virt 13603)) (a (.wire 150 39)) := by
  have c9756 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9757 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9758 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9759 := (publicBatchWrapper89_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9760 := (publicBatchWrapper89_copies305 a h).1
  have c9761 := (publicBatchWrapper89_copies305 a h).2.1
  have e_151_4 := arithEq_of_rows h (row := 151) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_4
  simp only [← c9756, ← c9757, ← c9758] at e_151_4
  have e_151_5 := arithEq_of_rows h (row := 151) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_5
  simp only [← c9759, ← c9760, ← c9761] at e_151_5
  have hr := e_151_5
  simp only [e_151_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 31) = bselect (a (.wire 141 43)) (a (.virt 13604)) (a (.wire 150 47)) := by
  have c9762 := (publicBatchWrapper89_copies305 a h).2.2.1
  have c9763 := (publicBatchWrapper89_copies305 a h).2.2.2.1
  have c9764 := (publicBatchWrapper89_copies305 a h).2.2.2.2.1
  have c9765 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.1
  have c9766 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.1
  have c9767 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.1
  have e_151_6 := arithEq_of_rows h (row := 151) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_6
  simp only [← c9762, ← c9763, ← c9764] at e_151_6
  have e_151_7 := arithEq_of_rows h (row := 151) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_7
  simp only [← c9765, ← c9766, ← c9767] at e_151_7
  have hr := e_151_7
  simp only [e_151_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 134 79) = bor (a (.wire 134 75)) (a (.wire 150 51)) := by
  have c9768 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.1
  have c9769 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.1
  have c9770 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9771 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9772 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9773 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_134_19 := arithEq_of_rows h (row := 134) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_19
  simp only [← c9771, ← c9772, k3, ← c9773] at e_134_19
  have e_145_7 := arithEq_of_rows h (row := 145) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_7
  simp only [← c9768, ← c9769, ← c9770] at e_145_7
  have hr := e_134_19
  simp only [e_145_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 35) = bnot (a (.wire 67 55)) := by
  have c9774 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9775 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9776 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_151_8 := arithEq_of_rows h (row := 151) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_8
  simp only [← c9774, k3, ← c9775, k3, ← c9776] at e_151_8
  have hr := e_151_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 39) = bnot (a (.wire 134 79)) := by
  have c9777 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9778 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9779 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_151_9 := arithEq_of_rows h (row := 151) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_9
  simp only [← c9777, k3, ← c9778, k3, ← c9779] at e_151_9
  have hr := e_151_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 47) = band (a (.wire 151 35)) (a (.wire 151 39)) := by
  have c9780 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9781 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9782 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_11 := arithEq_of_rows h (row := 141) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_11
  simp only [← c9780, ← c9781, ← c9782] at e_141_11
  have hr := e_141_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 47) = bselect (a (.wire 141 47)) (a (.virt 13805)) (a (.wire 150 63)) := by
  have c9783 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9784 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9785 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9786 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9787 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9788 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_10 := arithEq_of_rows h (row := 151) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_10
  simp only [← c9783, ← c9784, ← c9785] at e_151_10
  have e_151_11 := arithEq_of_rows h (row := 151) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_11
  simp only [← c9786, ← c9787, ← c9788] at e_151_11
  have hr := e_151_11
  simp only [e_151_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 55) = bselect (a (.wire 141 47)) (a (.virt 13806)) (a (.wire 150 71)) := by
  have c9789 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9790 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9791 := (publicBatchWrapper89_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9792 := (publicBatchWrapper89_copies306 a h).1
  have c9793 := (publicBatchWrapper89_copies306 a h).2.1
  have c9794 := (publicBatchWrapper89_copies306 a h).2.2.1
  have e_151_12 := arithEq_of_rows h (row := 151) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_12
  simp only [← c9789, ← c9790, ← c9791] at e_151_12
  have e_151_13 := arithEq_of_rows h (row := 151) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_13
  simp only [← c9792, ← c9793, ← c9794] at e_151_13
  have hr := e_151_13
  simp only [e_151_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 63) = bselect (a (.wire 141 47)) (a (.virt 13807)) (a (.wire 150 79)) := by
  have c9795 := (publicBatchWrapper89_copies306 a h).2.2.2.1
  have c9796 := (publicBatchWrapper89_copies306 a h).2.2.2.2.1
  have c9797 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.1
  have c9798 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.1
  have c9799 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.1
  have c9800 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.1
  have e_151_14 := arithEq_of_rows h (row := 151) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_14
  simp only [← c9795, ← c9796, ← c9797] at e_151_14
  have e_151_15 := arithEq_of_rows h (row := 151) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_15
  simp only [← c9798, ← c9799, ← c9800] at e_151_15
  have hr := e_151_15
  simp only [e_151_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 71) = bselect (a (.wire 141 47)) (a (.virt 13808)) (a (.wire 151 7)) := by
  have c9801 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.1
  have c9802 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9803 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9804 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9805 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9806 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_16 := arithEq_of_rows h (row := 151) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_16
  simp only [← c9801, ← c9802, ← c9803] at e_151_16
  have e_151_17 := arithEq_of_rows h (row := 151) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_17
  simp only [← c9804, ← c9805, ← c9806] at e_151_17
  have hr := e_151_17
  simp only [e_151_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 151 79) = bselect (a (.wire 141 47)) (a (.virt 13809)) (a (.wire 151 15)) := by
  have c9807 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9808 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9809 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9810 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9811 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9812 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_18 := arithEq_of_rows h (row := 151) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_18
  simp only [← c9807, ← c9808, ← c9809] at e_151_18
  have e_151_19 := arithEq_of_rows h (row := 151) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_19
  simp only [← c9810, ← c9811, ← c9812] at e_151_19
  have hr := e_151_19
  simp only [e_151_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 7) = bselect (a (.wire 141 47)) (a (.virt 13803)) (a (.wire 151 23)) := by
  have c9813 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9814 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9815 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9816 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9817 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9818 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_0 := arithEq_of_rows h (row := 152) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_0
  simp only [← c9813, ← c9814, ← c9815] at e_152_0
  have e_152_1 := arithEq_of_rows h (row := 152) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_1
  simp only [← c9816, ← c9817, ← c9818] at e_152_1
  have hr := e_152_1
  simp only [e_152_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 15) = bselect (a (.wire 141 47)) (a (.virt 13804)) (a (.wire 151 31)) := by
  have c9819 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9820 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9821 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9822 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9823 := (publicBatchWrapper89_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9824 := (publicBatchWrapper89_copies307 a h).1
  have e_152_2 := arithEq_of_rows h (row := 152) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_2
  simp only [← c9819, ← c9820, ← c9821] at e_152_2
  have e_152_3 := arithEq_of_rows h (row := 152) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_3
  simp only [← c9822, ← c9823, ← c9824] at e_152_3
  have hr := e_152_3
  simp only [e_152_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 3) = bor (a (.wire 134 79)) (a (.wire 151 35)) := by
  have c9825 := (publicBatchWrapper89_copies307 a h).2.1
  have c9826 := (publicBatchWrapper89_copies307 a h).2.2.1
  have c9827 := (publicBatchWrapper89_copies307 a h).2.2.2.1
  have c9828 := (publicBatchWrapper89_copies307 a h).2.2.2.2.1
  have c9829 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.1
  have c9830 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_8 := arithEq_of_rows h (row := 145) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_8
  simp only [← c9825, ← c9826, ← c9827] at e_145_8
  have e_153_0 := arithEq_of_rows h (row := 153) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_0
  simp only [← c9828, ← c9829, k3, ← c9830] at e_153_0
  have hr := e_153_0
  simp only [e_145_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 19) = bnot (a (.wire 69 19)) := by
  have c9831 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.1
  have c9832 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.1
  have c9833 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_152_4 := arithEq_of_rows h (row := 152) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_4
  simp only [← c9831, k3, ← c9832, k3, ← c9833] at e_152_4
  have hr := e_152_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 23) = bnot (a (.wire 153 3)) := by
  have c9834 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9835 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9836 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_152_5 := arithEq_of_rows h (row := 152) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_5
  simp only [← c9834, k3, ← c9835, k3, ← c9836] at e_152_5
  have hr := e_152_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 51) = band (a (.wire 152 19)) (a (.wire 152 23)) := by
  have c9837 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9838 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9839 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_12 := arithEq_of_rows h (row := 141) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_12
  simp only [← c9837, ← c9838, ← c9839] at e_141_12
  have hr := e_141_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 31) = bselect (a (.wire 141 51)) (a (.virt 14005)) (a (.wire 151 47)) := by
  have c9840 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9841 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9842 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9843 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9844 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9845 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_6 := arithEq_of_rows h (row := 152) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_6
  simp only [← c9840, ← c9841, ← c9842] at e_152_6
  have e_152_7 := arithEq_of_rows h (row := 152) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_7
  simp only [← c9843, ← c9844, ← c9845] at e_152_7
  have hr := e_152_7
  simp only [e_152_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 39) = bselect (a (.wire 141 51)) (a (.virt 14006)) (a (.wire 151 55)) := by
  have c9846 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9847 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9848 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9849 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9850 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9851 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_8 := arithEq_of_rows h (row := 152) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_8
  simp only [← c9846, ← c9847, ← c9848] at e_152_8
  have e_152_9 := arithEq_of_rows h (row := 152) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_9
  simp only [← c9849, ← c9850, ← c9851] at e_152_9
  have hr := e_152_9
  simp only [e_152_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 47) = bselect (a (.wire 141 51)) (a (.virt 14007)) (a (.wire 151 63)) := by
  have c9852 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9853 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9854 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9855 := (publicBatchWrapper89_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9856 := (publicBatchWrapper89_copies308 a h).1
  have c9857 := (publicBatchWrapper89_copies308 a h).2.1
  have e_152_10 := arithEq_of_rows h (row := 152) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_10
  simp only [← c9852, ← c9853, ← c9854] at e_152_10
  have e_152_11 := arithEq_of_rows h (row := 152) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_11
  simp only [← c9855, ← c9856, ← c9857] at e_152_11
  have hr := e_152_11
  simp only [e_152_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 55) = bselect (a (.wire 141 51)) (a (.virt 14008)) (a (.wire 151 71)) := by
  have c9858 := (publicBatchWrapper89_copies308 a h).2.2.1
  have c9859 := (publicBatchWrapper89_copies308 a h).2.2.2.1
  have c9860 := (publicBatchWrapper89_copies308 a h).2.2.2.2.1
  have c9861 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.1
  have c9862 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.1
  have c9863 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.1
  have e_152_12 := arithEq_of_rows h (row := 152) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_12
  simp only [← c9858, ← c9859, ← c9860] at e_152_12
  have e_152_13 := arithEq_of_rows h (row := 152) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_13
  simp only [← c9861, ← c9862, ← c9863] at e_152_13
  have hr := e_152_13
  simp only [e_152_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 63) = bselect (a (.wire 141 51)) (a (.virt 14009)) (a (.wire 151 79)) := by
  have c9864 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.1
  have c9865 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.1
  have c9866 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9867 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9868 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9869 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_14 := arithEq_of_rows h (row := 152) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_14
  simp only [← c9864, ← c9865, ← c9866] at e_152_14
  have e_152_15 := arithEq_of_rows h (row := 152) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_15
  simp only [← c9867, ← c9868, ← c9869] at e_152_15
  have hr := e_152_15
  simp only [e_152_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 71) = bselect (a (.wire 141 51)) (a (.virt 14003)) (a (.wire 152 7)) := by
  have c9870 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9871 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9872 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9873 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9874 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9875 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_16 := arithEq_of_rows h (row := 152) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_16
  simp only [← c9870, ← c9871, ← c9872] at e_152_16
  have e_152_17 := arithEq_of_rows h (row := 152) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_17
  simp only [← c9873, ← c9874, ← c9875] at e_152_17
  have hr := e_152_17
  simp only [e_152_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 152 79) = bselect (a (.wire 141 51)) (a (.virt 14004)) (a (.wire 152 15)) := by
  have c9876 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9877 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9878 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9879 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9880 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9881 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_18 := arithEq_of_rows h (row := 152) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_18
  simp only [← c9876, ← c9877, ← c9878] at e_152_18
  have e_152_19 := arithEq_of_rows h (row := 152) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_19
  simp only [← c9879, ← c9880, ← c9881] at e_152_19
  have hr := e_152_19
  simp only [e_152_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 7) = bor (a (.wire 153 3)) (a (.wire 152 19)) := by
  have c9882 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9883 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9884 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9885 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9886 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9887 := (publicBatchWrapper89_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_9 := arithEq_of_rows h (row := 145) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_9
  simp only [← c9882, ← c9883, ← c9884] at e_145_9
  have e_153_1 := arithEq_of_rows h (row := 153) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_1
  simp only [← c9885, ← c9886, k3, ← c9887] at e_153_1
  have hr := e_153_1
  simp only [e_145_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 3) = bnot (a (.wire 69 63)) := by
  have c9888 := (publicBatchWrapper89_copies309 a h).1
  have c9889 := (publicBatchWrapper89_copies309 a h).2.1
  have c9890 := (publicBatchWrapper89_copies309 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_154_0 := arithEq_of_rows h (row := 154) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_0
  simp only [← c9888, k3, ← c9889, k3, ← c9890] at e_154_0
  have hr := e_154_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 7) = bnot (a (.wire 153 7)) := by
  have c9891 := (publicBatchWrapper89_copies309 a h).2.2.2.1
  have c9892 := (publicBatchWrapper89_copies309 a h).2.2.2.2.1
  have c9893 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_154_1 := arithEq_of_rows h (row := 154) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_1
  simp only [← c9891, k3, ← c9892, k3, ← c9893] at e_154_1
  have hr := e_154_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 55) = band (a (.wire 154 3)) (a (.wire 154 7)) := by
  have c9894 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.1
  have c9895 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.1
  have c9896 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.1
  have e_141_13 := arithEq_of_rows h (row := 141) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_13
  simp only [← c9894, ← c9895, ← c9896] at e_141_13
  have hr := e_141_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 15) = bselect (a (.wire 141 55)) (a (.virt 14205)) (a (.wire 152 31)) := by
  have c9897 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.1
  have c9898 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9899 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9900 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9901 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9902 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_154_2 := arithEq_of_rows h (row := 154) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_2
  simp only [← c9897, ← c9898, ← c9899] at e_154_2
  have e_154_3 := arithEq_of_rows h (row := 154) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_3
  simp only [← c9900, ← c9901, ← c9902] at e_154_3
  have hr := e_154_3
  simp only [e_154_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 23) = bselect (a (.wire 141 55)) (a (.virt 14206)) (a (.wire 152 39)) := by
  have c9903 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9904 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9905 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9906 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9907 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9908 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_154_4 := arithEq_of_rows h (row := 154) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_4
  simp only [← c9903, ← c9904, ← c9905] at e_154_4
  have e_154_5 := arithEq_of_rows h (row := 154) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_5
  simp only [← c9906, ← c9907, ← c9908] at e_154_5
  have hr := e_154_5
  simp only [e_154_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 31) = bselect (a (.wire 141 55)) (a (.virt 14207)) (a (.wire 152 47)) := by
  have c9909 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9910 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9911 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9912 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9913 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9914 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_154_6 := arithEq_of_rows h (row := 154) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_6
  simp only [← c9909, ← c9910, ← c9911] at e_154_6
  have e_154_7 := arithEq_of_rows h (row := 154) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_7
  simp only [← c9912, ← c9913, ← c9914] at e_154_7
  have hr := e_154_7
  simp only [e_154_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 39) = bselect (a (.wire 141 55)) (a (.virt 14208)) (a (.wire 152 55)) := by
  have c9915 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9916 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9917 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9918 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9919 := (publicBatchWrapper89_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9920 := (publicBatchWrapper89_copies310 a h).1
  have e_154_8 := arithEq_of_rows h (row := 154) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_8
  simp only [← c9915, ← c9916, ← c9917] at e_154_8
  have e_154_9 := arithEq_of_rows h (row := 154) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_9
  simp only [← c9918, ← c9919, ← c9920] at e_154_9
  have hr := e_154_9
  simp only [e_154_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 47) = bselect (a (.wire 141 55)) (a (.virt 14209)) (a (.wire 152 63)) := by
  have c9921 := (publicBatchWrapper89_copies310 a h).2.1
  have c9922 := (publicBatchWrapper89_copies310 a h).2.2.1
  have c9923 := (publicBatchWrapper89_copies310 a h).2.2.2.1
  have c9924 := (publicBatchWrapper89_copies310 a h).2.2.2.2.1
  have c9925 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.1
  have c9926 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.1
  have e_154_10 := arithEq_of_rows h (row := 154) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_10
  simp only [← c9921, ← c9922, ← c9923] at e_154_10
  have e_154_11 := arithEq_of_rows h (row := 154) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_11
  simp only [← c9924, ← c9925, ← c9926] at e_154_11
  have hr := e_154_11
  simp only [e_154_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 55) = bselect (a (.wire 141 55)) (a (.virt 14203)) (a (.wire 152 71)) := by
  have c9927 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.1
  have c9928 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.1
  have c9929 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.1
  have c9930 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9931 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9932 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_154_12 := arithEq_of_rows h (row := 154) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_12
  simp only [← c9927, ← c9928, ← c9929] at e_154_12
  have e_154_13 := arithEq_of_rows h (row := 154) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_13
  simp only [← c9930, ← c9931, ← c9932] at e_154_13
  have hr := e_154_13
  simp only [e_154_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 63) = bselect (a (.wire 141 55)) (a (.virt 14204)) (a (.wire 152 79)) := by
  have c9933 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9934 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9935 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9936 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9937 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9938 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_154_14 := arithEq_of_rows h (row := 154) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_14
  simp only [← c9933, ← c9934, ← c9935] at e_154_14
  have e_154_15 := arithEq_of_rows h (row := 154) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_15
  simp only [← c9936, ← c9937, ← c9938] at e_154_15
  have hr := e_154_15
  simp only [e_154_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 11) = bor (a (.wire 153 7)) (a (.wire 154 3)) := by
  have c9939 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9940 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9941 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9942 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9943 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9944 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_10 := arithEq_of_rows h (row := 145) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_10
  simp only [← c9939, ← c9940, ← c9941] at e_145_10
  have e_153_2 := arithEq_of_rows h (row := 153) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_2
  simp only [← c9942, ← c9943, k3, ← c9944] at e_153_2
  have hr := e_153_2
  simp only [e_145_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 67) = bnot (a (.wire 71 27)) := by
  have c9945 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9946 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9947 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_154_16 := arithEq_of_rows h (row := 154) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_16
  simp only [← c9945, k3, ← c9946, k3, ← c9947] at e_154_16
  have hr := e_154_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 71) = bnot (a (.wire 153 11)) := by
  have c9948 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9949 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9950 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_154_17 := arithEq_of_rows h (row := 154) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_17
  simp only [← c9948, k3, ← c9949, k3, ← c9950] at e_154_17
  have hr := e_154_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 59) = band (a (.wire 154 67)) (a (.wire 154 71)) := by
  have c9951 := (publicBatchWrapper89_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9952 := (publicBatchWrapper89_copies311 a h).1
  have c9953 := (publicBatchWrapper89_copies311 a h).2.1
  have e_141_14 := arithEq_of_rows h (row := 141) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_14
  simp only [← c9951, ← c9952, ← c9953] at e_141_14
  have hr := e_141_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 154 79) = bselect (a (.wire 141 59)) (a (.virt 14405)) (a (.wire 154 15)) := by
  have c9954 := (publicBatchWrapper89_copies311 a h).2.2.1
  have c9955 := (publicBatchWrapper89_copies311 a h).2.2.2.1
  have c9956 := (publicBatchWrapper89_copies311 a h).2.2.2.2.1
  have c9957 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.1
  have c9958 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.1
  have c9959 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.1
  have e_154_18 := arithEq_of_rows h (row := 154) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_18
  simp only [← c9954, ← c9955, ← c9956] at e_154_18
  have e_154_19 := arithEq_of_rows h (row := 154) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_19
  simp only [← c9957, ← c9958, ← c9959] at e_154_19
  have hr := e_154_19
  simp only [e_154_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 7) = bselect (a (.wire 141 59)) (a (.virt 14406)) (a (.wire 154 23)) := by
  have c9960 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.1
  have c9961 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.1
  have c9962 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9963 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9964 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9965 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_0 := arithEq_of_rows h (row := 155) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_0
  simp only [← c9960, ← c9961, ← c9962] at e_155_0
  have e_155_1 := arithEq_of_rows h (row := 155) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_1
  simp only [← c9963, ← c9964, ← c9965] at e_155_1
  have hr := e_155_1
  simp only [e_155_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 15) = bselect (a (.wire 141 59)) (a (.virt 14407)) (a (.wire 154 31)) := by
  have c9966 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9967 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9968 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9969 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9970 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9971 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_2 := arithEq_of_rows h (row := 155) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_2
  simp only [← c9966, ← c9967, ← c9968] at e_155_2
  have e_155_3 := arithEq_of_rows h (row := 155) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_3
  simp only [← c9969, ← c9970, ← c9971] at e_155_3
  have hr := e_155_3
  simp only [e_155_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 23) = bselect (a (.wire 141 59)) (a (.virt 14408)) (a (.wire 154 39)) := by
  have c9972 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9973 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9974 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9975 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9976 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9977 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_4 := arithEq_of_rows h (row := 155) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_4
  simp only [← c9972, ← c9973, ← c9974] at e_155_4
  have e_155_5 := arithEq_of_rows h (row := 155) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_5
  simp only [← c9975, ← c9976, ← c9977] at e_155_5
  have hr := e_155_5
  simp only [e_155_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 31) = bselect (a (.wire 141 59)) (a (.virt 14409)) (a (.wire 154 47)) := by
  have c9978 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9979 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9980 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9981 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9982 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9983 := (publicBatchWrapper89_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_155_6 := arithEq_of_rows h (row := 155) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_6
  simp only [← c9978, ← c9979, ← c9980] at e_155_6
  have e_155_7 := arithEq_of_rows h (row := 155) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_7
  simp only [← c9981, ← c9982, ← c9983] at e_155_7
  have hr := e_155_7
  simp only [e_155_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 39) = bselect (a (.wire 141 59)) (a (.virt 14403)) (a (.wire 154 55)) := by
  have c9984 := (publicBatchWrapper89_copies312 a h).1
  have c9985 := (publicBatchWrapper89_copies312 a h).2.1
  have c9986 := (publicBatchWrapper89_copies312 a h).2.2.1
  have c9987 := (publicBatchWrapper89_copies312 a h).2.2.2.1
  have c9988 := (publicBatchWrapper89_copies312 a h).2.2.2.2.1
  have c9989 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.1
  have e_155_8 := arithEq_of_rows h (row := 155) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_8
  simp only [← c9984, ← c9985, ← c9986] at e_155_8
  have e_155_9 := arithEq_of_rows h (row := 155) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_9
  simp only [← c9987, ← c9988, ← c9989] at e_155_9
  have hr := e_155_9
  simp only [e_155_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 47) = bselect (a (.wire 141 59)) (a (.virt 14404)) (a (.wire 154 63)) := by
  have c9990 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.1
  have c9991 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.1
  have c9992 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.1
  have c9993 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.1
  have c9994 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9995 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_10 := arithEq_of_rows h (row := 155) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_10
  simp only [← c9990, ← c9991, ← c9992] at e_155_10
  have e_155_11 := arithEq_of_rows h (row := 155) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_11
  simp only [← c9993, ← c9994, ← c9995] at e_155_11
  have hr := e_155_11
  simp only [e_155_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 15) = bor (a (.wire 153 11)) (a (.wire 154 67)) := by
  have c9996 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9997 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9998 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9999 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10000 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10001 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_11 := arithEq_of_rows h (row := 145) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_11
  simp only [← c9996, ← c9997, ← c9998] at e_145_11
  have e_153_3 := arithEq_of_rows h (row := 153) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_3
  simp only [← c9999, ← c10000, k3, ← c10001] at e_153_3
  have hr := e_153_3
  simp only [e_145_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 51) = bnot (a (.wire 71 71)) := by
  have c10002 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10003 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10004 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_155_12 := arithEq_of_rows h (row := 155) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_12
  simp only [← c10002, k3, ← c10003, k3, ← c10004] at e_155_12
  have hr := e_155_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 55) = bnot (a (.wire 153 15)) := by
  have c10005 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10006 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10007 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_155_13 := arithEq_of_rows h (row := 155) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_13
  simp only [← c10005, k3, ← c10006, k3, ← c10007] at e_155_13
  have hr := e_155_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 63) = band (a (.wire 155 51)) (a (.wire 155 55)) := by
  have c10008 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10009 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10010 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_15 := arithEq_of_rows h (row := 141) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_15
  simp only [← c10008, ← c10009, ← c10010] at e_141_15
  have hr := e_141_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 63) = bselect (a (.wire 141 63)) (a (.virt 14605)) (a (.wire 154 79)) := by
  have c10011 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10012 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10013 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10014 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10015 := (publicBatchWrapper89_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10016 := (publicBatchWrapper89_copies313 a h).1
  have e_155_14 := arithEq_of_rows h (row := 155) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_14
  simp only [← c10011, ← c10012, ← c10013] at e_155_14
  have e_155_15 := arithEq_of_rows h (row := 155) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_15
  simp only [← c10014, ← c10015, ← c10016] at e_155_15
  have hr := e_155_15
  simp only [e_155_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 71) = bselect (a (.wire 141 63)) (a (.virt 14606)) (a (.wire 155 7)) := by
  have c10017 := (publicBatchWrapper89_copies313 a h).2.1
  have c10018 := (publicBatchWrapper89_copies313 a h).2.2.1
  have c10019 := (publicBatchWrapper89_copies313 a h).2.2.2.1
  have c10020 := (publicBatchWrapper89_copies313 a h).2.2.2.2.1
  have c10021 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.1
  have c10022 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.1
  have e_155_16 := arithEq_of_rows h (row := 155) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_16
  simp only [← c10017, ← c10018, ← c10019] at e_155_16
  have e_155_17 := arithEq_of_rows h (row := 155) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_17
  simp only [← c10020, ← c10021, ← c10022] at e_155_17
  have hr := e_155_17
  simp only [e_155_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 155 79) = bselect (a (.wire 141 63)) (a (.virt 14607)) (a (.wire 155 15)) := by
  have c10023 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.1
  have c10024 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.1
  have c10025 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.1
  have c10026 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10027 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10028 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_18 := arithEq_of_rows h (row := 155) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_18
  simp only [← c10023, ← c10024, ← c10025] at e_155_18
  have e_155_19 := arithEq_of_rows h (row := 155) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_19
  simp only [← c10026, ← c10027, ← c10028] at e_155_19
  have hr := e_155_19
  simp only [e_155_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 7) = bselect (a (.wire 141 63)) (a (.virt 14608)) (a (.wire 155 23)) := by
  have c10029 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10030 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10031 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10032 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10033 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10034 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_0 := arithEq_of_rows h (row := 156) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_0
  simp only [← c10029, ← c10030, ← c10031] at e_156_0
  have e_156_1 := arithEq_of_rows h (row := 156) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_1
  simp only [← c10032, ← c10033, ← c10034] at e_156_1
  have hr := e_156_1
  simp only [e_156_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 15) = bselect (a (.wire 141 63)) (a (.virt 14609)) (a (.wire 155 31)) := by
  have c10035 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10036 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10037 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10038 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10039 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10040 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_2 := arithEq_of_rows h (row := 156) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_2
  simp only [← c10035, ← c10036, ← c10037] at e_156_2
  have e_156_3 := arithEq_of_rows h (row := 156) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_3
  simp only [← c10038, ← c10039, ← c10040] at e_156_3
  have hr := e_156_3
  simp only [e_156_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 23) = bselect (a (.wire 141 63)) (a (.virt 14603)) (a (.wire 155 39)) := by
  have c10041 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10042 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10043 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10044 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10045 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10046 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_4 := arithEq_of_rows h (row := 156) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_4
  simp only [← c10041, ← c10042, ← c10043] at e_156_4
  have e_156_5 := arithEq_of_rows h (row := 156) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_5
  simp only [← c10044, ← c10045, ← c10046] at e_156_5
  have hr := e_156_5
  simp only [e_156_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 31) = bselect (a (.wire 141 63)) (a (.virt 14604)) (a (.wire 155 47)) := by
  have c10047 := (publicBatchWrapper89_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10048 := (publicBatchWrapper89_copies314 a h).1
  have c10049 := (publicBatchWrapper89_copies314 a h).2.1
  have c10050 := (publicBatchWrapper89_copies314 a h).2.2.1
  have c10051 := (publicBatchWrapper89_copies314 a h).2.2.2.1
  have c10052 := (publicBatchWrapper89_copies314 a h).2.2.2.2.1
  have e_156_6 := arithEq_of_rows h (row := 156) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_6
  simp only [← c10047, ← c10048, ← c10049] at e_156_6
  have e_156_7 := arithEq_of_rows h (row := 156) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_7
  simp only [← c10050, ← c10051, ← c10052] at e_156_7
  have hr := e_156_7
  simp only [e_156_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 19) = bor (a (.wire 153 15)) (a (.wire 155 51)) := by
  have c10053 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.1
  have c10054 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.1
  have c10055 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.1
  have c10056 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.1
  have c10057 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.1
  have c10058 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_12 := arithEq_of_rows h (row := 145) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_12
  simp only [← c10053, ← c10054, ← c10055] at e_145_12
  have e_153_4 := arithEq_of_rows h (row := 153) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_4
  simp only [← c10056, ← c10057, k3, ← c10058] at e_153_4
  have hr := e_153_4
  simp only [e_145_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 35) = bnot (a (.wire 72 35)) := by
  have c10059 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10060 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10061 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_156_8 := arithEq_of_rows h (row := 156) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_8
  simp only [← c10059, k3, ← c10060, k3, ← c10061] at e_156_8
  have hr := e_156_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 39) = bnot (a (.wire 153 19)) := by
  have c10062 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10063 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10064 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_156_9 := arithEq_of_rows h (row := 156) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_9
  simp only [← c10062, k3, ← c10063, k3, ← c10064] at e_156_9
  have hr := e_156_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 67) = band (a (.wire 156 35)) (a (.wire 156 39)) := by
  have c10065 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10066 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10067 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_16 := arithEq_of_rows h (row := 141) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_16
  simp only [← c10065, ← c10066, ← c10067] at e_141_16
  have hr := e_141_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 47) = bselect (a (.wire 141 67)) (a (.virt 14805)) (a (.wire 155 63)) := by
  have c10068 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10069 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10070 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10071 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10072 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10073 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_10 := arithEq_of_rows h (row := 156) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_10
  simp only [← c10068, ← c10069, ← c10070] at e_156_10
  have e_156_11 := arithEq_of_rows h (row := 156) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_11
  simp only [← c10071, ← c10072, ← c10073] at e_156_11
  have hr := e_156_11
  simp only [e_156_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 55) = bselect (a (.wire 141 67)) (a (.virt 14806)) (a (.wire 155 71)) := by
  have c10074 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10075 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10076 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10077 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10078 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10079 := (publicBatchWrapper89_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_156_12 := arithEq_of_rows h (row := 156) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_12
  simp only [← c10074, ← c10075, ← c10076] at e_156_12
  have e_156_13 := arithEq_of_rows h (row := 156) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_13
  simp only [← c10077, ← c10078, ← c10079] at e_156_13
  have hr := e_156_13
  simp only [e_156_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 63) = bselect (a (.wire 141 67)) (a (.virt 14807)) (a (.wire 155 79)) := by
  have c10080 := (publicBatchWrapper89_copies315 a h).1
  have c10081 := (publicBatchWrapper89_copies315 a h).2.1
  have c10082 := (publicBatchWrapper89_copies315 a h).2.2.1
  have c10083 := (publicBatchWrapper89_copies315 a h).2.2.2.1
  have c10084 := (publicBatchWrapper89_copies315 a h).2.2.2.2.1
  have c10085 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.1
  have e_156_14 := arithEq_of_rows h (row := 156) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_14
  simp only [← c10080, ← c10081, ← c10082] at e_156_14
  have e_156_15 := arithEq_of_rows h (row := 156) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_15
  simp only [← c10083, ← c10084, ← c10085] at e_156_15
  have hr := e_156_15
  simp only [e_156_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 71) = bselect (a (.wire 141 67)) (a (.virt 14808)) (a (.wire 156 7)) := by
  have c10086 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.1
  have c10087 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.1
  have c10088 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.1
  have c10089 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.1
  have c10090 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10091 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_16 := arithEq_of_rows h (row := 156) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_16
  simp only [← c10086, ← c10087, ← c10088] at e_156_16
  have e_156_17 := arithEq_of_rows h (row := 156) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_17
  simp only [← c10089, ← c10090, ← c10091] at e_156_17
  have hr := e_156_17
  simp only [e_156_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 156 79) = bselect (a (.wire 141 67)) (a (.virt 14809)) (a (.wire 156 15)) := by
  have c10092 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10093 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10094 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10095 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10096 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10097 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_18 := arithEq_of_rows h (row := 156) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_18
  simp only [← c10092, ← c10093, ← c10094] at e_156_18
  have e_156_19 := arithEq_of_rows h (row := 156) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_19
  simp only [← c10095, ← c10096, ← c10097] at e_156_19
  have hr := e_156_19
  simp only [e_156_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 7) = bselect (a (.wire 141 67)) (a (.virt 14803)) (a (.wire 156 23)) := by
  have c10098 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10099 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10100 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10101 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10102 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10103 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_0 := arithEq_of_rows h (row := 157) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_0
  simp only [← c10098, ← c10099, ← c10100] at e_157_0
  have e_157_1 := arithEq_of_rows h (row := 157) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_1
  simp only [← c10101, ← c10102, ← c10103] at e_157_1
  have hr := e_157_1
  simp only [e_157_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 15) = bselect (a (.wire 141 67)) (a (.virt 14804)) (a (.wire 156 31)) := by
  have c10104 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10105 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10106 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10107 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10108 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10109 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_2 := arithEq_of_rows h (row := 157) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_2
  simp only [← c10104, ← c10105, ← c10106] at e_157_2
  have e_157_3 := arithEq_of_rows h (row := 157) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_3
  simp only [← c10107, ← c10108, ← c10109] at e_157_3
  have hr := e_157_3
  simp only [e_157_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 23) = bor (a (.wire 153 19)) (a (.wire 156 35)) := by
  have c10110 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10111 := (publicBatchWrapper89_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10112 := (publicBatchWrapper89_copies316 a h).1
  have c10113 := (publicBatchWrapper89_copies316 a h).2.1
  have c10114 := (publicBatchWrapper89_copies316 a h).2.2.1
  have c10115 := (publicBatchWrapper89_copies316 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_13 := arithEq_of_rows h (row := 145) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_13
  simp only [← c10110, ← c10111, ← c10112] at e_145_13
  have e_153_5 := arithEq_of_rows h (row := 153) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_5
  simp only [← c10113, ← c10114, k3, ← c10115] at e_153_5
  have hr := e_153_5
  simp only [e_145_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 19) = bnot (a (.wire 72 79)) := by
  have c10116 := (publicBatchWrapper89_copies316 a h).2.2.2.2.1
  have c10117 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.1
  have c10118 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_157_4 := arithEq_of_rows h (row := 157) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_4
  simp only [← c10116, k3, ← c10117, k3, ← c10118] at e_157_4
  have hr := e_157_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 23) = bnot (a (.wire 153 23)) := by
  have c10119 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.1
  have c10120 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.1
  have c10121 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_157_5 := arithEq_of_rows h (row := 157) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_5
  simp only [← c10119, k3, ← c10120, k3, ← c10121] at e_157_5
  have hr := e_157_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 71) = band (a (.wire 157 19)) (a (.wire 157 23)) := by
  have c10122 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10123 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10124 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_17 := arithEq_of_rows h (row := 141) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_17
  simp only [← c10122, ← c10123, ← c10124] at e_141_17
  have hr := e_141_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 31) = bselect (a (.wire 141 71)) (a (.virt 15005)) (a (.wire 156 47)) := by
  have c10125 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10126 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10127 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10128 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10129 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10130 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_6 := arithEq_of_rows h (row := 157) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_6
  simp only [← c10125, ← c10126, ← c10127] at e_157_6
  have e_157_7 := arithEq_of_rows h (row := 157) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_7
  simp only [← c10128, ← c10129, ← c10130] at e_157_7
  have hr := e_157_7
  simp only [e_157_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 39) = bselect (a (.wire 141 71)) (a (.virt 15006)) (a (.wire 156 55)) := by
  have c10131 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10132 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10133 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10134 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10135 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10136 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_8 := arithEq_of_rows h (row := 157) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_8
  simp only [← c10131, ← c10132, ← c10133] at e_157_8
  have e_157_9 := arithEq_of_rows h (row := 157) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_9
  simp only [← c10134, ← c10135, ← c10136] at e_157_9
  have hr := e_157_9
  simp only [e_157_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 47) = bselect (a (.wire 141 71)) (a (.virt 15007)) (a (.wire 156 63)) := by
  have c10137 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10138 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10139 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10140 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10141 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10142 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_10 := arithEq_of_rows h (row := 157) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_10
  simp only [← c10137, ← c10138, ← c10139] at e_157_10
  have e_157_11 := arithEq_of_rows h (row := 157) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_11
  simp only [← c10140, ← c10141, ← c10142] at e_157_11
  have hr := e_157_11
  simp only [e_157_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 55) = bselect (a (.wire 141 71)) (a (.virt 15008)) (a (.wire 156 71)) := by
  have c10143 := (publicBatchWrapper89_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10144 := (publicBatchWrapper89_copies317 a h).1
  have c10145 := (publicBatchWrapper89_copies317 a h).2.1
  have c10146 := (publicBatchWrapper89_copies317 a h).2.2.1
  have c10147 := (publicBatchWrapper89_copies317 a h).2.2.2.1
  have c10148 := (publicBatchWrapper89_copies317 a h).2.2.2.2.1
  have e_157_12 := arithEq_of_rows h (row := 157) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_12
  simp only [← c10143, ← c10144, ← c10145] at e_157_12
  have e_157_13 := arithEq_of_rows h (row := 157) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_13
  simp only [← c10146, ← c10147, ← c10148] at e_157_13
  have hr := e_157_13
  simp only [e_157_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 63) = bselect (a (.wire 141 71)) (a (.virt 15009)) (a (.wire 156 79)) := by
  have c10149 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.1
  have c10150 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.1
  have c10151 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.1
  have c10152 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.1
  have c10153 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.1
  have c10154 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_157_14 := arithEq_of_rows h (row := 157) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_14
  simp only [← c10149, ← c10150, ← c10151] at e_157_14
  have e_157_15 := arithEq_of_rows h (row := 157) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_15
  simp only [← c10152, ← c10153, ← c10154] at e_157_15
  have hr := e_157_15
  simp only [e_157_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 71) = bselect (a (.wire 141 71)) (a (.virt 15003)) (a (.wire 157 7)) := by
  have c10155 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10156 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10157 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10158 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10159 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10160 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_16 := arithEq_of_rows h (row := 157) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_16
  simp only [← c10155, ← c10156, ← c10157] at e_157_16
  have e_157_17 := arithEq_of_rows h (row := 157) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_17
  simp only [← c10158, ← c10159, ← c10160] at e_157_17
  have hr := e_157_17
  simp only [e_157_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 157 79) = bselect (a (.wire 141 71)) (a (.virt 15004)) (a (.wire 157 15)) := by
  have c10161 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10162 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10163 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10164 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10165 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10166 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_18 := arithEq_of_rows h (row := 157) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_18
  simp only [← c10161, ← c10162, ← c10163] at e_157_18
  have e_157_19 := arithEq_of_rows h (row := 157) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_19
  simp only [← c10164, ← c10165, ← c10166] at e_157_19
  have hr := e_157_19
  simp only [e_157_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 27) = bor (a (.wire 153 23)) (a (.wire 157 19)) := by
  have c10167 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10168 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10169 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10170 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10171 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10172 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_14 := arithEq_of_rows h (row := 145) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_14
  simp only [← c10167, ← c10168, ← c10169] at e_145_14
  have e_153_6 := arithEq_of_rows h (row := 153) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_6
  simp only [← c10170, ← c10171, k3, ← c10172] at e_153_6
  have hr := e_153_6
  simp only [e_145_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 3) = bnot (a (.wire 74 43)) := by
  have c10173 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10174 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10175 := (publicBatchWrapper89_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_158_0 := arithEq_of_rows h (row := 158) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_0
  simp only [← c10173, k3, ← c10174, k3, ← c10175] at e_158_0
  have hr := e_158_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 7) = bnot (a (.wire 153 27)) := by
  have c10176 := (publicBatchWrapper89_copies318 a h).1
  have c10177 := (publicBatchWrapper89_copies318 a h).2.1
  have c10178 := (publicBatchWrapper89_copies318 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_158_1 := arithEq_of_rows h (row := 158) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_1
  simp only [← c10176, k3, ← c10177, k3, ← c10178] at e_158_1
  have hr := e_158_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 75) = band (a (.wire 158 3)) (a (.wire 158 7)) := by
  have c10179 := (publicBatchWrapper89_copies318 a h).2.2.2.1
  have c10180 := (publicBatchWrapper89_copies318 a h).2.2.2.2.1
  have c10181 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.1
  have e_141_18 := arithEq_of_rows h (row := 141) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_18
  simp only [← c10179, ← c10180, ← c10181] at e_141_18
  have hr := e_141_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 15) = bselect (a (.wire 141 75)) (a (.virt 15205)) (a (.wire 157 31)) := by
  have c10182 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.1
  have c10183 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.1
  have c10184 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.1
  have c10185 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.1
  have c10186 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10187 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_2 := arithEq_of_rows h (row := 158) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_2
  simp only [← c10182, ← c10183, ← c10184] at e_158_2
  have e_158_3 := arithEq_of_rows h (row := 158) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_3
  simp only [← c10185, ← c10186, ← c10187] at e_158_3
  have hr := e_158_3
  simp only [e_158_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 23) = bselect (a (.wire 141 75)) (a (.virt 15206)) (a (.wire 157 39)) := by
  have c10188 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10189 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10190 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10191 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10192 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10193 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_4 := arithEq_of_rows h (row := 158) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_4
  simp only [← c10188, ← c10189, ← c10190] at e_158_4
  have e_158_5 := arithEq_of_rows h (row := 158) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_5
  simp only [← c10191, ← c10192, ← c10193] at e_158_5
  have hr := e_158_5
  simp only [e_158_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 31) = bselect (a (.wire 141 75)) (a (.virt 15207)) (a (.wire 157 47)) := by
  have c10194 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10195 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10196 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10197 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10198 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10199 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_6 := arithEq_of_rows h (row := 158) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_6
  simp only [← c10194, ← c10195, ← c10196] at e_158_6
  have e_158_7 := arithEq_of_rows h (row := 158) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_7
  simp only [← c10197, ← c10198, ← c10199] at e_158_7
  have hr := e_158_7
  simp only [e_158_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 39) = bselect (a (.wire 141 75)) (a (.virt 15208)) (a (.wire 157 55)) := by
  have c10200 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10201 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10202 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10203 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10204 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10205 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_8 := arithEq_of_rows h (row := 158) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_8
  simp only [← c10200, ← c10201, ← c10202] at e_158_8
  have e_158_9 := arithEq_of_rows h (row := 158) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_9
  simp only [← c10203, ← c10204, ← c10205] at e_158_9
  have hr := e_158_9
  simp only [e_158_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 47) = bselect (a (.wire 141 75)) (a (.virt 15209)) (a (.wire 157 63)) := by
  have c10206 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10207 := (publicBatchWrapper89_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10208 := (publicBatchWrapper89_copies319 a h).1
  have c10209 := (publicBatchWrapper89_copies319 a h).2.1
  have c10210 := (publicBatchWrapper89_copies319 a h).2.2.1
  have c10211 := (publicBatchWrapper89_copies319 a h).2.2.2.1
  have e_158_10 := arithEq_of_rows h (row := 158) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_10
  simp only [← c10206, ← c10207, ← c10208] at e_158_10
  have e_158_11 := arithEq_of_rows h (row := 158) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_11
  simp only [← c10209, ← c10210, ← c10211] at e_158_11
  have hr := e_158_11
  simp only [e_158_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 55) = bselect (a (.wire 141 75)) (a (.virt 15203)) (a (.wire 157 71)) := by
  have c10212 := (publicBatchWrapper89_copies319 a h).2.2.2.2.1
  have c10213 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.1
  have c10214 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.1
  have c10215 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.1
  have c10216 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.1
  have c10217 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.1
  have e_158_12 := arithEq_of_rows h (row := 158) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_12
  simp only [← c10212, ← c10213, ← c10214] at e_158_12
  have e_158_13 := arithEq_of_rows h (row := 158) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_13
  simp only [← c10215, ← c10216, ← c10217] at e_158_13
  have hr := e_158_13
  simp only [e_158_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 63) = bselect (a (.wire 141 75)) (a (.virt 15204)) (a (.wire 157 79)) := by
  have c10218 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10219 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10220 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10221 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10222 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10223 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_14 := arithEq_of_rows h (row := 158) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_14
  simp only [← c10218, ← c10219, ← c10220] at e_158_14
  have e_158_15 := arithEq_of_rows h (row := 158) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_15
  simp only [← c10221, ← c10222, ← c10223] at e_158_15
  have hr := e_158_15
  simp only [e_158_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 31) = bor (a (.wire 153 27)) (a (.wire 158 3)) := by
  have c10224 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10225 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10226 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10227 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10228 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10229 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_15 := arithEq_of_rows h (row := 145) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_15
  simp only [← c10224, ← c10225, ← c10226] at e_145_15
  have e_153_7 := arithEq_of_rows h (row := 153) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_7
  simp only [← c10227, ← c10228, k3, ← c10229] at e_153_7
  have hr := e_153_7
  simp only [e_145_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 67) = bnot (a (.wire 76 7)) := by
  have c10230 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10231 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10232 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_158_16 := arithEq_of_rows h (row := 158) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_16
  simp only [← c10230, k3, ← c10231, k3, ← c10232] at e_158_16
  have hr := e_158_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 71) = bnot (a (.wire 153 31)) := by
  have c10233 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10234 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10235 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_158_17 := arithEq_of_rows h (row := 158) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_17
  simp only [← c10233, k3, ← c10234, k3, ← c10235] at e_158_17
  have hr := e_158_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 141 79) = band (a (.wire 158 67)) (a (.wire 158 71)) := by
  have c10236 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10237 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10238 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_19 := arithEq_of_rows h (row := 141) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_19
  simp only [← c10236, ← c10237, ← c10238] at e_141_19
  have hr := e_141_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 158 79) = bselect (a (.wire 141 79)) (a (.virt 15405)) (a (.wire 158 15)) := by
  have c10239 := (publicBatchWrapper89_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10240 := (publicBatchWrapper89_copies320 a h).1
  have c10241 := (publicBatchWrapper89_copies320 a h).2.1
  have c10242 := (publicBatchWrapper89_copies320 a h).2.2.1
  have c10243 := (publicBatchWrapper89_copies320 a h).2.2.2.1
  have c10244 := (publicBatchWrapper89_copies320 a h).2.2.2.2.1
  have e_158_18 := arithEq_of_rows h (row := 158) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_18
  simp only [← c10239, ← c10240, ← c10241] at e_158_18
  have e_158_19 := arithEq_of_rows h (row := 158) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_19
  simp only [← c10242, ← c10243, ← c10244] at e_158_19
  have hr := e_158_19
  simp only [e_158_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 7) = bselect (a (.wire 141 79)) (a (.virt 15406)) (a (.wire 158 23)) := by
  have c10245 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.1
  have c10246 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.1
  have c10247 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.1
  have c10248 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.1
  have c10249 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.1
  have c10250 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_159_0 := arithEq_of_rows h (row := 159) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_0
  simp only [← c10245, ← c10246, ← c10247] at e_159_0
  have e_159_1 := arithEq_of_rows h (row := 159) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_1
  simp only [← c10248, ← c10249, ← c10250] at e_159_1
  have hr := e_159_1
  simp only [e_159_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 15) = bselect (a (.wire 141 79)) (a (.virt 15407)) (a (.wire 158 31)) := by
  have c10251 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10252 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10253 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10254 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10255 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10256 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_159_2 := arithEq_of_rows h (row := 159) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_2
  simp only [← c10251, ← c10252, ← c10253] at e_159_2
  have e_159_3 := arithEq_of_rows h (row := 159) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_3
  simp only [← c10254, ← c10255, ← c10256] at e_159_3
  have hr := e_159_3
  simp only [e_159_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 23) = bselect (a (.wire 141 79)) (a (.virt 15408)) (a (.wire 158 39)) := by
  have c10257 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10258 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10259 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10260 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10261 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10262 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_159_4 := arithEq_of_rows h (row := 159) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_4
  simp only [← c10257, ← c10258, ← c10259] at e_159_4
  have e_159_5 := arithEq_of_rows h (row := 159) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_5
  simp only [← c10260, ← c10261, ← c10262] at e_159_5
  have hr := e_159_5
  simp only [e_159_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 31) = bselect (a (.wire 141 79)) (a (.virt 15409)) (a (.wire 158 47)) := by
  have c10263 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10264 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10265 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10266 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10267 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10268 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_159_6 := arithEq_of_rows h (row := 159) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_6
  simp only [← c10263, ← c10264, ← c10265] at e_159_6
  have e_159_7 := arithEq_of_rows h (row := 159) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_7
  simp only [← c10266, ← c10267, ← c10268] at e_159_7
  have hr := e_159_7
  simp only [e_159_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 39) = bselect (a (.wire 141 79)) (a (.virt 15403)) (a (.wire 158 55)) := by
  have c10269 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10270 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10271 := (publicBatchWrapper89_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10272 := (publicBatchWrapper89_copies321 a h).1
  have c10273 := (publicBatchWrapper89_copies321 a h).2.1
  have c10274 := (publicBatchWrapper89_copies321 a h).2.2.1
  have e_159_8 := arithEq_of_rows h (row := 159) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_8
  simp only [← c10269, ← c10270, ← c10271] at e_159_8
  have e_159_9 := arithEq_of_rows h (row := 159) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_9
  simp only [← c10272, ← c10273, ← c10274] at e_159_9
  have hr := e_159_9
  simp only [e_159_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 47) = bselect (a (.wire 141 79)) (a (.virt 15404)) (a (.wire 158 63)) := by
  have c10275 := (publicBatchWrapper89_copies321 a h).2.2.2.1
  have c10276 := (publicBatchWrapper89_copies321 a h).2.2.2.2.1
  have c10277 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.1
  have c10278 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.1
  have c10279 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.1
  have c10280 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.1
  have e_159_10 := arithEq_of_rows h (row := 159) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_10
  simp only [← c10275, ← c10276, ← c10277] at e_159_10
  have e_159_11 := arithEq_of_rows h (row := 159) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_11
  simp only [← c10278, ← c10279, ← c10280] at e_159_11
  have hr := e_159_11
  simp only [e_159_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 35) = bor (a (.wire 153 31)) (a (.wire 158 67)) := by
  have c10281 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.1
  have c10282 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10283 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10284 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10285 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10286 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_16 := arithEq_of_rows h (row := 145) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_16
  simp only [← c10281, ← c10282, ← c10283] at e_145_16
  have e_153_8 := arithEq_of_rows h (row := 153) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_8
  simp only [← c10284, ← c10285, k3, ← c10286] at e_153_8
  have hr := e_153_8
  simp only [e_145_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 51) = bnot (a (.wire 76 51)) := by
  have c10287 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10288 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10289 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_159_12 := arithEq_of_rows h (row := 159) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_12
  simp only [← c10287, k3, ← c10288, k3, ← c10289] at e_159_12
  have hr := e_159_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 55) = bnot (a (.wire 153 35)) := by
  have c10290 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10291 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10292 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_159_13 := arithEq_of_rows h (row := 159) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_13
  simp only [← c10290, k3, ← c10291, k3, ← c10292] at e_159_13
  have hr := e_159_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 3) = band (a (.wire 159 51)) (a (.wire 159 55)) := by
  have c10293 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10294 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10295 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_0 := arithEq_of_rows h (row := 160) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_0
  simp only [← c10293, ← c10294, ← c10295] at e_160_0
  have hr := e_160_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 63) = bselect (a (.wire 160 3)) (a (.virt 15605)) (a (.wire 158 79)) := by
  have c10296 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10297 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10298 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10299 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10300 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10301 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_159_14 := arithEq_of_rows h (row := 159) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_14
  simp only [← c10296, ← c10297, ← c10298] at e_159_14
  have e_159_15 := arithEq_of_rows h (row := 159) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_15
  simp only [← c10299, ← c10300, ← c10301] at e_159_15
  have hr := e_159_15
  simp only [e_159_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 71) = bselect (a (.wire 160 3)) (a (.virt 15606)) (a (.wire 159 7)) := by
  have c10302 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10303 := (publicBatchWrapper89_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10304 := (publicBatchWrapper89_copies322 a h).1
  have c10305 := (publicBatchWrapper89_copies322 a h).2.1
  have c10306 := (publicBatchWrapper89_copies322 a h).2.2.1
  have c10307 := (publicBatchWrapper89_copies322 a h).2.2.2.1
  have e_159_16 := arithEq_of_rows h (row := 159) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_16
  simp only [← c10302, ← c10303, ← c10304] at e_159_16
  have e_159_17 := arithEq_of_rows h (row := 159) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_17
  simp only [← c10305, ← c10306, ← c10307] at e_159_17
  have hr := e_159_17
  simp only [e_159_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 159 79) = bselect (a (.wire 160 3)) (a (.virt 15607)) (a (.wire 159 15)) := by
  have c10308 := (publicBatchWrapper89_copies322 a h).2.2.2.2.1
  have c10309 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.1
  have c10310 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.1
  have c10311 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.1
  have c10312 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.1
  have c10313 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.1
  have e_159_18 := arithEq_of_rows h (row := 159) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_18
  simp only [← c10308, ← c10309, ← c10310] at e_159_18
  have e_159_19 := arithEq_of_rows h (row := 159) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_19
  simp only [← c10311, ← c10312, ← c10313] at e_159_19
  have hr := e_159_19
  simp only [e_159_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 7) = bselect (a (.wire 160 3)) (a (.virt 15608)) (a (.wire 159 23)) := by
  have c10314 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10315 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10316 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10317 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10318 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10319 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_0 := arithEq_of_rows h (row := 161) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_0
  simp only [← c10314, ← c10315, ← c10316] at e_161_0
  have e_161_1 := arithEq_of_rows h (row := 161) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_1
  simp only [← c10317, ← c10318, ← c10319] at e_161_1
  have hr := e_161_1
  simp only [e_161_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 15) = bselect (a (.wire 160 3)) (a (.virt 15609)) (a (.wire 159 31)) := by
  have c10320 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10321 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10322 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10323 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10324 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10325 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_2 := arithEq_of_rows h (row := 161) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_2
  simp only [← c10320, ← c10321, ← c10322] at e_161_2
  have e_161_3 := arithEq_of_rows h (row := 161) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_3
  simp only [← c10323, ← c10324, ← c10325] at e_161_3
  have hr := e_161_3
  simp only [e_161_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 23) = bselect (a (.wire 160 3)) (a (.virt 15603)) (a (.wire 159 39)) := by
  have c10326 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10327 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10328 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10329 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10330 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10331 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_4 := arithEq_of_rows h (row := 161) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_4
  simp only [← c10326, ← c10327, ← c10328] at e_161_4
  have e_161_5 := arithEq_of_rows h (row := 161) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_5
  simp only [← c10329, ← c10330, ← c10331] at e_161_5
  have hr := e_161_5
  simp only [e_161_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 31) = bselect (a (.wire 160 3)) (a (.virt 15604)) (a (.wire 159 47)) := by
  have c10332 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10333 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10334 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10335 := (publicBatchWrapper89_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10336 := (publicBatchWrapper89_copies323 a h).1
  have c10337 := (publicBatchWrapper89_copies323 a h).2.1
  have e_161_6 := arithEq_of_rows h (row := 161) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_6
  simp only [← c10332, ← c10333, ← c10334] at e_161_6
  have e_161_7 := arithEq_of_rows h (row := 161) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_7
  simp only [← c10335, ← c10336, ← c10337] at e_161_7
  have hr := e_161_7
  simp only [e_161_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 39) = bor (a (.wire 153 35)) (a (.wire 159 51)) := by
  have c10338 := (publicBatchWrapper89_copies323 a h).2.2.1
  have c10339 := (publicBatchWrapper89_copies323 a h).2.2.2.1
  have c10340 := (publicBatchWrapper89_copies323 a h).2.2.2.2.1
  have c10341 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.1
  have c10342 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.1
  have c10343 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_17 := arithEq_of_rows h (row := 145) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_17
  simp only [← c10338, ← c10339, ← c10340] at e_145_17
  have e_153_9 := arithEq_of_rows h (row := 153) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_9
  simp only [← c10341, ← c10342, k3, ← c10343] at e_153_9
  have hr := e_153_9
  simp only [e_145_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 35) = bnot (a (.wire 77 15)) := by
  have c10344 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.1
  have c10345 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.1
  have c10346 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_161_8 := arithEq_of_rows h (row := 161) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_8
  simp only [← c10344, k3, ← c10345, k3, ← c10346] at e_161_8
  have hr := e_161_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 39) = bnot (a (.wire 153 39)) := by
  have c10347 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10348 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10349 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_161_9 := arithEq_of_rows h (row := 161) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_9
  simp only [← c10347, k3, ← c10348, k3, ← c10349] at e_161_9
  have hr := e_161_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 7) = band (a (.wire 161 35)) (a (.wire 161 39)) := by
  have c10350 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10351 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10352 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_1 := arithEq_of_rows h (row := 160) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_1
  simp only [← c10350, ← c10351, ← c10352] at e_160_1
  have hr := e_160_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 47) = bselect (a (.wire 160 7)) (a (.virt 15805)) (a (.wire 159 63)) := by
  have c10353 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10354 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10355 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10356 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10357 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10358 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_10 := arithEq_of_rows h (row := 161) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_10
  simp only [← c10353, ← c10354, ← c10355] at e_161_10
  have e_161_11 := arithEq_of_rows h (row := 161) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_11
  simp only [← c10356, ← c10357, ← c10358] at e_161_11
  have hr := e_161_11
  simp only [e_161_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 55) = bselect (a (.wire 160 7)) (a (.virt 15806)) (a (.wire 159 71)) := by
  have c10359 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10360 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10361 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10362 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10363 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10364 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_12 := arithEq_of_rows h (row := 161) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_12
  simp only [← c10359, ← c10360, ← c10361] at e_161_12
  have e_161_13 := arithEq_of_rows h (row := 161) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_13
  simp only [← c10362, ← c10363, ← c10364] at e_161_13
  have hr := e_161_13
  simp only [e_161_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 63) = bselect (a (.wire 160 7)) (a (.virt 15807)) (a (.wire 159 79)) := by
  have c10365 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10366 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10367 := (publicBatchWrapper89_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10368 := (publicBatchWrapper89_copies324 a h).1
  have c10369 := (publicBatchWrapper89_copies324 a h).2.1
  have c10370 := (publicBatchWrapper89_copies324 a h).2.2.1
  have e_161_14 := arithEq_of_rows h (row := 161) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_14
  simp only [← c10365, ← c10366, ← c10367] at e_161_14
  have e_161_15 := arithEq_of_rows h (row := 161) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_15
  simp only [← c10368, ← c10369, ← c10370] at e_161_15
  have hr := e_161_15
  simp only [e_161_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 71) = bselect (a (.wire 160 7)) (a (.virt 15808)) (a (.wire 161 7)) := by
  have c10371 := (publicBatchWrapper89_copies324 a h).2.2.2.1
  have c10372 := (publicBatchWrapper89_copies324 a h).2.2.2.2.1
  have c10373 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.1
  have c10374 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.1
  have c10375 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.1
  have c10376 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.1
  have e_161_16 := arithEq_of_rows h (row := 161) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_16
  simp only [← c10371, ← c10372, ← c10373] at e_161_16
  have e_161_17 := arithEq_of_rows h (row := 161) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_17
  simp only [← c10374, ← c10375, ← c10376] at e_161_17
  have hr := e_161_17
  simp only [e_161_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 161 79) = bselect (a (.wire 160 7)) (a (.virt 15809)) (a (.wire 161 15)) := by
  have c10377 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.1
  have c10378 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10379 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10380 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10381 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10382 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_18 := arithEq_of_rows h (row := 161) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_18
  simp only [← c10377, ← c10378, ← c10379] at e_161_18
  have e_161_19 := arithEq_of_rows h (row := 161) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_19
  simp only [← c10380, ← c10381, ← c10382] at e_161_19
  have hr := e_161_19
  simp only [e_161_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 7) = bselect (a (.wire 160 7)) (a (.virt 15803)) (a (.wire 161 23)) := by
  have c10383 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10384 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10385 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10386 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10387 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10388 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_0 := arithEq_of_rows h (row := 162) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_0
  simp only [← c10383, ← c10384, ← c10385] at e_162_0
  have e_162_1 := arithEq_of_rows h (row := 162) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_1
  simp only [← c10386, ← c10387, ← c10388] at e_162_1
  have hr := e_162_1
  simp only [e_162_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 15) = bselect (a (.wire 160 7)) (a (.virt 15804)) (a (.wire 161 31)) := by
  have c10389 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10390 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10391 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10392 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10393 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10394 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_2 := arithEq_of_rows h (row := 162) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_2
  simp only [← c10389, ← c10390, ← c10391] at e_162_2
  have e_162_3 := arithEq_of_rows h (row := 162) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_3
  simp only [← c10392, ← c10393, ← c10394] at e_162_3
  have hr := e_162_3
  simp only [e_162_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 43) = bor (a (.wire 153 39)) (a (.wire 161 35)) := by
  have c10395 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10396 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10397 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10398 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10399 := (publicBatchWrapper89_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10400 := (publicBatchWrapper89_copies325 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_18 := arithEq_of_rows h (row := 145) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_18
  simp only [← c10395, ← c10396, ← c10397] at e_145_18
  have e_153_10 := arithEq_of_rows h (row := 153) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_10
  simp only [← c10398, ← c10399, k3, ← c10400] at e_153_10
  have hr := e_153_10
  simp only [e_145_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 19) = bnot (a (.wire 77 59)) := by
  have c10401 := (publicBatchWrapper89_copies325 a h).2.1
  have c10402 := (publicBatchWrapper89_copies325 a h).2.2.1
  have c10403 := (publicBatchWrapper89_copies325 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_162_4 := arithEq_of_rows h (row := 162) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_4
  simp only [← c10401, k3, ← c10402, k3, ← c10403] at e_162_4
  have hr := e_162_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 23) = bnot (a (.wire 153 43)) := by
  have c10404 := (publicBatchWrapper89_copies325 a h).2.2.2.2.1
  have c10405 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.1
  have c10406 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_162_5 := arithEq_of_rows h (row := 162) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_5
  simp only [← c10404, k3, ← c10405, k3, ← c10406] at e_162_5
  have hr := e_162_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 11) = band (a (.wire 162 19)) (a (.wire 162 23)) := by
  have c10407 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.1
  have c10408 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.1
  have c10409 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.1
  have e_160_2 := arithEq_of_rows h (row := 160) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_2
  simp only [← c10407, ← c10408, ← c10409] at e_160_2
  have hr := e_160_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 31) = bselect (a (.wire 160 11)) (a (.virt 16005)) (a (.wire 161 47)) := by
  have c10410 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10411 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10412 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10413 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10414 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10415 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_6 := arithEq_of_rows h (row := 162) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_6
  simp only [← c10410, ← c10411, ← c10412] at e_162_6
  have e_162_7 := arithEq_of_rows h (row := 162) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_7
  simp only [← c10413, ← c10414, ← c10415] at e_162_7
  have hr := e_162_7
  simp only [e_162_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 39) = bselect (a (.wire 160 11)) (a (.virt 16006)) (a (.wire 161 55)) := by
  have c10416 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10417 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10418 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10419 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10420 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10421 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_8 := arithEq_of_rows h (row := 162) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_8
  simp only [← c10416, ← c10417, ← c10418] at e_162_8
  have e_162_9 := arithEq_of_rows h (row := 162) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_9
  simp only [← c10419, ← c10420, ← c10421] at e_162_9
  have hr := e_162_9
  simp only [e_162_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 47) = bselect (a (.wire 160 11)) (a (.virt 16007)) (a (.wire 161 63)) := by
  have c10422 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10423 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10424 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10425 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10426 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10427 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_10 := arithEq_of_rows h (row := 162) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_10
  simp only [← c10422, ← c10423, ← c10424] at e_162_10
  have e_162_11 := arithEq_of_rows h (row := 162) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_11
  simp only [← c10425, ← c10426, ← c10427] at e_162_11
  have hr := e_162_11
  simp only [e_162_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 55) = bselect (a (.wire 160 11)) (a (.virt 16008)) (a (.wire 161 71)) := by
  have c10428 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10429 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10430 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10431 := (publicBatchWrapper89_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10432 := (publicBatchWrapper89_copies326 a h).1
  have c10433 := (publicBatchWrapper89_copies326 a h).2.1
  have e_162_12 := arithEq_of_rows h (row := 162) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_12
  simp only [← c10428, ← c10429, ← c10430] at e_162_12
  have e_162_13 := arithEq_of_rows h (row := 162) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_13
  simp only [← c10431, ← c10432, ← c10433] at e_162_13
  have hr := e_162_13
  simp only [e_162_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 63) = bselect (a (.wire 160 11)) (a (.virt 16009)) (a (.wire 161 79)) := by
  have c10434 := (publicBatchWrapper89_copies326 a h).2.2.1
  have c10435 := (publicBatchWrapper89_copies326 a h).2.2.2.1
  have c10436 := (publicBatchWrapper89_copies326 a h).2.2.2.2.1
  have c10437 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.1
  have c10438 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.1
  have c10439 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.1
  have e_162_14 := arithEq_of_rows h (row := 162) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_14
  simp only [← c10434, ← c10435, ← c10436] at e_162_14
  have e_162_15 := arithEq_of_rows h (row := 162) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_15
  simp only [← c10437, ← c10438, ← c10439] at e_162_15
  have hr := e_162_15
  simp only [e_162_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 71) = bselect (a (.wire 160 11)) (a (.virt 16003)) (a (.wire 162 7)) := by
  have c10440 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.1
  have c10441 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.1
  have c10442 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10443 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10444 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10445 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_16 := arithEq_of_rows h (row := 162) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_16
  simp only [← c10440, ← c10441, ← c10442] at e_162_16
  have e_162_17 := arithEq_of_rows h (row := 162) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_17
  simp only [← c10443, ← c10444, ← c10445] at e_162_17
  have hr := e_162_17
  simp only [e_162_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 162 79) = bselect (a (.wire 160 11)) (a (.virt 16004)) (a (.wire 162 15)) := by
  have c10446 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10447 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10448 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10449 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10450 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10451 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_162_18 := arithEq_of_rows h (row := 162) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_18
  simp only [← c10446, ← c10447, ← c10448] at e_162_18
  have e_162_19 := arithEq_of_rows h (row := 162) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_19
  simp only [← c10449, ← c10450, ← c10451] at e_162_19
  have hr := e_162_19
  simp only [e_162_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 47) = bor (a (.wire 153 43)) (a (.wire 162 19)) := by
  have c10452 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10453 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10454 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10455 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10456 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10457 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_145_19 := arithEq_of_rows h (row := 145) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_19
  simp only [← c10452, ← c10453, ← c10454] at e_145_19
  have e_153_11 := arithEq_of_rows h (row := 153) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_11
  simp only [← c10455, ← c10456, k3, ← c10457] at e_153_11
  have hr := e_153_11
  simp only [e_145_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f1603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 3) = bnot (a (.wire 79 23)) := by
  have c10458 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10459 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10460 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_163_0 := arithEq_of_rows h (row := 163) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_0
  simp only [← c10458, k3, ← c10459, k3, ← c10460] at e_163_0
  have hr := e_163_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 7) = bnot (a (.wire 153 47)) := by
  have c10461 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10462 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10463 := (publicBatchWrapper89_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_163_1 := arithEq_of_rows h (row := 163) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_1
  simp only [← c10461, k3, ← c10462, k3, ← c10463] at e_163_1
  have hr := e_163_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper89_f1605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 160 15) = band (a (.wire 163 3)) (a (.wire 163 7)) := by
  have c10464 := (publicBatchWrapper89_copies327 a h).1
  have c10465 := (publicBatchWrapper89_copies327 a h).2.1
  have c10466 := (publicBatchWrapper89_copies327 a h).2.2.1
  have e_160_3 := arithEq_of_rows h (row := 160) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_3
  simp only [← c10464, ← c10465, ← c10466] at e_160_3
  have hr := e_160_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f1606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 15) = bselect (a (.wire 160 15)) (a (.virt 16205)) (a (.wire 162 31)) := by
  have c10467 := (publicBatchWrapper89_copies327 a h).2.2.2.1
  have c10468 := (publicBatchWrapper89_copies327 a h).2.2.2.2.1
  have c10469 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.1
  have c10470 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.1
  have c10471 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.1
  have c10472 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.1
  have e_163_2 := arithEq_of_rows h (row := 163) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_2
  simp only [← c10467, ← c10468, ← c10469] at e_163_2
  have e_163_3 := arithEq_of_rows h (row := 163) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_3
  simp only [← c10470, ← c10471, ← c10472] at e_163_3
  have hr := e_163_3
  simp only [e_163_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 23) = bselect (a (.wire 160 15)) (a (.virt 16206)) (a (.wire 162 39)) := by
  have c10473 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.1
  have c10474 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10475 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10476 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10477 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10478 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_4 := arithEq_of_rows h (row := 163) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_4
  simp only [← c10473, ← c10474, ← c10475] at e_163_4
  have e_163_5 := arithEq_of_rows h (row := 163) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_5
  simp only [← c10476, ← c10477, ← c10478] at e_163_5
  have hr := e_163_5
  simp only [e_163_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 31) = bselect (a (.wire 160 15)) (a (.virt 16207)) (a (.wire 162 47)) := by
  have c10479 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10480 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10481 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10482 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10483 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10484 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_6 := arithEq_of_rows h (row := 163) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_6
  simp only [← c10479, ← c10480, ← c10481] at e_163_6
  have e_163_7 := arithEq_of_rows h (row := 163) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_7
  simp only [← c10482, ← c10483, ← c10484] at e_163_7
  have hr := e_163_7
  simp only [e_163_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 39) = bselect (a (.wire 160 15)) (a (.virt 16208)) (a (.wire 162 55)) := by
  have c10485 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10486 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10487 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10488 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10489 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10490 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_8 := arithEq_of_rows h (row := 163) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_8
  simp only [← c10485, ← c10486, ← c10487] at e_163_8
  have e_163_9 := arithEq_of_rows h (row := 163) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_9
  simp only [← c10488, ← c10489, ← c10490] at e_163_9
  have hr := e_163_9
  simp only [e_163_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 47) = bselect (a (.wire 160 15)) (a (.virt 16209)) (a (.wire 162 63)) := by
  have c10491 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10492 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10493 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10494 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10495 := (publicBatchWrapper89_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10496 := (publicBatchWrapper89_copies328 a h).1
  have e_163_10 := arithEq_of_rows h (row := 163) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_10
  simp only [← c10491, ← c10492, ← c10493] at e_163_10
  have e_163_11 := arithEq_of_rows h (row := 163) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_11
  simp only [← c10494, ← c10495, ← c10496] at e_163_11
  have hr := e_163_11
  simp only [e_163_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 55) = bselect (a (.wire 160 15)) (a (.virt 16203)) (a (.wire 162 71)) := by
  have c10497 := (publicBatchWrapper89_copies328 a h).2.1
  have c10498 := (publicBatchWrapper89_copies328 a h).2.2.1
  have c10499 := (publicBatchWrapper89_copies328 a h).2.2.2.1
  have c10500 := (publicBatchWrapper89_copies328 a h).2.2.2.2.1
  have c10501 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.1
  have c10502 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.1
  have e_163_12 := arithEq_of_rows h (row := 163) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_12
  simp only [← c10497, ← c10498, ← c10499] at e_163_12
  have e_163_13 := arithEq_of_rows h (row := 163) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_13
  simp only [← c10500, ← c10501, ← c10502] at e_163_13
  have hr := e_163_13
  simp only [e_163_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 163 63) = bselect (a (.wire 160 15)) (a (.virt 16204)) (a (.wire 162 79)) := by
  have c10503 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.1
  have c10504 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.1
  have c10505 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.1
  have c10506 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10507 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10508 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_14 := arithEq_of_rows h (row := 163) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_14
  simp only [← c10503, ← c10504, ← c10505] at e_163_14
  have e_163_15 := arithEq_of_rows h (row := 163) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_15
  simp only [← c10506, ← c10507, ← c10508] at e_163_15
  have hr := e_163_15
  simp only [e_163_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper89_f1613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 153 51) = bor (a (.wire 153 47)) (a (.wire 163 3)) := by
  have c10509 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10510 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10511 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10512 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10513 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10514 := (publicBatchWrapper89_copies328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_153_12 := arithEq_of_rows h (row := 153) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_12
  simp only [← c10512, ← c10513, k3, ← c10514] at e_153_12
  have e_164_0 := arithEq_of_rows h (row := 164) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_0
  simp only [← c10509, ← c10510, ← c10511] at e_164_0
  have hr := e_153_12
  simp only [e_164_0] at hr
  simp only [bor]
  linear_combination hr

end Plonky2Spec.Generated
