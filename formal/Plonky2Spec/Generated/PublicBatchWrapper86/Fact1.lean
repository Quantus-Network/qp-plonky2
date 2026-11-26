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

theorem publicBatchWrapper86_f677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16808)) (a (.virt 200)) (a (.virt 17885)) (a (.virt 17886)) := by
  have c5588 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5589 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5590 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5591 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5592 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5593 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5594 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5595 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5596 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5597 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5598 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5599 := (publicBatchWrapper86_copies174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5600 := (publicBatchWrapper86_copies175 a h).1
  have c5601 := (publicBatchWrapper86_copies175 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_81_18 := arithEq_of_rows h (row := 81) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_18
  simp only [← c5588, k3, ← c5589, k3, ← c5590] at e_81_18
  have e_81_19 := arithEq_of_rows h (row := 81) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_19
  simp only [← c5597, ← c5598, k3, ← c5599] at e_81_19
  have e_82_6 := arithEq_of_rows h (row := 82) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_6
  simp only [← c5591, ← c5592, ← c5593] at e_82_6
  have e_82_7 := arithEq_of_rows h (row := 82) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_7
  simp only [← c5594, ← c5595, ← c5596] at e_82_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_82_6
    linear_combination c5600.trans k2 - hc
  · have hc := e_81_19
    simp only [e_81_18, e_82_7] at hc
    linear_combination c5601.trans k2 - hc

theorem publicBatchWrapper86_f678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 82 35) = band (a (.virt 17879)) (a (.virt 17881)) := by
  have c5602 := (publicBatchWrapper86_copies175 a h).2.2.1
  have c5603 := (publicBatchWrapper86_copies175 a h).2.2.2.1
  have c5604 := (publicBatchWrapper86_copies175 a h).2.2.2.2.1
  have e_82_8 := arithEq_of_rows h (row := 82) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_8
  simp only [← c5602, ← c5603, ← c5604] at e_82_8
  have hr := e_82_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 82 39) = band (a (.virt 17883)) (a (.virt 17885)) := by
  have c5605 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.1
  have c5606 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.1
  have c5607 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.1
  have e_82_9 := arithEq_of_rows h (row := 82) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_9
  simp only [← c5605, ← c5606, ← c5607] at e_82_9
  have hr := e_82_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 82 43) = band (a (.wire 82 35)) (a (.wire 82 39)) := by
  have c5608 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.1
  have c5609 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.1
  have c5610 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_82_10 := arithEq_of_rows h (row := 82) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_10
  simp only [← c5608, ← c5609, ← c5610] at e_82_10
  have hr := e_82_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17005)) (a (.virt 200)) (a (.virt 17887)) (a (.virt 17888)) := by
  have c5611 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5612 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5613 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5614 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5615 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5616 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5617 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5618 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5619 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5620 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5621 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5622 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5623 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5624 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_82_11 := arithEq_of_rows h (row := 82) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_11
  simp only [← c5614, ← c5615, ← c5616] at e_82_11
  have e_82_12 := arithEq_of_rows h (row := 82) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_12
  simp only [← c5617, ← c5618, ← c5619] at e_82_12
  have e_83_0 := arithEq_of_rows h (row := 83) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_0
  simp only [← c5611, k3, ← c5612, k3, ← c5613] at e_83_0
  have e_83_1 := arithEq_of_rows h (row := 83) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_1
  simp only [← c5620, ← c5621, k3, ← c5622] at e_83_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_82_11
    linear_combination c5623.trans k2 - hc
  · have hc := e_83_1
    simp only [e_83_0, e_82_12] at hc
    linear_combination c5624.trans k2 - hc

theorem publicBatchWrapper86_f682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17006)) (a (.virt 200)) (a (.virt 17889)) (a (.virt 17890)) := by
  have c5625 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5626 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5627 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5628 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5629 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5630 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5631 := (publicBatchWrapper86_copies175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5632 := (publicBatchWrapper86_copies176 a h).1
  have c5633 := (publicBatchWrapper86_copies176 a h).2.1
  have c5634 := (publicBatchWrapper86_copies176 a h).2.2.1
  have c5635 := (publicBatchWrapper86_copies176 a h).2.2.2.1
  have c5636 := (publicBatchWrapper86_copies176 a h).2.2.2.2.1
  have c5637 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.1
  have c5638 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_82_13 := arithEq_of_rows h (row := 82) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_13
  simp only [← c5628, ← c5629, ← c5630] at e_82_13
  have e_82_14 := arithEq_of_rows h (row := 82) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_14
  simp only [← c5631, ← c5632, ← c5633] at e_82_14
  have e_83_2 := arithEq_of_rows h (row := 83) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_2
  simp only [← c5625, k3, ← c5626, k3, ← c5627] at e_83_2
  have e_83_3 := arithEq_of_rows h (row := 83) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_3
  simp only [← c5634, ← c5635, k3, ← c5636] at e_83_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_82_13
    linear_combination c5637.trans k2 - hc
  · have hc := e_83_3
    simp only [e_83_2, e_82_14] at hc
    linear_combination c5638.trans k2 - hc

theorem publicBatchWrapper86_f683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17007)) (a (.virt 200)) (a (.virt 17891)) (a (.virt 17892)) := by
  have c5639 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.1
  have c5640 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.1
  have c5641 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.1
  have c5642 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5643 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5644 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5645 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5646 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5647 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5648 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5649 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5650 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5651 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5652 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_82_15 := arithEq_of_rows h (row := 82) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_15
  simp only [← c5642, ← c5643, ← c5644] at e_82_15
  have e_82_16 := arithEq_of_rows h (row := 82) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_16
  simp only [← c5645, ← c5646, ← c5647] at e_82_16
  have e_83_4 := arithEq_of_rows h (row := 83) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_4
  simp only [← c5639, k3, ← c5640, k3, ← c5641] at e_83_4
  have e_83_5 := arithEq_of_rows h (row := 83) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_5
  simp only [← c5648, ← c5649, k3, ← c5650] at e_83_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_82_15
    linear_combination c5651.trans k2 - hc
  · have hc := e_83_5
    simp only [e_83_4, e_82_16] at hc
    linear_combination c5652.trans k2 - hc

theorem publicBatchWrapper86_f684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17008)) (a (.virt 200)) (a (.virt 17893)) (a (.virt 17894)) := by
  have c5653 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5654 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5655 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5656 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5657 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5658 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5659 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5660 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5661 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5662 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5663 := (publicBatchWrapper86_copies176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5664 := (publicBatchWrapper86_copies177 a h).1
  have c5665 := (publicBatchWrapper86_copies177 a h).2.1
  have c5666 := (publicBatchWrapper86_copies177 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_82_17 := arithEq_of_rows h (row := 82) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_17
  simp only [← c5656, ← c5657, ← c5658] at e_82_17
  have e_82_18 := arithEq_of_rows h (row := 82) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_18
  simp only [← c5659, ← c5660, ← c5661] at e_82_18
  have e_83_6 := arithEq_of_rows h (row := 83) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_6
  simp only [← c5653, k3, ← c5654, k3, ← c5655] at e_83_6
  have e_83_7 := arithEq_of_rows h (row := 83) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_7
  simp only [← c5662, ← c5663, k3, ← c5664] at e_83_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_82_17
    linear_combination c5665.trans k2 - hc
  · have hc := e_83_7
    simp only [e_83_6, e_82_18] at hc
    linear_combination c5666.trans k2 - hc

theorem publicBatchWrapper86_f685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 82 79) = band (a (.virt 17887)) (a (.virt 17889)) := by
  have c5667 := (publicBatchWrapper86_copies177 a h).2.2.2.1
  have c5668 := (publicBatchWrapper86_copies177 a h).2.2.2.2.1
  have c5669 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.1
  have e_82_19 := arithEq_of_rows h (row := 82) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_82_19
  simp only [← c5667, ← c5668, ← c5669] at e_82_19
  have hr := e_82_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 3) = band (a (.virt 17891)) (a (.virt 17893)) := by
  have c5670 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.1
  have c5671 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.1
  have c5672 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.1
  have e_84_0 := arithEq_of_rows h (row := 84) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_0
  simp only [← c5670, ← c5671, ← c5672] at e_84_0
  have hr := e_84_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 7) = band (a (.wire 82 79)) (a (.wire 84 3)) := by
  have c5673 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.1
  have c5674 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5675 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_1 := arithEq_of_rows h (row := 84) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_1
  simp only [← c5673, ← c5674, ← c5675] at e_84_1
  have hr := e_84_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 35) = bnot (a (.wire 3 27)) := by
  have c5676 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5677 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5678 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_8 := arithEq_of_rows h (row := 83) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_8
  simp only [← c5676, k3, ← c5677, k3, ← c5678] at e_83_8
  have hr := e_83_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 201) = bnot (a (.virt 200)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  simp only [bnot, k2, k3]
  ring

theorem publicBatchWrapper86_f690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 35) = band (a (.wire 83 35)) (a (.virt 201)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  simp only [band, k3]
  ring

theorem publicBatchWrapper86_f691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 39) = bselect (a (.wire 83 35)) (a (.virt 3)) (a (.virt 200)) := by
  have c5679 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5680 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5681 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_9 := arithEq_of_rows h (row := 83) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_9
  simp only [← c5679, ← c5680, ← c5681, k2] at e_83_9
  have hr := e_83_9
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 43) = bselect (a (.wire 83 35)) (a (.virt 4)) (a (.virt 200)) := by
  have c5682 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5683 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5684 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_10 := arithEq_of_rows h (row := 83) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_10
  simp only [← c5682, ← c5683, ← c5684, k2] at e_83_10
  have hr := e_83_10
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 47) = bselect (a (.wire 83 35)) (a (.virt 5)) (a (.virt 200)) := by
  have c5685 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5686 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5687 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_11 := arithEq_of_rows h (row := 83) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_11
  simp only [← c5685, ← c5686, ← c5687, k2] at e_83_11
  have hr := e_83_11
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 51) = bselect (a (.wire 83 35)) (a (.virt 6)) (a (.virt 200)) := by
  have c5688 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5689 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5690 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_12 := arithEq_of_rows h (row := 83) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_12
  simp only [← c5688, ← c5689, ← c5690, k2] at e_83_12
  have hr := e_83_12
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 55) = bselect (a (.wire 83 35)) (a (.virt 7)) (a (.virt 200)) := by
  have c5691 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5692 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5693 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_13 := arithEq_of_rows h (row := 83) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_13
  simp only [← c5691, ← c5692, ← c5693, k2] at e_83_13
  have hr := e_83_13
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 59) = bselect (a (.wire 83 35)) (a (.virt 1)) (a (.virt 200)) := by
  have c5694 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5695 := (publicBatchWrapper86_copies177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5696 := (publicBatchWrapper86_copies178 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_14 := arithEq_of_rows h (row := 83) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_14
  simp only [← c5694, ← c5695, ← c5696, k2] at e_83_14
  have hr := e_83_14
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 63) = bselect (a (.wire 83 35)) (a (.virt 2)) (a (.virt 200)) := by
  have c5697 := (publicBatchWrapper86_copies178 a h).2.1
  have c5698 := (publicBatchWrapper86_copies178 a h).2.2.1
  have c5699 := (publicBatchWrapper86_copies178 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_15 := arithEq_of_rows h (row := 83) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_15
  simp only [← c5697, ← c5698, ← c5699, k2] at e_83_15
  have hr := e_83_15
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 35) = bor (a (.virt 200)) (a (.wire 83 35)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  simp only [bor, k2]
  ring

theorem publicBatchWrapper86_f699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 67) = bnot (a (.wire 3 71)) := by
  have c5700 := (publicBatchWrapper86_copies178 a h).2.2.2.2.1
  have c5701 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.1
  have c5702 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_16 := arithEq_of_rows h (row := 83) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_16
  simp only [← c5700, k3, ← c5701, k3, ← c5702] at e_83_16
  have hr := e_83_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 71) = bnot (a (.wire 83 35)) := by
  have c5703 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.1
  have c5704 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.1
  have c5705 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_83_17 := arithEq_of_rows h (row := 83) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_17
  simp only [← c5703, k3, ← c5704, k3, ← c5705] at e_83_17
  have hr := e_83_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 11) = band (a (.wire 83 67)) (a (.wire 83 71)) := by
  have c5706 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5707 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5708 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_2 := arithEq_of_rows h (row := 84) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_2
  simp only [← c5706, ← c5707, ← c5708] at e_84_2
  have hr := e_84_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 83 79) = bselect (a (.wire 84 11)) (a (.virt 205)) (a (.wire 83 39)) := by
  have c5709 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5710 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5711 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5712 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5713 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5714 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_83_18 := arithEq_of_rows h (row := 83) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_18
  simp only [← c5709, ← c5710, ← c5711] at e_83_18
  have e_83_19 := arithEq_of_rows h (row := 83) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_83_19
  simp only [← c5712, ← c5713, ← c5714] at e_83_19
  have hr := e_83_19
  simp only [e_83_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 7) = bselect (a (.wire 84 11)) (a (.virt 206)) (a (.wire 83 43)) := by
  have c5715 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5716 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5717 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5718 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5719 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5720 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_0 := arithEq_of_rows h (row := 85) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_0
  simp only [← c5715, ← c5716, ← c5717] at e_85_0
  have e_85_1 := arithEq_of_rows h (row := 85) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_1
  simp only [← c5718, ← c5719, ← c5720] at e_85_1
  have hr := e_85_1
  simp only [e_85_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 15) = bselect (a (.wire 84 11)) (a (.virt 207)) (a (.wire 83 47)) := by
  have c5721 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5722 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5723 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5724 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5725 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5726 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_2 := arithEq_of_rows h (row := 85) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_2
  simp only [← c5721, ← c5722, ← c5723] at e_85_2
  have e_85_3 := arithEq_of_rows h (row := 85) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_3
  simp only [← c5724, ← c5725, ← c5726] at e_85_3
  have hr := e_85_3
  simp only [e_85_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 23) = bselect (a (.wire 84 11)) (a (.virt 208)) (a (.wire 83 51)) := by
  have c5727 := (publicBatchWrapper86_copies178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5728 := (publicBatchWrapper86_copies179 a h).1
  have c5729 := (publicBatchWrapper86_copies179 a h).2.1
  have c5730 := (publicBatchWrapper86_copies179 a h).2.2.1
  have c5731 := (publicBatchWrapper86_copies179 a h).2.2.2.1
  have c5732 := (publicBatchWrapper86_copies179 a h).2.2.2.2.1
  have e_85_4 := arithEq_of_rows h (row := 85) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_4
  simp only [← c5727, ← c5728, ← c5729] at e_85_4
  have e_85_5 := arithEq_of_rows h (row := 85) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_5
  simp only [← c5730, ← c5731, ← c5732] at e_85_5
  have hr := e_85_5
  simp only [e_85_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 31) = bselect (a (.wire 84 11)) (a (.virt 209)) (a (.wire 83 55)) := by
  have c5733 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.1
  have c5734 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.1
  have c5735 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.1
  have c5736 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.1
  have c5737 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.1
  have c5738 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_85_6 := arithEq_of_rows h (row := 85) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_6
  simp only [← c5733, ← c5734, ← c5735] at e_85_6
  have e_85_7 := arithEq_of_rows h (row := 85) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_7
  simp only [← c5736, ← c5737, ← c5738] at e_85_7
  have hr := e_85_7
  simp only [e_85_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 39) = bselect (a (.wire 84 11)) (a (.virt 203)) (a (.wire 83 59)) := by
  have c5739 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5740 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5741 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5742 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5743 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5744 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_8 := arithEq_of_rows h (row := 85) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_8
  simp only [← c5739, ← c5740, ← c5741] at e_85_8
  have e_85_9 := arithEq_of_rows h (row := 85) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_9
  simp only [← c5742, ← c5743, ← c5744] at e_85_9
  have hr := e_85_9
  simp only [e_85_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 47) = bselect (a (.wire 84 11)) (a (.virt 204)) (a (.wire 83 63)) := by
  have c5745 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5746 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5747 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5748 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5749 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5750 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_10 := arithEq_of_rows h (row := 85) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_10
  simp only [← c5745, ← c5746, ← c5747] at e_85_10
  have e_85_11 := arithEq_of_rows h (row := 85) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_11
  simp only [← c5748, ← c5749, ← c5750] at e_85_11
  have hr := e_85_11
  simp only [e_85_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 35) = bor (a (.wire 83 35)) (a (.wire 83 67)) := by
  have c5751 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5752 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5753 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5754 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5755 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5756 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c5754, ← c5755, k3, ← c5756] at e_1_8
  have e_86_0 := arithEq_of_rows h (row := 86) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_0
  simp only [← c5751, ← c5752, ← c5753] at e_86_0
  have hr := e_1_8
  simp only [e_86_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 51) = bnot (a (.wire 4 35)) := by
  have c5757 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5758 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5759 := (publicBatchWrapper86_copies179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_85_12 := arithEq_of_rows h (row := 85) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_12
  simp only [← c5757, k3, ← c5758, k3, ← c5759] at e_85_12
  have hr := e_85_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 55) = bnot (a (.wire 1 35)) := by
  have c5760 := (publicBatchWrapper86_copies180 a h).1
  have c5761 := (publicBatchWrapper86_copies180 a h).2.1
  have c5762 := (publicBatchWrapper86_copies180 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_85_13 := arithEq_of_rows h (row := 85) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_13
  simp only [← c5760, k3, ← c5761, k3, ← c5762] at e_85_13
  have hr := e_85_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 15) = band (a (.wire 85 51)) (a (.wire 85 55)) := by
  have c5763 := (publicBatchWrapper86_copies180 a h).2.2.2.1
  have c5764 := (publicBatchWrapper86_copies180 a h).2.2.2.2.1
  have c5765 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.1
  have e_84_3 := arithEq_of_rows h (row := 84) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_3
  simp only [← c5763, ← c5764, ← c5765] at e_84_3
  have hr := e_84_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 63) = bselect (a (.wire 84 15)) (a (.virt 405)) (a (.wire 83 79)) := by
  have c5766 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.1
  have c5767 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.1
  have c5768 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.1
  have c5769 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.1
  have c5770 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5771 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_14 := arithEq_of_rows h (row := 85) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_14
  simp only [← c5766, ← c5767, ← c5768] at e_85_14
  have e_85_15 := arithEq_of_rows h (row := 85) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_15
  simp only [← c5769, ← c5770, ← c5771] at e_85_15
  have hr := e_85_15
  simp only [e_85_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 71) = bselect (a (.wire 84 15)) (a (.virt 406)) (a (.wire 85 7)) := by
  have c5772 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5773 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5774 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5775 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5776 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5777 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_16 := arithEq_of_rows h (row := 85) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_16
  simp only [← c5772, ← c5773, ← c5774] at e_85_16
  have e_85_17 := arithEq_of_rows h (row := 85) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_17
  simp only [← c5775, ← c5776, ← c5777] at e_85_17
  have hr := e_85_17
  simp only [e_85_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 85 79) = bselect (a (.wire 84 15)) (a (.virt 407)) (a (.wire 85 15)) := by
  have c5778 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5779 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5780 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5781 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5782 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5783 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_85_18 := arithEq_of_rows h (row := 85) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_18
  simp only [← c5778, ← c5779, ← c5780] at e_85_18
  have e_85_19 := arithEq_of_rows h (row := 85) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_85_19
  simp only [← c5781, ← c5782, ← c5783] at e_85_19
  have hr := e_85_19
  simp only [e_85_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 7) = bselect (a (.wire 84 15)) (a (.virt 408)) (a (.wire 85 23)) := by
  have c5784 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5785 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5786 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5787 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5788 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5789 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_0 := arithEq_of_rows h (row := 87) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_0
  simp only [← c5784, ← c5785, ← c5786] at e_87_0
  have e_87_1 := arithEq_of_rows h (row := 87) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_1
  simp only [← c5787, ← c5788, ← c5789] at e_87_1
  have hr := e_87_1
  simp only [e_87_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 15) = bselect (a (.wire 84 15)) (a (.virt 409)) (a (.wire 85 31)) := by
  have c5790 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5791 := (publicBatchWrapper86_copies180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5792 := (publicBatchWrapper86_copies181 a h).1
  have c5793 := (publicBatchWrapper86_copies181 a h).2.1
  have c5794 := (publicBatchWrapper86_copies181 a h).2.2.1
  have c5795 := (publicBatchWrapper86_copies181 a h).2.2.2.1
  have e_87_2 := arithEq_of_rows h (row := 87) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_2
  simp only [← c5790, ← c5791, ← c5792] at e_87_2
  have e_87_3 := arithEq_of_rows h (row := 87) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_3
  simp only [← c5793, ← c5794, ← c5795] at e_87_3
  have hr := e_87_3
  simp only [e_87_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 23) = bselect (a (.wire 84 15)) (a (.virt 403)) (a (.wire 85 39)) := by
  have c5796 := (publicBatchWrapper86_copies181 a h).2.2.2.2.1
  have c5797 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.1
  have c5798 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.1
  have c5799 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.1
  have c5800 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.1
  have c5801 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.1
  have e_87_4 := arithEq_of_rows h (row := 87) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_4
  simp only [← c5796, ← c5797, ← c5798] at e_87_4
  have e_87_5 := arithEq_of_rows h (row := 87) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_5
  simp only [← c5799, ← c5800, ← c5801] at e_87_5
  have hr := e_87_5
  simp only [e_87_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 31) = bselect (a (.wire 84 15)) (a (.virt 404)) (a (.wire 85 47)) := by
  have c5802 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5803 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5804 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5805 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5806 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5807 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_6 := arithEq_of_rows h (row := 87) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_6
  simp only [← c5802, ← c5803, ← c5804] at e_87_6
  have e_87_7 := arithEq_of_rows h (row := 87) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_7
  simp only [← c5805, ← c5806, ← c5807] at e_87_7
  have hr := e_87_7
  simp only [e_87_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 39) = bor (a (.wire 1 35)) (a (.wire 85 51)) := by
  have c5808 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5809 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5810 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5811 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5812 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5813 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_9 := arithEq_of_rows h (row := 1) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_9
  simp only [← c5811, ← c5812, k3, ← c5813] at e_1_9
  have e_86_1 := arithEq_of_rows h (row := 86) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_1
  simp only [← c5808, ← c5809, ← c5810] at e_86_1
  have hr := e_1_9
  simp only [e_86_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 35) = bnot (a (.wire 4 79)) := by
  have c5814 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5815 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5816 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_87_8 := arithEq_of_rows h (row := 87) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_8
  simp only [← c5814, k3, ← c5815, k3, ← c5816] at e_87_8
  have hr := e_87_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 39) = bnot (a (.wire 1 39)) := by
  have c5817 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5818 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5819 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_87_9 := arithEq_of_rows h (row := 87) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_9
  simp only [← c5817, k3, ← c5818, k3, ← c5819] at e_87_9
  have hr := e_87_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 19) = band (a (.wire 87 35)) (a (.wire 87 39)) := by
  have c5820 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5821 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5822 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_4 := arithEq_of_rows h (row := 84) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_4
  simp only [← c5820, ← c5821, ← c5822] at e_84_4
  have hr := e_84_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 47) = bselect (a (.wire 84 19)) (a (.virt 605)) (a (.wire 85 63)) := by
  have c5823 := (publicBatchWrapper86_copies181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5824 := (publicBatchWrapper86_copies182 a h).1
  have c5825 := (publicBatchWrapper86_copies182 a h).2.1
  have c5826 := (publicBatchWrapper86_copies182 a h).2.2.1
  have c5827 := (publicBatchWrapper86_copies182 a h).2.2.2.1
  have c5828 := (publicBatchWrapper86_copies182 a h).2.2.2.2.1
  have e_87_10 := arithEq_of_rows h (row := 87) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_10
  simp only [← c5823, ← c5824, ← c5825] at e_87_10
  have e_87_11 := arithEq_of_rows h (row := 87) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_11
  simp only [← c5826, ← c5827, ← c5828] at e_87_11
  have hr := e_87_11
  simp only [e_87_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 55) = bselect (a (.wire 84 19)) (a (.virt 606)) (a (.wire 85 71)) := by
  have c5829 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.1
  have c5830 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.1
  have c5831 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.1
  have c5832 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.1
  have c5833 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.1
  have c5834 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_87_12 := arithEq_of_rows h (row := 87) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_12
  simp only [← c5829, ← c5830, ← c5831] at e_87_12
  have e_87_13 := arithEq_of_rows h (row := 87) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_13
  simp only [← c5832, ← c5833, ← c5834] at e_87_13
  have hr := e_87_13
  simp only [e_87_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 63) = bselect (a (.wire 84 19)) (a (.virt 607)) (a (.wire 85 79)) := by
  have c5835 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5836 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5837 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5838 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5839 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5840 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_14 := arithEq_of_rows h (row := 87) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_14
  simp only [← c5835, ← c5836, ← c5837] at e_87_14
  have e_87_15 := arithEq_of_rows h (row := 87) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_15
  simp only [← c5838, ← c5839, ← c5840] at e_87_15
  have hr := e_87_15
  simp only [e_87_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 71) = bselect (a (.wire 84 19)) (a (.virt 608)) (a (.wire 87 7)) := by
  have c5841 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5842 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5843 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5844 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5845 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5846 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_16 := arithEq_of_rows h (row := 87) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_16
  simp only [← c5841, ← c5842, ← c5843] at e_87_16
  have e_87_17 := arithEq_of_rows h (row := 87) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_17
  simp only [← c5844, ← c5845, ← c5846] at e_87_17
  have hr := e_87_17
  simp only [e_87_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 87 79) = bselect (a (.wire 84 19)) (a (.virt 609)) (a (.wire 87 15)) := by
  have c5847 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5848 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5849 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5850 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5851 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5852 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_87_18 := arithEq_of_rows h (row := 87) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_18
  simp only [← c5847, ← c5848, ← c5849] at e_87_18
  have e_87_19 := arithEq_of_rows h (row := 87) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_87_19
  simp only [← c5850, ← c5851, ← c5852] at e_87_19
  have hr := e_87_19
  simp only [e_87_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 7) = bselect (a (.wire 84 19)) (a (.virt 603)) (a (.wire 87 23)) := by
  have c5853 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5854 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5855 := (publicBatchWrapper86_copies182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5856 := (publicBatchWrapper86_copies183 a h).1
  have c5857 := (publicBatchWrapper86_copies183 a h).2.1
  have c5858 := (publicBatchWrapper86_copies183 a h).2.2.1
  have e_88_0 := arithEq_of_rows h (row := 88) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_0
  simp only [← c5853, ← c5854, ← c5855] at e_88_0
  have e_88_1 := arithEq_of_rows h (row := 88) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_1
  simp only [← c5856, ← c5857, ← c5858] at e_88_1
  have hr := e_88_1
  simp only [e_88_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 15) = bselect (a (.wire 84 19)) (a (.virt 604)) (a (.wire 87 31)) := by
  have c5859 := (publicBatchWrapper86_copies183 a h).2.2.2.1
  have c5860 := (publicBatchWrapper86_copies183 a h).2.2.2.2.1
  have c5861 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.1
  have c5862 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.1
  have c5863 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.1
  have c5864 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.1
  have e_88_2 := arithEq_of_rows h (row := 88) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_2
  simp only [← c5859, ← c5860, ← c5861] at e_88_2
  have e_88_3 := arithEq_of_rows h (row := 88) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_3
  simp only [← c5862, ← c5863, ← c5864] at e_88_3
  have hr := e_88_3
  simp only [e_88_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 43) = bor (a (.wire 1 39)) (a (.wire 87 35)) := by
  have c5865 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.1
  have c5866 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5867 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5868 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5869 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5870 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_10 := arithEq_of_rows h (row := 1) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_10
  simp only [← c5868, ← c5869, k3, ← c5870] at e_1_10
  have e_86_2 := arithEq_of_rows h (row := 86) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_2
  simp only [← c5865, ← c5866, ← c5867] at e_86_2
  have hr := e_1_10
  simp only [e_86_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 19) = bnot (a (.wire 6 43)) := by
  have c5871 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5872 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5873 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_88_4 := arithEq_of_rows h (row := 88) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_4
  simp only [← c5871, k3, ← c5872, k3, ← c5873] at e_88_4
  have hr := e_88_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 23) = bnot (a (.wire 1 43)) := by
  have c5874 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5875 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5876 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_88_5 := arithEq_of_rows h (row := 88) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_5
  simp only [← c5874, k3, ← c5875, k3, ← c5876] at e_88_5
  have hr := e_88_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 23) = band (a (.wire 88 19)) (a (.wire 88 23)) := by
  have c5877 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5878 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5879 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_5 := arithEq_of_rows h (row := 84) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_5
  simp only [← c5877, ← c5878, ← c5879] at e_84_5
  have hr := e_84_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 31) = bselect (a (.wire 84 23)) (a (.virt 805)) (a (.wire 87 47)) := by
  have c5880 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5881 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5882 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5883 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5884 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5885 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_88_6 := arithEq_of_rows h (row := 88) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_6
  simp only [← c5880, ← c5881, ← c5882] at e_88_6
  have e_88_7 := arithEq_of_rows h (row := 88) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_7
  simp only [← c5883, ← c5884, ← c5885] at e_88_7
  have hr := e_88_7
  simp only [e_88_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 39) = bselect (a (.wire 84 23)) (a (.virt 806)) (a (.wire 87 55)) := by
  have c5886 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5887 := (publicBatchWrapper86_copies183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5888 := (publicBatchWrapper86_copies184 a h).1
  have c5889 := (publicBatchWrapper86_copies184 a h).2.1
  have c5890 := (publicBatchWrapper86_copies184 a h).2.2.1
  have c5891 := (publicBatchWrapper86_copies184 a h).2.2.2.1
  have e_88_8 := arithEq_of_rows h (row := 88) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_8
  simp only [← c5886, ← c5887, ← c5888] at e_88_8
  have e_88_9 := arithEq_of_rows h (row := 88) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_9
  simp only [← c5889, ← c5890, ← c5891] at e_88_9
  have hr := e_88_9
  simp only [e_88_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 47) = bselect (a (.wire 84 23)) (a (.virt 807)) (a (.wire 87 63)) := by
  have c5892 := (publicBatchWrapper86_copies184 a h).2.2.2.2.1
  have c5893 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.1
  have c5894 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.1
  have c5895 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.1
  have c5896 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.1
  have c5897 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.1
  have e_88_10 := arithEq_of_rows h (row := 88) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_10
  simp only [← c5892, ← c5893, ← c5894] at e_88_10
  have e_88_11 := arithEq_of_rows h (row := 88) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_11
  simp only [← c5895, ← c5896, ← c5897] at e_88_11
  have hr := e_88_11
  simp only [e_88_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 55) = bselect (a (.wire 84 23)) (a (.virt 808)) (a (.wire 87 71)) := by
  have c5898 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5899 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5900 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5901 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5902 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5903 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_88_12 := arithEq_of_rows h (row := 88) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_12
  simp only [← c5898, ← c5899, ← c5900] at e_88_12
  have e_88_13 := arithEq_of_rows h (row := 88) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_13
  simp only [← c5901, ← c5902, ← c5903] at e_88_13
  have hr := e_88_13
  simp only [e_88_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 63) = bselect (a (.wire 84 23)) (a (.virt 809)) (a (.wire 87 79)) := by
  have c5904 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5905 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5906 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5907 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5908 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5909 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_88_14 := arithEq_of_rows h (row := 88) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_14
  simp only [← c5904, ← c5905, ← c5906] at e_88_14
  have e_88_15 := arithEq_of_rows h (row := 88) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_15
  simp only [← c5907, ← c5908, ← c5909] at e_88_15
  have hr := e_88_15
  simp only [e_88_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 71) = bselect (a (.wire 84 23)) (a (.virt 803)) (a (.wire 88 7)) := by
  have c5910 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5911 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5912 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5913 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5914 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5915 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_88_16 := arithEq_of_rows h (row := 88) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_16
  simp only [← c5910, ← c5911, ← c5912] at e_88_16
  have e_88_17 := arithEq_of_rows h (row := 88) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_17
  simp only [← c5913, ← c5914, ← c5915] at e_88_17
  have hr := e_88_17
  simp only [e_88_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 88 79) = bselect (a (.wire 84 23)) (a (.virt 804)) (a (.wire 88 15)) := by
  have c5916 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5917 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5918 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5919 := (publicBatchWrapper86_copies184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5920 := (publicBatchWrapper86_copies185 a h).1
  have c5921 := (publicBatchWrapper86_copies185 a h).2.1
  have e_88_18 := arithEq_of_rows h (row := 88) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_18
  simp only [← c5916, ← c5917, ← c5918] at e_88_18
  have e_88_19 := arithEq_of_rows h (row := 88) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_88_19
  simp only [← c5919, ← c5920, ← c5921] at e_88_19
  have hr := e_88_19
  simp only [e_88_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 47) = bor (a (.wire 1 43)) (a (.wire 88 19)) := by
  have c5922 := (publicBatchWrapper86_copies185 a h).2.2.1
  have c5923 := (publicBatchWrapper86_copies185 a h).2.2.2.1
  have c5924 := (publicBatchWrapper86_copies185 a h).2.2.2.2.1
  have c5925 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.1
  have c5926 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.1
  have c5927 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_11 := arithEq_of_rows h (row := 1) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_11
  simp only [← c5925, ← c5926, k3, ← c5927] at e_1_11
  have e_86_3 := arithEq_of_rows h (row := 86) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_3
  simp only [← c5922, ← c5923, ← c5924] at e_86_3
  have hr := e_1_11
  simp only [e_86_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 3) = bnot (a (.wire 8 7)) := by
  have c5928 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.1
  have c5929 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.1
  have c5930 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_89_0 := arithEq_of_rows h (row := 89) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_0
  simp only [← c5928, k3, ← c5929, k3, ← c5930] at e_89_0
  have hr := e_89_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 7) = bnot (a (.wire 1 47)) := by
  have c5931 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5932 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5933 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_89_1 := arithEq_of_rows h (row := 89) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_1
  simp only [← c5931, k3, ← c5932, k3, ← c5933] at e_89_1
  have hr := e_89_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 27) = band (a (.wire 89 3)) (a (.wire 89 7)) := by
  have c5934 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5935 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5936 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_6 := arithEq_of_rows h (row := 84) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_6
  simp only [← c5934, ← c5935, ← c5936] at e_84_6
  have hr := e_84_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 15) = bselect (a (.wire 84 27)) (a (.virt 1005)) (a (.wire 88 31)) := by
  have c5937 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5938 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5939 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5940 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5941 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5942 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_2 := arithEq_of_rows h (row := 89) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_2
  simp only [← c5937, ← c5938, ← c5939] at e_89_2
  have e_89_3 := arithEq_of_rows h (row := 89) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_3
  simp only [← c5940, ← c5941, ← c5942] at e_89_3
  have hr := e_89_3
  simp only [e_89_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 23) = bselect (a (.wire 84 27)) (a (.virt 1006)) (a (.wire 88 39)) := by
  have c5943 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5944 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5945 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5946 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5947 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5948 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_4 := arithEq_of_rows h (row := 89) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_4
  simp only [← c5943, ← c5944, ← c5945] at e_89_4
  have e_89_5 := arithEq_of_rows h (row := 89) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_5
  simp only [← c5946, ← c5947, ← c5948] at e_89_5
  have hr := e_89_5
  simp only [e_89_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 31) = bselect (a (.wire 84 27)) (a (.virt 1007)) (a (.wire 88 47)) := by
  have c5949 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5950 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5951 := (publicBatchWrapper86_copies185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5952 := (publicBatchWrapper86_copies186 a h).1
  have c5953 := (publicBatchWrapper86_copies186 a h).2.1
  have c5954 := (publicBatchWrapper86_copies186 a h).2.2.1
  have e_89_6 := arithEq_of_rows h (row := 89) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_6
  simp only [← c5949, ← c5950, ← c5951] at e_89_6
  have e_89_7 := arithEq_of_rows h (row := 89) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_7
  simp only [← c5952, ← c5953, ← c5954] at e_89_7
  have hr := e_89_7
  simp only [e_89_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 39) = bselect (a (.wire 84 27)) (a (.virt 1008)) (a (.wire 88 55)) := by
  have c5955 := (publicBatchWrapper86_copies186 a h).2.2.2.1
  have c5956 := (publicBatchWrapper86_copies186 a h).2.2.2.2.1
  have c5957 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.1
  have c5958 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.1
  have c5959 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.1
  have c5960 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.1
  have e_89_8 := arithEq_of_rows h (row := 89) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_8
  simp only [← c5955, ← c5956, ← c5957] at e_89_8
  have e_89_9 := arithEq_of_rows h (row := 89) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_9
  simp only [← c5958, ← c5959, ← c5960] at e_89_9
  have hr := e_89_9
  simp only [e_89_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 47) = bselect (a (.wire 84 27)) (a (.virt 1009)) (a (.wire 88 63)) := by
  have c5961 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.1
  have c5962 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5963 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5964 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5965 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5966 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_10 := arithEq_of_rows h (row := 89) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_10
  simp only [← c5961, ← c5962, ← c5963] at e_89_10
  have e_89_11 := arithEq_of_rows h (row := 89) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_11
  simp only [← c5964, ← c5965, ← c5966] at e_89_11
  have hr := e_89_11
  simp only [e_89_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 55) = bselect (a (.wire 84 27)) (a (.virt 1003)) (a (.wire 88 71)) := by
  have c5967 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5968 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5969 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5970 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5971 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5972 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_12 := arithEq_of_rows h (row := 89) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_12
  simp only [← c5967, ← c5968, ← c5969] at e_89_12
  have e_89_13 := arithEq_of_rows h (row := 89) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_13
  simp only [← c5970, ← c5971, ← c5972] at e_89_13
  have hr := e_89_13
  simp only [e_89_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 63) = bselect (a (.wire 84 27)) (a (.virt 1004)) (a (.wire 88 79)) := by
  have c5973 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5974 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5975 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5976 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5977 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5978 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_14 := arithEq_of_rows h (row := 89) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_14
  simp only [← c5973, ← c5974, ← c5975] at e_89_14
  have e_89_15 := arithEq_of_rows h (row := 89) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_15
  simp only [← c5976, ← c5977, ← c5978] at e_89_15
  have hr := e_89_15
  simp only [e_89_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 51) = bor (a (.wire 1 47)) (a (.wire 89 3)) := by
  have c5979 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5980 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5981 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5982 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5983 := (publicBatchWrapper86_copies186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5984 := (publicBatchWrapper86_copies187 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_12 := arithEq_of_rows h (row := 1) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_12
  simp only [← c5982, ← c5983, k3, ← c5984] at e_1_12
  have e_86_4 := arithEq_of_rows h (row := 86) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_4
  simp only [← c5979, ← c5980, ← c5981] at e_86_4
  have hr := e_1_12
  simp only [e_86_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 67) = bnot (a (.wire 8 51)) := by
  have c5985 := (publicBatchWrapper86_copies187 a h).2.1
  have c5986 := (publicBatchWrapper86_copies187 a h).2.2.1
  have c5987 := (publicBatchWrapper86_copies187 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_89_16 := arithEq_of_rows h (row := 89) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_16
  simp only [← c5985, k3, ← c5986, k3, ← c5987] at e_89_16
  have hr := e_89_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 71) = bnot (a (.wire 1 51)) := by
  have c5988 := (publicBatchWrapper86_copies187 a h).2.2.2.2.1
  have c5989 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.1
  have c5990 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_89_17 := arithEq_of_rows h (row := 89) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_17
  simp only [← c5988, k3, ← c5989, k3, ← c5990] at e_89_17
  have hr := e_89_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 31) = band (a (.wire 89 67)) (a (.wire 89 71)) := by
  have c5991 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.1
  have c5992 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.1
  have c5993 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.1
  have e_84_7 := arithEq_of_rows h (row := 84) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_7
  simp only [← c5991, ← c5992, ← c5993] at e_84_7
  have hr := e_84_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 89 79) = bselect (a (.wire 84 31)) (a (.virt 1205)) (a (.wire 89 15)) := by
  have c5994 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5995 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5996 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5997 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5998 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5999 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_89_18 := arithEq_of_rows h (row := 89) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_18
  simp only [← c5994, ← c5995, ← c5996] at e_89_18
  have e_89_19 := arithEq_of_rows h (row := 89) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_89_19
  simp only [← c5997, ← c5998, ← c5999] at e_89_19
  have hr := e_89_19
  simp only [e_89_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 7) = bselect (a (.wire 84 31)) (a (.virt 1206)) (a (.wire 89 23)) := by
  have c6000 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6001 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6002 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6003 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6004 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6005 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_0 := arithEq_of_rows h (row := 90) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_0
  simp only [← c6000, ← c6001, ← c6002] at e_90_0
  have e_90_1 := arithEq_of_rows h (row := 90) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_1
  simp only [← c6003, ← c6004, ← c6005] at e_90_1
  have hr := e_90_1
  simp only [e_90_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 15) = bselect (a (.wire 84 31)) (a (.virt 1207)) (a (.wire 89 31)) := by
  have c6006 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6007 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6008 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6009 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6010 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6011 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_2 := arithEq_of_rows h (row := 90) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_2
  simp only [← c6006, ← c6007, ← c6008] at e_90_2
  have e_90_3 := arithEq_of_rows h (row := 90) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_3
  simp only [← c6009, ← c6010, ← c6011] at e_90_3
  have hr := e_90_3
  simp only [e_90_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 23) = bselect (a (.wire 84 31)) (a (.virt 1208)) (a (.wire 89 39)) := by
  have c6012 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6013 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6014 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6015 := (publicBatchWrapper86_copies187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6016 := (publicBatchWrapper86_copies188 a h).1
  have c6017 := (publicBatchWrapper86_copies188 a h).2.1
  have e_90_4 := arithEq_of_rows h (row := 90) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_4
  simp only [← c6012, ← c6013, ← c6014] at e_90_4
  have e_90_5 := arithEq_of_rows h (row := 90) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_5
  simp only [← c6015, ← c6016, ← c6017] at e_90_5
  have hr := e_90_5
  simp only [e_90_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 31) = bselect (a (.wire 84 31)) (a (.virt 1209)) (a (.wire 89 47)) := by
  have c6018 := (publicBatchWrapper86_copies188 a h).2.2.1
  have c6019 := (publicBatchWrapper86_copies188 a h).2.2.2.1
  have c6020 := (publicBatchWrapper86_copies188 a h).2.2.2.2.1
  have c6021 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.1
  have c6022 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.1
  have c6023 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.1
  have e_90_6 := arithEq_of_rows h (row := 90) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_6
  simp only [← c6018, ← c6019, ← c6020] at e_90_6
  have e_90_7 := arithEq_of_rows h (row := 90) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_7
  simp only [← c6021, ← c6022, ← c6023] at e_90_7
  have hr := e_90_7
  simp only [e_90_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 39) = bselect (a (.wire 84 31)) (a (.virt 1203)) (a (.wire 89 55)) := by
  have c6024 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.1
  have c6025 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.1
  have c6026 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6027 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6028 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6029 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_8 := arithEq_of_rows h (row := 90) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_8
  simp only [← c6024, ← c6025, ← c6026] at e_90_8
  have e_90_9 := arithEq_of_rows h (row := 90) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_9
  simp only [← c6027, ← c6028, ← c6029] at e_90_9
  have hr := e_90_9
  simp only [e_90_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 47) = bselect (a (.wire 84 31)) (a (.virt 1204)) (a (.wire 89 63)) := by
  have c6030 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6031 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6032 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6033 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6034 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6035 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_10 := arithEq_of_rows h (row := 90) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_10
  simp only [← c6030, ← c6031, ← c6032] at e_90_10
  have e_90_11 := arithEq_of_rows h (row := 90) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_11
  simp only [← c6033, ← c6034, ← c6035] at e_90_11
  have hr := e_90_11
  simp only [e_90_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 55) = bor (a (.wire 1 51)) (a (.wire 89 67)) := by
  have c6036 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6037 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6038 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6039 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6040 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6041 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_13 := arithEq_of_rows h (row := 1) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_13
  simp only [← c6039, ← c6040, k3, ← c6041] at e_1_13
  have e_86_5 := arithEq_of_rows h (row := 86) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_5
  simp only [← c6036, ← c6037, ← c6038] at e_86_5
  have hr := e_1_13
  simp only [e_86_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 51) = bnot (a (.wire 10 15)) := by
  have c6042 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6043 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6044 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_90_12 := arithEq_of_rows h (row := 90) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_12
  simp only [← c6042, k3, ← c6043, k3, ← c6044] at e_90_12
  have hr := e_90_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 55) = bnot (a (.wire 1 55)) := by
  have c6045 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6046 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6047 := (publicBatchWrapper86_copies188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_90_13 := arithEq_of_rows h (row := 90) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_13
  simp only [← c6045, k3, ← c6046, k3, ← c6047] at e_90_13
  have hr := e_90_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 35) = band (a (.wire 90 51)) (a (.wire 90 55)) := by
  have c6048 := (publicBatchWrapper86_copies189 a h).1
  have c6049 := (publicBatchWrapper86_copies189 a h).2.1
  have c6050 := (publicBatchWrapper86_copies189 a h).2.2.1
  have e_84_8 := arithEq_of_rows h (row := 84) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_8
  simp only [← c6048, ← c6049, ← c6050] at e_84_8
  have hr := e_84_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 63) = bselect (a (.wire 84 35)) (a (.virt 1405)) (a (.wire 89 79)) := by
  have c6051 := (publicBatchWrapper86_copies189 a h).2.2.2.1
  have c6052 := (publicBatchWrapper86_copies189 a h).2.2.2.2.1
  have c6053 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.1
  have c6054 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.1
  have c6055 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.1
  have c6056 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.1
  have e_90_14 := arithEq_of_rows h (row := 90) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_14
  simp only [← c6051, ← c6052, ← c6053] at e_90_14
  have e_90_15 := arithEq_of_rows h (row := 90) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_15
  simp only [← c6054, ← c6055, ← c6056] at e_90_15
  have hr := e_90_15
  simp only [e_90_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 71) = bselect (a (.wire 84 35)) (a (.virt 1406)) (a (.wire 90 7)) := by
  have c6057 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.1
  have c6058 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6059 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6060 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6061 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6062 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_16 := arithEq_of_rows h (row := 90) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_16
  simp only [← c6057, ← c6058, ← c6059] at e_90_16
  have e_90_17 := arithEq_of_rows h (row := 90) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_17
  simp only [← c6060, ← c6061, ← c6062] at e_90_17
  have hr := e_90_17
  simp only [e_90_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 90 79) = bselect (a (.wire 84 35)) (a (.virt 1407)) (a (.wire 90 15)) := by
  have c6063 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6064 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6065 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6066 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6067 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6068 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_90_18 := arithEq_of_rows h (row := 90) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_18
  simp only [← c6063, ← c6064, ← c6065] at e_90_18
  have e_90_19 := arithEq_of_rows h (row := 90) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_90_19
  simp only [← c6066, ← c6067, ← c6068] at e_90_19
  have hr := e_90_19
  simp only [e_90_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 7) = bselect (a (.wire 84 35)) (a (.virt 1408)) (a (.wire 90 23)) := by
  have c6069 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6070 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6071 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6072 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6073 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6074 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_0 := arithEq_of_rows h (row := 91) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_0
  simp only [← c6069, ← c6070, ← c6071] at e_91_0
  have e_91_1 := arithEq_of_rows h (row := 91) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_1
  simp only [← c6072, ← c6073, ← c6074] at e_91_1
  have hr := e_91_1
  simp only [e_91_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 15) = bselect (a (.wire 84 35)) (a (.virt 1409)) (a (.wire 90 31)) := by
  have c6075 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6076 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6077 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6078 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6079 := (publicBatchWrapper86_copies189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6080 := (publicBatchWrapper86_copies190 a h).1
  have e_91_2 := arithEq_of_rows h (row := 91) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_2
  simp only [← c6075, ← c6076, ← c6077] at e_91_2
  have e_91_3 := arithEq_of_rows h (row := 91) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_3
  simp only [← c6078, ← c6079, ← c6080] at e_91_3
  have hr := e_91_3
  simp only [e_91_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 23) = bselect (a (.wire 84 35)) (a (.virt 1403)) (a (.wire 90 39)) := by
  have c6081 := (publicBatchWrapper86_copies190 a h).2.1
  have c6082 := (publicBatchWrapper86_copies190 a h).2.2.1
  have c6083 := (publicBatchWrapper86_copies190 a h).2.2.2.1
  have c6084 := (publicBatchWrapper86_copies190 a h).2.2.2.2.1
  have c6085 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.1
  have c6086 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.1
  have e_91_4 := arithEq_of_rows h (row := 91) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_4
  simp only [← c6081, ← c6082, ← c6083] at e_91_4
  have e_91_5 := arithEq_of_rows h (row := 91) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_5
  simp only [← c6084, ← c6085, ← c6086] at e_91_5
  have hr := e_91_5
  simp only [e_91_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 31) = bselect (a (.wire 84 35)) (a (.virt 1404)) (a (.wire 90 47)) := by
  have c6087 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.1
  have c6088 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.1
  have c6089 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.1
  have c6090 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6091 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6092 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_6 := arithEq_of_rows h (row := 91) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_6
  simp only [← c6087, ← c6088, ← c6089] at e_91_6
  have e_91_7 := arithEq_of_rows h (row := 91) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_7
  simp only [← c6090, ← c6091, ← c6092] at e_91_7
  have hr := e_91_7
  simp only [e_91_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 59) = bor (a (.wire 1 55)) (a (.wire 90 51)) := by
  have c6093 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6094 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6095 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6096 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6097 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6098 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_14 := arithEq_of_rows h (row := 1) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_14
  simp only [← c6096, ← c6097, k3, ← c6098] at e_1_14
  have e_86_6 := arithEq_of_rows h (row := 86) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_6
  simp only [← c6093, ← c6094, ← c6095] at e_86_6
  have hr := e_1_14
  simp only [e_86_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 35) = bnot (a (.wire 10 59)) := by
  have c6099 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6100 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6101 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_91_8 := arithEq_of_rows h (row := 91) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_8
  simp only [← c6099, k3, ← c6100, k3, ← c6101] at e_91_8
  have hr := e_91_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 39) = bnot (a (.wire 1 59)) := by
  have c6102 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6103 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6104 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_91_9 := arithEq_of_rows h (row := 91) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_9
  simp only [← c6102, k3, ← c6103, k3, ← c6104] at e_91_9
  have hr := e_91_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 39) = band (a (.wire 91 35)) (a (.wire 91 39)) := by
  have c6105 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6106 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6107 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_9 := arithEq_of_rows h (row := 84) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_9
  simp only [← c6105, ← c6106, ← c6107] at e_84_9
  have hr := e_84_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 47) = bselect (a (.wire 84 39)) (a (.virt 1605)) (a (.wire 90 63)) := by
  have c6108 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6109 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6110 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6111 := (publicBatchWrapper86_copies190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6112 := (publicBatchWrapper86_copies191 a h).1
  have c6113 := (publicBatchWrapper86_copies191 a h).2.1
  have e_91_10 := arithEq_of_rows h (row := 91) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_10
  simp only [← c6108, ← c6109, ← c6110] at e_91_10
  have e_91_11 := arithEq_of_rows h (row := 91) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_11
  simp only [← c6111, ← c6112, ← c6113] at e_91_11
  have hr := e_91_11
  simp only [e_91_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 55) = bselect (a (.wire 84 39)) (a (.virt 1606)) (a (.wire 90 71)) := by
  have c6114 := (publicBatchWrapper86_copies191 a h).2.2.1
  have c6115 := (publicBatchWrapper86_copies191 a h).2.2.2.1
  have c6116 := (publicBatchWrapper86_copies191 a h).2.2.2.2.1
  have c6117 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.1
  have c6118 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.1
  have c6119 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.1
  have e_91_12 := arithEq_of_rows h (row := 91) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_12
  simp only [← c6114, ← c6115, ← c6116] at e_91_12
  have e_91_13 := arithEq_of_rows h (row := 91) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_13
  simp only [← c6117, ← c6118, ← c6119] at e_91_13
  have hr := e_91_13
  simp only [e_91_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 63) = bselect (a (.wire 84 39)) (a (.virt 1607)) (a (.wire 90 79)) := by
  have c6120 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.1
  have c6121 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.1
  have c6122 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6123 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6124 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6125 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_14 := arithEq_of_rows h (row := 91) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_14
  simp only [← c6120, ← c6121, ← c6122] at e_91_14
  have e_91_15 := arithEq_of_rows h (row := 91) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_15
  simp only [← c6123, ← c6124, ← c6125] at e_91_15
  have hr := e_91_15
  simp only [e_91_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 71) = bselect (a (.wire 84 39)) (a (.virt 1608)) (a (.wire 91 7)) := by
  have c6126 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6127 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6128 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6129 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6130 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6131 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_16 := arithEq_of_rows h (row := 91) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_16
  simp only [← c6126, ← c6127, ← c6128] at e_91_16
  have e_91_17 := arithEq_of_rows h (row := 91) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_17
  simp only [← c6129, ← c6130, ← c6131] at e_91_17
  have hr := e_91_17
  simp only [e_91_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 91 79) = bselect (a (.wire 84 39)) (a (.virt 1609)) (a (.wire 91 15)) := by
  have c6132 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6133 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6134 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6135 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6136 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6137 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_91_18 := arithEq_of_rows h (row := 91) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_18
  simp only [← c6132, ← c6133, ← c6134] at e_91_18
  have e_91_19 := arithEq_of_rows h (row := 91) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_91_19
  simp only [← c6135, ← c6136, ← c6137] at e_91_19
  have hr := e_91_19
  simp only [e_91_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 7) = bselect (a (.wire 84 39)) (a (.virt 1603)) (a (.wire 91 23)) := by
  have c6138 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6139 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6140 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6141 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6142 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6143 := (publicBatchWrapper86_copies191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_92_0 := arithEq_of_rows h (row := 92) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_0
  simp only [← c6138, ← c6139, ← c6140] at e_92_0
  have e_92_1 := arithEq_of_rows h (row := 92) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_1
  simp only [← c6141, ← c6142, ← c6143] at e_92_1
  have hr := e_92_1
  simp only [e_92_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 15) = bselect (a (.wire 84 39)) (a (.virt 1604)) (a (.wire 91 31)) := by
  have c6144 := (publicBatchWrapper86_copies192 a h).1
  have c6145 := (publicBatchWrapper86_copies192 a h).2.1
  have c6146 := (publicBatchWrapper86_copies192 a h).2.2.1
  have c6147 := (publicBatchWrapper86_copies192 a h).2.2.2.1
  have c6148 := (publicBatchWrapper86_copies192 a h).2.2.2.2.1
  have c6149 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.1
  have e_92_2 := arithEq_of_rows h (row := 92) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_2
  simp only [← c6144, ← c6145, ← c6146] at e_92_2
  have e_92_3 := arithEq_of_rows h (row := 92) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_3
  simp only [← c6147, ← c6148, ← c6149] at e_92_3
  have hr := e_92_3
  simp only [e_92_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 63) = bor (a (.wire 1 59)) (a (.wire 91 35)) := by
  have c6150 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.1
  have c6151 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.1
  have c6152 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.1
  have c6153 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.1
  have c6154 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6155 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_15 := arithEq_of_rows h (row := 1) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_15
  simp only [← c6153, ← c6154, k3, ← c6155] at e_1_15
  have e_86_7 := arithEq_of_rows h (row := 86) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_7
  simp only [← c6150, ← c6151, ← c6152] at e_86_7
  have hr := e_1_15
  simp only [e_86_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 19) = bnot (a (.wire 11 23)) := by
  have c6156 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6157 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6158 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_92_4 := arithEq_of_rows h (row := 92) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_4
  simp only [← c6156, k3, ← c6157, k3, ← c6158] at e_92_4
  have hr := e_92_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 23) = bnot (a (.wire 1 63)) := by
  have c6159 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6160 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6161 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_92_5 := arithEq_of_rows h (row := 92) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_5
  simp only [← c6159, k3, ← c6160, k3, ← c6161] at e_92_5
  have hr := e_92_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 43) = band (a (.wire 92 19)) (a (.wire 92 23)) := by
  have c6162 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6163 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6164 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_10 := arithEq_of_rows h (row := 84) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_10
  simp only [← c6162, ← c6163, ← c6164] at e_84_10
  have hr := e_84_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 31) = bselect (a (.wire 84 43)) (a (.virt 1805)) (a (.wire 91 47)) := by
  have c6165 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6166 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6167 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6168 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6169 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6170 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_6 := arithEq_of_rows h (row := 92) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_6
  simp only [← c6165, ← c6166, ← c6167] at e_92_6
  have e_92_7 := arithEq_of_rows h (row := 92) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_7
  simp only [← c6168, ← c6169, ← c6170] at e_92_7
  have hr := e_92_7
  simp only [e_92_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 39) = bselect (a (.wire 84 43)) (a (.virt 1806)) (a (.wire 91 55)) := by
  have c6171 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6172 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6173 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6174 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6175 := (publicBatchWrapper86_copies192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6176 := (publicBatchWrapper86_copies193 a h).1
  have e_92_8 := arithEq_of_rows h (row := 92) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_8
  simp only [← c6171, ← c6172, ← c6173] at e_92_8
  have e_92_9 := arithEq_of_rows h (row := 92) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_9
  simp only [← c6174, ← c6175, ← c6176] at e_92_9
  have hr := e_92_9
  simp only [e_92_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 47) = bselect (a (.wire 84 43)) (a (.virt 1807)) (a (.wire 91 63)) := by
  have c6177 := (publicBatchWrapper86_copies193 a h).2.1
  have c6178 := (publicBatchWrapper86_copies193 a h).2.2.1
  have c6179 := (publicBatchWrapper86_copies193 a h).2.2.2.1
  have c6180 := (publicBatchWrapper86_copies193 a h).2.2.2.2.1
  have c6181 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.1
  have c6182 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.1
  have e_92_10 := arithEq_of_rows h (row := 92) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_10
  simp only [← c6177, ← c6178, ← c6179] at e_92_10
  have e_92_11 := arithEq_of_rows h (row := 92) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_11
  simp only [← c6180, ← c6181, ← c6182] at e_92_11
  have hr := e_92_11
  simp only [e_92_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 55) = bselect (a (.wire 84 43)) (a (.virt 1808)) (a (.wire 91 71)) := by
  have c6183 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.1
  have c6184 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.1
  have c6185 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.1
  have c6186 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6187 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6188 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_12 := arithEq_of_rows h (row := 92) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_12
  simp only [← c6183, ← c6184, ← c6185] at e_92_12
  have e_92_13 := arithEq_of_rows h (row := 92) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_13
  simp only [← c6186, ← c6187, ← c6188] at e_92_13
  have hr := e_92_13
  simp only [e_92_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 63) = bselect (a (.wire 84 43)) (a (.virt 1809)) (a (.wire 91 79)) := by
  have c6189 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6190 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6191 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6192 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6193 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6194 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_14 := arithEq_of_rows h (row := 92) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_14
  simp only [← c6189, ← c6190, ← c6191] at e_92_14
  have e_92_15 := arithEq_of_rows h (row := 92) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_15
  simp only [← c6192, ← c6193, ← c6194] at e_92_15
  have hr := e_92_15
  simp only [e_92_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 71) = bselect (a (.wire 84 43)) (a (.virt 1803)) (a (.wire 92 7)) := by
  have c6195 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6196 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6197 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6198 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6199 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6200 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_16 := arithEq_of_rows h (row := 92) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_16
  simp only [← c6195, ← c6196, ← c6197] at e_92_16
  have e_92_17 := arithEq_of_rows h (row := 92) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_17
  simp only [← c6198, ← c6199, ← c6200] at e_92_17
  have hr := e_92_17
  simp only [e_92_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 92 79) = bselect (a (.wire 84 43)) (a (.virt 1804)) (a (.wire 92 15)) := by
  have c6201 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6202 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6203 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6204 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6205 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6206 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_92_18 := arithEq_of_rows h (row := 92) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_18
  simp only [← c6201, ← c6202, ← c6203] at e_92_18
  have e_92_19 := arithEq_of_rows h (row := 92) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_92_19
  simp only [← c6204, ← c6205, ← c6206] at e_92_19
  have hr := e_92_19
  simp only [e_92_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 67) = bor (a (.wire 1 63)) (a (.wire 92 19)) := by
  have c6207 := (publicBatchWrapper86_copies193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6208 := (publicBatchWrapper86_copies194 a h).1
  have c6209 := (publicBatchWrapper86_copies194 a h).2.1
  have c6210 := (publicBatchWrapper86_copies194 a h).2.2.1
  have c6211 := (publicBatchWrapper86_copies194 a h).2.2.2.1
  have c6212 := (publicBatchWrapper86_copies194 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_16 := arithEq_of_rows h (row := 1) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_16
  simp only [← c6210, ← c6211, k3, ← c6212] at e_1_16
  have e_86_8 := arithEq_of_rows h (row := 86) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_8
  simp only [← c6207, ← c6208, ← c6209] at e_86_8
  have hr := e_1_16
  simp only [e_86_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 3) = bnot (a (.wire 11 67)) := by
  have c6213 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.1
  have c6214 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.1
  have c6215 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_93_0 := arithEq_of_rows h (row := 93) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_0
  simp only [← c6213, k3, ← c6214, k3, ← c6215] at e_93_0
  have hr := e_93_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 7) = bnot (a (.wire 1 67)) := by
  have c6216 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.1
  have c6217 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.1
  have c6218 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_93_1 := arithEq_of_rows h (row := 93) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_1
  simp only [← c6216, k3, ← c6217, k3, ← c6218] at e_93_1
  have hr := e_93_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 47) = band (a (.wire 93 3)) (a (.wire 93 7)) := by
  have c6219 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6220 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6221 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_11 := arithEq_of_rows h (row := 84) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_11
  simp only [← c6219, ← c6220, ← c6221] at e_84_11
  have hr := e_84_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 15) = bselect (a (.wire 84 47)) (a (.virt 2005)) (a (.wire 92 31)) := by
  have c6222 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6223 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6224 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6225 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6226 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6227 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_2 := arithEq_of_rows h (row := 93) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_2
  simp only [← c6222, ← c6223, ← c6224] at e_93_2
  have e_93_3 := arithEq_of_rows h (row := 93) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_3
  simp only [← c6225, ← c6226, ← c6227] at e_93_3
  have hr := e_93_3
  simp only [e_93_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 23) = bselect (a (.wire 84 47)) (a (.virt 2006)) (a (.wire 92 39)) := by
  have c6228 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6229 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6230 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6231 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6232 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6233 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_4 := arithEq_of_rows h (row := 93) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_4
  simp only [← c6228, ← c6229, ← c6230] at e_93_4
  have e_93_5 := arithEq_of_rows h (row := 93) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_5
  simp only [← c6231, ← c6232, ← c6233] at e_93_5
  have hr := e_93_5
  simp only [e_93_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 31) = bselect (a (.wire 84 47)) (a (.virt 2007)) (a (.wire 92 47)) := by
  have c6234 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6235 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6236 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6237 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6238 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6239 := (publicBatchWrapper86_copies194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_93_6 := arithEq_of_rows h (row := 93) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_6
  simp only [← c6234, ← c6235, ← c6236] at e_93_6
  have e_93_7 := arithEq_of_rows h (row := 93) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_7
  simp only [← c6237, ← c6238, ← c6239] at e_93_7
  have hr := e_93_7
  simp only [e_93_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 39) = bselect (a (.wire 84 47)) (a (.virt 2008)) (a (.wire 92 55)) := by
  have c6240 := (publicBatchWrapper86_copies195 a h).1
  have c6241 := (publicBatchWrapper86_copies195 a h).2.1
  have c6242 := (publicBatchWrapper86_copies195 a h).2.2.1
  have c6243 := (publicBatchWrapper86_copies195 a h).2.2.2.1
  have c6244 := (publicBatchWrapper86_copies195 a h).2.2.2.2.1
  have c6245 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.1
  have e_93_8 := arithEq_of_rows h (row := 93) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_8
  simp only [← c6240, ← c6241, ← c6242] at e_93_8
  have e_93_9 := arithEq_of_rows h (row := 93) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_9
  simp only [← c6243, ← c6244, ← c6245] at e_93_9
  have hr := e_93_9
  simp only [e_93_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 47) = bselect (a (.wire 84 47)) (a (.virt 2009)) (a (.wire 92 63)) := by
  have c6246 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.1
  have c6247 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.1
  have c6248 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.1
  have c6249 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.1
  have c6250 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6251 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_10 := arithEq_of_rows h (row := 93) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_10
  simp only [← c6246, ← c6247, ← c6248] at e_93_10
  have e_93_11 := arithEq_of_rows h (row := 93) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_11
  simp only [← c6249, ← c6250, ← c6251] at e_93_11
  have hr := e_93_11
  simp only [e_93_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 55) = bselect (a (.wire 84 47)) (a (.virt 2003)) (a (.wire 92 71)) := by
  have c6252 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6253 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6254 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6255 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6256 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6257 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_12 := arithEq_of_rows h (row := 93) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_12
  simp only [← c6252, ← c6253, ← c6254] at e_93_12
  have e_93_13 := arithEq_of_rows h (row := 93) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_13
  simp only [← c6255, ← c6256, ← c6257] at e_93_13
  have hr := e_93_13
  simp only [e_93_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 63) = bselect (a (.wire 84 47)) (a (.virt 2004)) (a (.wire 92 79)) := by
  have c6258 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6259 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6260 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6261 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6262 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6263 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_14 := arithEq_of_rows h (row := 93) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_14
  simp only [← c6258, ← c6259, ← c6260] at e_93_14
  have e_93_15 := arithEq_of_rows h (row := 93) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_15
  simp only [← c6261, ← c6262, ← c6263] at e_93_15
  have hr := e_93_15
  simp only [e_93_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 71) = bor (a (.wire 1 67)) (a (.wire 93 3)) := by
  have c6264 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6265 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6266 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6267 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6268 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6269 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_17 := arithEq_of_rows h (row := 1) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_17
  simp only [← c6267, ← c6268, k3, ← c6269] at e_1_17
  have e_86_9 := arithEq_of_rows h (row := 86) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_9
  simp only [← c6264, ← c6265, ← c6266] at e_86_9
  have hr := e_1_17
  simp only [e_86_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 67) = bnot (a (.wire 13 31)) := by
  have c6270 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6271 := (publicBatchWrapper86_copies195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6272 := (publicBatchWrapper86_copies196 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_93_16 := arithEq_of_rows h (row := 93) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_16
  simp only [← c6270, k3, ← c6271, k3, ← c6272] at e_93_16
  have hr := e_93_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 71) = bnot (a (.wire 1 71)) := by
  have c6273 := (publicBatchWrapper86_copies196 a h).2.1
  have c6274 := (publicBatchWrapper86_copies196 a h).2.2.1
  have c6275 := (publicBatchWrapper86_copies196 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_93_17 := arithEq_of_rows h (row := 93) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_17
  simp only [← c6273, k3, ← c6274, k3, ← c6275] at e_93_17
  have hr := e_93_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 51) = band (a (.wire 93 67)) (a (.wire 93 71)) := by
  have c6276 := (publicBatchWrapper86_copies196 a h).2.2.2.2.1
  have c6277 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.1
  have c6278 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.1
  have e_84_12 := arithEq_of_rows h (row := 84) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_12
  simp only [← c6276, ← c6277, ← c6278] at e_84_12
  have hr := e_84_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 93 79) = bselect (a (.wire 84 51)) (a (.virt 2205)) (a (.wire 93 15)) := by
  have c6279 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.1
  have c6280 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.1
  have c6281 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.1
  have c6282 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6283 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6284 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_93_18 := arithEq_of_rows h (row := 93) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_18
  simp only [← c6279, ← c6280, ← c6281] at e_93_18
  have e_93_19 := arithEq_of_rows h (row := 93) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_93_19
  simp only [← c6282, ← c6283, ← c6284] at e_93_19
  have hr := e_93_19
  simp only [e_93_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 7) = bselect (a (.wire 84 51)) (a (.virt 2206)) (a (.wire 93 23)) := by
  have c6285 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6286 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6287 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6288 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6289 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6290 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_0 := arithEq_of_rows h (row := 94) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_0
  simp only [← c6285, ← c6286, ← c6287] at e_94_0
  have e_94_1 := arithEq_of_rows h (row := 94) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_1
  simp only [← c6288, ← c6289, ← c6290] at e_94_1
  have hr := e_94_1
  simp only [e_94_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 15) = bselect (a (.wire 84 51)) (a (.virt 2207)) (a (.wire 93 31)) := by
  have c6291 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6292 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6293 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6294 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6295 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6296 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_2 := arithEq_of_rows h (row := 94) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_2
  simp only [← c6291, ← c6292, ← c6293] at e_94_2
  have e_94_3 := arithEq_of_rows h (row := 94) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_3
  simp only [← c6294, ← c6295, ← c6296] at e_94_3
  have hr := e_94_3
  simp only [e_94_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 23) = bselect (a (.wire 84 51)) (a (.virt 2208)) (a (.wire 93 39)) := by
  have c6297 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6298 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6299 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6300 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6301 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6302 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_4 := arithEq_of_rows h (row := 94) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_4
  simp only [← c6297, ← c6298, ← c6299] at e_94_4
  have e_94_5 := arithEq_of_rows h (row := 94) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_5
  simp only [← c6300, ← c6301, ← c6302] at e_94_5
  have hr := e_94_5
  simp only [e_94_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 31) = bselect (a (.wire 84 51)) (a (.virt 2209)) (a (.wire 93 47)) := by
  have c6303 := (publicBatchWrapper86_copies196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6304 := (publicBatchWrapper86_copies197 a h).1
  have c6305 := (publicBatchWrapper86_copies197 a h).2.1
  have c6306 := (publicBatchWrapper86_copies197 a h).2.2.1
  have c6307 := (publicBatchWrapper86_copies197 a h).2.2.2.1
  have c6308 := (publicBatchWrapper86_copies197 a h).2.2.2.2.1
  have e_94_6 := arithEq_of_rows h (row := 94) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_6
  simp only [← c6303, ← c6304, ← c6305] at e_94_6
  have e_94_7 := arithEq_of_rows h (row := 94) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_7
  simp only [← c6306, ← c6307, ← c6308] at e_94_7
  have hr := e_94_7
  simp only [e_94_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 39) = bselect (a (.wire 84 51)) (a (.virt 2203)) (a (.wire 93 55)) := by
  have c6309 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.1
  have c6310 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.1
  have c6311 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.1
  have c6312 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.1
  have c6313 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.1
  have c6314 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_94_8 := arithEq_of_rows h (row := 94) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_8
  simp only [← c6309, ← c6310, ← c6311] at e_94_8
  have e_94_9 := arithEq_of_rows h (row := 94) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_9
  simp only [← c6312, ← c6313, ← c6314] at e_94_9
  have hr := e_94_9
  simp only [e_94_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 47) = bselect (a (.wire 84 51)) (a (.virt 2204)) (a (.wire 93 63)) := by
  have c6315 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6316 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6317 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6318 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6319 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6320 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_10 := arithEq_of_rows h (row := 94) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_10
  simp only [← c6315, ← c6316, ← c6317] at e_94_10
  have e_94_11 := arithEq_of_rows h (row := 94) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_11
  simp only [← c6318, ← c6319, ← c6320] at e_94_11
  have hr := e_94_11
  simp only [e_94_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 75) = bor (a (.wire 1 71)) (a (.wire 93 67)) := by
  have c6321 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6322 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6323 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6324 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6325 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6326 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_18 := arithEq_of_rows h (row := 1) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_18
  simp only [← c6324, ← c6325, k3, ← c6326] at e_1_18
  have e_86_10 := arithEq_of_rows h (row := 86) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_10
  simp only [← c6321, ← c6322, ← c6323] at e_86_10
  have hr := e_1_18
  simp only [e_86_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 51) = bnot (a (.wire 13 75)) := by
  have c6327 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6328 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6329 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_94_12 := arithEq_of_rows h (row := 94) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_12
  simp only [← c6327, k3, ← c6328, k3, ← c6329] at e_94_12
  have hr := e_94_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 55) = bnot (a (.wire 1 75)) := by
  have c6330 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6331 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6332 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_94_13 := arithEq_of_rows h (row := 94) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_13
  simp only [← c6330, k3, ← c6331, k3, ← c6332] at e_94_13
  have hr := e_94_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 55) = band (a (.wire 94 51)) (a (.wire 94 55)) := by
  have c6333 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6334 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6335 := (publicBatchWrapper86_copies197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_84_13 := arithEq_of_rows h (row := 84) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_13
  simp only [← c6333, ← c6334, ← c6335] at e_84_13
  have hr := e_84_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 63) = bselect (a (.wire 84 55)) (a (.virt 2405)) (a (.wire 93 79)) := by
  have c6336 := (publicBatchWrapper86_copies198 a h).1
  have c6337 := (publicBatchWrapper86_copies198 a h).2.1
  have c6338 := (publicBatchWrapper86_copies198 a h).2.2.1
  have c6339 := (publicBatchWrapper86_copies198 a h).2.2.2.1
  have c6340 := (publicBatchWrapper86_copies198 a h).2.2.2.2.1
  have c6341 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.1
  have e_94_14 := arithEq_of_rows h (row := 94) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_14
  simp only [← c6336, ← c6337, ← c6338] at e_94_14
  have e_94_15 := arithEq_of_rows h (row := 94) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_15
  simp only [← c6339, ← c6340, ← c6341] at e_94_15
  have hr := e_94_15
  simp only [e_94_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 71) = bselect (a (.wire 84 55)) (a (.virt 2406)) (a (.wire 94 7)) := by
  have c6342 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.1
  have c6343 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.1
  have c6344 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.1
  have c6345 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.1
  have c6346 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6347 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_16 := arithEq_of_rows h (row := 94) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_16
  simp only [← c6342, ← c6343, ← c6344] at e_94_16
  have e_94_17 := arithEq_of_rows h (row := 94) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_17
  simp only [← c6345, ← c6346, ← c6347] at e_94_17
  have hr := e_94_17
  simp only [e_94_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 94 79) = bselect (a (.wire 84 55)) (a (.virt 2407)) (a (.wire 94 15)) := by
  have c6348 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6349 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6350 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6351 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6352 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6353 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_94_18 := arithEq_of_rows h (row := 94) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_18
  simp only [← c6348, ← c6349, ← c6350] at e_94_18
  have e_94_19 := arithEq_of_rows h (row := 94) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_94_19
  simp only [← c6351, ← c6352, ← c6353] at e_94_19
  have hr := e_94_19
  simp only [e_94_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 7) = bselect (a (.wire 84 55)) (a (.virt 2408)) (a (.wire 94 23)) := by
  have c6354 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6355 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6356 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6357 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6358 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6359 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_0 := arithEq_of_rows h (row := 95) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_0
  simp only [← c6354, ← c6355, ← c6356] at e_95_0
  have e_95_1 := arithEq_of_rows h (row := 95) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_1
  simp only [← c6357, ← c6358, ← c6359] at e_95_1
  have hr := e_95_1
  simp only [e_95_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 15) = bselect (a (.wire 84 55)) (a (.virt 2409)) (a (.wire 94 31)) := by
  have c6360 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6361 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6362 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6363 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6364 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6365 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_2 := arithEq_of_rows h (row := 95) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_2
  simp only [← c6360, ← c6361, ← c6362] at e_95_2
  have e_95_3 := arithEq_of_rows h (row := 95) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_3
  simp only [← c6363, ← c6364, ← c6365] at e_95_3
  have hr := e_95_3
  simp only [e_95_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 23) = bselect (a (.wire 84 55)) (a (.virt 2403)) (a (.wire 94 39)) := by
  have c6366 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6367 := (publicBatchWrapper86_copies198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6368 := (publicBatchWrapper86_copies199 a h).1
  have c6369 := (publicBatchWrapper86_copies199 a h).2.1
  have c6370 := (publicBatchWrapper86_copies199 a h).2.2.1
  have c6371 := (publicBatchWrapper86_copies199 a h).2.2.2.1
  have e_95_4 := arithEq_of_rows h (row := 95) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_4
  simp only [← c6366, ← c6367, ← c6368] at e_95_4
  have e_95_5 := arithEq_of_rows h (row := 95) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_5
  simp only [← c6369, ← c6370, ← c6371] at e_95_5
  have hr := e_95_5
  simp only [e_95_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 31) = bselect (a (.wire 84 55)) (a (.virt 2404)) (a (.wire 94 47)) := by
  have c6372 := (publicBatchWrapper86_copies199 a h).2.2.2.2.1
  have c6373 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.1
  have c6374 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.1
  have c6375 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.1
  have c6376 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.1
  have c6377 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.1
  have e_95_6 := arithEq_of_rows h (row := 95) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_6
  simp only [← c6372, ← c6373, ← c6374] at e_95_6
  have e_95_7 := arithEq_of_rows h (row := 95) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_7
  simp only [← c6375, ← c6376, ← c6377] at e_95_7
  have hr := e_95_7
  simp only [e_95_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1 79) = bor (a (.wire 1 75)) (a (.wire 94 51)) := by
  have c6378 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6379 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6380 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6381 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6382 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6383 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1_19 := arithEq_of_rows h (row := 1) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_19
  simp only [← c6381, ← c6382, k3, ← c6383] at e_1_19
  have e_86_11 := arithEq_of_rows h (row := 86) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_11
  simp only [← c6378, ← c6379, ← c6380] at e_86_11
  have hr := e_1_19
  simp only [e_86_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 35) = bnot (a (.wire 15 39)) := by
  have c6384 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6385 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6386 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_95_8 := arithEq_of_rows h (row := 95) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_8
  simp only [← c6384, k3, ← c6385, k3, ← c6386] at e_95_8
  have hr := e_95_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 39) = bnot (a (.wire 1 79)) := by
  have c6387 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6388 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6389 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_95_9 := arithEq_of_rows h (row := 95) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_9
  simp only [← c6387, k3, ← c6388, k3, ← c6389] at e_95_9
  have hr := e_95_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 59) = band (a (.wire 95 35)) (a (.wire 95 39)) := by
  have c6390 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6391 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6392 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_14 := arithEq_of_rows h (row := 84) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_14
  simp only [← c6390, ← c6391, ← c6392] at e_84_14
  have hr := e_84_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 47) = bselect (a (.wire 84 59)) (a (.virt 2605)) (a (.wire 94 63)) := by
  have c6393 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6394 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6395 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6396 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6397 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6398 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_10 := arithEq_of_rows h (row := 95) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_10
  simp only [← c6393, ← c6394, ← c6395] at e_95_10
  have e_95_11 := arithEq_of_rows h (row := 95) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_11
  simp only [← c6396, ← c6397, ← c6398] at e_95_11
  have hr := e_95_11
  simp only [e_95_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 55) = bselect (a (.wire 84 59)) (a (.virt 2606)) (a (.wire 94 71)) := by
  have c6399 := (publicBatchWrapper86_copies199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6400 := (publicBatchWrapper86_copies200 a h).1
  have c6401 := (publicBatchWrapper86_copies200 a h).2.1
  have c6402 := (publicBatchWrapper86_copies200 a h).2.2.1
  have c6403 := (publicBatchWrapper86_copies200 a h).2.2.2.1
  have c6404 := (publicBatchWrapper86_copies200 a h).2.2.2.2.1
  have e_95_12 := arithEq_of_rows h (row := 95) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_12
  simp only [← c6399, ← c6400, ← c6401] at e_95_12
  have e_95_13 := arithEq_of_rows h (row := 95) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_13
  simp only [← c6402, ← c6403, ← c6404] at e_95_13
  have hr := e_95_13
  simp only [e_95_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 63) = bselect (a (.wire 84 59)) (a (.virt 2607)) (a (.wire 94 79)) := by
  have c6405 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.1
  have c6406 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.1
  have c6407 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.1
  have c6408 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.1
  have c6409 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.1
  have c6410 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_95_14 := arithEq_of_rows h (row := 95) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_14
  simp only [← c6405, ← c6406, ← c6407] at e_95_14
  have e_95_15 := arithEq_of_rows h (row := 95) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_15
  simp only [← c6408, ← c6409, ← c6410] at e_95_15
  have hr := e_95_15
  simp only [e_95_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 71) = bselect (a (.wire 84 59)) (a (.virt 2608)) (a (.wire 95 7)) := by
  have c6411 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6412 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6413 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6414 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6415 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6416 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_16 := arithEq_of_rows h (row := 95) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_16
  simp only [← c6411, ← c6412, ← c6413] at e_95_16
  have e_95_17 := arithEq_of_rows h (row := 95) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_17
  simp only [← c6414, ← c6415, ← c6416] at e_95_17
  have hr := e_95_17
  simp only [e_95_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 95 79) = bselect (a (.wire 84 59)) (a (.virt 2609)) (a (.wire 95 15)) := by
  have c6417 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6418 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6419 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6420 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6421 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6422 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_95_18 := arithEq_of_rows h (row := 95) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_18
  simp only [← c6417, ← c6418, ← c6419] at e_95_18
  have e_95_19 := arithEq_of_rows h (row := 95) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_95_19
  simp only [← c6420, ← c6421, ← c6422] at e_95_19
  have hr := e_95_19
  simp only [e_95_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 7) = bselect (a (.wire 84 59)) (a (.virt 2603)) (a (.wire 95 23)) := by
  have c6423 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6424 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6425 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6426 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6427 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6428 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_96_0 := arithEq_of_rows h (row := 96) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_0
  simp only [← c6423, ← c6424, ← c6425] at e_96_0
  have e_96_1 := arithEq_of_rows h (row := 96) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_1
  simp only [← c6426, ← c6427, ← c6428] at e_96_1
  have hr := e_96_1
  simp only [e_96_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 15) = bselect (a (.wire 84 59)) (a (.virt 2604)) (a (.wire 95 31)) := by
  have c6429 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6430 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6431 := (publicBatchWrapper86_copies200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6432 := (publicBatchWrapper86_copies201 a h).1
  have c6433 := (publicBatchWrapper86_copies201 a h).2.1
  have c6434 := (publicBatchWrapper86_copies201 a h).2.2.1
  have e_96_2 := arithEq_of_rows h (row := 96) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_2
  simp only [← c6429, ← c6430, ← c6431] at e_96_2
  have e_96_3 := arithEq_of_rows h (row := 96) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_3
  simp only [← c6432, ← c6433, ← c6434] at e_96_3
  have hr := e_96_3
  simp only [e_96_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 3) = bor (a (.wire 1 79)) (a (.wire 95 35)) := by
  have c6435 := (publicBatchWrapper86_copies201 a h).2.2.2.1
  have c6436 := (publicBatchWrapper86_copies201 a h).2.2.2.2.1
  have c6437 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.1
  have c6438 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.1
  have c6439 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.1
  have c6440 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_86_12 := arithEq_of_rows h (row := 86) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_12
  simp only [← c6435, ← c6436, ← c6437] at e_86_12
  have e_97_0 := arithEq_of_rows h (row := 97) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_0
  simp only [← c6438, ← c6439, k3, ← c6440] at e_97_0
  have hr := e_97_0
  simp only [e_86_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 19) = bnot (a (.wire 16 3)) := by
  have c6441 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.1
  have c6442 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6443 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_96_4 := arithEq_of_rows h (row := 96) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_4
  simp only [← c6441, k3, ← c6442, k3, ← c6443] at e_96_4
  have hr := e_96_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 23) = bnot (a (.wire 97 3)) := by
  have c6444 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6445 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6446 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_96_5 := arithEq_of_rows h (row := 96) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_5
  simp only [← c6444, k3, ← c6445, k3, ← c6446] at e_96_5
  have hr := e_96_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 63) = band (a (.wire 96 19)) (a (.wire 96 23)) := by
  have c6447 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6448 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6449 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_15 := arithEq_of_rows h (row := 84) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_15
  simp only [← c6447, ← c6448, ← c6449] at e_84_15
  have hr := e_84_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 31) = bselect (a (.wire 84 63)) (a (.virt 2805)) (a (.wire 95 47)) := by
  have c6450 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6451 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6452 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6453 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6454 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6455 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_96_6 := arithEq_of_rows h (row := 96) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_6
  simp only [← c6450, ← c6451, ← c6452] at e_96_6
  have e_96_7 := arithEq_of_rows h (row := 96) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_7
  simp only [← c6453, ← c6454, ← c6455] at e_96_7
  have hr := e_96_7
  simp only [e_96_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 39) = bselect (a (.wire 84 63)) (a (.virt 2806)) (a (.wire 95 55)) := by
  have c6456 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6457 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6458 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6459 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6460 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6461 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_96_8 := arithEq_of_rows h (row := 96) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_8
  simp only [← c6456, ← c6457, ← c6458] at e_96_8
  have e_96_9 := arithEq_of_rows h (row := 96) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_9
  simp only [← c6459, ← c6460, ← c6461] at e_96_9
  have hr := e_96_9
  simp only [e_96_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 47) = bselect (a (.wire 84 63)) (a (.virt 2807)) (a (.wire 95 63)) := by
  have c6462 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6463 := (publicBatchWrapper86_copies201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6464 := (publicBatchWrapper86_copies202 a h).1
  have c6465 := (publicBatchWrapper86_copies202 a h).2.1
  have c6466 := (publicBatchWrapper86_copies202 a h).2.2.1
  have c6467 := (publicBatchWrapper86_copies202 a h).2.2.2.1
  have e_96_10 := arithEq_of_rows h (row := 96) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_10
  simp only [← c6462, ← c6463, ← c6464] at e_96_10
  have e_96_11 := arithEq_of_rows h (row := 96) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_11
  simp only [← c6465, ← c6466, ← c6467] at e_96_11
  have hr := e_96_11
  simp only [e_96_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 55) = bselect (a (.wire 84 63)) (a (.virt 2808)) (a (.wire 95 71)) := by
  have c6468 := (publicBatchWrapper86_copies202 a h).2.2.2.2.1
  have c6469 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.1
  have c6470 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.1
  have c6471 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.1
  have c6472 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.1
  have c6473 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.1
  have e_96_12 := arithEq_of_rows h (row := 96) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_12
  simp only [← c6468, ← c6469, ← c6470] at e_96_12
  have e_96_13 := arithEq_of_rows h (row := 96) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_13
  simp only [← c6471, ← c6472, ← c6473] at e_96_13
  have hr := e_96_13
  simp only [e_96_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 63) = bselect (a (.wire 84 63)) (a (.virt 2809)) (a (.wire 95 79)) := by
  have c6474 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6475 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6476 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6477 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6478 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6479 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_96_14 := arithEq_of_rows h (row := 96) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_14
  simp only [← c6474, ← c6475, ← c6476] at e_96_14
  have e_96_15 := arithEq_of_rows h (row := 96) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_15
  simp only [← c6477, ← c6478, ← c6479] at e_96_15
  have hr := e_96_15
  simp only [e_96_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 71) = bselect (a (.wire 84 63)) (a (.virt 2803)) (a (.wire 96 7)) := by
  have c6480 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6481 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6482 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6483 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6484 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6485 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_96_16 := arithEq_of_rows h (row := 96) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_16
  simp only [← c6480, ← c6481, ← c6482] at e_96_16
  have e_96_17 := arithEq_of_rows h (row := 96) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_17
  simp only [← c6483, ← c6484, ← c6485] at e_96_17
  have hr := e_96_17
  simp only [e_96_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 96 79) = bselect (a (.wire 84 63)) (a (.virt 2804)) (a (.wire 96 15)) := by
  have c6486 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6487 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6488 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6489 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6490 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6491 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_96_18 := arithEq_of_rows h (row := 96) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_18
  simp only [← c6486, ← c6487, ← c6488] at e_96_18
  have e_96_19 := arithEq_of_rows h (row := 96) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_96_19
  simp only [← c6489, ← c6490, ← c6491] at e_96_19
  have hr := e_96_19
  simp only [e_96_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 7) = bor (a (.wire 97 3)) (a (.wire 96 19)) := by
  have c6492 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6493 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6494 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6495 := (publicBatchWrapper86_copies202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6496 := (publicBatchWrapper86_copies203 a h).1
  have c6497 := (publicBatchWrapper86_copies203 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_86_13 := arithEq_of_rows h (row := 86) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_13
  simp only [← c6492, ← c6493, ← c6494] at e_86_13
  have e_97_1 := arithEq_of_rows h (row := 97) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_1
  simp only [← c6495, ← c6496, k3, ← c6497] at e_97_1
  have hr := e_97_1
  simp only [e_86_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 3) = bnot (a (.wire 16 47)) := by
  have c6498 := (publicBatchWrapper86_copies203 a h).2.2.1
  have c6499 := (publicBatchWrapper86_copies203 a h).2.2.2.1
  have c6500 := (publicBatchWrapper86_copies203 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_98_0 := arithEq_of_rows h (row := 98) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_0
  simp only [← c6498, k3, ← c6499, k3, ← c6500] at e_98_0
  have hr := e_98_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 7) = bnot (a (.wire 97 7)) := by
  have c6501 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.1
  have c6502 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.1
  have c6503 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_98_1 := arithEq_of_rows h (row := 98) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_1
  simp only [← c6501, k3, ← c6502, k3, ← c6503] at e_98_1
  have hr := e_98_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 67) = band (a (.wire 98 3)) (a (.wire 98 7)) := by
  have c6504 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.1
  have c6505 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.1
  have c6506 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_84_16 := arithEq_of_rows h (row := 84) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_16
  simp only [← c6504, ← c6505, ← c6506] at e_84_16
  have hr := e_84_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 15) = bselect (a (.wire 84 67)) (a (.virt 3005)) (a (.wire 96 31)) := by
  have c6507 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6508 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6509 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6510 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6511 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6512 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_2 := arithEq_of_rows h (row := 98) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_2
  simp only [← c6507, ← c6508, ← c6509] at e_98_2
  have e_98_3 := arithEq_of_rows h (row := 98) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_3
  simp only [← c6510, ← c6511, ← c6512] at e_98_3
  have hr := e_98_3
  simp only [e_98_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 23) = bselect (a (.wire 84 67)) (a (.virt 3006)) (a (.wire 96 39)) := by
  have c6513 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6514 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6515 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6516 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6517 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6518 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_4 := arithEq_of_rows h (row := 98) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_4
  simp only [← c6513, ← c6514, ← c6515] at e_98_4
  have e_98_5 := arithEq_of_rows h (row := 98) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_5
  simp only [← c6516, ← c6517, ← c6518] at e_98_5
  have hr := e_98_5
  simp only [e_98_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 31) = bselect (a (.wire 84 67)) (a (.virt 3007)) (a (.wire 96 47)) := by
  have c6519 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6520 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6521 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6522 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6523 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6524 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_6 := arithEq_of_rows h (row := 98) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_6
  simp only [← c6519, ← c6520, ← c6521] at e_98_6
  have e_98_7 := arithEq_of_rows h (row := 98) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_7
  simp only [← c6522, ← c6523, ← c6524] at e_98_7
  have hr := e_98_7
  simp only [e_98_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 39) = bselect (a (.wire 84 67)) (a (.virt 3008)) (a (.wire 96 55)) := by
  have c6525 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6526 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6527 := (publicBatchWrapper86_copies203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6528 := (publicBatchWrapper86_copies204 a h).1
  have c6529 := (publicBatchWrapper86_copies204 a h).2.1
  have c6530 := (publicBatchWrapper86_copies204 a h).2.2.1
  have e_98_8 := arithEq_of_rows h (row := 98) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_8
  simp only [← c6525, ← c6526, ← c6527] at e_98_8
  have e_98_9 := arithEq_of_rows h (row := 98) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_9
  simp only [← c6528, ← c6529, ← c6530] at e_98_9
  have hr := e_98_9
  simp only [e_98_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 47) = bselect (a (.wire 84 67)) (a (.virt 3009)) (a (.wire 96 63)) := by
  have c6531 := (publicBatchWrapper86_copies204 a h).2.2.2.1
  have c6532 := (publicBatchWrapper86_copies204 a h).2.2.2.2.1
  have c6533 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.1
  have c6534 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.1
  have c6535 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.1
  have c6536 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.1
  have e_98_10 := arithEq_of_rows h (row := 98) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_10
  simp only [← c6531, ← c6532, ← c6533] at e_98_10
  have e_98_11 := arithEq_of_rows h (row := 98) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_11
  simp only [← c6534, ← c6535, ← c6536] at e_98_11
  have hr := e_98_11
  simp only [e_98_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 55) = bselect (a (.wire 84 67)) (a (.virt 3003)) (a (.wire 96 71)) := by
  have c6537 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.1
  have c6538 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6539 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6540 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6541 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6542 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_12 := arithEq_of_rows h (row := 98) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_12
  simp only [← c6537, ← c6538, ← c6539] at e_98_12
  have e_98_13 := arithEq_of_rows h (row := 98) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_13
  simp only [← c6540, ← c6541, ← c6542] at e_98_13
  have hr := e_98_13
  simp only [e_98_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 63) = bselect (a (.wire 84 67)) (a (.virt 3004)) (a (.wire 96 79)) := by
  have c6543 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6544 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6545 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6546 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6547 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6548 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_98_14 := arithEq_of_rows h (row := 98) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_14
  simp only [← c6543, ← c6544, ← c6545] at e_98_14
  have e_98_15 := arithEq_of_rows h (row := 98) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_15
  simp only [← c6546, ← c6547, ← c6548] at e_98_15
  have hr := e_98_15
  simp only [e_98_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 11) = bor (a (.wire 97 7)) (a (.wire 98 3)) := by
  have c6549 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6550 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6551 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6552 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6553 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6554 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_86_14 := arithEq_of_rows h (row := 86) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_14
  simp only [← c6549, ← c6550, ← c6551] at e_86_14
  have e_97_2 := arithEq_of_rows h (row := 97) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_2
  simp only [← c6552, ← c6553, k3, ← c6554] at e_97_2
  have hr := e_97_2
  simp only [e_86_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 67) = bnot (a (.wire 18 11)) := by
  have c6555 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6556 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6557 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_98_16 := arithEq_of_rows h (row := 98) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_16
  simp only [← c6555, k3, ← c6556, k3, ← c6557] at e_98_16
  have hr := e_98_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 71) = bnot (a (.wire 97 11)) := by
  have c6558 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6559 := (publicBatchWrapper86_copies204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6560 := (publicBatchWrapper86_copies205 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_98_17 := arithEq_of_rows h (row := 98) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_17
  simp only [← c6558, k3, ← c6559, k3, ← c6560] at e_98_17
  have hr := e_98_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 71) = band (a (.wire 98 67)) (a (.wire 98 71)) := by
  have c6561 := (publicBatchWrapper86_copies205 a h).2.1
  have c6562 := (publicBatchWrapper86_copies205 a h).2.2.1
  have c6563 := (publicBatchWrapper86_copies205 a h).2.2.2.1
  have e_84_17 := arithEq_of_rows h (row := 84) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_17
  simp only [← c6561, ← c6562, ← c6563] at e_84_17
  have hr := e_84_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 98 79) = bselect (a (.wire 84 71)) (a (.virt 3205)) (a (.wire 98 15)) := by
  have c6564 := (publicBatchWrapper86_copies205 a h).2.2.2.2.1
  have c6565 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.1
  have c6566 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.1
  have c6567 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.1
  have c6568 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.1
  have c6569 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.1
  have e_98_18 := arithEq_of_rows h (row := 98) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_18
  simp only [← c6564, ← c6565, ← c6566] at e_98_18
  have e_98_19 := arithEq_of_rows h (row := 98) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_98_19
  simp only [← c6567, ← c6568, ← c6569] at e_98_19
  have hr := e_98_19
  simp only [e_98_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 7) = bselect (a (.wire 84 71)) (a (.virt 3206)) (a (.wire 98 23)) := by
  have c6570 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6571 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6572 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6573 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6574 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6575 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_0 := arithEq_of_rows h (row := 99) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_0
  simp only [← c6570, ← c6571, ← c6572] at e_99_0
  have e_99_1 := arithEq_of_rows h (row := 99) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_1
  simp only [← c6573, ← c6574, ← c6575] at e_99_1
  have hr := e_99_1
  simp only [e_99_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 15) = bselect (a (.wire 84 71)) (a (.virt 3207)) (a (.wire 98 31)) := by
  have c6576 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6577 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6578 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6579 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6580 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6581 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_2 := arithEq_of_rows h (row := 99) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_2
  simp only [← c6576, ← c6577, ← c6578] at e_99_2
  have e_99_3 := arithEq_of_rows h (row := 99) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_3
  simp only [← c6579, ← c6580, ← c6581] at e_99_3
  have hr := e_99_3
  simp only [e_99_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 23) = bselect (a (.wire 84 71)) (a (.virt 3208)) (a (.wire 98 39)) := by
  have c6582 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6583 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6584 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6585 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6586 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6587 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_4 := arithEq_of_rows h (row := 99) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_4
  simp only [← c6582, ← c6583, ← c6584] at e_99_4
  have e_99_5 := arithEq_of_rows h (row := 99) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_5
  simp only [← c6585, ← c6586, ← c6587] at e_99_5
  have hr := e_99_5
  simp only [e_99_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 31) = bselect (a (.wire 84 71)) (a (.virt 3209)) (a (.wire 98 47)) := by
  have c6588 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6589 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6590 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6591 := (publicBatchWrapper86_copies205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6592 := (publicBatchWrapper86_copies206 a h).1
  have c6593 := (publicBatchWrapper86_copies206 a h).2.1
  have e_99_6 := arithEq_of_rows h (row := 99) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_6
  simp only [← c6588, ← c6589, ← c6590] at e_99_6
  have e_99_7 := arithEq_of_rows h (row := 99) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_7
  simp only [← c6591, ← c6592, ← c6593] at e_99_7
  have hr := e_99_7
  simp only [e_99_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 39) = bselect (a (.wire 84 71)) (a (.virt 3203)) (a (.wire 98 55)) := by
  have c6594 := (publicBatchWrapper86_copies206 a h).2.2.1
  have c6595 := (publicBatchWrapper86_copies206 a h).2.2.2.1
  have c6596 := (publicBatchWrapper86_copies206 a h).2.2.2.2.1
  have c6597 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.1
  have c6598 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.1
  have c6599 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.1
  have e_99_8 := arithEq_of_rows h (row := 99) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_8
  simp only [← c6594, ← c6595, ← c6596] at e_99_8
  have e_99_9 := arithEq_of_rows h (row := 99) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_9
  simp only [← c6597, ← c6598, ← c6599] at e_99_9
  have hr := e_99_9
  simp only [e_99_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 47) = bselect (a (.wire 84 71)) (a (.virt 3204)) (a (.wire 98 63)) := by
  have c6600 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.1
  have c6601 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.1
  have c6602 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6603 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6604 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6605 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_10 := arithEq_of_rows h (row := 99) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_10
  simp only [← c6600, ← c6601, ← c6602] at e_99_10
  have e_99_11 := arithEq_of_rows h (row := 99) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_11
  simp only [← c6603, ← c6604, ← c6605] at e_99_11
  have hr := e_99_11
  simp only [e_99_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 15) = bor (a (.wire 97 11)) (a (.wire 98 67)) := by
  have c6606 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6607 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6608 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6609 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6610 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6611 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_86_15 := arithEq_of_rows h (row := 86) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_15
  simp only [← c6606, ← c6607, ← c6608] at e_86_15
  have e_97_3 := arithEq_of_rows h (row := 97) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_3
  simp only [← c6609, ← c6610, k3, ← c6611] at e_97_3
  have hr := e_97_3
  simp only [e_86_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 51) = bnot (a (.wire 18 55)) := by
  have c6612 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6613 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6614 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_99_12 := arithEq_of_rows h (row := 99) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_12
  simp only [← c6612, k3, ← c6613, k3, ← c6614] at e_99_12
  have hr := e_99_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 55) = bnot (a (.wire 97 15)) := by
  have c6615 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6616 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6617 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_99_13 := arithEq_of_rows h (row := 99) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_13
  simp only [← c6615, k3, ← c6616, k3, ← c6617] at e_99_13
  have hr := e_99_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 75) = band (a (.wire 99 51)) (a (.wire 99 55)) := by
  have c6618 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6619 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6620 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_18 := arithEq_of_rows h (row := 84) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_18
  simp only [← c6618, ← c6619, ← c6620] at e_84_18
  have hr := e_84_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 63) = bselect (a (.wire 84 75)) (a (.virt 3405)) (a (.wire 98 79)) := by
  have c6621 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6622 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6623 := (publicBatchWrapper86_copies206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6624 := (publicBatchWrapper86_copies207 a h).1
  have c6625 := (publicBatchWrapper86_copies207 a h).2.1
  have c6626 := (publicBatchWrapper86_copies207 a h).2.2.1
  have e_99_14 := arithEq_of_rows h (row := 99) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_14
  simp only [← c6621, ← c6622, ← c6623] at e_99_14
  have e_99_15 := arithEq_of_rows h (row := 99) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_15
  simp only [← c6624, ← c6625, ← c6626] at e_99_15
  have hr := e_99_15
  simp only [e_99_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 71) = bselect (a (.wire 84 75)) (a (.virt 3406)) (a (.wire 99 7)) := by
  have c6627 := (publicBatchWrapper86_copies207 a h).2.2.2.1
  have c6628 := (publicBatchWrapper86_copies207 a h).2.2.2.2.1
  have c6629 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.1
  have c6630 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.1
  have c6631 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.1
  have c6632 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.1
  have e_99_16 := arithEq_of_rows h (row := 99) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_16
  simp only [← c6627, ← c6628, ← c6629] at e_99_16
  have e_99_17 := arithEq_of_rows h (row := 99) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_17
  simp only [← c6630, ← c6631, ← c6632] at e_99_17
  have hr := e_99_17
  simp only [e_99_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 99 79) = bselect (a (.wire 84 75)) (a (.virt 3407)) (a (.wire 99 15)) := by
  have c6633 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.1
  have c6634 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6635 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6636 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6637 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6638 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_99_18 := arithEq_of_rows h (row := 99) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_18
  simp only [← c6633, ← c6634, ← c6635] at e_99_18
  have e_99_19 := arithEq_of_rows h (row := 99) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_99_19
  simp only [← c6636, ← c6637, ← c6638] at e_99_19
  have hr := e_99_19
  simp only [e_99_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 7) = bselect (a (.wire 84 75)) (a (.virt 3408)) (a (.wire 99 23)) := by
  have c6639 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6640 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6641 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6642 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6643 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6644 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_0 := arithEq_of_rows h (row := 100) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_0
  simp only [← c6639, ← c6640, ← c6641] at e_100_0
  have e_100_1 := arithEq_of_rows h (row := 100) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_1
  simp only [← c6642, ← c6643, ← c6644] at e_100_1
  have hr := e_100_1
  simp only [e_100_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 15) = bselect (a (.wire 84 75)) (a (.virt 3409)) (a (.wire 99 31)) := by
  have c6645 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6646 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6647 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6648 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6649 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6650 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_2 := arithEq_of_rows h (row := 100) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_2
  simp only [← c6645, ← c6646, ← c6647] at e_100_2
  have e_100_3 := arithEq_of_rows h (row := 100) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_3
  simp only [← c6648, ← c6649, ← c6650] at e_100_3
  have hr := e_100_3
  simp only [e_100_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 23) = bselect (a (.wire 84 75)) (a (.virt 3403)) (a (.wire 99 39)) := by
  have c6651 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6652 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6653 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6654 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6655 := (publicBatchWrapper86_copies207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6656 := (publicBatchWrapper86_copies208 a h).1
  have e_100_4 := arithEq_of_rows h (row := 100) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_4
  simp only [← c6651, ← c6652, ← c6653] at e_100_4
  have e_100_5 := arithEq_of_rows h (row := 100) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_5
  simp only [← c6654, ← c6655, ← c6656] at e_100_5
  have hr := e_100_5
  simp only [e_100_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 31) = bselect (a (.wire 84 75)) (a (.virt 3404)) (a (.wire 99 47)) := by
  have c6657 := (publicBatchWrapper86_copies208 a h).2.1
  have c6658 := (publicBatchWrapper86_copies208 a h).2.2.1
  have c6659 := (publicBatchWrapper86_copies208 a h).2.2.2.1
  have c6660 := (publicBatchWrapper86_copies208 a h).2.2.2.2.1
  have c6661 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.1
  have c6662 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.1
  have e_100_6 := arithEq_of_rows h (row := 100) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_6
  simp only [← c6657, ← c6658, ← c6659] at e_100_6
  have e_100_7 := arithEq_of_rows h (row := 100) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_7
  simp only [← c6660, ← c6661, ← c6662] at e_100_7
  have hr := e_100_7
  simp only [e_100_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 19) = bor (a (.wire 97 15)) (a (.wire 99 51)) := by
  have c6663 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.1
  have c6664 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.1
  have c6665 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.1
  have c6666 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6667 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6668 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_86_16 := arithEq_of_rows h (row := 86) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_16
  simp only [← c6663, ← c6664, ← c6665] at e_86_16
  have e_97_4 := arithEq_of_rows h (row := 97) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_4
  simp only [← c6666, ← c6667, k3, ← c6668] at e_97_4
  have hr := e_97_4
  simp only [e_86_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 35) = bnot (a (.wire 20 19)) := by
  have c6669 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6670 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6671 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_100_8 := arithEq_of_rows h (row := 100) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_8
  simp only [← c6669, k3, ← c6670, k3, ← c6671] at e_100_8
  have hr := e_100_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 39) = bnot (a (.wire 97 19)) := by
  have c6672 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6673 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6674 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_100_9 := arithEq_of_rows h (row := 100) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_9
  simp only [← c6672, k3, ← c6673, k3, ← c6674] at e_100_9
  have hr := e_100_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 84 79) = band (a (.wire 100 35)) (a (.wire 100 39)) := by
  have c6675 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6676 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6677 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_84_19 := arithEq_of_rows h (row := 84) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_84_19
  simp only [← c6675, ← c6676, ← c6677] at e_84_19
  have hr := e_84_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 47) = bselect (a (.wire 84 79)) (a (.virt 3605)) (a (.wire 99 63)) := by
  have c6678 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6679 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6680 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6681 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6682 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6683 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_10 := arithEq_of_rows h (row := 100) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_10
  simp only [← c6678, ← c6679, ← c6680] at e_100_10
  have e_100_11 := arithEq_of_rows h (row := 100) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_11
  simp only [← c6681, ← c6682, ← c6683] at e_100_11
  have hr := e_100_11
  simp only [e_100_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 55) = bselect (a (.wire 84 79)) (a (.virt 3606)) (a (.wire 99 71)) := by
  have c6684 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6685 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6686 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6687 := (publicBatchWrapper86_copies208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6688 := (publicBatchWrapper86_copies209 a h).1
  have c6689 := (publicBatchWrapper86_copies209 a h).2.1
  have e_100_12 := arithEq_of_rows h (row := 100) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_12
  simp only [← c6684, ← c6685, ← c6686] at e_100_12
  have e_100_13 := arithEq_of_rows h (row := 100) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_13
  simp only [← c6687, ← c6688, ← c6689] at e_100_13
  have hr := e_100_13
  simp only [e_100_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 63) = bselect (a (.wire 84 79)) (a (.virt 3607)) (a (.wire 99 79)) := by
  have c6690 := (publicBatchWrapper86_copies209 a h).2.2.1
  have c6691 := (publicBatchWrapper86_copies209 a h).2.2.2.1
  have c6692 := (publicBatchWrapper86_copies209 a h).2.2.2.2.1
  have c6693 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.1
  have c6694 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.1
  have c6695 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.1
  have e_100_14 := arithEq_of_rows h (row := 100) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_14
  simp only [← c6690, ← c6691, ← c6692] at e_100_14
  have e_100_15 := arithEq_of_rows h (row := 100) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_15
  simp only [← c6693, ← c6694, ← c6695] at e_100_15
  have hr := e_100_15
  simp only [e_100_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 71) = bselect (a (.wire 84 79)) (a (.virt 3608)) (a (.wire 100 7)) := by
  have c6696 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.1
  have c6697 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.1
  have c6698 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6699 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6700 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6701 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_16 := arithEq_of_rows h (row := 100) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_16
  simp only [← c6696, ← c6697, ← c6698] at e_100_16
  have e_100_17 := arithEq_of_rows h (row := 100) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_17
  simp only [← c6699, ← c6700, ← c6701] at e_100_17
  have hr := e_100_17
  simp only [e_100_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 100 79) = bselect (a (.wire 84 79)) (a (.virt 3609)) (a (.wire 100 15)) := by
  have c6702 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6703 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6704 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6705 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6706 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6707 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_100_18 := arithEq_of_rows h (row := 100) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_18
  simp only [← c6702, ← c6703, ← c6704] at e_100_18
  have e_100_19 := arithEq_of_rows h (row := 100) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_100_19
  simp only [← c6705, ← c6706, ← c6707] at e_100_19
  have hr := e_100_19
  simp only [e_100_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 7) = bselect (a (.wire 84 79)) (a (.virt 3603)) (a (.wire 100 23)) := by
  have c6708 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6709 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6710 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6711 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6712 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6713 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_0 := arithEq_of_rows h (row := 101) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_0
  simp only [← c6708, ← c6709, ← c6710] at e_101_0
  have e_101_1 := arithEq_of_rows h (row := 101) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_1
  simp only [← c6711, ← c6712, ← c6713] at e_101_1
  have hr := e_101_1
  simp only [e_101_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 15) = bselect (a (.wire 84 79)) (a (.virt 3604)) (a (.wire 100 31)) := by
  have c6714 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6715 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6716 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6717 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6718 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6719 := (publicBatchWrapper86_copies209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_101_2 := arithEq_of_rows h (row := 101) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_2
  simp only [← c6714, ← c6715, ← c6716] at e_101_2
  have e_101_3 := arithEq_of_rows h (row := 101) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_3
  simp only [← c6717, ← c6718, ← c6719] at e_101_3
  have hr := e_101_3
  simp only [e_101_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 23) = bor (a (.wire 97 19)) (a (.wire 100 35)) := by
  have c6720 := (publicBatchWrapper86_copies210 a h).1
  have c6721 := (publicBatchWrapper86_copies210 a h).2.1
  have c6722 := (publicBatchWrapper86_copies210 a h).2.2.1
  have c6723 := (publicBatchWrapper86_copies210 a h).2.2.2.1
  have c6724 := (publicBatchWrapper86_copies210 a h).2.2.2.2.1
  have c6725 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_86_17 := arithEq_of_rows h (row := 86) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_17
  simp only [← c6720, ← c6721, ← c6722] at e_86_17
  have e_97_5 := arithEq_of_rows h (row := 97) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_5
  simp only [← c6723, ← c6724, k3, ← c6725] at e_97_5
  have hr := e_97_5
  simp only [e_86_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 19) = bnot (a (.wire 20 63)) := by
  have c6726 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.1
  have c6727 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.1
  have c6728 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_101_4 := arithEq_of_rows h (row := 101) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_4
  simp only [← c6726, k3, ← c6727, k3, ← c6728] at e_101_4
  have hr := e_101_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 23) = bnot (a (.wire 97 23)) := by
  have c6729 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.1
  have c6730 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6731 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_101_5 := arithEq_of_rows h (row := 101) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_5
  simp only [← c6729, k3, ← c6730, k3, ← c6731] at e_101_5
  have hr := e_101_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 3) = band (a (.wire 101 19)) (a (.wire 101 23)) := by
  have c6732 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6733 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6734 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_0 := arithEq_of_rows h (row := 102) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_0
  simp only [← c6732, ← c6733, ← c6734] at e_102_0
  have hr := e_102_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 31) = bselect (a (.wire 102 3)) (a (.virt 3805)) (a (.wire 100 47)) := by
  have c6735 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6736 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6737 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6738 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6739 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6740 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_6 := arithEq_of_rows h (row := 101) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_6
  simp only [← c6735, ← c6736, ← c6737] at e_101_6
  have e_101_7 := arithEq_of_rows h (row := 101) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_7
  simp only [← c6738, ← c6739, ← c6740] at e_101_7
  have hr := e_101_7
  simp only [e_101_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 39) = bselect (a (.wire 102 3)) (a (.virt 3806)) (a (.wire 100 55)) := by
  have c6741 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6742 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6743 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6744 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6745 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6746 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_8 := arithEq_of_rows h (row := 101) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_8
  simp only [← c6741, ← c6742, ← c6743] at e_101_8
  have e_101_9 := arithEq_of_rows h (row := 101) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_9
  simp only [← c6744, ← c6745, ← c6746] at e_101_9
  have hr := e_101_9
  simp only [e_101_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 47) = bselect (a (.wire 102 3)) (a (.virt 3807)) (a (.wire 100 63)) := by
  have c6747 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6748 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6749 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6750 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6751 := (publicBatchWrapper86_copies210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6752 := (publicBatchWrapper86_copies211 a h).1
  have e_101_10 := arithEq_of_rows h (row := 101) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_10
  simp only [← c6747, ← c6748, ← c6749] at e_101_10
  have e_101_11 := arithEq_of_rows h (row := 101) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_11
  simp only [← c6750, ← c6751, ← c6752] at e_101_11
  have hr := e_101_11
  simp only [e_101_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 55) = bselect (a (.wire 102 3)) (a (.virt 3808)) (a (.wire 100 71)) := by
  have c6753 := (publicBatchWrapper86_copies211 a h).2.1
  have c6754 := (publicBatchWrapper86_copies211 a h).2.2.1
  have c6755 := (publicBatchWrapper86_copies211 a h).2.2.2.1
  have c6756 := (publicBatchWrapper86_copies211 a h).2.2.2.2.1
  have c6757 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.1
  have c6758 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.1
  have e_101_12 := arithEq_of_rows h (row := 101) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_12
  simp only [← c6753, ← c6754, ← c6755] at e_101_12
  have e_101_13 := arithEq_of_rows h (row := 101) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_13
  simp only [← c6756, ← c6757, ← c6758] at e_101_13
  have hr := e_101_13
  simp only [e_101_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 63) = bselect (a (.wire 102 3)) (a (.virt 3809)) (a (.wire 100 79)) := by
  have c6759 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.1
  have c6760 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.1
  have c6761 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.1
  have c6762 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6763 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6764 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_14 := arithEq_of_rows h (row := 101) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_14
  simp only [← c6759, ← c6760, ← c6761] at e_101_14
  have e_101_15 := arithEq_of_rows h (row := 101) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_15
  simp only [← c6762, ← c6763, ← c6764] at e_101_15
  have hr := e_101_15
  simp only [e_101_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 71) = bselect (a (.wire 102 3)) (a (.virt 3803)) (a (.wire 101 7)) := by
  have c6765 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6766 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6767 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6768 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6769 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6770 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_16 := arithEq_of_rows h (row := 101) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_16
  simp only [← c6765, ← c6766, ← c6767] at e_101_16
  have e_101_17 := arithEq_of_rows h (row := 101) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_17
  simp only [← c6768, ← c6769, ← c6770] at e_101_17
  have hr := e_101_17
  simp only [e_101_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 101 79) = bselect (a (.wire 102 3)) (a (.virt 3804)) (a (.wire 101 15)) := by
  have c6771 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6772 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6773 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6774 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6775 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6776 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_101_18 := arithEq_of_rows h (row := 101) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_18
  simp only [← c6771, ← c6772, ← c6773] at e_101_18
  have e_101_19 := arithEq_of_rows h (row := 101) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_101_19
  simp only [← c6774, ← c6775, ← c6776] at e_101_19
  have hr := e_101_19
  simp only [e_101_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 27) = bor (a (.wire 97 23)) (a (.wire 101 19)) := by
  have c6777 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6778 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6779 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6780 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6781 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6782 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_86_18 := arithEq_of_rows h (row := 86) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_18
  simp only [← c6777, ← c6778, ← c6779] at e_86_18
  have e_97_6 := arithEq_of_rows h (row := 97) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_6
  simp only [← c6780, ← c6781, k3, ← c6782] at e_97_6
  have hr := e_97_6
  simp only [e_86_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 3) = bnot (a (.wire 22 27)) := by
  have c6783 := (publicBatchWrapper86_copies211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6784 := (publicBatchWrapper86_copies212 a h).1
  have c6785 := (publicBatchWrapper86_copies212 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_103_0 := arithEq_of_rows h (row := 103) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_0
  simp only [← c6783, k3, ← c6784, k3, ← c6785] at e_103_0
  have hr := e_103_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 7) = bnot (a (.wire 97 27)) := by
  have c6786 := (publicBatchWrapper86_copies212 a h).2.2.1
  have c6787 := (publicBatchWrapper86_copies212 a h).2.2.2.1
  have c6788 := (publicBatchWrapper86_copies212 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_103_1 := arithEq_of_rows h (row := 103) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_1
  simp only [← c6786, k3, ← c6787, k3, ← c6788] at e_103_1
  have hr := e_103_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 7) = band (a (.wire 103 3)) (a (.wire 103 7)) := by
  have c6789 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.1
  have c6790 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.1
  have c6791 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.1
  have e_102_1 := arithEq_of_rows h (row := 102) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_1
  simp only [← c6789, ← c6790, ← c6791] at e_102_1
  have hr := e_102_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 15) = bselect (a (.wire 102 7)) (a (.virt 4005)) (a (.wire 101 31)) := by
  have c6792 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.1
  have c6793 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.1
  have c6794 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6795 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6796 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6797 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_2 := arithEq_of_rows h (row := 103) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_2
  simp only [← c6792, ← c6793, ← c6794] at e_103_2
  have e_103_3 := arithEq_of_rows h (row := 103) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_3
  simp only [← c6795, ← c6796, ← c6797] at e_103_3
  have hr := e_103_3
  simp only [e_103_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 23) = bselect (a (.wire 102 7)) (a (.virt 4006)) (a (.wire 101 39)) := by
  have c6798 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6799 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6800 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6801 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6802 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6803 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_4 := arithEq_of_rows h (row := 103) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_4
  simp only [← c6798, ← c6799, ← c6800] at e_103_4
  have e_103_5 := arithEq_of_rows h (row := 103) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_5
  simp only [← c6801, ← c6802, ← c6803] at e_103_5
  have hr := e_103_5
  simp only [e_103_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 31) = bselect (a (.wire 102 7)) (a (.virt 4007)) (a (.wire 101 47)) := by
  have c6804 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6805 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6806 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6807 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6808 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6809 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_6 := arithEq_of_rows h (row := 103) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_6
  simp only [← c6804, ← c6805, ← c6806] at e_103_6
  have e_103_7 := arithEq_of_rows h (row := 103) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_7
  simp only [← c6807, ← c6808, ← c6809] at e_103_7
  have hr := e_103_7
  simp only [e_103_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 39) = bselect (a (.wire 102 7)) (a (.virt 4008)) (a (.wire 101 55)) := by
  have c6810 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6811 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6812 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6813 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6814 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6815 := (publicBatchWrapper86_copies212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_103_8 := arithEq_of_rows h (row := 103) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_8
  simp only [← c6810, ← c6811, ← c6812] at e_103_8
  have e_103_9 := arithEq_of_rows h (row := 103) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_9
  simp only [← c6813, ← c6814, ← c6815] at e_103_9
  have hr := e_103_9
  simp only [e_103_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 47) = bselect (a (.wire 102 7)) (a (.virt 4009)) (a (.wire 101 63)) := by
  have c6816 := (publicBatchWrapper86_copies213 a h).1
  have c6817 := (publicBatchWrapper86_copies213 a h).2.1
  have c6818 := (publicBatchWrapper86_copies213 a h).2.2.1
  have c6819 := (publicBatchWrapper86_copies213 a h).2.2.2.1
  have c6820 := (publicBatchWrapper86_copies213 a h).2.2.2.2.1
  have c6821 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.1
  have e_103_10 := arithEq_of_rows h (row := 103) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_10
  simp only [← c6816, ← c6817, ← c6818] at e_103_10
  have e_103_11 := arithEq_of_rows h (row := 103) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_11
  simp only [← c6819, ← c6820, ← c6821] at e_103_11
  have hr := e_103_11
  simp only [e_103_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 55) = bselect (a (.wire 102 7)) (a (.virt 4003)) (a (.wire 101 71)) := by
  have c6822 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.1
  have c6823 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.1
  have c6824 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.1
  have c6825 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.1
  have c6826 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6827 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_12 := arithEq_of_rows h (row := 103) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_12
  simp only [← c6822, ← c6823, ← c6824] at e_103_12
  have e_103_13 := arithEq_of_rows h (row := 103) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_13
  simp only [← c6825, ← c6826, ← c6827] at e_103_13
  have hr := e_103_13
  simp only [e_103_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 63) = bselect (a (.wire 102 7)) (a (.virt 4004)) (a (.wire 101 79)) := by
  have c6828 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6829 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6830 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6831 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6832 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6833 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_103_14 := arithEq_of_rows h (row := 103) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_14
  simp only [← c6828, ← c6829, ← c6830] at e_103_14
  have e_103_15 := arithEq_of_rows h (row := 103) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_15
  simp only [← c6831, ← c6832, ← c6833] at e_103_15
  have hr := e_103_15
  simp only [e_103_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 31) = bor (a (.wire 97 27)) (a (.wire 103 3)) := by
  have c6834 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6835 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6836 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6837 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6838 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6839 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_86_19 := arithEq_of_rows h (row := 86) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_86_19
  simp only [← c6834, ← c6835, ← c6836] at e_86_19
  have e_97_7 := arithEq_of_rows h (row := 97) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_7
  simp only [← c6837, ← c6838, k3, ← c6839] at e_97_7
  have hr := e_97_7
  simp only [e_86_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 67) = bnot (a (.wire 22 71)) := by
  have c6840 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6841 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6842 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_103_16 := arithEq_of_rows h (row := 103) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_16
  simp only [← c6840, k3, ← c6841, k3, ← c6842] at e_103_16
  have hr := e_103_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 71) = bnot (a (.wire 97 31)) := by
  have c6843 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6844 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6845 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_103_17 := arithEq_of_rows h (row := 103) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_17
  simp only [← c6843, k3, ← c6844, k3, ← c6845] at e_103_17
  have hr := e_103_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 11) = band (a (.wire 103 67)) (a (.wire 103 71)) := by
  have c6846 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6847 := (publicBatchWrapper86_copies213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6848 := (publicBatchWrapper86_copies214 a h).1
  have e_102_2 := arithEq_of_rows h (row := 102) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_2
  simp only [← c6846, ← c6847, ← c6848] at e_102_2
  have hr := e_102_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 103 79) = bselect (a (.wire 102 11)) (a (.virt 4205)) (a (.wire 103 15)) := by
  have c6849 := (publicBatchWrapper86_copies214 a h).2.1
  have c6850 := (publicBatchWrapper86_copies214 a h).2.2.1
  have c6851 := (publicBatchWrapper86_copies214 a h).2.2.2.1
  have c6852 := (publicBatchWrapper86_copies214 a h).2.2.2.2.1
  have c6853 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.1
  have c6854 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.1
  have e_103_18 := arithEq_of_rows h (row := 103) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_18
  simp only [← c6849, ← c6850, ← c6851] at e_103_18
  have e_103_19 := arithEq_of_rows h (row := 103) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_103_19
  simp only [← c6852, ← c6853, ← c6854] at e_103_19
  have hr := e_103_19
  simp only [e_103_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 7) = bselect (a (.wire 102 11)) (a (.virt 4206)) (a (.wire 103 23)) := by
  have c6855 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.1
  have c6856 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.1
  have c6857 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.1
  have c6858 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6859 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6860 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_0 := arithEq_of_rows h (row := 104) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_0
  simp only [← c6855, ← c6856, ← c6857] at e_104_0
  have e_104_1 := arithEq_of_rows h (row := 104) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_1
  simp only [← c6858, ← c6859, ← c6860] at e_104_1
  have hr := e_104_1
  simp only [e_104_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 15) = bselect (a (.wire 102 11)) (a (.virt 4207)) (a (.wire 103 31)) := by
  have c6861 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6862 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6863 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6864 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6865 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6866 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_2 := arithEq_of_rows h (row := 104) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_2
  simp only [← c6861, ← c6862, ← c6863] at e_104_2
  have e_104_3 := arithEq_of_rows h (row := 104) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_3
  simp only [← c6864, ← c6865, ← c6866] at e_104_3
  have hr := e_104_3
  simp only [e_104_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 23) = bselect (a (.wire 102 11)) (a (.virt 4208)) (a (.wire 103 39)) := by
  have c6867 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6868 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6869 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6870 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6871 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6872 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_4 := arithEq_of_rows h (row := 104) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_4
  simp only [← c6867, ← c6868, ← c6869] at e_104_4
  have e_104_5 := arithEq_of_rows h (row := 104) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_5
  simp only [← c6870, ← c6871, ← c6872] at e_104_5
  have hr := e_104_5
  simp only [e_104_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 31) = bselect (a (.wire 102 11)) (a (.virt 4209)) (a (.wire 103 47)) := by
  have c6873 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6874 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6875 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6876 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6877 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6878 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_6 := arithEq_of_rows h (row := 104) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_6
  simp only [← c6873, ← c6874, ← c6875] at e_104_6
  have e_104_7 := arithEq_of_rows h (row := 104) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_7
  simp only [← c6876, ← c6877, ← c6878] at e_104_7
  have hr := e_104_7
  simp only [e_104_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 39) = bselect (a (.wire 102 11)) (a (.virt 4203)) (a (.wire 103 55)) := by
  have c6879 := (publicBatchWrapper86_copies214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6880 := (publicBatchWrapper86_copies215 a h).1
  have c6881 := (publicBatchWrapper86_copies215 a h).2.1
  have c6882 := (publicBatchWrapper86_copies215 a h).2.2.1
  have c6883 := (publicBatchWrapper86_copies215 a h).2.2.2.1
  have c6884 := (publicBatchWrapper86_copies215 a h).2.2.2.2.1
  have e_104_8 := arithEq_of_rows h (row := 104) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_8
  simp only [← c6879, ← c6880, ← c6881] at e_104_8
  have e_104_9 := arithEq_of_rows h (row := 104) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_9
  simp only [← c6882, ← c6883, ← c6884] at e_104_9
  have hr := e_104_9
  simp only [e_104_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 47) = bselect (a (.wire 102 11)) (a (.virt 4204)) (a (.wire 103 63)) := by
  have c6885 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.1
  have c6886 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.1
  have c6887 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.1
  have c6888 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.1
  have c6889 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.1
  have c6890 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_104_10 := arithEq_of_rows h (row := 104) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_10
  simp only [← c6885, ← c6886, ← c6887] at e_104_10
  have e_104_11 := arithEq_of_rows h (row := 104) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_11
  simp only [← c6888, ← c6889, ← c6890] at e_104_11
  have hr := e_104_11
  simp only [e_104_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 35) = bor (a (.wire 97 31)) (a (.wire 103 67)) := by
  have c6891 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6892 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6893 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6894 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6895 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6896 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_8 := arithEq_of_rows h (row := 97) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_8
  simp only [← c6894, ← c6895, k3, ← c6896] at e_97_8
  have e_105_0 := arithEq_of_rows h (row := 105) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_0
  simp only [← c6891, ← c6892, ← c6893] at e_105_0
  have hr := e_97_8
  simp only [e_105_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 51) = bnot (a (.wire 23 35)) := by
  have c6897 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6898 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6899 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_104_12 := arithEq_of_rows h (row := 104) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_12
  simp only [← c6897, k3, ← c6898, k3, ← c6899] at e_104_12
  have hr := e_104_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 55) = bnot (a (.wire 97 35)) := by
  have c6900 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6901 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6902 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_104_13 := arithEq_of_rows h (row := 104) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_13
  simp only [← c6900, k3, ← c6901, k3, ← c6902] at e_104_13
  have hr := e_104_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 15) = band (a (.wire 104 51)) (a (.wire 104 55)) := by
  have c6903 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6904 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6905 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_3 := arithEq_of_rows h (row := 102) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_3
  simp only [← c6903, ← c6904, ← c6905] at e_102_3
  have hr := e_102_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 63) = bselect (a (.wire 102 15)) (a (.virt 4405)) (a (.wire 103 79)) := by
  have c6906 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6907 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6908 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6909 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6910 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6911 := (publicBatchWrapper86_copies215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_104_14 := arithEq_of_rows h (row := 104) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_14
  simp only [← c6906, ← c6907, ← c6908] at e_104_14
  have e_104_15 := arithEq_of_rows h (row := 104) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_15
  simp only [← c6909, ← c6910, ← c6911] at e_104_15
  have hr := e_104_15
  simp only [e_104_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 71) = bselect (a (.wire 102 15)) (a (.virt 4406)) (a (.wire 104 7)) := by
  have c6912 := (publicBatchWrapper86_copies216 a h).1
  have c6913 := (publicBatchWrapper86_copies216 a h).2.1
  have c6914 := (publicBatchWrapper86_copies216 a h).2.2.1
  have c6915 := (publicBatchWrapper86_copies216 a h).2.2.2.1
  have c6916 := (publicBatchWrapper86_copies216 a h).2.2.2.2.1
  have c6917 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.1
  have e_104_16 := arithEq_of_rows h (row := 104) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_16
  simp only [← c6912, ← c6913, ← c6914] at e_104_16
  have e_104_17 := arithEq_of_rows h (row := 104) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_17
  simp only [← c6915, ← c6916, ← c6917] at e_104_17
  have hr := e_104_17
  simp only [e_104_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 104 79) = bselect (a (.wire 102 15)) (a (.virt 4407)) (a (.wire 104 15)) := by
  have c6918 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.1
  have c6919 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.1
  have c6920 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.1
  have c6921 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.1
  have c6922 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6923 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_104_18 := arithEq_of_rows h (row := 104) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_18
  simp only [← c6918, ← c6919, ← c6920] at e_104_18
  have e_104_19 := arithEq_of_rows h (row := 104) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_104_19
  simp only [← c6921, ← c6922, ← c6923] at e_104_19
  have hr := e_104_19
  simp only [e_104_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 7) = bselect (a (.wire 102 15)) (a (.virt 4408)) (a (.wire 104 23)) := by
  have c6924 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6925 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6926 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6927 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6928 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6929 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_0 := arithEq_of_rows h (row := 106) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_0
  simp only [← c6924, ← c6925, ← c6926] at e_106_0
  have e_106_1 := arithEq_of_rows h (row := 106) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_1
  simp only [← c6927, ← c6928, ← c6929] at e_106_1
  have hr := e_106_1
  simp only [e_106_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 15) = bselect (a (.wire 102 15)) (a (.virt 4409)) (a (.wire 104 31)) := by
  have c6930 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6931 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6932 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6933 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6934 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6935 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_2 := arithEq_of_rows h (row := 106) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_2
  simp only [← c6930, ← c6931, ← c6932] at e_106_2
  have e_106_3 := arithEq_of_rows h (row := 106) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_3
  simp only [← c6933, ← c6934, ← c6935] at e_106_3
  have hr := e_106_3
  simp only [e_106_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 23) = bselect (a (.wire 102 15)) (a (.virt 4403)) (a (.wire 104 39)) := by
  have c6936 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6937 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6938 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6939 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6940 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6941 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_4 := arithEq_of_rows h (row := 106) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_4
  simp only [← c6936, ← c6937, ← c6938] at e_106_4
  have e_106_5 := arithEq_of_rows h (row := 106) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_5
  simp only [← c6939, ← c6940, ← c6941] at e_106_5
  have hr := e_106_5
  simp only [e_106_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 31) = bselect (a (.wire 102 15)) (a (.virt 4404)) (a (.wire 104 47)) := by
  have c6942 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6943 := (publicBatchWrapper86_copies216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6944 := (publicBatchWrapper86_copies217 a h).1
  have c6945 := (publicBatchWrapper86_copies217 a h).2.1
  have c6946 := (publicBatchWrapper86_copies217 a h).2.2.1
  have c6947 := (publicBatchWrapper86_copies217 a h).2.2.2.1
  have e_106_6 := arithEq_of_rows h (row := 106) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_6
  simp only [← c6942, ← c6943, ← c6944] at e_106_6
  have e_106_7 := arithEq_of_rows h (row := 106) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_7
  simp only [← c6945, ← c6946, ← c6947] at e_106_7
  have hr := e_106_7
  simp only [e_106_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 39) = bor (a (.wire 97 35)) (a (.wire 104 51)) := by
  have c6948 := (publicBatchWrapper86_copies217 a h).2.2.2.2.1
  have c6949 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.1
  have c6950 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.1
  have c6951 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.1
  have c6952 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.1
  have c6953 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_9 := arithEq_of_rows h (row := 97) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_9
  simp only [← c6951, ← c6952, k3, ← c6953] at e_97_9
  have e_105_1 := arithEq_of_rows h (row := 105) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_1
  simp only [← c6948, ← c6949, ← c6950] at e_105_1
  have hr := e_97_9
  simp only [e_105_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 35) = bnot (a (.wire 23 79)) := by
  have c6954 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.1
  have c6955 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6956 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_106_8 := arithEq_of_rows h (row := 106) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_8
  simp only [← c6954, k3, ← c6955, k3, ← c6956] at e_106_8
  have hr := e_106_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 39) = bnot (a (.wire 97 39)) := by
  have c6957 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6958 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6959 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_106_9 := arithEq_of_rows h (row := 106) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_9
  simp only [← c6957, k3, ← c6958, k3, ← c6959] at e_106_9
  have hr := e_106_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 19) = band (a (.wire 106 35)) (a (.wire 106 39)) := by
  have c6960 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6961 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6962 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_4 := arithEq_of_rows h (row := 102) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_4
  simp only [← c6960, ← c6961, ← c6962] at e_102_4
  have hr := e_102_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 47) = bselect (a (.wire 102 19)) (a (.virt 4605)) (a (.wire 104 63)) := by
  have c6963 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6964 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6965 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6966 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6967 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6968 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_10 := arithEq_of_rows h (row := 106) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_10
  simp only [← c6963, ← c6964, ← c6965] at e_106_10
  have e_106_11 := arithEq_of_rows h (row := 106) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_11
  simp only [← c6966, ← c6967, ← c6968] at e_106_11
  have hr := e_106_11
  simp only [e_106_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 55) = bselect (a (.wire 102 19)) (a (.virt 4606)) (a (.wire 104 71)) := by
  have c6969 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6970 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6971 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6972 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6973 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6974 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_12 := arithEq_of_rows h (row := 106) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_12
  simp only [← c6969, ← c6970, ← c6971] at e_106_12
  have e_106_13 := arithEq_of_rows h (row := 106) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_13
  simp only [← c6972, ← c6973, ← c6974] at e_106_13
  have hr := e_106_13
  simp only [e_106_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 63) = bselect (a (.wire 102 19)) (a (.virt 4607)) (a (.wire 104 79)) := by
  have c6975 := (publicBatchWrapper86_copies217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c6976 := (publicBatchWrapper86_copies218 a h).1
  have c6977 := (publicBatchWrapper86_copies218 a h).2.1
  have c6978 := (publicBatchWrapper86_copies218 a h).2.2.1
  have c6979 := (publicBatchWrapper86_copies218 a h).2.2.2.1
  have c6980 := (publicBatchWrapper86_copies218 a h).2.2.2.2.1
  have e_106_14 := arithEq_of_rows h (row := 106) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_14
  simp only [← c6975, ← c6976, ← c6977] at e_106_14
  have e_106_15 := arithEq_of_rows h (row := 106) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_15
  simp only [← c6978, ← c6979, ← c6980] at e_106_15
  have hr := e_106_15
  simp only [e_106_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 71) = bselect (a (.wire 102 19)) (a (.virt 4608)) (a (.wire 106 7)) := by
  have c6981 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.1
  have c6982 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.1
  have c6983 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.1
  have c6984 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.1
  have c6985 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.1
  have c6986 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_106_16 := arithEq_of_rows h (row := 106) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_16
  simp only [← c6981, ← c6982, ← c6983] at e_106_16
  have e_106_17 := arithEq_of_rows h (row := 106) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_17
  simp only [← c6984, ← c6985, ← c6986] at e_106_17
  have hr := e_106_17
  simp only [e_106_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 106 79) = bselect (a (.wire 102 19)) (a (.virt 4609)) (a (.wire 106 15)) := by
  have c6987 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c6988 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6989 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6990 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6991 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6992 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_106_18 := arithEq_of_rows h (row := 106) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_18
  simp only [← c6987, ← c6988, ← c6989] at e_106_18
  have e_106_19 := arithEq_of_rows h (row := 106) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_106_19
  simp only [← c6990, ← c6991, ← c6992] at e_106_19
  have hr := e_106_19
  simp only [e_106_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 7) = bselect (a (.wire 102 19)) (a (.virt 4603)) (a (.wire 106 23)) := by
  have c6993 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6994 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6995 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6996 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6997 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c6998 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_0 := arithEq_of_rows h (row := 107) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_0
  simp only [← c6993, ← c6994, ← c6995] at e_107_0
  have e_107_1 := arithEq_of_rows h (row := 107) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_1
  simp only [← c6996, ← c6997, ← c6998] at e_107_1
  have hr := e_107_1
  simp only [e_107_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 15) = bselect (a (.wire 102 19)) (a (.virt 4604)) (a (.wire 106 31)) := by
  have c6999 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7000 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7001 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7002 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7003 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7004 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_2 := arithEq_of_rows h (row := 107) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_2
  simp only [← c6999, ← c7000, ← c7001] at e_107_2
  have e_107_3 := arithEq_of_rows h (row := 107) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_3
  simp only [← c7002, ← c7003, ← c7004] at e_107_3
  have hr := e_107_3
  simp only [e_107_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 43) = bor (a (.wire 97 39)) (a (.wire 106 35)) := by
  have c7005 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7006 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7007 := (publicBatchWrapper86_copies218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7008 := (publicBatchWrapper86_copies219 a h).1
  have c7009 := (publicBatchWrapper86_copies219 a h).2.1
  have c7010 := (publicBatchWrapper86_copies219 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_10 := arithEq_of_rows h (row := 97) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_10
  simp only [← c7008, ← c7009, k3, ← c7010] at e_97_10
  have e_105_2 := arithEq_of_rows h (row := 105) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_2
  simp only [← c7005, ← c7006, ← c7007] at e_105_2
  have hr := e_97_10
  simp only [e_105_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 19) = bnot (a (.wire 25 43)) := by
  have c7011 := (publicBatchWrapper86_copies219 a h).2.2.2.1
  have c7012 := (publicBatchWrapper86_copies219 a h).2.2.2.2.1
  have c7013 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_107_4 := arithEq_of_rows h (row := 107) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_4
  simp only [← c7011, k3, ← c7012, k3, ← c7013] at e_107_4
  have hr := e_107_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 23) = bnot (a (.wire 97 43)) := by
  have c7014 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.1
  have c7015 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.1
  have c7016 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_107_5 := arithEq_of_rows h (row := 107) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_5
  simp only [← c7014, k3, ← c7015, k3, ← c7016] at e_107_5
  have hr := e_107_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 23) = band (a (.wire 107 19)) (a (.wire 107 23)) := by
  have c7017 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.1
  have c7018 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7019 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_5 := arithEq_of_rows h (row := 102) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_5
  simp only [← c7017, ← c7018, ← c7019] at e_102_5
  have hr := e_102_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 31) = bselect (a (.wire 102 23)) (a (.virt 4805)) (a (.wire 106 47)) := by
  have c7020 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7021 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7022 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7023 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7024 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7025 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_6 := arithEq_of_rows h (row := 107) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_6
  simp only [← c7020, ← c7021, ← c7022] at e_107_6
  have e_107_7 := arithEq_of_rows h (row := 107) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_7
  simp only [← c7023, ← c7024, ← c7025] at e_107_7
  have hr := e_107_7
  simp only [e_107_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 39) = bselect (a (.wire 102 23)) (a (.virt 4806)) (a (.wire 106 55)) := by
  have c7026 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7027 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7028 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7029 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7030 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7031 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_8 := arithEq_of_rows h (row := 107) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_8
  simp only [← c7026, ← c7027, ← c7028] at e_107_8
  have e_107_9 := arithEq_of_rows h (row := 107) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_9
  simp only [← c7029, ← c7030, ← c7031] at e_107_9
  have hr := e_107_9
  simp only [e_107_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 47) = bselect (a (.wire 102 23)) (a (.virt 4807)) (a (.wire 106 63)) := by
  have c7032 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7033 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7034 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7035 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7036 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7037 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_10 := arithEq_of_rows h (row := 107) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_10
  simp only [← c7032, ← c7033, ← c7034] at e_107_10
  have e_107_11 := arithEq_of_rows h (row := 107) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_11
  simp only [← c7035, ← c7036, ← c7037] at e_107_11
  have hr := e_107_11
  simp only [e_107_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 55) = bselect (a (.wire 102 23)) (a (.virt 4808)) (a (.wire 106 71)) := by
  have c7038 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7039 := (publicBatchWrapper86_copies219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7040 := (publicBatchWrapper86_copies220 a h).1
  have c7041 := (publicBatchWrapper86_copies220 a h).2.1
  have c7042 := (publicBatchWrapper86_copies220 a h).2.2.1
  have c7043 := (publicBatchWrapper86_copies220 a h).2.2.2.1
  have e_107_12 := arithEq_of_rows h (row := 107) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_12
  simp only [← c7038, ← c7039, ← c7040] at e_107_12
  have e_107_13 := arithEq_of_rows h (row := 107) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_13
  simp only [← c7041, ← c7042, ← c7043] at e_107_13
  have hr := e_107_13
  simp only [e_107_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 63) = bselect (a (.wire 102 23)) (a (.virt 4809)) (a (.wire 106 79)) := by
  have c7044 := (publicBatchWrapper86_copies220 a h).2.2.2.2.1
  have c7045 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.1
  have c7046 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.1
  have c7047 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.1
  have c7048 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.1
  have c7049 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.1
  have e_107_14 := arithEq_of_rows h (row := 107) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_14
  simp only [← c7044, ← c7045, ← c7046] at e_107_14
  have e_107_15 := arithEq_of_rows h (row := 107) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_15
  simp only [← c7047, ← c7048, ← c7049] at e_107_15
  have hr := e_107_15
  simp only [e_107_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 71) = bselect (a (.wire 102 23)) (a (.virt 4803)) (a (.wire 107 7)) := by
  have c7050 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7051 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7052 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7053 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7054 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7055 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_16 := arithEq_of_rows h (row := 107) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_16
  simp only [← c7050, ← c7051, ← c7052] at e_107_16
  have e_107_17 := arithEq_of_rows h (row := 107) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_17
  simp only [← c7053, ← c7054, ← c7055] at e_107_17
  have hr := e_107_17
  simp only [e_107_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 107 79) = bselect (a (.wire 102 23)) (a (.virt 4804)) (a (.wire 107 15)) := by
  have c7056 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7057 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7058 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7059 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7060 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7061 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_107_18 := arithEq_of_rows h (row := 107) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_18
  simp only [← c7056, ← c7057, ← c7058] at e_107_18
  have e_107_19 := arithEq_of_rows h (row := 107) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_107_19
  simp only [← c7059, ← c7060, ← c7061] at e_107_19
  have hr := e_107_19
  simp only [e_107_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 47) = bor (a (.wire 97 43)) (a (.wire 107 19)) := by
  have c7062 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7063 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7064 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7065 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7066 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7067 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_11 := arithEq_of_rows h (row := 97) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_11
  simp only [← c7065, ← c7066, k3, ← c7067] at e_97_11
  have e_105_3 := arithEq_of_rows h (row := 105) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_3
  simp only [← c7062, ← c7063, ← c7064] at e_105_3
  have hr := e_97_11
  simp only [e_105_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 3) = bnot (a (.wire 27 7)) := by
  have c7068 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7069 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7070 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_108_0 := arithEq_of_rows h (row := 108) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_0
  simp only [← c7068, k3, ← c7069, k3, ← c7070] at e_108_0
  have hr := e_108_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 7) = bnot (a (.wire 97 47)) := by
  have c7071 := (publicBatchWrapper86_copies220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7072 := (publicBatchWrapper86_copies221 a h).1
  have c7073 := (publicBatchWrapper86_copies221 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_108_1 := arithEq_of_rows h (row := 108) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_1
  simp only [← c7071, k3, ← c7072, k3, ← c7073] at e_108_1
  have hr := e_108_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 27) = band (a (.wire 108 3)) (a (.wire 108 7)) := by
  have c7074 := (publicBatchWrapper86_copies221 a h).2.2.1
  have c7075 := (publicBatchWrapper86_copies221 a h).2.2.2.1
  have c7076 := (publicBatchWrapper86_copies221 a h).2.2.2.2.1
  have e_102_6 := arithEq_of_rows h (row := 102) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_6
  simp only [← c7074, ← c7075, ← c7076] at e_102_6
  have hr := e_102_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 15) = bselect (a (.wire 102 27)) (a (.virt 5005)) (a (.wire 107 31)) := by
  have c7077 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.1
  have c7078 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.1
  have c7079 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.1
  have c7080 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.1
  have c7081 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.1
  have c7082 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_108_2 := arithEq_of_rows h (row := 108) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_2
  simp only [← c7077, ← c7078, ← c7079] at e_108_2
  have e_108_3 := arithEq_of_rows h (row := 108) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_3
  simp only [← c7080, ← c7081, ← c7082] at e_108_3
  have hr := e_108_3
  simp only [e_108_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 23) = bselect (a (.wire 102 27)) (a (.virt 5006)) (a (.wire 107 39)) := by
  have c7083 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7084 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7085 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7086 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7087 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7088 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_108_4 := arithEq_of_rows h (row := 108) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_4
  simp only [← c7083, ← c7084, ← c7085] at e_108_4
  have e_108_5 := arithEq_of_rows h (row := 108) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_5
  simp only [← c7086, ← c7087, ← c7088] at e_108_5
  have hr := e_108_5
  simp only [e_108_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 31) = bselect (a (.wire 102 27)) (a (.virt 5007)) (a (.wire 107 47)) := by
  have c7089 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7090 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7091 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7092 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7093 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7094 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_108_6 := arithEq_of_rows h (row := 108) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_6
  simp only [← c7089, ← c7090, ← c7091] at e_108_6
  have e_108_7 := arithEq_of_rows h (row := 108) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_7
  simp only [← c7092, ← c7093, ← c7094] at e_108_7
  have hr := e_108_7
  simp only [e_108_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 39) = bselect (a (.wire 102 27)) (a (.virt 5008)) (a (.wire 107 55)) := by
  have c7095 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7096 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7097 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7098 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7099 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7100 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_108_8 := arithEq_of_rows h (row := 108) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_8
  simp only [← c7095, ← c7096, ← c7097] at e_108_8
  have e_108_9 := arithEq_of_rows h (row := 108) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_9
  simp only [← c7098, ← c7099, ← c7100] at e_108_9
  have hr := e_108_9
  simp only [e_108_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 47) = bselect (a (.wire 102 27)) (a (.virt 5009)) (a (.wire 107 63)) := by
  have c7101 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7102 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7103 := (publicBatchWrapper86_copies221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7104 := (publicBatchWrapper86_copies222 a h).1
  have c7105 := (publicBatchWrapper86_copies222 a h).2.1
  have c7106 := (publicBatchWrapper86_copies222 a h).2.2.1
  have e_108_10 := arithEq_of_rows h (row := 108) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_10
  simp only [← c7101, ← c7102, ← c7103] at e_108_10
  have e_108_11 := arithEq_of_rows h (row := 108) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_11
  simp only [← c7104, ← c7105, ← c7106] at e_108_11
  have hr := e_108_11
  simp only [e_108_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 55) = bselect (a (.wire 102 27)) (a (.virt 5003)) (a (.wire 107 71)) := by
  have c7107 := (publicBatchWrapper86_copies222 a h).2.2.2.1
  have c7108 := (publicBatchWrapper86_copies222 a h).2.2.2.2.1
  have c7109 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.1
  have c7110 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.1
  have c7111 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.1
  have c7112 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.1
  have e_108_12 := arithEq_of_rows h (row := 108) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_12
  simp only [← c7107, ← c7108, ← c7109] at e_108_12
  have e_108_13 := arithEq_of_rows h (row := 108) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_13
  simp only [← c7110, ← c7111, ← c7112] at e_108_13
  have hr := e_108_13
  simp only [e_108_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 63) = bselect (a (.wire 102 27)) (a (.virt 5004)) (a (.wire 107 79)) := by
  have c7113 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.1
  have c7114 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7115 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7116 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7117 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7118 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_108_14 := arithEq_of_rows h (row := 108) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_14
  simp only [← c7113, ← c7114, ← c7115] at e_108_14
  have e_108_15 := arithEq_of_rows h (row := 108) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_15
  simp only [← c7116, ← c7117, ← c7118] at e_108_15
  have hr := e_108_15
  simp only [e_108_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 51) = bor (a (.wire 97 47)) (a (.wire 108 3)) := by
  have c7119 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7120 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7121 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7122 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7123 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7124 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_12 := arithEq_of_rows h (row := 97) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_12
  simp only [← c7122, ← c7123, k3, ← c7124] at e_97_12
  have e_105_4 := arithEq_of_rows h (row := 105) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_4
  simp only [← c7119, ← c7120, ← c7121] at e_105_4
  have hr := e_97_12
  simp only [e_105_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 67) = bnot (a (.wire 27 51)) := by
  have c7125 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7126 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7127 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_108_16 := arithEq_of_rows h (row := 108) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_16
  simp only [← c7125, k3, ← c7126, k3, ← c7127] at e_108_16
  have hr := e_108_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 71) = bnot (a (.wire 97 51)) := by
  have c7128 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7129 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7130 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_108_17 := arithEq_of_rows h (row := 108) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_17
  simp only [← c7128, k3, ← c7129, k3, ← c7130] at e_108_17
  have hr := e_108_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 31) = band (a (.wire 108 67)) (a (.wire 108 71)) := by
  have c7131 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7132 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7133 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_7 := arithEq_of_rows h (row := 102) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_7
  simp only [← c7131, ← c7132, ← c7133] at e_102_7
  have hr := e_102_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 108 79) = bselect (a (.wire 102 31)) (a (.virt 5205)) (a (.wire 108 15)) := by
  have c7134 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7135 := (publicBatchWrapper86_copies222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7136 := (publicBatchWrapper86_copies223 a h).1
  have c7137 := (publicBatchWrapper86_copies223 a h).2.1
  have c7138 := (publicBatchWrapper86_copies223 a h).2.2.1
  have c7139 := (publicBatchWrapper86_copies223 a h).2.2.2.1
  have e_108_18 := arithEq_of_rows h (row := 108) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_18
  simp only [← c7134, ← c7135, ← c7136] at e_108_18
  have e_108_19 := arithEq_of_rows h (row := 108) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_108_19
  simp only [← c7137, ← c7138, ← c7139] at e_108_19
  have hr := e_108_19
  simp only [e_108_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 7) = bselect (a (.wire 102 31)) (a (.virt 5206)) (a (.wire 108 23)) := by
  have c7140 := (publicBatchWrapper86_copies223 a h).2.2.2.2.1
  have c7141 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.1
  have c7142 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.1
  have c7143 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.1
  have c7144 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.1
  have c7145 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.1
  have e_109_0 := arithEq_of_rows h (row := 109) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_0
  simp only [← c7140, ← c7141, ← c7142] at e_109_0
  have e_109_1 := arithEq_of_rows h (row := 109) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_1
  simp only [← c7143, ← c7144, ← c7145] at e_109_1
  have hr := e_109_1
  simp only [e_109_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 15) = bselect (a (.wire 102 31)) (a (.virt 5207)) (a (.wire 108 31)) := by
  have c7146 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7147 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7148 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7149 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7150 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7151 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_2 := arithEq_of_rows h (row := 109) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_2
  simp only [← c7146, ← c7147, ← c7148] at e_109_2
  have e_109_3 := arithEq_of_rows h (row := 109) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_3
  simp only [← c7149, ← c7150, ← c7151] at e_109_3
  have hr := e_109_3
  simp only [e_109_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 23) = bselect (a (.wire 102 31)) (a (.virt 5208)) (a (.wire 108 39)) := by
  have c7152 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7153 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7154 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7155 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7156 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7157 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_4 := arithEq_of_rows h (row := 109) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_4
  simp only [← c7152, ← c7153, ← c7154] at e_109_4
  have e_109_5 := arithEq_of_rows h (row := 109) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_5
  simp only [← c7155, ← c7156, ← c7157] at e_109_5
  have hr := e_109_5
  simp only [e_109_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 31) = bselect (a (.wire 102 31)) (a (.virt 5209)) (a (.wire 108 47)) := by
  have c7158 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7159 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7160 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7161 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7162 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7163 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_6 := arithEq_of_rows h (row := 109) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_6
  simp only [← c7158, ← c7159, ← c7160] at e_109_6
  have e_109_7 := arithEq_of_rows h (row := 109) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_7
  simp only [← c7161, ← c7162, ← c7163] at e_109_7
  have hr := e_109_7
  simp only [e_109_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 39) = bselect (a (.wire 102 31)) (a (.virt 5203)) (a (.wire 108 55)) := by
  have c7164 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7165 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7166 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7167 := (publicBatchWrapper86_copies223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7168 := (publicBatchWrapper86_copies224 a h).1
  have c7169 := (publicBatchWrapper86_copies224 a h).2.1
  have e_109_8 := arithEq_of_rows h (row := 109) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_8
  simp only [← c7164, ← c7165, ← c7166] at e_109_8
  have e_109_9 := arithEq_of_rows h (row := 109) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_9
  simp only [← c7167, ← c7168, ← c7169] at e_109_9
  have hr := e_109_9
  simp only [e_109_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 47) = bselect (a (.wire 102 31)) (a (.virt 5204)) (a (.wire 108 63)) := by
  have c7170 := (publicBatchWrapper86_copies224 a h).2.2.1
  have c7171 := (publicBatchWrapper86_copies224 a h).2.2.2.1
  have c7172 := (publicBatchWrapper86_copies224 a h).2.2.2.2.1
  have c7173 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.1
  have c7174 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.1
  have c7175 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.1
  have e_109_10 := arithEq_of_rows h (row := 109) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_10
  simp only [← c7170, ← c7171, ← c7172] at e_109_10
  have e_109_11 := arithEq_of_rows h (row := 109) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_11
  simp only [← c7173, ← c7174, ← c7175] at e_109_11
  have hr := e_109_11
  simp only [e_109_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 55) = bor (a (.wire 97 51)) (a (.wire 108 67)) := by
  have c7176 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.1
  have c7177 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.1
  have c7178 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7179 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7180 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7181 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_13 := arithEq_of_rows h (row := 97) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_13
  simp only [← c7179, ← c7180, k3, ← c7181] at e_97_13
  have e_105_5 := arithEq_of_rows h (row := 105) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_5
  simp only [← c7176, ← c7177, ← c7178] at e_105_5
  have hr := e_97_13
  simp only [e_105_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 51) = bnot (a (.wire 29 15)) := by
  have c7182 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7183 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7184 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_109_12 := arithEq_of_rows h (row := 109) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_12
  simp only [← c7182, k3, ← c7183, k3, ← c7184] at e_109_12
  have hr := e_109_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 55) = bnot (a (.wire 97 55)) := by
  have c7185 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7186 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7187 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_109_13 := arithEq_of_rows h (row := 109) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_13
  simp only [← c7185, k3, ← c7186, k3, ← c7187] at e_109_13
  have hr := e_109_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 35) = band (a (.wire 109 51)) (a (.wire 109 55)) := by
  have c7188 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7189 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7190 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_8 := arithEq_of_rows h (row := 102) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_8
  simp only [← c7188, ← c7189, ← c7190] at e_102_8
  have hr := e_102_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 63) = bselect (a (.wire 102 35)) (a (.virt 5405)) (a (.wire 108 79)) := by
  have c7191 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7192 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7193 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7194 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7195 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7196 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_109_14 := arithEq_of_rows h (row := 109) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_14
  simp only [← c7191, ← c7192, ← c7193] at e_109_14
  have e_109_15 := arithEq_of_rows h (row := 109) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_15
  simp only [← c7194, ← c7195, ← c7196] at e_109_15
  have hr := e_109_15
  simp only [e_109_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 71) = bselect (a (.wire 102 35)) (a (.virt 5406)) (a (.wire 109 7)) := by
  have c7197 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7198 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7199 := (publicBatchWrapper86_copies224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7200 := (publicBatchWrapper86_copies225 a h).1
  have c7201 := (publicBatchWrapper86_copies225 a h).2.1
  have c7202 := (publicBatchWrapper86_copies225 a h).2.2.1
  have e_109_16 := arithEq_of_rows h (row := 109) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_16
  simp only [← c7197, ← c7198, ← c7199] at e_109_16
  have e_109_17 := arithEq_of_rows h (row := 109) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_17
  simp only [← c7200, ← c7201, ← c7202] at e_109_17
  have hr := e_109_17
  simp only [e_109_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 109 79) = bselect (a (.wire 102 35)) (a (.virt 5407)) (a (.wire 109 15)) := by
  have c7203 := (publicBatchWrapper86_copies225 a h).2.2.2.1
  have c7204 := (publicBatchWrapper86_copies225 a h).2.2.2.2.1
  have c7205 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.1
  have c7206 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.1
  have c7207 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.1
  have c7208 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.1
  have e_109_18 := arithEq_of_rows h (row := 109) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_18
  simp only [← c7203, ← c7204, ← c7205] at e_109_18
  have e_109_19 := arithEq_of_rows h (row := 109) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_109_19
  simp only [← c7206, ← c7207, ← c7208] at e_109_19
  have hr := e_109_19
  simp only [e_109_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 7) = bselect (a (.wire 102 35)) (a (.virt 5408)) (a (.wire 109 23)) := by
  have c7209 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.1
  have c7210 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7211 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7212 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7213 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7214 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_0 := arithEq_of_rows h (row := 110) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_0
  simp only [← c7209, ← c7210, ← c7211] at e_110_0
  have e_110_1 := arithEq_of_rows h (row := 110) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_1
  simp only [← c7212, ← c7213, ← c7214] at e_110_1
  have hr := e_110_1
  simp only [e_110_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 15) = bselect (a (.wire 102 35)) (a (.virt 5409)) (a (.wire 109 31)) := by
  have c7215 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7216 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7217 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7218 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7219 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7220 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_2 := arithEq_of_rows h (row := 110) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_2
  simp only [← c7215, ← c7216, ← c7217] at e_110_2
  have e_110_3 := arithEq_of_rows h (row := 110) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_3
  simp only [← c7218, ← c7219, ← c7220] at e_110_3
  have hr := e_110_3
  simp only [e_110_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 23) = bselect (a (.wire 102 35)) (a (.virt 5403)) (a (.wire 109 39)) := by
  have c7221 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7222 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7223 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7224 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7225 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7226 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_4 := arithEq_of_rows h (row := 110) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_4
  simp only [← c7221, ← c7222, ← c7223] at e_110_4
  have e_110_5 := arithEq_of_rows h (row := 110) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_5
  simp only [← c7224, ← c7225, ← c7226] at e_110_5
  have hr := e_110_5
  simp only [e_110_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 31) = bselect (a (.wire 102 35)) (a (.virt 5404)) (a (.wire 109 47)) := by
  have c7227 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7228 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7229 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7230 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7231 := (publicBatchWrapper86_copies225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7232 := (publicBatchWrapper86_copies226 a h).1
  have e_110_6 := arithEq_of_rows h (row := 110) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_6
  simp only [← c7227, ← c7228, ← c7229] at e_110_6
  have e_110_7 := arithEq_of_rows h (row := 110) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_7
  simp only [← c7230, ← c7231, ← c7232] at e_110_7
  have hr := e_110_7
  simp only [e_110_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 59) = bor (a (.wire 97 55)) (a (.wire 109 51)) := by
  have c7233 := (publicBatchWrapper86_copies226 a h).2.1
  have c7234 := (publicBatchWrapper86_copies226 a h).2.2.1
  have c7235 := (publicBatchWrapper86_copies226 a h).2.2.2.1
  have c7236 := (publicBatchWrapper86_copies226 a h).2.2.2.2.1
  have c7237 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.1
  have c7238 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_14 := arithEq_of_rows h (row := 97) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_14
  simp only [← c7236, ← c7237, k3, ← c7238] at e_97_14
  have e_105_6 := arithEq_of_rows h (row := 105) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_6
  simp only [← c7233, ← c7234, ← c7235] at e_105_6
  have hr := e_97_14
  simp only [e_105_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 35) = bnot (a (.wire 29 59)) := by
  have c7239 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.1
  have c7240 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.1
  have c7241 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_110_8 := arithEq_of_rows h (row := 110) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_8
  simp only [← c7239, k3, ← c7240, k3, ← c7241] at e_110_8
  have hr := e_110_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 39) = bnot (a (.wire 97 59)) := by
  have c7242 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7243 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7244 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_110_9 := arithEq_of_rows h (row := 110) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_9
  simp only [← c7242, k3, ← c7243, k3, ← c7244] at e_110_9
  have hr := e_110_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 39) = band (a (.wire 110 35)) (a (.wire 110 39)) := by
  have c7245 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7246 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7247 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_9 := arithEq_of_rows h (row := 102) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_9
  simp only [← c7245, ← c7246, ← c7247] at e_102_9
  have hr := e_102_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 47) = bselect (a (.wire 102 39)) (a (.virt 5605)) (a (.wire 109 63)) := by
  have c7248 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7249 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7250 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7251 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7252 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7253 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_10 := arithEq_of_rows h (row := 110) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_10
  simp only [← c7248, ← c7249, ← c7250] at e_110_10
  have e_110_11 := arithEq_of_rows h (row := 110) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_11
  simp only [← c7251, ← c7252, ← c7253] at e_110_11
  have hr := e_110_11
  simp only [e_110_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 55) = bselect (a (.wire 102 39)) (a (.virt 5606)) (a (.wire 109 71)) := by
  have c7254 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7255 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7256 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7257 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7258 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7259 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_12 := arithEq_of_rows h (row := 110) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_12
  simp only [← c7254, ← c7255, ← c7256] at e_110_12
  have e_110_13 := arithEq_of_rows h (row := 110) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_13
  simp only [← c7257, ← c7258, ← c7259] at e_110_13
  have hr := e_110_13
  simp only [e_110_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 63) = bselect (a (.wire 102 39)) (a (.virt 5607)) (a (.wire 109 79)) := by
  have c7260 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7261 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7262 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7263 := (publicBatchWrapper86_copies226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7264 := (publicBatchWrapper86_copies227 a h).1
  have c7265 := (publicBatchWrapper86_copies227 a h).2.1
  have e_110_14 := arithEq_of_rows h (row := 110) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_14
  simp only [← c7260, ← c7261, ← c7262] at e_110_14
  have e_110_15 := arithEq_of_rows h (row := 110) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_15
  simp only [← c7263, ← c7264, ← c7265] at e_110_15
  have hr := e_110_15
  simp only [e_110_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 71) = bselect (a (.wire 102 39)) (a (.virt 5608)) (a (.wire 110 7)) := by
  have c7266 := (publicBatchWrapper86_copies227 a h).2.2.1
  have c7267 := (publicBatchWrapper86_copies227 a h).2.2.2.1
  have c7268 := (publicBatchWrapper86_copies227 a h).2.2.2.2.1
  have c7269 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.1
  have c7270 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.1
  have c7271 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.1
  have e_110_16 := arithEq_of_rows h (row := 110) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_16
  simp only [← c7266, ← c7267, ← c7268] at e_110_16
  have e_110_17 := arithEq_of_rows h (row := 110) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_17
  simp only [← c7269, ← c7270, ← c7271] at e_110_17
  have hr := e_110_17
  simp only [e_110_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 110 79) = bselect (a (.wire 102 39)) (a (.virt 5609)) (a (.wire 110 15)) := by
  have c7272 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.1
  have c7273 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.1
  have c7274 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7275 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7276 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7277 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_110_18 := arithEq_of_rows h (row := 110) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_18
  simp only [← c7272, ← c7273, ← c7274] at e_110_18
  have e_110_19 := arithEq_of_rows h (row := 110) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_110_19
  simp only [← c7275, ← c7276, ← c7277] at e_110_19
  have hr := e_110_19
  simp only [e_110_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 7) = bselect (a (.wire 102 39)) (a (.virt 5603)) (a (.wire 110 23)) := by
  have c7278 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7279 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7280 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7281 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7282 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7283 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_0 := arithEq_of_rows h (row := 111) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_0
  simp only [← c7278, ← c7279, ← c7280] at e_111_0
  have e_111_1 := arithEq_of_rows h (row := 111) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_1
  simp only [← c7281, ← c7282, ← c7283] at e_111_1
  have hr := e_111_1
  simp only [e_111_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 15) = bselect (a (.wire 102 39)) (a (.virt 5604)) (a (.wire 110 31)) := by
  have c7284 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7285 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7286 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7287 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7288 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7289 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_2 := arithEq_of_rows h (row := 111) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_2
  simp only [← c7284, ← c7285, ← c7286] at e_111_2
  have e_111_3 := arithEq_of_rows h (row := 111) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_3
  simp only [← c7287, ← c7288, ← c7289] at e_111_3
  have hr := e_111_3
  simp only [e_111_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 63) = bor (a (.wire 97 59)) (a (.wire 110 35)) := by
  have c7290 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7291 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7292 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7293 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7294 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7295 := (publicBatchWrapper86_copies227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_15 := arithEq_of_rows h (row := 97) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_15
  simp only [← c7293, ← c7294, k3, ← c7295] at e_97_15
  have e_105_7 := arithEq_of_rows h (row := 105) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_7
  simp only [← c7290, ← c7291, ← c7292] at e_105_7
  have hr := e_97_15
  simp only [e_105_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 19) = bnot (a (.wire 30 23)) := by
  have c7296 := (publicBatchWrapper86_copies228 a h).1
  have c7297 := (publicBatchWrapper86_copies228 a h).2.1
  have c7298 := (publicBatchWrapper86_copies228 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_111_4 := arithEq_of_rows h (row := 111) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_4
  simp only [← c7296, k3, ← c7297, k3, ← c7298] at e_111_4
  have hr := e_111_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 23) = bnot (a (.wire 97 63)) := by
  have c7299 := (publicBatchWrapper86_copies228 a h).2.2.2.1
  have c7300 := (publicBatchWrapper86_copies228 a h).2.2.2.2.1
  have c7301 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_111_5 := arithEq_of_rows h (row := 111) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_5
  simp only [← c7299, k3, ← c7300, k3, ← c7301] at e_111_5
  have hr := e_111_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 43) = band (a (.wire 111 19)) (a (.wire 111 23)) := by
  have c7302 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.1
  have c7303 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.1
  have c7304 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.1
  have e_102_10 := arithEq_of_rows h (row := 102) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_10
  simp only [← c7302, ← c7303, ← c7304] at e_102_10
  have hr := e_102_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 31) = bselect (a (.wire 102 43)) (a (.virt 5805)) (a (.wire 110 47)) := by
  have c7305 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.1
  have c7306 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7307 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7308 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7309 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7310 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_6 := arithEq_of_rows h (row := 111) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_6
  simp only [← c7305, ← c7306, ← c7307] at e_111_6
  have e_111_7 := arithEq_of_rows h (row := 111) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_7
  simp only [← c7308, ← c7309, ← c7310] at e_111_7
  have hr := e_111_7
  simp only [e_111_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 39) = bselect (a (.wire 102 43)) (a (.virt 5806)) (a (.wire 110 55)) := by
  have c7311 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7312 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7313 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7314 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7315 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7316 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_8 := arithEq_of_rows h (row := 111) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_8
  simp only [← c7311, ← c7312, ← c7313] at e_111_8
  have e_111_9 := arithEq_of_rows h (row := 111) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_9
  simp only [← c7314, ← c7315, ← c7316] at e_111_9
  have hr := e_111_9
  simp only [e_111_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 47) = bselect (a (.wire 102 43)) (a (.virt 5807)) (a (.wire 110 63)) := by
  have c7317 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7318 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7319 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7320 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7321 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7322 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_10 := arithEq_of_rows h (row := 111) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_10
  simp only [← c7317, ← c7318, ← c7319] at e_111_10
  have e_111_11 := arithEq_of_rows h (row := 111) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_11
  simp only [← c7320, ← c7321, ← c7322] at e_111_11
  have hr := e_111_11
  simp only [e_111_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 55) = bselect (a (.wire 102 43)) (a (.virt 5808)) (a (.wire 110 71)) := by
  have c7323 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7324 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7325 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7326 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7327 := (publicBatchWrapper86_copies228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7328 := (publicBatchWrapper86_copies229 a h).1
  have e_111_12 := arithEq_of_rows h (row := 111) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_12
  simp only [← c7323, ← c7324, ← c7325] at e_111_12
  have e_111_13 := arithEq_of_rows h (row := 111) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_13
  simp only [← c7326, ← c7327, ← c7328] at e_111_13
  have hr := e_111_13
  simp only [e_111_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 63) = bselect (a (.wire 102 43)) (a (.virt 5809)) (a (.wire 110 79)) := by
  have c7329 := (publicBatchWrapper86_copies229 a h).2.1
  have c7330 := (publicBatchWrapper86_copies229 a h).2.2.1
  have c7331 := (publicBatchWrapper86_copies229 a h).2.2.2.1
  have c7332 := (publicBatchWrapper86_copies229 a h).2.2.2.2.1
  have c7333 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.1
  have c7334 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.1
  have e_111_14 := arithEq_of_rows h (row := 111) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_14
  simp only [← c7329, ← c7330, ← c7331] at e_111_14
  have e_111_15 := arithEq_of_rows h (row := 111) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_15
  simp only [← c7332, ← c7333, ← c7334] at e_111_15
  have hr := e_111_15
  simp only [e_111_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 71) = bselect (a (.wire 102 43)) (a (.virt 5803)) (a (.wire 111 7)) := by
  have c7335 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.1
  have c7336 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.1
  have c7337 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.1
  have c7338 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7339 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7340 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_16 := arithEq_of_rows h (row := 111) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_16
  simp only [← c7335, ← c7336, ← c7337] at e_111_16
  have e_111_17 := arithEq_of_rows h (row := 111) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_17
  simp only [← c7338, ← c7339, ← c7340] at e_111_17
  have hr := e_111_17
  simp only [e_111_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 111 79) = bselect (a (.wire 102 43)) (a (.virt 5804)) (a (.wire 111 15)) := by
  have c7341 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7342 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7343 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7344 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7345 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7346 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_111_18 := arithEq_of_rows h (row := 111) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_18
  simp only [← c7341, ← c7342, ← c7343] at e_111_18
  have e_111_19 := arithEq_of_rows h (row := 111) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_111_19
  simp only [← c7344, ← c7345, ← c7346] at e_111_19
  have hr := e_111_19
  simp only [e_111_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 67) = bor (a (.wire 97 63)) (a (.wire 111 19)) := by
  have c7347 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7348 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7349 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7350 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7351 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7352 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_16 := arithEq_of_rows h (row := 97) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_16
  simp only [← c7350, ← c7351, k3, ← c7352] at e_97_16
  have e_105_8 := arithEq_of_rows h (row := 105) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_8
  simp only [← c7347, ← c7348, ← c7349] at e_105_8
  have hr := e_97_16
  simp only [e_105_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 3) = bnot (a (.wire 30 67)) := by
  have c7353 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7354 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7355 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_112_0 := arithEq_of_rows h (row := 112) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_0
  simp only [← c7353, k3, ← c7354, k3, ← c7355] at e_112_0
  have hr := e_112_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 7) = bnot (a (.wire 97 67)) := by
  have c7356 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7357 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7358 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_112_1 := arithEq_of_rows h (row := 112) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_1
  simp only [← c7356, k3, ← c7357, k3, ← c7358] at e_112_1
  have hr := e_112_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 47) = band (a (.wire 112 3)) (a (.wire 112 7)) := by
  have c7359 := (publicBatchWrapper86_copies229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7360 := (publicBatchWrapper86_copies230 a h).1
  have c7361 := (publicBatchWrapper86_copies230 a h).2.1
  have e_102_11 := arithEq_of_rows h (row := 102) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_11
  simp only [← c7359, ← c7360, ← c7361] at e_102_11
  have hr := e_102_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 15) = bselect (a (.wire 102 47)) (a (.virt 6005)) (a (.wire 111 31)) := by
  have c7362 := (publicBatchWrapper86_copies230 a h).2.2.1
  have c7363 := (publicBatchWrapper86_copies230 a h).2.2.2.1
  have c7364 := (publicBatchWrapper86_copies230 a h).2.2.2.2.1
  have c7365 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.1
  have c7366 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.1
  have c7367 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.1
  have e_112_2 := arithEq_of_rows h (row := 112) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_2
  simp only [← c7362, ← c7363, ← c7364] at e_112_2
  have e_112_3 := arithEq_of_rows h (row := 112) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_3
  simp only [← c7365, ← c7366, ← c7367] at e_112_3
  have hr := e_112_3
  simp only [e_112_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 23) = bselect (a (.wire 102 47)) (a (.virt 6006)) (a (.wire 111 39)) := by
  have c7368 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.1
  have c7369 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.1
  have c7370 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7371 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7372 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7373 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_4 := arithEq_of_rows h (row := 112) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_4
  simp only [← c7368, ← c7369, ← c7370] at e_112_4
  have e_112_5 := arithEq_of_rows h (row := 112) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_5
  simp only [← c7371, ← c7372, ← c7373] at e_112_5
  have hr := e_112_5
  simp only [e_112_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 31) = bselect (a (.wire 102 47)) (a (.virt 6007)) (a (.wire 111 47)) := by
  have c7374 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7375 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7376 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7377 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7378 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7379 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_6 := arithEq_of_rows h (row := 112) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_6
  simp only [← c7374, ← c7375, ← c7376] at e_112_6
  have e_112_7 := arithEq_of_rows h (row := 112) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_7
  simp only [← c7377, ← c7378, ← c7379] at e_112_7
  have hr := e_112_7
  simp only [e_112_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 39) = bselect (a (.wire 102 47)) (a (.virt 6008)) (a (.wire 111 55)) := by
  have c7380 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7381 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7382 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7383 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7384 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7385 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_8 := arithEq_of_rows h (row := 112) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_8
  simp only [← c7380, ← c7381, ← c7382] at e_112_8
  have e_112_9 := arithEq_of_rows h (row := 112) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_9
  simp only [← c7383, ← c7384, ← c7385] at e_112_9
  have hr := e_112_9
  simp only [e_112_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 47) = bselect (a (.wire 102 47)) (a (.virt 6009)) (a (.wire 111 63)) := by
  have c7386 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7387 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7388 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7389 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7390 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7391 := (publicBatchWrapper86_copies230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_112_10 := arithEq_of_rows h (row := 112) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_10
  simp only [← c7386, ← c7387, ← c7388] at e_112_10
  have e_112_11 := arithEq_of_rows h (row := 112) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_11
  simp only [← c7389, ← c7390, ← c7391] at e_112_11
  have hr := e_112_11
  simp only [e_112_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 55) = bselect (a (.wire 102 47)) (a (.virt 6003)) (a (.wire 111 71)) := by
  have c7392 := (publicBatchWrapper86_copies231 a h).1
  have c7393 := (publicBatchWrapper86_copies231 a h).2.1
  have c7394 := (publicBatchWrapper86_copies231 a h).2.2.1
  have c7395 := (publicBatchWrapper86_copies231 a h).2.2.2.1
  have c7396 := (publicBatchWrapper86_copies231 a h).2.2.2.2.1
  have c7397 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.1
  have e_112_12 := arithEq_of_rows h (row := 112) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_12
  simp only [← c7392, ← c7393, ← c7394] at e_112_12
  have e_112_13 := arithEq_of_rows h (row := 112) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_13
  simp only [← c7395, ← c7396, ← c7397] at e_112_13
  have hr := e_112_13
  simp only [e_112_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 63) = bselect (a (.wire 102 47)) (a (.virt 6004)) (a (.wire 111 79)) := by
  have c7398 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.1
  have c7399 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.1
  have c7400 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.1
  have c7401 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.1
  have c7402 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7403 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_112_14 := arithEq_of_rows h (row := 112) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_14
  simp only [← c7398, ← c7399, ← c7400] at e_112_14
  have e_112_15 := arithEq_of_rows h (row := 112) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_15
  simp only [← c7401, ← c7402, ← c7403] at e_112_15
  have hr := e_112_15
  simp only [e_112_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 71) = bor (a (.wire 97 67)) (a (.wire 112 3)) := by
  have c7404 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7405 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7406 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7407 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7408 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7409 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_17 := arithEq_of_rows h (row := 97) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_17
  simp only [← c7407, ← c7408, k3, ← c7409] at e_97_17
  have e_105_9 := arithEq_of_rows h (row := 105) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_9
  simp only [← c7404, ← c7405, ← c7406] at e_105_9
  have hr := e_97_17
  simp only [e_105_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 67) = bnot (a (.wire 32 31)) := by
  have c7410 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7411 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7412 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_112_16 := arithEq_of_rows h (row := 112) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_16
  simp only [← c7410, k3, ← c7411, k3, ← c7412] at e_112_16
  have hr := e_112_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 71) = bnot (a (.wire 97 71)) := by
  have c7413 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7414 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7415 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_112_17 := arithEq_of_rows h (row := 112) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_17
  simp only [← c7413, k3, ← c7414, k3, ← c7415] at e_112_17
  have hr := e_112_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 51) = band (a (.wire 112 67)) (a (.wire 112 71)) := by
  have c7416 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7417 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7418 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_12 := arithEq_of_rows h (row := 102) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_12
  simp only [← c7416, ← c7417, ← c7418] at e_102_12
  have hr := e_102_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 112 79) = bselect (a (.wire 102 51)) (a (.virt 6205)) (a (.wire 112 15)) := by
  have c7419 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7420 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7421 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7422 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7423 := (publicBatchWrapper86_copies231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7424 := (publicBatchWrapper86_copies232 a h).1
  have e_112_18 := arithEq_of_rows h (row := 112) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_18
  simp only [← c7419, ← c7420, ← c7421] at e_112_18
  have e_112_19 := arithEq_of_rows h (row := 112) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_112_19
  simp only [← c7422, ← c7423, ← c7424] at e_112_19
  have hr := e_112_19
  simp only [e_112_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 7) = bselect (a (.wire 102 51)) (a (.virt 6206)) (a (.wire 112 23)) := by
  have c7425 := (publicBatchWrapper86_copies232 a h).2.1
  have c7426 := (publicBatchWrapper86_copies232 a h).2.2.1
  have c7427 := (publicBatchWrapper86_copies232 a h).2.2.2.1
  have c7428 := (publicBatchWrapper86_copies232 a h).2.2.2.2.1
  have c7429 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.1
  have c7430 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.1
  have e_113_0 := arithEq_of_rows h (row := 113) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_0
  simp only [← c7425, ← c7426, ← c7427] at e_113_0
  have e_113_1 := arithEq_of_rows h (row := 113) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_1
  simp only [← c7428, ← c7429, ← c7430] at e_113_1
  have hr := e_113_1
  simp only [e_113_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 15) = bselect (a (.wire 102 51)) (a (.virt 6207)) (a (.wire 112 31)) := by
  have c7431 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.1
  have c7432 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.1
  have c7433 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.1
  have c7434 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7435 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7436 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_2 := arithEq_of_rows h (row := 113) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_2
  simp only [← c7431, ← c7432, ← c7433] at e_113_2
  have e_113_3 := arithEq_of_rows h (row := 113) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_3
  simp only [← c7434, ← c7435, ← c7436] at e_113_3
  have hr := e_113_3
  simp only [e_113_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 23) = bselect (a (.wire 102 51)) (a (.virt 6208)) (a (.wire 112 39)) := by
  have c7437 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7438 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7439 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7440 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7441 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7442 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_4 := arithEq_of_rows h (row := 113) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_4
  simp only [← c7437, ← c7438, ← c7439] at e_113_4
  have e_113_5 := arithEq_of_rows h (row := 113) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_5
  simp only [← c7440, ← c7441, ← c7442] at e_113_5
  have hr := e_113_5
  simp only [e_113_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 31) = bselect (a (.wire 102 51)) (a (.virt 6209)) (a (.wire 112 47)) := by
  have c7443 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7444 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7445 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7446 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7447 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7448 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_6 := arithEq_of_rows h (row := 113) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_6
  simp only [← c7443, ← c7444, ← c7445] at e_113_6
  have e_113_7 := arithEq_of_rows h (row := 113) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_7
  simp only [← c7446, ← c7447, ← c7448] at e_113_7
  have hr := e_113_7
  simp only [e_113_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 39) = bselect (a (.wire 102 51)) (a (.virt 6203)) (a (.wire 112 55)) := by
  have c7449 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7450 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7451 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7452 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7453 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7454 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_8 := arithEq_of_rows h (row := 113) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_8
  simp only [← c7449, ← c7450, ← c7451] at e_113_8
  have e_113_9 := arithEq_of_rows h (row := 113) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_9
  simp only [← c7452, ← c7453, ← c7454] at e_113_9
  have hr := e_113_9
  simp only [e_113_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 47) = bselect (a (.wire 102 51)) (a (.virt 6204)) (a (.wire 112 63)) := by
  have c7455 := (publicBatchWrapper86_copies232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7456 := (publicBatchWrapper86_copies233 a h).1
  have c7457 := (publicBatchWrapper86_copies233 a h).2.1
  have c7458 := (publicBatchWrapper86_copies233 a h).2.2.1
  have c7459 := (publicBatchWrapper86_copies233 a h).2.2.2.1
  have c7460 := (publicBatchWrapper86_copies233 a h).2.2.2.2.1
  have e_113_10 := arithEq_of_rows h (row := 113) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_10
  simp only [← c7455, ← c7456, ← c7457] at e_113_10
  have e_113_11 := arithEq_of_rows h (row := 113) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_11
  simp only [← c7458, ← c7459, ← c7460] at e_113_11
  have hr := e_113_11
  simp only [e_113_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 75) = bor (a (.wire 97 71)) (a (.wire 112 67)) := by
  have c7461 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.1
  have c7462 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.1
  have c7463 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.1
  have c7464 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.1
  have c7465 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.1
  have c7466 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_18 := arithEq_of_rows h (row := 97) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_18
  simp only [← c7464, ← c7465, k3, ← c7466] at e_97_18
  have e_105_10 := arithEq_of_rows h (row := 105) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_10
  simp only [← c7461, ← c7462, ← c7463] at e_105_10
  have hr := e_97_18
  simp only [e_105_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 51) = bnot (a (.wire 32 75)) := by
  have c7467 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7468 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7469 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_113_12 := arithEq_of_rows h (row := 113) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_12
  simp only [← c7467, k3, ← c7468, k3, ← c7469] at e_113_12
  have hr := e_113_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 55) = bnot (a (.wire 97 75)) := by
  have c7470 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7471 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7472 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_113_13 := arithEq_of_rows h (row := 113) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_13
  simp only [← c7470, k3, ← c7471, k3, ← c7472] at e_113_13
  have hr := e_113_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 55) = band (a (.wire 113 51)) (a (.wire 113 55)) := by
  have c7473 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7474 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7475 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_13 := arithEq_of_rows h (row := 102) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_13
  simp only [← c7473, ← c7474, ← c7475] at e_102_13
  have hr := e_102_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 63) = bselect (a (.wire 102 55)) (a (.virt 6405)) (a (.wire 112 79)) := by
  have c7476 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7477 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7478 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7479 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7480 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7481 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_113_14 := arithEq_of_rows h (row := 113) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_14
  simp only [← c7476, ← c7477, ← c7478] at e_113_14
  have e_113_15 := arithEq_of_rows h (row := 113) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_15
  simp only [← c7479, ← c7480, ← c7481] at e_113_15
  have hr := e_113_15
  simp only [e_113_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 71) = bselect (a (.wire 102 55)) (a (.virt 6406)) (a (.wire 113 7)) := by
  have c7482 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7483 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7484 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7485 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7486 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7487 := (publicBatchWrapper86_copies233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_113_16 := arithEq_of_rows h (row := 113) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_16
  simp only [← c7482, ← c7483, ← c7484] at e_113_16
  have e_113_17 := arithEq_of_rows h (row := 113) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_17
  simp only [← c7485, ← c7486, ← c7487] at e_113_17
  have hr := e_113_17
  simp only [e_113_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 113 79) = bselect (a (.wire 102 55)) (a (.virt 6407)) (a (.wire 113 15)) := by
  have c7488 := (publicBatchWrapper86_copies234 a h).1
  have c7489 := (publicBatchWrapper86_copies234 a h).2.1
  have c7490 := (publicBatchWrapper86_copies234 a h).2.2.1
  have c7491 := (publicBatchWrapper86_copies234 a h).2.2.2.1
  have c7492 := (publicBatchWrapper86_copies234 a h).2.2.2.2.1
  have c7493 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.1
  have e_113_18 := arithEq_of_rows h (row := 113) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_18
  simp only [← c7488, ← c7489, ← c7490] at e_113_18
  have e_113_19 := arithEq_of_rows h (row := 113) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_113_19
  simp only [← c7491, ← c7492, ← c7493] at e_113_19
  have hr := e_113_19
  simp only [e_113_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 7) = bselect (a (.wire 102 55)) (a (.virt 6408)) (a (.wire 113 23)) := by
  have c7494 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.1
  have c7495 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.1
  have c7496 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.1
  have c7497 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.1
  have c7498 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7499 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_0 := arithEq_of_rows h (row := 114) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_0
  simp only [← c7494, ← c7495, ← c7496] at e_114_0
  have e_114_1 := arithEq_of_rows h (row := 114) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_1
  simp only [← c7497, ← c7498, ← c7499] at e_114_1
  have hr := e_114_1
  simp only [e_114_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 15) = bselect (a (.wire 102 55)) (a (.virt 6409)) (a (.wire 113 31)) := by
  have c7500 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7501 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7502 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7503 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7504 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7505 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_2 := arithEq_of_rows h (row := 114) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_2
  simp only [← c7500, ← c7501, ← c7502] at e_114_2
  have e_114_3 := arithEq_of_rows h (row := 114) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_3
  simp only [← c7503, ← c7504, ← c7505] at e_114_3
  have hr := e_114_3
  simp only [e_114_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 23) = bselect (a (.wire 102 55)) (a (.virt 6403)) (a (.wire 113 39)) := by
  have c7506 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7507 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7508 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7509 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7510 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7511 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_4 := arithEq_of_rows h (row := 114) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_4
  simp only [← c7506, ← c7507, ← c7508] at e_114_4
  have e_114_5 := arithEq_of_rows h (row := 114) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_5
  simp only [← c7509, ← c7510, ← c7511] at e_114_5
  have hr := e_114_5
  simp only [e_114_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 31) = bselect (a (.wire 102 55)) (a (.virt 6404)) (a (.wire 113 47)) := by
  have c7512 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7513 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7514 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7515 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7516 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7517 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_6 := arithEq_of_rows h (row := 114) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_6
  simp only [← c7512, ← c7513, ← c7514] at e_114_6
  have e_114_7 := arithEq_of_rows h (row := 114) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_7
  simp only [← c7515, ← c7516, ← c7517] at e_114_7
  have hr := e_114_7
  simp only [e_114_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 97 79) = bor (a (.wire 97 75)) (a (.wire 113 51)) := by
  have c7518 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7519 := (publicBatchWrapper86_copies234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7520 := (publicBatchWrapper86_copies235 a h).1
  have c7521 := (publicBatchWrapper86_copies235 a h).2.1
  have c7522 := (publicBatchWrapper86_copies235 a h).2.2.1
  have c7523 := (publicBatchWrapper86_copies235 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_97_19 := arithEq_of_rows h (row := 97) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_97_19
  simp only [← c7521, ← c7522, k3, ← c7523] at e_97_19
  have e_105_11 := arithEq_of_rows h (row := 105) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_11
  simp only [← c7518, ← c7519, ← c7520] at e_105_11
  have hr := e_97_19
  simp only [e_105_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 35) = bnot (a (.wire 34 39)) := by
  have c7524 := (publicBatchWrapper86_copies235 a h).2.2.2.2.1
  have c7525 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.1
  have c7526 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_114_8 := arithEq_of_rows h (row := 114) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_8
  simp only [← c7524, k3, ← c7525, k3, ← c7526] at e_114_8
  have hr := e_114_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 39) = bnot (a (.wire 97 79)) := by
  have c7527 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.1
  have c7528 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.1
  have c7529 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_114_9 := arithEq_of_rows h (row := 114) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_9
  simp only [← c7527, k3, ← c7528, k3, ← c7529] at e_114_9
  have hr := e_114_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 59) = band (a (.wire 114 35)) (a (.wire 114 39)) := by
  have c7530 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7531 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7532 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_14 := arithEq_of_rows h (row := 102) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_14
  simp only [← c7530, ← c7531, ← c7532] at e_102_14
  have hr := e_102_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 47) = bselect (a (.wire 102 59)) (a (.virt 6605)) (a (.wire 113 63)) := by
  have c7533 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7534 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7535 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7536 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7537 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7538 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_10 := arithEq_of_rows h (row := 114) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_10
  simp only [← c7533, ← c7534, ← c7535] at e_114_10
  have e_114_11 := arithEq_of_rows h (row := 114) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_11
  simp only [← c7536, ← c7537, ← c7538] at e_114_11
  have hr := e_114_11
  simp only [e_114_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 55) = bselect (a (.wire 102 59)) (a (.virt 6606)) (a (.wire 113 71)) := by
  have c7539 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7540 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7541 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7542 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7543 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7544 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_12 := arithEq_of_rows h (row := 114) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_12
  simp only [← c7539, ← c7540, ← c7541] at e_114_12
  have e_114_13 := arithEq_of_rows h (row := 114) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_13
  simp only [← c7542, ← c7543, ← c7544] at e_114_13
  have hr := e_114_13
  simp only [e_114_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 63) = bselect (a (.wire 102 59)) (a (.virt 6607)) (a (.wire 113 79)) := by
  have c7545 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7546 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7547 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7548 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7549 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7550 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_114_14 := arithEq_of_rows h (row := 114) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_14
  simp only [← c7545, ← c7546, ← c7547] at e_114_14
  have e_114_15 := arithEq_of_rows h (row := 114) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_15
  simp only [← c7548, ← c7549, ← c7550] at e_114_15
  have hr := e_114_15
  simp only [e_114_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 71) = bselect (a (.wire 102 59)) (a (.virt 6608)) (a (.wire 114 7)) := by
  have c7551 := (publicBatchWrapper86_copies235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7552 := (publicBatchWrapper86_copies236 a h).1
  have c7553 := (publicBatchWrapper86_copies236 a h).2.1
  have c7554 := (publicBatchWrapper86_copies236 a h).2.2.1
  have c7555 := (publicBatchWrapper86_copies236 a h).2.2.2.1
  have c7556 := (publicBatchWrapper86_copies236 a h).2.2.2.2.1
  have e_114_16 := arithEq_of_rows h (row := 114) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_16
  simp only [← c7551, ← c7552, ← c7553] at e_114_16
  have e_114_17 := arithEq_of_rows h (row := 114) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_17
  simp only [← c7554, ← c7555, ← c7556] at e_114_17
  have hr := e_114_17
  simp only [e_114_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 114 79) = bselect (a (.wire 102 59)) (a (.virt 6609)) (a (.wire 114 15)) := by
  have c7557 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.1
  have c7558 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.1
  have c7559 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.1
  have c7560 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.1
  have c7561 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.1
  have c7562 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_114_18 := arithEq_of_rows h (row := 114) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_18
  simp only [← c7557, ← c7558, ← c7559] at e_114_18
  have e_114_19 := arithEq_of_rows h (row := 114) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_114_19
  simp only [← c7560, ← c7561, ← c7562] at e_114_19
  have hr := e_114_19
  simp only [e_114_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 7) = bselect (a (.wire 102 59)) (a (.virt 6603)) (a (.wire 114 23)) := by
  have c7563 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7564 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7565 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7566 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7567 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7568 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_0 := arithEq_of_rows h (row := 115) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_0
  simp only [← c7563, ← c7564, ← c7565] at e_115_0
  have e_115_1 := arithEq_of_rows h (row := 115) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_1
  simp only [← c7566, ← c7567, ← c7568] at e_115_1
  have hr := e_115_1
  simp only [e_115_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 15) = bselect (a (.wire 102 59)) (a (.virt 6604)) (a (.wire 114 31)) := by
  have c7569 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7570 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7571 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7572 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7573 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7574 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_2 := arithEq_of_rows h (row := 115) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_2
  simp only [← c7569, ← c7570, ← c7571] at e_115_2
  have e_115_3 := arithEq_of_rows h (row := 115) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_3
  simp only [← c7572, ← c7573, ← c7574] at e_115_3
  have hr := e_115_3
  simp only [e_115_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 3) = bor (a (.wire 97 79)) (a (.wire 114 35)) := by
  have c7575 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7576 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7577 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7578 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7579 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7580 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_105_12 := arithEq_of_rows h (row := 105) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_12
  simp only [← c7575, ← c7576, ← c7577] at e_105_12
  have e_116_0 := arithEq_of_rows h (row := 116) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_0
  simp only [← c7578, ← c7579, k3, ← c7580] at e_116_0
  have hr := e_116_0
  simp only [e_105_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 19) = bnot (a (.wire 35 3)) := by
  have c7581 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7582 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7583 := (publicBatchWrapper86_copies236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_115_4 := arithEq_of_rows h (row := 115) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_4
  simp only [← c7581, k3, ← c7582, k3, ← c7583] at e_115_4
  have hr := e_115_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 23) = bnot (a (.wire 116 3)) := by
  have c7584 := (publicBatchWrapper86_copies237 a h).1
  have c7585 := (publicBatchWrapper86_copies237 a h).2.1
  have c7586 := (publicBatchWrapper86_copies237 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_115_5 := arithEq_of_rows h (row := 115) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_5
  simp only [← c7584, k3, ← c7585, k3, ← c7586] at e_115_5
  have hr := e_115_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 63) = band (a (.wire 115 19)) (a (.wire 115 23)) := by
  have c7587 := (publicBatchWrapper86_copies237 a h).2.2.2.1
  have c7588 := (publicBatchWrapper86_copies237 a h).2.2.2.2.1
  have c7589 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.1
  have e_102_15 := arithEq_of_rows h (row := 102) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_15
  simp only [← c7587, ← c7588, ← c7589] at e_102_15
  have hr := e_102_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 31) = bselect (a (.wire 102 63)) (a (.virt 6805)) (a (.wire 114 47)) := by
  have c7590 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.1
  have c7591 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.1
  have c7592 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.1
  have c7593 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.1
  have c7594 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7595 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_6 := arithEq_of_rows h (row := 115) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_6
  simp only [← c7590, ← c7591, ← c7592] at e_115_6
  have e_115_7 := arithEq_of_rows h (row := 115) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_7
  simp only [← c7593, ← c7594, ← c7595] at e_115_7
  have hr := e_115_7
  simp only [e_115_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 39) = bselect (a (.wire 102 63)) (a (.virt 6806)) (a (.wire 114 55)) := by
  have c7596 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7597 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7598 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7599 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7600 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7601 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_8 := arithEq_of_rows h (row := 115) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_8
  simp only [← c7596, ← c7597, ← c7598] at e_115_8
  have e_115_9 := arithEq_of_rows h (row := 115) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_9
  simp only [← c7599, ← c7600, ← c7601] at e_115_9
  have hr := e_115_9
  simp only [e_115_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 47) = bselect (a (.wire 102 63)) (a (.virt 6807)) (a (.wire 114 63)) := by
  have c7602 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7603 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7604 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7605 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7606 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7607 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_10 := arithEq_of_rows h (row := 115) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_10
  simp only [← c7602, ← c7603, ← c7604] at e_115_10
  have e_115_11 := arithEq_of_rows h (row := 115) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_11
  simp only [← c7605, ← c7606, ← c7607] at e_115_11
  have hr := e_115_11
  simp only [e_115_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 55) = bselect (a (.wire 102 63)) (a (.virt 6808)) (a (.wire 114 71)) := by
  have c7608 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7609 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7610 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7611 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7612 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7613 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_12 := arithEq_of_rows h (row := 115) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_12
  simp only [← c7608, ← c7609, ← c7610] at e_115_12
  have e_115_13 := arithEq_of_rows h (row := 115) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_13
  simp only [← c7611, ← c7612, ← c7613] at e_115_13
  have hr := e_115_13
  simp only [e_115_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 63) = bselect (a (.wire 102 63)) (a (.virt 6809)) (a (.wire 114 79)) := by
  have c7614 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7615 := (publicBatchWrapper86_copies237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7616 := (publicBatchWrapper86_copies238 a h).1
  have c7617 := (publicBatchWrapper86_copies238 a h).2.1
  have c7618 := (publicBatchWrapper86_copies238 a h).2.2.1
  have c7619 := (publicBatchWrapper86_copies238 a h).2.2.2.1
  have e_115_14 := arithEq_of_rows h (row := 115) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_14
  simp only [← c7614, ← c7615, ← c7616] at e_115_14
  have e_115_15 := arithEq_of_rows h (row := 115) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_15
  simp only [← c7617, ← c7618, ← c7619] at e_115_15
  have hr := e_115_15
  simp only [e_115_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 71) = bselect (a (.wire 102 63)) (a (.virt 6803)) (a (.wire 115 7)) := by
  have c7620 := (publicBatchWrapper86_copies238 a h).2.2.2.2.1
  have c7621 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.1
  have c7622 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.1
  have c7623 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.1
  have c7624 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.1
  have c7625 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.1
  have e_115_16 := arithEq_of_rows h (row := 115) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_16
  simp only [← c7620, ← c7621, ← c7622] at e_115_16
  have e_115_17 := arithEq_of_rows h (row := 115) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_17
  simp only [← c7623, ← c7624, ← c7625] at e_115_17
  have hr := e_115_17
  simp only [e_115_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 115 79) = bselect (a (.wire 102 63)) (a (.virt 6804)) (a (.wire 115 15)) := by
  have c7626 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7627 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7628 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7629 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7630 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7631 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_115_18 := arithEq_of_rows h (row := 115) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_18
  simp only [← c7626, ← c7627, ← c7628] at e_115_18
  have e_115_19 := arithEq_of_rows h (row := 115) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_115_19
  simp only [← c7629, ← c7630, ← c7631] at e_115_19
  have hr := e_115_19
  simp only [e_115_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 7) = bor (a (.wire 116 3)) (a (.wire 115 19)) := by
  have c7632 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7633 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7634 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7635 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7636 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7637 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_105_13 := arithEq_of_rows h (row := 105) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_13
  simp only [← c7632, ← c7633, ← c7634] at e_105_13
  have e_116_1 := arithEq_of_rows h (row := 116) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_1
  simp only [← c7635, ← c7636, k3, ← c7637] at e_116_1
  have hr := e_116_1
  simp only [e_105_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 3) = bnot (a (.wire 35 47)) := by
  have c7638 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7639 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7640 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_117_0 := arithEq_of_rows h (row := 117) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_0
  simp only [← c7638, k3, ← c7639, k3, ← c7640] at e_117_0
  have hr := e_117_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 7) = bnot (a (.wire 116 7)) := by
  have c7641 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7642 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7643 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_117_1 := arithEq_of_rows h (row := 117) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_1
  simp only [← c7641, k3, ← c7642, k3, ← c7643] at e_117_1
  have hr := e_117_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 67) = band (a (.wire 117 3)) (a (.wire 117 7)) := by
  have c7644 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7645 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7646 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_16 := arithEq_of_rows h (row := 102) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_16
  simp only [← c7644, ← c7645, ← c7646] at e_102_16
  have hr := e_102_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 15) = bselect (a (.wire 102 67)) (a (.virt 7005)) (a (.wire 115 31)) := by
  have c7647 := (publicBatchWrapper86_copies238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7648 := (publicBatchWrapper86_copies239 a h).1
  have c7649 := (publicBatchWrapper86_copies239 a h).2.1
  have c7650 := (publicBatchWrapper86_copies239 a h).2.2.1
  have c7651 := (publicBatchWrapper86_copies239 a h).2.2.2.1
  have c7652 := (publicBatchWrapper86_copies239 a h).2.2.2.2.1
  have e_117_2 := arithEq_of_rows h (row := 117) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_2
  simp only [← c7647, ← c7648, ← c7649] at e_117_2
  have e_117_3 := arithEq_of_rows h (row := 117) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_3
  simp only [← c7650, ← c7651, ← c7652] at e_117_3
  have hr := e_117_3
  simp only [e_117_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 23) = bselect (a (.wire 102 67)) (a (.virt 7006)) (a (.wire 115 39)) := by
  have c7653 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.1
  have c7654 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.1
  have c7655 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.1
  have c7656 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.1
  have c7657 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.1
  have c7658 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_117_4 := arithEq_of_rows h (row := 117) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_4
  simp only [← c7653, ← c7654, ← c7655] at e_117_4
  have e_117_5 := arithEq_of_rows h (row := 117) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_5
  simp only [← c7656, ← c7657, ← c7658] at e_117_5
  have hr := e_117_5
  simp only [e_117_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 31) = bselect (a (.wire 102 67)) (a (.virt 7007)) (a (.wire 115 47)) := by
  have c7659 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7660 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7661 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7662 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7663 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7664 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_6 := arithEq_of_rows h (row := 117) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_6
  simp only [← c7659, ← c7660, ← c7661] at e_117_6
  have e_117_7 := arithEq_of_rows h (row := 117) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_7
  simp only [← c7662, ← c7663, ← c7664] at e_117_7
  have hr := e_117_7
  simp only [e_117_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 39) = bselect (a (.wire 102 67)) (a (.virt 7008)) (a (.wire 115 55)) := by
  have c7665 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7666 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7667 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7668 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7669 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7670 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_8 := arithEq_of_rows h (row := 117) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_8
  simp only [← c7665, ← c7666, ← c7667] at e_117_8
  have e_117_9 := arithEq_of_rows h (row := 117) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_9
  simp only [← c7668, ← c7669, ← c7670] at e_117_9
  have hr := e_117_9
  simp only [e_117_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 47) = bselect (a (.wire 102 67)) (a (.virt 7009)) (a (.wire 115 63)) := by
  have c7671 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7672 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7673 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7674 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7675 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7676 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_10 := arithEq_of_rows h (row := 117) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_10
  simp only [← c7671, ← c7672, ← c7673] at e_117_10
  have e_117_11 := arithEq_of_rows h (row := 117) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_11
  simp only [← c7674, ← c7675, ← c7676] at e_117_11
  have hr := e_117_11
  simp only [e_117_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 55) = bselect (a (.wire 102 67)) (a (.virt 7003)) (a (.wire 115 71)) := by
  have c7677 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7678 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7679 := (publicBatchWrapper86_copies239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7680 := (publicBatchWrapper86_copies240 a h).1
  have c7681 := (publicBatchWrapper86_copies240 a h).2.1
  have c7682 := (publicBatchWrapper86_copies240 a h).2.2.1
  have e_117_12 := arithEq_of_rows h (row := 117) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_12
  simp only [← c7677, ← c7678, ← c7679] at e_117_12
  have e_117_13 := arithEq_of_rows h (row := 117) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_13
  simp only [← c7680, ← c7681, ← c7682] at e_117_13
  have hr := e_117_13
  simp only [e_117_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 63) = bselect (a (.wire 102 67)) (a (.virt 7004)) (a (.wire 115 79)) := by
  have c7683 := (publicBatchWrapper86_copies240 a h).2.2.2.1
  have c7684 := (publicBatchWrapper86_copies240 a h).2.2.2.2.1
  have c7685 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.1
  have c7686 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.1
  have c7687 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.1
  have c7688 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.1
  have e_117_14 := arithEq_of_rows h (row := 117) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_14
  simp only [← c7683, ← c7684, ← c7685] at e_117_14
  have e_117_15 := arithEq_of_rows h (row := 117) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_15
  simp only [← c7686, ← c7687, ← c7688] at e_117_15
  have hr := e_117_15
  simp only [e_117_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 11) = bor (a (.wire 116 7)) (a (.wire 117 3)) := by
  have c7689 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.1
  have c7690 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7691 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7692 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7693 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7694 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_105_14 := arithEq_of_rows h (row := 105) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_14
  simp only [← c7689, ← c7690, ← c7691] at e_105_14
  have e_116_2 := arithEq_of_rows h (row := 116) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_2
  simp only [← c7692, ← c7693, k3, ← c7694] at e_116_2
  have hr := e_116_2
  simp only [e_105_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 67) = bnot (a (.wire 37 11)) := by
  have c7695 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7696 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7697 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_117_16 := arithEq_of_rows h (row := 117) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_16
  simp only [← c7695, k3, ← c7696, k3, ← c7697] at e_117_16
  have hr := e_117_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 71) = bnot (a (.wire 116 11)) := by
  have c7698 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7699 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7700 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_117_17 := arithEq_of_rows h (row := 117) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_17
  simp only [← c7698, k3, ← c7699, k3, ← c7700] at e_117_17
  have hr := e_117_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 71) = band (a (.wire 117 67)) (a (.wire 117 71)) := by
  have c7701 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7702 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7703 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_17 := arithEq_of_rows h (row := 102) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_17
  simp only [← c7701, ← c7702, ← c7703] at e_102_17
  have hr := e_102_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 117 79) = bselect (a (.wire 102 71)) (a (.virt 7205)) (a (.wire 117 15)) := by
  have c7704 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7705 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7706 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7707 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7708 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7709 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_117_18 := arithEq_of_rows h (row := 117) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_18
  simp only [← c7704, ← c7705, ← c7706] at e_117_18
  have e_117_19 := arithEq_of_rows h (row := 117) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_117_19
  simp only [← c7707, ← c7708, ← c7709] at e_117_19
  have hr := e_117_19
  simp only [e_117_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 7) = bselect (a (.wire 102 71)) (a (.virt 7206)) (a (.wire 117 23)) := by
  have c7710 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7711 := (publicBatchWrapper86_copies240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7712 := (publicBatchWrapper86_copies241 a h).1
  have c7713 := (publicBatchWrapper86_copies241 a h).2.1
  have c7714 := (publicBatchWrapper86_copies241 a h).2.2.1
  have c7715 := (publicBatchWrapper86_copies241 a h).2.2.2.1
  have e_118_0 := arithEq_of_rows h (row := 118) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_0
  simp only [← c7710, ← c7711, ← c7712] at e_118_0
  have e_118_1 := arithEq_of_rows h (row := 118) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_1
  simp only [← c7713, ← c7714, ← c7715] at e_118_1
  have hr := e_118_1
  simp only [e_118_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 15) = bselect (a (.wire 102 71)) (a (.virt 7207)) (a (.wire 117 31)) := by
  have c7716 := (publicBatchWrapper86_copies241 a h).2.2.2.2.1
  have c7717 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.1
  have c7718 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.1
  have c7719 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.1
  have c7720 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.1
  have c7721 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.1
  have e_118_2 := arithEq_of_rows h (row := 118) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_2
  simp only [← c7716, ← c7717, ← c7718] at e_118_2
  have e_118_3 := arithEq_of_rows h (row := 118) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_3
  simp only [← c7719, ← c7720, ← c7721] at e_118_3
  have hr := e_118_3
  simp only [e_118_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 23) = bselect (a (.wire 102 71)) (a (.virt 7208)) (a (.wire 117 39)) := by
  have c7722 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7723 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7724 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7725 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7726 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7727 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_4 := arithEq_of_rows h (row := 118) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_4
  simp only [← c7722, ← c7723, ← c7724] at e_118_4
  have e_118_5 := arithEq_of_rows h (row := 118) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_5
  simp only [← c7725, ← c7726, ← c7727] at e_118_5
  have hr := e_118_5
  simp only [e_118_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 31) = bselect (a (.wire 102 71)) (a (.virt 7209)) (a (.wire 117 47)) := by
  have c7728 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7729 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7730 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7731 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7732 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7733 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_6 := arithEq_of_rows h (row := 118) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_6
  simp only [← c7728, ← c7729, ← c7730] at e_118_6
  have e_118_7 := arithEq_of_rows h (row := 118) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_7
  simp only [← c7731, ← c7732, ← c7733] at e_118_7
  have hr := e_118_7
  simp only [e_118_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 39) = bselect (a (.wire 102 71)) (a (.virt 7203)) (a (.wire 117 55)) := by
  have c7734 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7735 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7736 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7737 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7738 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7739 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_8 := arithEq_of_rows h (row := 118) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_8
  simp only [← c7734, ← c7735, ← c7736] at e_118_8
  have e_118_9 := arithEq_of_rows h (row := 118) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_9
  simp only [← c7737, ← c7738, ← c7739] at e_118_9
  have hr := e_118_9
  simp only [e_118_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 47) = bselect (a (.wire 102 71)) (a (.virt 7204)) (a (.wire 117 63)) := by
  have c7740 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7741 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7742 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7743 := (publicBatchWrapper86_copies241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7744 := (publicBatchWrapper86_copies242 a h).1
  have c7745 := (publicBatchWrapper86_copies242 a h).2.1
  have e_118_10 := arithEq_of_rows h (row := 118) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_10
  simp only [← c7740, ← c7741, ← c7742] at e_118_10
  have e_118_11 := arithEq_of_rows h (row := 118) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_11
  simp only [← c7743, ← c7744, ← c7745] at e_118_11
  have hr := e_118_11
  simp only [e_118_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 15) = bor (a (.wire 116 11)) (a (.wire 117 67)) := by
  have c7746 := (publicBatchWrapper86_copies242 a h).2.2.1
  have c7747 := (publicBatchWrapper86_copies242 a h).2.2.2.1
  have c7748 := (publicBatchWrapper86_copies242 a h).2.2.2.2.1
  have c7749 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.1
  have c7750 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.1
  have c7751 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_105_15 := arithEq_of_rows h (row := 105) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_15
  simp only [← c7746, ← c7747, ← c7748] at e_105_15
  have e_116_3 := arithEq_of_rows h (row := 116) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_3
  simp only [← c7749, ← c7750, k3, ← c7751] at e_116_3
  have hr := e_116_3
  simp only [e_105_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 51) = bnot (a (.wire 37 55)) := by
  have c7752 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.1
  have c7753 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.1
  have c7754 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_118_12 := arithEq_of_rows h (row := 118) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_12
  simp only [← c7752, k3, ← c7753, k3, ← c7754] at e_118_12
  have hr := e_118_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 55) = bnot (a (.wire 116 15)) := by
  have c7755 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7756 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7757 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_118_13 := arithEq_of_rows h (row := 118) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_13
  simp only [← c7755, k3, ← c7756, k3, ← c7757] at e_118_13
  have hr := e_118_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 75) = band (a (.wire 118 51)) (a (.wire 118 55)) := by
  have c7758 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7759 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7760 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_102_18 := arithEq_of_rows h (row := 102) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_18
  simp only [← c7758, ← c7759, ← c7760] at e_102_18
  have hr := e_102_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 63) = bselect (a (.wire 102 75)) (a (.virt 7405)) (a (.wire 117 79)) := by
  have c7761 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7762 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7763 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7764 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7765 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7766 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_14 := arithEq_of_rows h (row := 118) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_14
  simp only [← c7761, ← c7762, ← c7763] at e_118_14
  have e_118_15 := arithEq_of_rows h (row := 118) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_15
  simp only [← c7764, ← c7765, ← c7766] at e_118_15
  have hr := e_118_15
  simp only [e_118_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 71) = bselect (a (.wire 102 75)) (a (.virt 7406)) (a (.wire 118 7)) := by
  have c7767 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7768 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7769 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7770 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7771 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7772 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_118_16 := arithEq_of_rows h (row := 118) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_16
  simp only [← c7767, ← c7768, ← c7769] at e_118_16
  have e_118_17 := arithEq_of_rows h (row := 118) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_17
  simp only [← c7770, ← c7771, ← c7772] at e_118_17
  have hr := e_118_17
  simp only [e_118_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 118 79) = bselect (a (.wire 102 75)) (a (.virt 7407)) (a (.wire 118 15)) := by
  have c7773 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7774 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7775 := (publicBatchWrapper86_copies242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7776 := (publicBatchWrapper86_copies243 a h).1
  have c7777 := (publicBatchWrapper86_copies243 a h).2.1
  have c7778 := (publicBatchWrapper86_copies243 a h).2.2.1
  have e_118_18 := arithEq_of_rows h (row := 118) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_18
  simp only [← c7773, ← c7774, ← c7775] at e_118_18
  have e_118_19 := arithEq_of_rows h (row := 118) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_118_19
  simp only [← c7776, ← c7777, ← c7778] at e_118_19
  have hr := e_118_19
  simp only [e_118_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 7) = bselect (a (.wire 102 75)) (a (.virt 7408)) (a (.wire 118 23)) := by
  have c7779 := (publicBatchWrapper86_copies243 a h).2.2.2.1
  have c7780 := (publicBatchWrapper86_copies243 a h).2.2.2.2.1
  have c7781 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.1
  have c7782 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.1
  have c7783 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.1
  have c7784 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.1
  have e_119_0 := arithEq_of_rows h (row := 119) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_0
  simp only [← c7779, ← c7780, ← c7781] at e_119_0
  have e_119_1 := arithEq_of_rows h (row := 119) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_1
  simp only [← c7782, ← c7783, ← c7784] at e_119_1
  have hr := e_119_1
  simp only [e_119_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 15) = bselect (a (.wire 102 75)) (a (.virt 7409)) (a (.wire 118 31)) := by
  have c7785 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.1
  have c7786 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7787 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7788 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7789 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7790 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_2 := arithEq_of_rows h (row := 119) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_2
  simp only [← c7785, ← c7786, ← c7787] at e_119_2
  have e_119_3 := arithEq_of_rows h (row := 119) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_3
  simp only [← c7788, ← c7789, ← c7790] at e_119_3
  have hr := e_119_3
  simp only [e_119_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 23) = bselect (a (.wire 102 75)) (a (.virt 7403)) (a (.wire 118 39)) := by
  have c7791 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7792 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7793 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7794 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7795 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7796 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_4 := arithEq_of_rows h (row := 119) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_4
  simp only [← c7791, ← c7792, ← c7793] at e_119_4
  have e_119_5 := arithEq_of_rows h (row := 119) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_5
  simp only [← c7794, ← c7795, ← c7796] at e_119_5
  have hr := e_119_5
  simp only [e_119_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 31) = bselect (a (.wire 102 75)) (a (.virt 7404)) (a (.wire 118 47)) := by
  have c7797 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7798 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7799 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7800 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7801 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7802 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_6 := arithEq_of_rows h (row := 119) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_6
  simp only [← c7797, ← c7798, ← c7799] at e_119_6
  have e_119_7 := arithEq_of_rows h (row := 119) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_7
  simp only [← c7800, ← c7801, ← c7802] at e_119_7
  have hr := e_119_7
  simp only [e_119_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 19) = bor (a (.wire 116 15)) (a (.wire 118 51)) := by
  have c7803 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7804 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7805 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7806 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7807 := (publicBatchWrapper86_copies243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7808 := (publicBatchWrapper86_copies244 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_105_16 := arithEq_of_rows h (row := 105) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_16
  simp only [← c7803, ← c7804, ← c7805] at e_105_16
  have e_116_4 := arithEq_of_rows h (row := 116) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_4
  simp only [← c7806, ← c7807, k3, ← c7808] at e_116_4
  have hr := e_116_4
  simp only [e_105_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 35) = bnot (a (.wire 39 19)) := by
  have c7809 := (publicBatchWrapper86_copies244 a h).2.1
  have c7810 := (publicBatchWrapper86_copies244 a h).2.2.1
  have c7811 := (publicBatchWrapper86_copies244 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_119_8 := arithEq_of_rows h (row := 119) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_8
  simp only [← c7809, k3, ← c7810, k3, ← c7811] at e_119_8
  have hr := e_119_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 39) = bnot (a (.wire 116 19)) := by
  have c7812 := (publicBatchWrapper86_copies244 a h).2.2.2.2.1
  have c7813 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.1
  have c7814 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_119_9 := arithEq_of_rows h (row := 119) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_9
  simp only [← c7812, k3, ← c7813, k3, ← c7814] at e_119_9
  have hr := e_119_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 102 79) = band (a (.wire 119 35)) (a (.wire 119 39)) := by
  have c7815 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.1
  have c7816 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.1
  have c7817 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.1
  have e_102_19 := arithEq_of_rows h (row := 102) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_102_19
  simp only [← c7815, ← c7816, ← c7817] at e_102_19
  have hr := e_102_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 47) = bselect (a (.wire 102 79)) (a (.virt 7605)) (a (.wire 118 63)) := by
  have c7818 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7819 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7820 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7821 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7822 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7823 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_10 := arithEq_of_rows h (row := 119) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_10
  simp only [← c7818, ← c7819, ← c7820] at e_119_10
  have e_119_11 := arithEq_of_rows h (row := 119) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_11
  simp only [← c7821, ← c7822, ← c7823] at e_119_11
  have hr := e_119_11
  simp only [e_119_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 55) = bselect (a (.wire 102 79)) (a (.virt 7606)) (a (.wire 118 71)) := by
  have c7824 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7825 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7826 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7827 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7828 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7829 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_12 := arithEq_of_rows h (row := 119) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_12
  simp only [← c7824, ← c7825, ← c7826] at e_119_12
  have e_119_13 := arithEq_of_rows h (row := 119) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_13
  simp only [← c7827, ← c7828, ← c7829] at e_119_13
  have hr := e_119_13
  simp only [e_119_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 63) = bselect (a (.wire 102 79)) (a (.virt 7607)) (a (.wire 118 79)) := by
  have c7830 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7831 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7832 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7833 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7834 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7835 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_119_14 := arithEq_of_rows h (row := 119) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_14
  simp only [← c7830, ← c7831, ← c7832] at e_119_14
  have e_119_15 := arithEq_of_rows h (row := 119) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_15
  simp only [← c7833, ← c7834, ← c7835] at e_119_15
  have hr := e_119_15
  simp only [e_119_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 71) = bselect (a (.wire 102 79)) (a (.virt 7608)) (a (.wire 119 7)) := by
  have c7836 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7837 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7838 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7839 := (publicBatchWrapper86_copies244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7840 := (publicBatchWrapper86_copies245 a h).1
  have c7841 := (publicBatchWrapper86_copies245 a h).2.1
  have e_119_16 := arithEq_of_rows h (row := 119) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_16
  simp only [← c7836, ← c7837, ← c7838] at e_119_16
  have e_119_17 := arithEq_of_rows h (row := 119) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_17
  simp only [← c7839, ← c7840, ← c7841] at e_119_17
  have hr := e_119_17
  simp only [e_119_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 119 79) = bselect (a (.wire 102 79)) (a (.virt 7609)) (a (.wire 119 15)) := by
  have c7842 := (publicBatchWrapper86_copies245 a h).2.2.1
  have c7843 := (publicBatchWrapper86_copies245 a h).2.2.2.1
  have c7844 := (publicBatchWrapper86_copies245 a h).2.2.2.2.1
  have c7845 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.1
  have c7846 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.1
  have c7847 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.1
  have e_119_18 := arithEq_of_rows h (row := 119) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_18
  simp only [← c7842, ← c7843, ← c7844] at e_119_18
  have e_119_19 := arithEq_of_rows h (row := 119) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_119_19
  simp only [← c7845, ← c7846, ← c7847] at e_119_19
  have hr := e_119_19
  simp only [e_119_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 7) = bselect (a (.wire 102 79)) (a (.virt 7603)) (a (.wire 119 23)) := by
  have c7848 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.1
  have c7849 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.1
  have c7850 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7851 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7852 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7853 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_0 := arithEq_of_rows h (row := 120) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_0
  simp only [← c7848, ← c7849, ← c7850] at e_120_0
  have e_120_1 := arithEq_of_rows h (row := 120) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_1
  simp only [← c7851, ← c7852, ← c7853] at e_120_1
  have hr := e_120_1
  simp only [e_120_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 15) = bselect (a (.wire 102 79)) (a (.virt 7604)) (a (.wire 119 31)) := by
  have c7854 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7855 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7856 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7857 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7858 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7859 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_2 := arithEq_of_rows h (row := 120) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_2
  simp only [← c7854, ← c7855, ← c7856] at e_120_2
  have e_120_3 := arithEq_of_rows h (row := 120) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_3
  simp only [← c7857, ← c7858, ← c7859] at e_120_3
  have hr := e_120_3
  simp only [e_120_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 23) = bor (a (.wire 116 19)) (a (.wire 119 35)) := by
  have c7860 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7861 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7862 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7863 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7864 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7865 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_105_17 := arithEq_of_rows h (row := 105) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_17
  simp only [← c7860, ← c7861, ← c7862] at e_105_17
  have e_116_5 := arithEq_of_rows h (row := 116) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_5
  simp only [← c7863, ← c7864, k3, ← c7865] at e_116_5
  have hr := e_116_5
  simp only [e_105_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 19) = bnot (a (.wire 39 63)) := by
  have c7866 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7867 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7868 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_120_4 := arithEq_of_rows h (row := 120) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_4
  simp only [← c7866, k3, ← c7867, k3, ← c7868] at e_120_4
  have hr := e_120_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 23) = bnot (a (.wire 116 23)) := by
  have c7869 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7870 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7871 := (publicBatchWrapper86_copies245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_120_5 := arithEq_of_rows h (row := 120) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_5
  simp only [← c7869, k3, ← c7870, k3, ← c7871] at e_120_5
  have hr := e_120_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 3) = band (a (.wire 120 19)) (a (.wire 120 23)) := by
  have c7872 := (publicBatchWrapper86_copies246 a h).1
  have c7873 := (publicBatchWrapper86_copies246 a h).2.1
  have c7874 := (publicBatchWrapper86_copies246 a h).2.2.1
  have e_121_0 := arithEq_of_rows h (row := 121) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_0
  simp only [← c7872, ← c7873, ← c7874] at e_121_0
  have hr := e_121_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 31) = bselect (a (.wire 121 3)) (a (.virt 7805)) (a (.wire 119 47)) := by
  have c7875 := (publicBatchWrapper86_copies246 a h).2.2.2.1
  have c7876 := (publicBatchWrapper86_copies246 a h).2.2.2.2.1
  have c7877 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.1
  have c7878 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.1
  have c7879 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.1
  have c7880 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.1
  have e_120_6 := arithEq_of_rows h (row := 120) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_6
  simp only [← c7875, ← c7876, ← c7877] at e_120_6
  have e_120_7 := arithEq_of_rows h (row := 120) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_7
  simp only [← c7878, ← c7879, ← c7880] at e_120_7
  have hr := e_120_7
  simp only [e_120_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 39) = bselect (a (.wire 121 3)) (a (.virt 7806)) (a (.wire 119 55)) := by
  have c7881 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.1
  have c7882 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7883 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7884 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7885 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7886 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_8 := arithEq_of_rows h (row := 120) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_8
  simp only [← c7881, ← c7882, ← c7883] at e_120_8
  have e_120_9 := arithEq_of_rows h (row := 120) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_9
  simp only [← c7884, ← c7885, ← c7886] at e_120_9
  have hr := e_120_9
  simp only [e_120_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 47) = bselect (a (.wire 121 3)) (a (.virt 7807)) (a (.wire 119 63)) := by
  have c7887 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7888 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7889 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7890 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7891 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7892 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_10 := arithEq_of_rows h (row := 120) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_10
  simp only [← c7887, ← c7888, ← c7889] at e_120_10
  have e_120_11 := arithEq_of_rows h (row := 120) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_11
  simp only [← c7890, ← c7891, ← c7892] at e_120_11
  have hr := e_120_11
  simp only [e_120_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 55) = bselect (a (.wire 121 3)) (a (.virt 7808)) (a (.wire 119 71)) := by
  have c7893 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7894 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7895 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7896 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7897 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7898 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_12 := arithEq_of_rows h (row := 120) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_12
  simp only [← c7893, ← c7894, ← c7895] at e_120_12
  have e_120_13 := arithEq_of_rows h (row := 120) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_13
  simp only [← c7896, ← c7897, ← c7898] at e_120_13
  have hr := e_120_13
  simp only [e_120_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 63) = bselect (a (.wire 121 3)) (a (.virt 7809)) (a (.wire 119 79)) := by
  have c7899 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7900 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7901 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7902 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7903 := (publicBatchWrapper86_copies246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7904 := (publicBatchWrapper86_copies247 a h).1
  have e_120_14 := arithEq_of_rows h (row := 120) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_14
  simp only [← c7899, ← c7900, ← c7901] at e_120_14
  have e_120_15 := arithEq_of_rows h (row := 120) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_15
  simp only [← c7902, ← c7903, ← c7904] at e_120_15
  have hr := e_120_15
  simp only [e_120_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 71) = bselect (a (.wire 121 3)) (a (.virt 7803)) (a (.wire 120 7)) := by
  have c7905 := (publicBatchWrapper86_copies247 a h).2.1
  have c7906 := (publicBatchWrapper86_copies247 a h).2.2.1
  have c7907 := (publicBatchWrapper86_copies247 a h).2.2.2.1
  have c7908 := (publicBatchWrapper86_copies247 a h).2.2.2.2.1
  have c7909 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.1
  have c7910 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.1
  have e_120_16 := arithEq_of_rows h (row := 120) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_16
  simp only [← c7905, ← c7906, ← c7907] at e_120_16
  have e_120_17 := arithEq_of_rows h (row := 120) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_17
  simp only [← c7908, ← c7909, ← c7910] at e_120_17
  have hr := e_120_17
  simp only [e_120_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 120 79) = bselect (a (.wire 121 3)) (a (.virt 7804)) (a (.wire 120 15)) := by
  have c7911 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.1
  have c7912 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.1
  have c7913 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.1
  have c7914 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7915 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7916 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_120_18 := arithEq_of_rows h (row := 120) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_18
  simp only [← c7911, ← c7912, ← c7913] at e_120_18
  have e_120_19 := arithEq_of_rows h (row := 120) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_120_19
  simp only [← c7914, ← c7915, ← c7916] at e_120_19
  have hr := e_120_19
  simp only [e_120_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 27) = bor (a (.wire 116 23)) (a (.wire 120 19)) := by
  have c7917 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7918 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7919 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7920 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7921 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7922 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_105_18 := arithEq_of_rows h (row := 105) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_18
  simp only [← c7917, ← c7918, ← c7919] at e_105_18
  have e_116_6 := arithEq_of_rows h (row := 116) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_6
  simp only [← c7920, ← c7921, k3, ← c7922] at e_116_6
  have hr := e_116_6
  simp only [e_105_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 3) = bnot (a (.wire 41 27)) := by
  have c7923 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7924 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7925 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_122_0 := arithEq_of_rows h (row := 122) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_0
  simp only [← c7923, k3, ← c7924, k3, ← c7925] at e_122_0
  have hr := e_122_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 7) = bnot (a (.wire 116 27)) := by
  have c7926 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7927 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7928 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_122_1 := arithEq_of_rows h (row := 122) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_1
  simp only [← c7926, k3, ← c7927, k3, ← c7928] at e_122_1
  have hr := e_122_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 7) = band (a (.wire 122 3)) (a (.wire 122 7)) := by
  have c7929 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7930 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7931 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_1 := arithEq_of_rows h (row := 121) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_1
  simp only [← c7929, ← c7930, ← c7931] at e_121_1
  have hr := e_121_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 15) = bselect (a (.wire 121 7)) (a (.virt 8005)) (a (.wire 120 31)) := by
  have c7932 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7933 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7934 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7935 := (publicBatchWrapper86_copies247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c7936 := (publicBatchWrapper86_copies248 a h).1
  have c7937 := (publicBatchWrapper86_copies248 a h).2.1
  have e_122_2 := arithEq_of_rows h (row := 122) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_2
  simp only [← c7932, ← c7933, ← c7934] at e_122_2
  have e_122_3 := arithEq_of_rows h (row := 122) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_3
  simp only [← c7935, ← c7936, ← c7937] at e_122_3
  have hr := e_122_3
  simp only [e_122_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 23) = bselect (a (.wire 121 7)) (a (.virt 8006)) (a (.wire 120 39)) := by
  have c7938 := (publicBatchWrapper86_copies248 a h).2.2.1
  have c7939 := (publicBatchWrapper86_copies248 a h).2.2.2.1
  have c7940 := (publicBatchWrapper86_copies248 a h).2.2.2.2.1
  have c7941 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.1
  have c7942 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.1
  have c7943 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.1
  have e_122_4 := arithEq_of_rows h (row := 122) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_4
  simp only [← c7938, ← c7939, ← c7940] at e_122_4
  have e_122_5 := arithEq_of_rows h (row := 122) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_5
  simp only [← c7941, ← c7942, ← c7943] at e_122_5
  have hr := e_122_5
  simp only [e_122_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 31) = bselect (a (.wire 121 7)) (a (.virt 8007)) (a (.wire 120 47)) := by
  have c7944 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.1
  have c7945 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.1
  have c7946 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7947 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c7948 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7949 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_6 := arithEq_of_rows h (row := 122) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_6
  simp only [← c7944, ← c7945, ← c7946] at e_122_6
  have e_122_7 := arithEq_of_rows h (row := 122) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_7
  simp only [← c7947, ← c7948, ← c7949] at e_122_7
  have hr := e_122_7
  simp only [e_122_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 39) = bselect (a (.wire 121 7)) (a (.virt 8008)) (a (.wire 120 55)) := by
  have c7950 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7951 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7952 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7953 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7954 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7955 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_8 := arithEq_of_rows h (row := 122) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_8
  simp only [← c7950, ← c7951, ← c7952] at e_122_8
  have e_122_9 := arithEq_of_rows h (row := 122) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_9
  simp only [← c7953, ← c7954, ← c7955] at e_122_9
  have hr := e_122_9
  simp only [e_122_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 47) = bselect (a (.wire 121 7)) (a (.virt 8009)) (a (.wire 120 63)) := by
  have c7956 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7957 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7958 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7959 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7960 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7961 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_10 := arithEq_of_rows h (row := 122) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_10
  simp only [← c7956, ← c7957, ← c7958] at e_122_10
  have e_122_11 := arithEq_of_rows h (row := 122) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_11
  simp only [← c7959, ← c7960, ← c7961] at e_122_11
  have hr := e_122_11
  simp only [e_122_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 55) = bselect (a (.wire 121 7)) (a (.virt 8003)) (a (.wire 120 71)) := by
  have c7962 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7963 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7964 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7965 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7966 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7967 := (publicBatchWrapper86_copies248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_122_12 := arithEq_of_rows h (row := 122) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_12
  simp only [← c7962, ← c7963, ← c7964] at e_122_12
  have e_122_13 := arithEq_of_rows h (row := 122) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_13
  simp only [← c7965, ← c7966, ← c7967] at e_122_13
  have hr := e_122_13
  simp only [e_122_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 63) = bselect (a (.wire 121 7)) (a (.virt 8004)) (a (.wire 120 79)) := by
  have c7968 := (publicBatchWrapper86_copies249 a h).1
  have c7969 := (publicBatchWrapper86_copies249 a h).2.1
  have c7970 := (publicBatchWrapper86_copies249 a h).2.2.1
  have c7971 := (publicBatchWrapper86_copies249 a h).2.2.2.1
  have c7972 := (publicBatchWrapper86_copies249 a h).2.2.2.2.1
  have c7973 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.1
  have e_122_14 := arithEq_of_rows h (row := 122) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_14
  simp only [← c7968, ← c7969, ← c7970] at e_122_14
  have e_122_15 := arithEq_of_rows h (row := 122) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_15
  simp only [← c7971, ← c7972, ← c7973] at e_122_15
  have hr := e_122_15
  simp only [e_122_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 31) = bor (a (.wire 116 27)) (a (.wire 122 3)) := by
  have c7974 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.1
  have c7975 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.1
  have c7976 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.1
  have c7977 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.1
  have c7978 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.1
  have c7979 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_105_19 := arithEq_of_rows h (row := 105) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_105_19
  simp only [← c7974, ← c7975, ← c7976] at e_105_19
  have e_116_7 := arithEq_of_rows h (row := 116) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_7
  simp only [← c7977, ← c7978, k3, ← c7979] at e_116_7
  have hr := e_116_7
  simp only [e_105_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 67) = bnot (a (.wire 41 71)) := by
  have c7980 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7981 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7982 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_122_16 := arithEq_of_rows h (row := 122) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_16
  simp only [← c7980, k3, ← c7981, k3, ← c7982] at e_122_16
  have hr := e_122_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 71) = bnot (a (.wire 116 31)) := by
  have c7983 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7984 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7985 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_122_17 := arithEq_of_rows h (row := 122) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_17
  simp only [← c7983, k3, ← c7984, k3, ← c7985] at e_122_17
  have hr := e_122_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 11) = band (a (.wire 122 67)) (a (.wire 122 71)) := by
  have c7986 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7987 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7988 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_2 := arithEq_of_rows h (row := 121) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_2
  simp only [← c7986, ← c7987, ← c7988] at e_121_2
  have hr := e_121_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 122 79) = bselect (a (.wire 121 11)) (a (.virt 8205)) (a (.wire 122 15)) := by
  have c7989 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7990 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7991 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7992 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7993 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7994 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_122_18 := arithEq_of_rows h (row := 122) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_18
  simp only [← c7989, ← c7990, ← c7991] at e_122_18
  have e_122_19 := arithEq_of_rows h (row := 122) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_122_19
  simp only [← c7992, ← c7993, ← c7994] at e_122_19
  have hr := e_122_19
  simp only [e_122_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 7) = bselect (a (.wire 121 11)) (a (.virt 8206)) (a (.wire 122 23)) := by
  have c7995 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7996 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7997 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7998 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c7999 := (publicBatchWrapper86_copies249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8000 := (publicBatchWrapper86_copies250 a h).1
  have e_123_0 := arithEq_of_rows h (row := 123) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_0
  simp only [← c7995, ← c7996, ← c7997] at e_123_0
  have e_123_1 := arithEq_of_rows h (row := 123) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_1
  simp only [← c7998, ← c7999, ← c8000] at e_123_1
  have hr := e_123_1
  simp only [e_123_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 15) = bselect (a (.wire 121 11)) (a (.virt 8207)) (a (.wire 122 31)) := by
  have c8001 := (publicBatchWrapper86_copies250 a h).2.1
  have c8002 := (publicBatchWrapper86_copies250 a h).2.2.1
  have c8003 := (publicBatchWrapper86_copies250 a h).2.2.2.1
  have c8004 := (publicBatchWrapper86_copies250 a h).2.2.2.2.1
  have c8005 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.1
  have c8006 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.1
  have e_123_2 := arithEq_of_rows h (row := 123) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_2
  simp only [← c8001, ← c8002, ← c8003] at e_123_2
  have e_123_3 := arithEq_of_rows h (row := 123) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_3
  simp only [← c8004, ← c8005, ← c8006] at e_123_3
  have hr := e_123_3
  simp only [e_123_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 23) = bselect (a (.wire 121 11)) (a (.virt 8208)) (a (.wire 122 39)) := by
  have c8007 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.1
  have c8008 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.1
  have c8009 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.1
  have c8010 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8011 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8012 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_4 := arithEq_of_rows h (row := 123) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_4
  simp only [← c8007, ← c8008, ← c8009] at e_123_4
  have e_123_5 := arithEq_of_rows h (row := 123) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_5
  simp only [← c8010, ← c8011, ← c8012] at e_123_5
  have hr := e_123_5
  simp only [e_123_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 31) = bselect (a (.wire 121 11)) (a (.virt 8209)) (a (.wire 122 47)) := by
  have c8013 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8014 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8015 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8016 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8017 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8018 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_6 := arithEq_of_rows h (row := 123) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_6
  simp only [← c8013, ← c8014, ← c8015] at e_123_6
  have e_123_7 := arithEq_of_rows h (row := 123) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_7
  simp only [← c8016, ← c8017, ← c8018] at e_123_7
  have hr := e_123_7
  simp only [e_123_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 39) = bselect (a (.wire 121 11)) (a (.virt 8203)) (a (.wire 122 55)) := by
  have c8019 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8020 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8021 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8022 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8023 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8024 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_8 := arithEq_of_rows h (row := 123) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_8
  simp only [← c8019, ← c8020, ← c8021] at e_123_8
  have e_123_9 := arithEq_of_rows h (row := 123) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_9
  simp only [← c8022, ← c8023, ← c8024] at e_123_9
  have hr := e_123_9
  simp only [e_123_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 47) = bselect (a (.wire 121 11)) (a (.virt 8204)) (a (.wire 122 63)) := by
  have c8025 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8026 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8027 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8028 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8029 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8030 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_10 := arithEq_of_rows h (row := 123) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_10
  simp only [← c8025, ← c8026, ← c8027] at e_123_10
  have e_123_11 := arithEq_of_rows h (row := 123) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_11
  simp only [← c8028, ← c8029, ← c8030] at e_123_11
  have hr := e_123_11
  simp only [e_123_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 35) = bor (a (.wire 116 31)) (a (.wire 122 67)) := by
  have c8031 := (publicBatchWrapper86_copies250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8032 := (publicBatchWrapper86_copies251 a h).1
  have c8033 := (publicBatchWrapper86_copies251 a h).2.1
  have c8034 := (publicBatchWrapper86_copies251 a h).2.2.1
  have c8035 := (publicBatchWrapper86_copies251 a h).2.2.2.1
  have c8036 := (publicBatchWrapper86_copies251 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_8 := arithEq_of_rows h (row := 116) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_8
  simp only [← c8034, ← c8035, k3, ← c8036] at e_116_8
  have e_124_0 := arithEq_of_rows h (row := 124) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_0
  simp only [← c8031, ← c8032, ← c8033] at e_124_0
  have hr := e_116_8
  simp only [e_124_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 51) = bnot (a (.wire 42 35)) := by
  have c8037 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.1
  have c8038 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.1
  have c8039 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_123_12 := arithEq_of_rows h (row := 123) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_12
  simp only [← c8037, k3, ← c8038, k3, ← c8039] at e_123_12
  have hr := e_123_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 55) = bnot (a (.wire 116 35)) := by
  have c8040 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.1
  have c8041 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.1
  have c8042 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_123_13 := arithEq_of_rows h (row := 123) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_13
  simp only [← c8040, k3, ← c8041, k3, ← c8042] at e_123_13
  have hr := e_123_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 15) = band (a (.wire 123 51)) (a (.wire 123 55)) := by
  have c8043 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8044 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8045 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_3 := arithEq_of_rows h (row := 121) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_3
  simp only [← c8043, ← c8044, ← c8045] at e_121_3
  have hr := e_121_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 63) = bselect (a (.wire 121 15)) (a (.virt 8405)) (a (.wire 122 79)) := by
  have c8046 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8047 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8048 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8049 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8050 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8051 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_14 := arithEq_of_rows h (row := 123) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_14
  simp only [← c8046, ← c8047, ← c8048] at e_123_14
  have e_123_15 := arithEq_of_rows h (row := 123) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_15
  simp only [← c8049, ← c8050, ← c8051] at e_123_15
  have hr := e_123_15
  simp only [e_123_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 71) = bselect (a (.wire 121 15)) (a (.virt 8406)) (a (.wire 123 7)) := by
  have c8052 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8053 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8054 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8055 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8056 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8057 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_123_16 := arithEq_of_rows h (row := 123) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_16
  simp only [← c8052, ← c8053, ← c8054] at e_123_16
  have e_123_17 := arithEq_of_rows h (row := 123) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_17
  simp only [← c8055, ← c8056, ← c8057] at e_123_17
  have hr := e_123_17
  simp only [e_123_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 123 79) = bselect (a (.wire 121 15)) (a (.virt 8407)) (a (.wire 123 15)) := by
  have c8058 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8059 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8060 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8061 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8062 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8063 := (publicBatchWrapper86_copies251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_123_18 := arithEq_of_rows h (row := 123) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_18
  simp only [← c8058, ← c8059, ← c8060] at e_123_18
  have e_123_19 := arithEq_of_rows h (row := 123) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_123_19
  simp only [← c8061, ← c8062, ← c8063] at e_123_19
  have hr := e_123_19
  simp only [e_123_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 7) = bselect (a (.wire 121 15)) (a (.virt 8408)) (a (.wire 123 23)) := by
  have c8064 := (publicBatchWrapper86_copies252 a h).1
  have c8065 := (publicBatchWrapper86_copies252 a h).2.1
  have c8066 := (publicBatchWrapper86_copies252 a h).2.2.1
  have c8067 := (publicBatchWrapper86_copies252 a h).2.2.2.1
  have c8068 := (publicBatchWrapper86_copies252 a h).2.2.2.2.1
  have c8069 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.1
  have e_125_0 := arithEq_of_rows h (row := 125) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_0
  simp only [← c8064, ← c8065, ← c8066] at e_125_0
  have e_125_1 := arithEq_of_rows h (row := 125) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_1
  simp only [← c8067, ← c8068, ← c8069] at e_125_1
  have hr := e_125_1
  simp only [e_125_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 15) = bselect (a (.wire 121 15)) (a (.virt 8409)) (a (.wire 123 31)) := by
  have c8070 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.1
  have c8071 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.1
  have c8072 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.1
  have c8073 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.1
  have c8074 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8075 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_2 := arithEq_of_rows h (row := 125) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_2
  simp only [← c8070, ← c8071, ← c8072] at e_125_2
  have e_125_3 := arithEq_of_rows h (row := 125) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_3
  simp only [← c8073, ← c8074, ← c8075] at e_125_3
  have hr := e_125_3
  simp only [e_125_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 23) = bselect (a (.wire 121 15)) (a (.virt 8403)) (a (.wire 123 39)) := by
  have c8076 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8077 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8078 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8079 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8080 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8081 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_4 := arithEq_of_rows h (row := 125) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_4
  simp only [← c8076, ← c8077, ← c8078] at e_125_4
  have e_125_5 := arithEq_of_rows h (row := 125) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_5
  simp only [← c8079, ← c8080, ← c8081] at e_125_5
  have hr := e_125_5
  simp only [e_125_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 31) = bselect (a (.wire 121 15)) (a (.virt 8404)) (a (.wire 123 47)) := by
  have c8082 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8083 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8084 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8085 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8086 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8087 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_6 := arithEq_of_rows h (row := 125) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_6
  simp only [← c8082, ← c8083, ← c8084] at e_125_6
  have e_125_7 := arithEq_of_rows h (row := 125) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_7
  simp only [← c8085, ← c8086, ← c8087] at e_125_7
  have hr := e_125_7
  simp only [e_125_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 39) = bor (a (.wire 116 35)) (a (.wire 123 51)) := by
  have c8088 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8089 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8090 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8091 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8092 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8093 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_9 := arithEq_of_rows h (row := 116) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_9
  simp only [← c8091, ← c8092, k3, ← c8093] at e_116_9
  have e_124_1 := arithEq_of_rows h (row := 124) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_1
  simp only [← c8088, ← c8089, ← c8090] at e_124_1
  have hr := e_116_9
  simp only [e_124_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 35) = bnot (a (.wire 42 79)) := by
  have c8094 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8095 := (publicBatchWrapper86_copies252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8096 := (publicBatchWrapper86_copies253 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_125_8 := arithEq_of_rows h (row := 125) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_8
  simp only [← c8094, k3, ← c8095, k3, ← c8096] at e_125_8
  have hr := e_125_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 39) = bnot (a (.wire 116 39)) := by
  have c8097 := (publicBatchWrapper86_copies253 a h).2.1
  have c8098 := (publicBatchWrapper86_copies253 a h).2.2.1
  have c8099 := (publicBatchWrapper86_copies253 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_125_9 := arithEq_of_rows h (row := 125) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_9
  simp only [← c8097, k3, ← c8098, k3, ← c8099] at e_125_9
  have hr := e_125_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 19) = band (a (.wire 125 35)) (a (.wire 125 39)) := by
  have c8100 := (publicBatchWrapper86_copies253 a h).2.2.2.2.1
  have c8101 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.1
  have c8102 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.1
  have e_121_4 := arithEq_of_rows h (row := 121) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_4
  simp only [← c8100, ← c8101, ← c8102] at e_121_4
  have hr := e_121_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 47) = bselect (a (.wire 121 19)) (a (.virt 8605)) (a (.wire 123 63)) := by
  have c8103 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.1
  have c8104 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.1
  have c8105 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.1
  have c8106 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8107 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8108 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_10 := arithEq_of_rows h (row := 125) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_10
  simp only [← c8103, ← c8104, ← c8105] at e_125_10
  have e_125_11 := arithEq_of_rows h (row := 125) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_11
  simp only [← c8106, ← c8107, ← c8108] at e_125_11
  have hr := e_125_11
  simp only [e_125_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 55) = bselect (a (.wire 121 19)) (a (.virt 8606)) (a (.wire 123 71)) := by
  have c8109 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8110 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8111 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8112 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8113 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8114 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_12 := arithEq_of_rows h (row := 125) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_12
  simp only [← c8109, ← c8110, ← c8111] at e_125_12
  have e_125_13 := arithEq_of_rows h (row := 125) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_13
  simp only [← c8112, ← c8113, ← c8114] at e_125_13
  have hr := e_125_13
  simp only [e_125_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 63) = bselect (a (.wire 121 19)) (a (.virt 8607)) (a (.wire 123 79)) := by
  have c8115 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8116 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8117 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8118 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8119 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8120 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_14 := arithEq_of_rows h (row := 125) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_14
  simp only [← c8115, ← c8116, ← c8117] at e_125_14
  have e_125_15 := arithEq_of_rows h (row := 125) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_15
  simp only [← c8118, ← c8119, ← c8120] at e_125_15
  have hr := e_125_15
  simp only [e_125_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 71) = bselect (a (.wire 121 19)) (a (.virt 8608)) (a (.wire 125 7)) := by
  have c8121 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8122 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8123 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8124 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8125 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8126 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_125_16 := arithEq_of_rows h (row := 125) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_16
  simp only [← c8121, ← c8122, ← c8123] at e_125_16
  have e_125_17 := arithEq_of_rows h (row := 125) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_17
  simp only [← c8124, ← c8125, ← c8126] at e_125_17
  have hr := e_125_17
  simp only [e_125_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 125 79) = bselect (a (.wire 121 19)) (a (.virt 8609)) (a (.wire 125 15)) := by
  have c8127 := (publicBatchWrapper86_copies253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8128 := (publicBatchWrapper86_copies254 a h).1
  have c8129 := (publicBatchWrapper86_copies254 a h).2.1
  have c8130 := (publicBatchWrapper86_copies254 a h).2.2.1
  have c8131 := (publicBatchWrapper86_copies254 a h).2.2.2.1
  have c8132 := (publicBatchWrapper86_copies254 a h).2.2.2.2.1
  have e_125_18 := arithEq_of_rows h (row := 125) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_18
  simp only [← c8127, ← c8128, ← c8129] at e_125_18
  have e_125_19 := arithEq_of_rows h (row := 125) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_125_19
  simp only [← c8130, ← c8131, ← c8132] at e_125_19
  have hr := e_125_19
  simp only [e_125_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 7) = bselect (a (.wire 121 19)) (a (.virt 8603)) (a (.wire 125 23)) := by
  have c8133 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.1
  have c8134 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.1
  have c8135 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.1
  have c8136 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.1
  have c8137 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.1
  have c8138 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_126_0 := arithEq_of_rows h (row := 126) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_0
  simp only [← c8133, ← c8134, ← c8135] at e_126_0
  have e_126_1 := arithEq_of_rows h (row := 126) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_1
  simp only [← c8136, ← c8137, ← c8138] at e_126_1
  have hr := e_126_1
  simp only [e_126_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 15) = bselect (a (.wire 121 19)) (a (.virt 8604)) (a (.wire 125 31)) := by
  have c8139 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8140 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8141 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8142 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8143 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8144 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_2 := arithEq_of_rows h (row := 126) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_2
  simp only [← c8139, ← c8140, ← c8141] at e_126_2
  have e_126_3 := arithEq_of_rows h (row := 126) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_3
  simp only [← c8142, ← c8143, ← c8144] at e_126_3
  have hr := e_126_3
  simp only [e_126_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 43) = bor (a (.wire 116 39)) (a (.wire 125 35)) := by
  have c8145 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8146 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8147 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8148 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8149 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8150 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_10 := arithEq_of_rows h (row := 116) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_10
  simp only [← c8148, ← c8149, k3, ← c8150] at e_116_10
  have e_124_2 := arithEq_of_rows h (row := 124) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_2
  simp only [← c8145, ← c8146, ← c8147] at e_124_2
  have hr := e_116_10
  simp only [e_124_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 19) = bnot (a (.wire 44 43)) := by
  have c8151 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8152 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8153 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_126_4 := arithEq_of_rows h (row := 126) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_4
  simp only [← c8151, k3, ← c8152, k3, ← c8153] at e_126_4
  have hr := e_126_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 23) = bnot (a (.wire 116 43)) := by
  have c8154 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8155 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8156 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_126_5 := arithEq_of_rows h (row := 126) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_5
  simp only [← c8154, k3, ← c8155, k3, ← c8156] at e_126_5
  have hr := e_126_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 23) = band (a (.wire 126 19)) (a (.wire 126 23)) := by
  have c8157 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8158 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8159 := (publicBatchWrapper86_copies254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_121_5 := arithEq_of_rows h (row := 121) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_5
  simp only [← c8157, ← c8158, ← c8159] at e_121_5
  have hr := e_121_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 31) = bselect (a (.wire 121 23)) (a (.virt 8805)) (a (.wire 125 47)) := by
  have c8160 := (publicBatchWrapper86_copies255 a h).1
  have c8161 := (publicBatchWrapper86_copies255 a h).2.1
  have c8162 := (publicBatchWrapper86_copies255 a h).2.2.1
  have c8163 := (publicBatchWrapper86_copies255 a h).2.2.2.1
  have c8164 := (publicBatchWrapper86_copies255 a h).2.2.2.2.1
  have c8165 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.1
  have e_126_6 := arithEq_of_rows h (row := 126) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_6
  simp only [← c8160, ← c8161, ← c8162] at e_126_6
  have e_126_7 := arithEq_of_rows h (row := 126) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_7
  simp only [← c8163, ← c8164, ← c8165] at e_126_7
  have hr := e_126_7
  simp only [e_126_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 39) = bselect (a (.wire 121 23)) (a (.virt 8806)) (a (.wire 125 55)) := by
  have c8166 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.1
  have c8167 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.1
  have c8168 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.1
  have c8169 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.1
  have c8170 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8171 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_8 := arithEq_of_rows h (row := 126) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_8
  simp only [← c8166, ← c8167, ← c8168] at e_126_8
  have e_126_9 := arithEq_of_rows h (row := 126) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_9
  simp only [← c8169, ← c8170, ← c8171] at e_126_9
  have hr := e_126_9
  simp only [e_126_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 47) = bselect (a (.wire 121 23)) (a (.virt 8807)) (a (.wire 125 63)) := by
  have c8172 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8173 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8174 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8175 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8176 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8177 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_10 := arithEq_of_rows h (row := 126) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_10
  simp only [← c8172, ← c8173, ← c8174] at e_126_10
  have e_126_11 := arithEq_of_rows h (row := 126) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_11
  simp only [← c8175, ← c8176, ← c8177] at e_126_11
  have hr := e_126_11
  simp only [e_126_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 55) = bselect (a (.wire 121 23)) (a (.virt 8808)) (a (.wire 125 71)) := by
  have c8178 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8179 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8180 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8181 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8182 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8183 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_12 := arithEq_of_rows h (row := 126) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_12
  simp only [← c8178, ← c8179, ← c8180] at e_126_12
  have e_126_13 := arithEq_of_rows h (row := 126) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_13
  simp only [← c8181, ← c8182, ← c8183] at e_126_13
  have hr := e_126_13
  simp only [e_126_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 63) = bselect (a (.wire 121 23)) (a (.virt 8809)) (a (.wire 125 79)) := by
  have c8184 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8185 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8186 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8187 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8188 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8189 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_126_14 := arithEq_of_rows h (row := 126) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_14
  simp only [← c8184, ← c8185, ← c8186] at e_126_14
  have e_126_15 := arithEq_of_rows h (row := 126) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_15
  simp only [← c8187, ← c8188, ← c8189] at e_126_15
  have hr := e_126_15
  simp only [e_126_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 71) = bselect (a (.wire 121 23)) (a (.virt 8803)) (a (.wire 126 7)) := by
  have c8190 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8191 := (publicBatchWrapper86_copies255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8192 := (publicBatchWrapper86_copies256 a h).1
  have c8193 := (publicBatchWrapper86_copies256 a h).2.1
  have c8194 := (publicBatchWrapper86_copies256 a h).2.2.1
  have c8195 := (publicBatchWrapper86_copies256 a h).2.2.2.1
  have e_126_16 := arithEq_of_rows h (row := 126) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_16
  simp only [← c8190, ← c8191, ← c8192] at e_126_16
  have e_126_17 := arithEq_of_rows h (row := 126) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_17
  simp only [← c8193, ← c8194, ← c8195] at e_126_17
  have hr := e_126_17
  simp only [e_126_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 126 79) = bselect (a (.wire 121 23)) (a (.virt 8804)) (a (.wire 126 15)) := by
  have c8196 := (publicBatchWrapper86_copies256 a h).2.2.2.2.1
  have c8197 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.1
  have c8198 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.1
  have c8199 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.1
  have c8200 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.1
  have c8201 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.1
  have e_126_18 := arithEq_of_rows h (row := 126) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_18
  simp only [← c8196, ← c8197, ← c8198] at e_126_18
  have e_126_19 := arithEq_of_rows h (row := 126) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_126_19
  simp only [← c8199, ← c8200, ← c8201] at e_126_19
  have hr := e_126_19
  simp only [e_126_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 47) = bor (a (.wire 116 43)) (a (.wire 126 19)) := by
  have c8202 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8203 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8204 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8205 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8206 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8207 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_11 := arithEq_of_rows h (row := 116) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_11
  simp only [← c8205, ← c8206, k3, ← c8207] at e_116_11
  have e_124_3 := arithEq_of_rows h (row := 124) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_3
  simp only [← c8202, ← c8203, ← c8204] at e_124_3
  have hr := e_116_11
  simp only [e_124_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 3) = bnot (a (.wire 46 7)) := by
  have c8208 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8209 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8210 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_127_0 := arithEq_of_rows h (row := 127) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_0
  simp only [← c8208, k3, ← c8209, k3, ← c8210] at e_127_0
  have hr := e_127_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 7) = bnot (a (.wire 116 47)) := by
  have c8211 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8212 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8213 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_127_1 := arithEq_of_rows h (row := 127) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_1
  simp only [← c8211, k3, ← c8212, k3, ← c8213] at e_127_1
  have hr := e_127_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 27) = band (a (.wire 127 3)) (a (.wire 127 7)) := by
  have c8214 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8215 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8216 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_6 := arithEq_of_rows h (row := 121) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_6
  simp only [← c8214, ← c8215, ← c8216] at e_121_6
  have hr := e_121_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 15) = bselect (a (.wire 121 27)) (a (.virt 9005)) (a (.wire 126 31)) := by
  have c8217 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8218 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8219 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8220 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8221 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8222 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_2 := arithEq_of_rows h (row := 127) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_2
  simp only [← c8217, ← c8218, ← c8219] at e_127_2
  have e_127_3 := arithEq_of_rows h (row := 127) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_3
  simp only [← c8220, ← c8221, ← c8222] at e_127_3
  have hr := e_127_3
  simp only [e_127_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 23) = bselect (a (.wire 121 27)) (a (.virt 9006)) (a (.wire 126 39)) := by
  have c8223 := (publicBatchWrapper86_copies256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8224 := (publicBatchWrapper86_copies257 a h).1
  have c8225 := (publicBatchWrapper86_copies257 a h).2.1
  have c8226 := (publicBatchWrapper86_copies257 a h).2.2.1
  have c8227 := (publicBatchWrapper86_copies257 a h).2.2.2.1
  have c8228 := (publicBatchWrapper86_copies257 a h).2.2.2.2.1
  have e_127_4 := arithEq_of_rows h (row := 127) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_4
  simp only [← c8223, ← c8224, ← c8225] at e_127_4
  have e_127_5 := arithEq_of_rows h (row := 127) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_5
  simp only [← c8226, ← c8227, ← c8228] at e_127_5
  have hr := e_127_5
  simp only [e_127_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 31) = bselect (a (.wire 121 27)) (a (.virt 9007)) (a (.wire 126 47)) := by
  have c8229 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.1
  have c8230 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.1
  have c8231 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.1
  have c8232 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.1
  have c8233 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.1
  have c8234 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_127_6 := arithEq_of_rows h (row := 127) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_6
  simp only [← c8229, ← c8230, ← c8231] at e_127_6
  have e_127_7 := arithEq_of_rows h (row := 127) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_7
  simp only [← c8232, ← c8233, ← c8234] at e_127_7
  have hr := e_127_7
  simp only [e_127_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 39) = bselect (a (.wire 121 27)) (a (.virt 9008)) (a (.wire 126 55)) := by
  have c8235 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8236 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8237 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8238 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8239 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8240 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_8 := arithEq_of_rows h (row := 127) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_8
  simp only [← c8235, ← c8236, ← c8237] at e_127_8
  have e_127_9 := arithEq_of_rows h (row := 127) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_9
  simp only [← c8238, ← c8239, ← c8240] at e_127_9
  have hr := e_127_9
  simp only [e_127_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 47) = bselect (a (.wire 121 27)) (a (.virt 9009)) (a (.wire 126 63)) := by
  have c8241 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8242 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8243 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8244 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8245 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8246 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_10 := arithEq_of_rows h (row := 127) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_10
  simp only [← c8241, ← c8242, ← c8243] at e_127_10
  have e_127_11 := arithEq_of_rows h (row := 127) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_11
  simp only [← c8244, ← c8245, ← c8246] at e_127_11
  have hr := e_127_11
  simp only [e_127_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 55) = bselect (a (.wire 121 27)) (a (.virt 9003)) (a (.wire 126 71)) := by
  have c8247 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8248 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8249 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8250 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8251 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8252 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_12 := arithEq_of_rows h (row := 127) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_12
  simp only [← c8247, ← c8248, ← c8249] at e_127_12
  have e_127_13 := arithEq_of_rows h (row := 127) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_13
  simp only [← c8250, ← c8251, ← c8252] at e_127_13
  have hr := e_127_13
  simp only [e_127_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 63) = bselect (a (.wire 121 27)) (a (.virt 9004)) (a (.wire 126 79)) := by
  have c8253 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8254 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8255 := (publicBatchWrapper86_copies257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8256 := (publicBatchWrapper86_copies258 a h).1
  have c8257 := (publicBatchWrapper86_copies258 a h).2.1
  have c8258 := (publicBatchWrapper86_copies258 a h).2.2.1
  have e_127_14 := arithEq_of_rows h (row := 127) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_14
  simp only [← c8253, ← c8254, ← c8255] at e_127_14
  have e_127_15 := arithEq_of_rows h (row := 127) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_15
  simp only [← c8256, ← c8257, ← c8258] at e_127_15
  have hr := e_127_15
  simp only [e_127_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 51) = bor (a (.wire 116 47)) (a (.wire 127 3)) := by
  have c8259 := (publicBatchWrapper86_copies258 a h).2.2.2.1
  have c8260 := (publicBatchWrapper86_copies258 a h).2.2.2.2.1
  have c8261 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.1
  have c8262 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.1
  have c8263 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.1
  have c8264 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_12 := arithEq_of_rows h (row := 116) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_12
  simp only [← c8262, ← c8263, k3, ← c8264] at e_116_12
  have e_124_4 := arithEq_of_rows h (row := 124) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_4
  simp only [← c8259, ← c8260, ← c8261] at e_124_4
  have hr := e_116_12
  simp only [e_124_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 67) = bnot (a (.wire 46 51)) := by
  have c8265 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.1
  have c8266 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8267 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_127_16 := arithEq_of_rows h (row := 127) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_16
  simp only [← c8265, k3, ← c8266, k3, ← c8267] at e_127_16
  have hr := e_127_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 71) = bnot (a (.wire 116 51)) := by
  have c8268 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8269 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8270 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_127_17 := arithEq_of_rows h (row := 127) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_17
  simp only [← c8268, k3, ← c8269, k3, ← c8270] at e_127_17
  have hr := e_127_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 31) = band (a (.wire 127 67)) (a (.wire 127 71)) := by
  have c8271 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8272 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8273 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_7 := arithEq_of_rows h (row := 121) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_7
  simp only [← c8271, ← c8272, ← c8273] at e_121_7
  have hr := e_121_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 127 79) = bselect (a (.wire 121 31)) (a (.virt 9205)) (a (.wire 127 15)) := by
  have c8274 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8275 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8276 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8277 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8278 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8279 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_127_18 := arithEq_of_rows h (row := 127) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_18
  simp only [← c8274, ← c8275, ← c8276] at e_127_18
  have e_127_19 := arithEq_of_rows h (row := 127) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_127_19
  simp only [← c8277, ← c8278, ← c8279] at e_127_19
  have hr := e_127_19
  simp only [e_127_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 7) = bselect (a (.wire 121 31)) (a (.virt 9206)) (a (.wire 127 23)) := by
  have c8280 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8281 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8282 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8283 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8284 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8285 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_0 := arithEq_of_rows h (row := 128) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_0
  simp only [← c8280, ← c8281, ← c8282] at e_128_0
  have e_128_1 := arithEq_of_rows h (row := 128) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_1
  simp only [← c8283, ← c8284, ← c8285] at e_128_1
  have hr := e_128_1
  simp only [e_128_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 15) = bselect (a (.wire 121 31)) (a (.virt 9207)) (a (.wire 127 31)) := by
  have c8286 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8287 := (publicBatchWrapper86_copies258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8288 := (publicBatchWrapper86_copies259 a h).1
  have c8289 := (publicBatchWrapper86_copies259 a h).2.1
  have c8290 := (publicBatchWrapper86_copies259 a h).2.2.1
  have c8291 := (publicBatchWrapper86_copies259 a h).2.2.2.1
  have e_128_2 := arithEq_of_rows h (row := 128) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_2
  simp only [← c8286, ← c8287, ← c8288] at e_128_2
  have e_128_3 := arithEq_of_rows h (row := 128) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_3
  simp only [← c8289, ← c8290, ← c8291] at e_128_3
  have hr := e_128_3
  simp only [e_128_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 23) = bselect (a (.wire 121 31)) (a (.virt 9208)) (a (.wire 127 39)) := by
  have c8292 := (publicBatchWrapper86_copies259 a h).2.2.2.2.1
  have c8293 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.1
  have c8294 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.1
  have c8295 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.1
  have c8296 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.1
  have c8297 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.1
  have e_128_4 := arithEq_of_rows h (row := 128) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_4
  simp only [← c8292, ← c8293, ← c8294] at e_128_4
  have e_128_5 := arithEq_of_rows h (row := 128) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_5
  simp only [← c8295, ← c8296, ← c8297] at e_128_5
  have hr := e_128_5
  simp only [e_128_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 31) = bselect (a (.wire 121 31)) (a (.virt 9209)) (a (.wire 127 47)) := by
  have c8298 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8299 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8300 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8301 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8302 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8303 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_6 := arithEq_of_rows h (row := 128) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_6
  simp only [← c8298, ← c8299, ← c8300] at e_128_6
  have e_128_7 := arithEq_of_rows h (row := 128) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_7
  simp only [← c8301, ← c8302, ← c8303] at e_128_7
  have hr := e_128_7
  simp only [e_128_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 39) = bselect (a (.wire 121 31)) (a (.virt 9203)) (a (.wire 127 55)) := by
  have c8304 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8305 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8306 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8307 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8308 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8309 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_8 := arithEq_of_rows h (row := 128) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_8
  simp only [← c8304, ← c8305, ← c8306] at e_128_8
  have e_128_9 := arithEq_of_rows h (row := 128) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_9
  simp only [← c8307, ← c8308, ← c8309] at e_128_9
  have hr := e_128_9
  simp only [e_128_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 47) = bselect (a (.wire 121 31)) (a (.virt 9204)) (a (.wire 127 63)) := by
  have c8310 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8311 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8312 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8313 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8314 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8315 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_10 := arithEq_of_rows h (row := 128) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_10
  simp only [← c8310, ← c8311, ← c8312] at e_128_10
  have e_128_11 := arithEq_of_rows h (row := 128) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_11
  simp only [← c8313, ← c8314, ← c8315] at e_128_11
  have hr := e_128_11
  simp only [e_128_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 55) = bor (a (.wire 116 51)) (a (.wire 127 67)) := by
  have c8316 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8317 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8318 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8319 := (publicBatchWrapper86_copies259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8320 := (publicBatchWrapper86_copies260 a h).1
  have c8321 := (publicBatchWrapper86_copies260 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_13 := arithEq_of_rows h (row := 116) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_13
  simp only [← c8319, ← c8320, k3, ← c8321] at e_116_13
  have e_124_5 := arithEq_of_rows h (row := 124) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_5
  simp only [← c8316, ← c8317, ← c8318] at e_124_5
  have hr := e_116_13
  simp only [e_124_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 51) = bnot (a (.wire 48 15)) := by
  have c8322 := (publicBatchWrapper86_copies260 a h).2.2.1
  have c8323 := (publicBatchWrapper86_copies260 a h).2.2.2.1
  have c8324 := (publicBatchWrapper86_copies260 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_128_12 := arithEq_of_rows h (row := 128) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_12
  simp only [← c8322, k3, ← c8323, k3, ← c8324] at e_128_12
  have hr := e_128_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 55) = bnot (a (.wire 116 55)) := by
  have c8325 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.1
  have c8326 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.1
  have c8327 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_128_13 := arithEq_of_rows h (row := 128) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_13
  simp only [← c8325, k3, ← c8326, k3, ← c8327] at e_128_13
  have hr := e_128_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 35) = band (a (.wire 128 51)) (a (.wire 128 55)) := by
  have c8328 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.1
  have c8329 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.1
  have c8330 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_121_8 := arithEq_of_rows h (row := 121) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_8
  simp only [← c8328, ← c8329, ← c8330] at e_121_8
  have hr := e_121_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 63) = bselect (a (.wire 121 35)) (a (.virt 9405)) (a (.wire 127 79)) := by
  have c8331 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8332 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8333 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8334 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8335 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8336 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_14 := arithEq_of_rows h (row := 128) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_14
  simp only [← c8331, ← c8332, ← c8333] at e_128_14
  have e_128_15 := arithEq_of_rows h (row := 128) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_15
  simp only [← c8334, ← c8335, ← c8336] at e_128_15
  have hr := e_128_15
  simp only [e_128_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 71) = bselect (a (.wire 121 35)) (a (.virt 9406)) (a (.wire 128 7)) := by
  have c8337 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8338 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8339 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8340 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8341 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8342 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_16 := arithEq_of_rows h (row := 128) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_16
  simp only [← c8337, ← c8338, ← c8339] at e_128_16
  have e_128_17 := arithEq_of_rows h (row := 128) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_17
  simp only [← c8340, ← c8341, ← c8342] at e_128_17
  have hr := e_128_17
  simp only [e_128_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 128 79) = bselect (a (.wire 121 35)) (a (.virt 9407)) (a (.wire 128 15)) := by
  have c8343 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8344 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8345 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8346 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8347 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8348 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_128_18 := arithEq_of_rows h (row := 128) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_18
  simp only [← c8343, ← c8344, ← c8345] at e_128_18
  have e_128_19 := arithEq_of_rows h (row := 128) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_128_19
  simp only [← c8346, ← c8347, ← c8348] at e_128_19
  have hr := e_128_19
  simp only [e_128_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 7) = bselect (a (.wire 121 35)) (a (.virt 9408)) (a (.wire 128 23)) := by
  have c8349 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8350 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8351 := (publicBatchWrapper86_copies260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8352 := (publicBatchWrapper86_copies261 a h).1
  have c8353 := (publicBatchWrapper86_copies261 a h).2.1
  have c8354 := (publicBatchWrapper86_copies261 a h).2.2.1
  have e_129_0 := arithEq_of_rows h (row := 129) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_0
  simp only [← c8349, ← c8350, ← c8351] at e_129_0
  have e_129_1 := arithEq_of_rows h (row := 129) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_1
  simp only [← c8352, ← c8353, ← c8354] at e_129_1
  have hr := e_129_1
  simp only [e_129_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 15) = bselect (a (.wire 121 35)) (a (.virt 9409)) (a (.wire 128 31)) := by
  have c8355 := (publicBatchWrapper86_copies261 a h).2.2.2.1
  have c8356 := (publicBatchWrapper86_copies261 a h).2.2.2.2.1
  have c8357 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.1
  have c8358 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.1
  have c8359 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.1
  have c8360 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.1
  have e_129_2 := arithEq_of_rows h (row := 129) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_2
  simp only [← c8355, ← c8356, ← c8357] at e_129_2
  have e_129_3 := arithEq_of_rows h (row := 129) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_3
  simp only [← c8358, ← c8359, ← c8360] at e_129_3
  have hr := e_129_3
  simp only [e_129_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 23) = bselect (a (.wire 121 35)) (a (.virt 9403)) (a (.wire 128 39)) := by
  have c8361 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.1
  have c8362 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8363 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8364 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8365 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8366 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_4 := arithEq_of_rows h (row := 129) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_4
  simp only [← c8361, ← c8362, ← c8363] at e_129_4
  have e_129_5 := arithEq_of_rows h (row := 129) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_5
  simp only [← c8364, ← c8365, ← c8366] at e_129_5
  have hr := e_129_5
  simp only [e_129_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 31) = bselect (a (.wire 121 35)) (a (.virt 9404)) (a (.wire 128 47)) := by
  have c8367 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8368 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8369 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8370 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8371 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8372 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_6 := arithEq_of_rows h (row := 129) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_6
  simp only [← c8367, ← c8368, ← c8369] at e_129_6
  have e_129_7 := arithEq_of_rows h (row := 129) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_7
  simp only [← c8370, ← c8371, ← c8372] at e_129_7
  have hr := e_129_7
  simp only [e_129_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 59) = bor (a (.wire 116 55)) (a (.wire 128 51)) := by
  have c8373 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8374 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8375 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8376 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8377 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8378 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_14 := arithEq_of_rows h (row := 116) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_14
  simp only [← c8376, ← c8377, k3, ← c8378] at e_116_14
  have e_124_6 := arithEq_of_rows h (row := 124) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_6
  simp only [← c8373, ← c8374, ← c8375] at e_124_6
  have hr := e_116_14
  simp only [e_124_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 35) = bnot (a (.wire 48 59)) := by
  have c8379 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8380 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8381 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_129_8 := arithEq_of_rows h (row := 129) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_8
  simp only [← c8379, k3, ← c8380, k3, ← c8381] at e_129_8
  have hr := e_129_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 39) = bnot (a (.wire 116 59)) := by
  have c8382 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8383 := (publicBatchWrapper86_copies261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8384 := (publicBatchWrapper86_copies262 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_129_9 := arithEq_of_rows h (row := 129) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_9
  simp only [← c8382, k3, ← c8383, k3, ← c8384] at e_129_9
  have hr := e_129_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 39) = band (a (.wire 129 35)) (a (.wire 129 39)) := by
  have c8385 := (publicBatchWrapper86_copies262 a h).2.1
  have c8386 := (publicBatchWrapper86_copies262 a h).2.2.1
  have c8387 := (publicBatchWrapper86_copies262 a h).2.2.2.1
  have e_121_9 := arithEq_of_rows h (row := 121) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_9
  simp only [← c8385, ← c8386, ← c8387] at e_121_9
  have hr := e_121_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 47) = bselect (a (.wire 121 39)) (a (.virt 9605)) (a (.wire 128 63)) := by
  have c8388 := (publicBatchWrapper86_copies262 a h).2.2.2.2.1
  have c8389 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.1
  have c8390 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.1
  have c8391 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.1
  have c8392 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.1
  have c8393 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.1
  have e_129_10 := arithEq_of_rows h (row := 129) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_10
  simp only [← c8388, ← c8389, ← c8390] at e_129_10
  have e_129_11 := arithEq_of_rows h (row := 129) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_11
  simp only [← c8391, ← c8392, ← c8393] at e_129_11
  have hr := e_129_11
  simp only [e_129_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 55) = bselect (a (.wire 121 39)) (a (.virt 9606)) (a (.wire 128 71)) := by
  have c8394 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8395 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8396 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8397 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8398 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8399 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_12 := arithEq_of_rows h (row := 129) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_12
  simp only [← c8394, ← c8395, ← c8396] at e_129_12
  have e_129_13 := arithEq_of_rows h (row := 129) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_13
  simp only [← c8397, ← c8398, ← c8399] at e_129_13
  have hr := e_129_13
  simp only [e_129_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 63) = bselect (a (.wire 121 39)) (a (.virt 9607)) (a (.wire 128 79)) := by
  have c8400 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8401 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8402 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8403 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8404 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8405 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_14 := arithEq_of_rows h (row := 129) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_14
  simp only [← c8400, ← c8401, ← c8402] at e_129_14
  have e_129_15 := arithEq_of_rows h (row := 129) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_15
  simp only [← c8403, ← c8404, ← c8405] at e_129_15
  have hr := e_129_15
  simp only [e_129_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 71) = bselect (a (.wire 121 39)) (a (.virt 9608)) (a (.wire 129 7)) := by
  have c8406 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8407 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8408 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8409 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8410 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8411 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_129_16 := arithEq_of_rows h (row := 129) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_16
  simp only [← c8406, ← c8407, ← c8408] at e_129_16
  have e_129_17 := arithEq_of_rows h (row := 129) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_17
  simp only [← c8409, ← c8410, ← c8411] at e_129_17
  have hr := e_129_17
  simp only [e_129_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 129 79) = bselect (a (.wire 121 39)) (a (.virt 9609)) (a (.wire 129 15)) := by
  have c8412 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8413 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8414 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8415 := (publicBatchWrapper86_copies262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8416 := (publicBatchWrapper86_copies263 a h).1
  have c8417 := (publicBatchWrapper86_copies263 a h).2.1
  have e_129_18 := arithEq_of_rows h (row := 129) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_18
  simp only [← c8412, ← c8413, ← c8414] at e_129_18
  have e_129_19 := arithEq_of_rows h (row := 129) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_129_19
  simp only [← c8415, ← c8416, ← c8417] at e_129_19
  have hr := e_129_19
  simp only [e_129_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 7) = bselect (a (.wire 121 39)) (a (.virt 9603)) (a (.wire 129 23)) := by
  have c8418 := (publicBatchWrapper86_copies263 a h).2.2.1
  have c8419 := (publicBatchWrapper86_copies263 a h).2.2.2.1
  have c8420 := (publicBatchWrapper86_copies263 a h).2.2.2.2.1
  have c8421 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.1
  have c8422 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.1
  have c8423 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.1
  have e_130_0 := arithEq_of_rows h (row := 130) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_0
  simp only [← c8418, ← c8419, ← c8420] at e_130_0
  have e_130_1 := arithEq_of_rows h (row := 130) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_1
  simp only [← c8421, ← c8422, ← c8423] at e_130_1
  have hr := e_130_1
  simp only [e_130_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 15) = bselect (a (.wire 121 39)) (a (.virt 9604)) (a (.wire 129 31)) := by
  have c8424 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.1
  have c8425 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.1
  have c8426 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8427 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8428 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8429 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_130_2 := arithEq_of_rows h (row := 130) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_2
  simp only [← c8424, ← c8425, ← c8426] at e_130_2
  have e_130_3 := arithEq_of_rows h (row := 130) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_3
  simp only [← c8427, ← c8428, ← c8429] at e_130_3
  have hr := e_130_3
  simp only [e_130_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 63) = bor (a (.wire 116 59)) (a (.wire 129 35)) := by
  have c8430 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8431 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8432 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8433 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8434 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8435 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_15 := arithEq_of_rows h (row := 116) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_15
  simp only [← c8433, ← c8434, k3, ← c8435] at e_116_15
  have e_124_7 := arithEq_of_rows h (row := 124) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_7
  simp only [← c8430, ← c8431, ← c8432] at e_124_7
  have hr := e_116_15
  simp only [e_124_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 19) = bnot (a (.wire 49 23)) := by
  have c8436 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8437 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8438 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_130_4 := arithEq_of_rows h (row := 130) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_4
  simp only [← c8436, k3, ← c8437, k3, ← c8438] at e_130_4
  have hr := e_130_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 23) = bnot (a (.wire 116 63)) := by
  have c8439 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8440 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8441 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_130_5 := arithEq_of_rows h (row := 130) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_5
  simp only [← c8439, k3, ← c8440, k3, ← c8441] at e_130_5
  have hr := e_130_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 43) = band (a (.wire 130 19)) (a (.wire 130 23)) := by
  have c8442 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8443 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8444 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_10 := arithEq_of_rows h (row := 121) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_10
  simp only [← c8442, ← c8443, ← c8444] at e_121_10
  have hr := e_121_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 31) = bselect (a (.wire 121 43)) (a (.virt 9805)) (a (.wire 129 47)) := by
  have c8445 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8446 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8447 := (publicBatchWrapper86_copies263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8448 := (publicBatchWrapper86_copies264 a h).1
  have c8449 := (publicBatchWrapper86_copies264 a h).2.1
  have c8450 := (publicBatchWrapper86_copies264 a h).2.2.1
  have e_130_6 := arithEq_of_rows h (row := 130) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_6
  simp only [← c8445, ← c8446, ← c8447] at e_130_6
  have e_130_7 := arithEq_of_rows h (row := 130) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_7
  simp only [← c8448, ← c8449, ← c8450] at e_130_7
  have hr := e_130_7
  simp only [e_130_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 39) = bselect (a (.wire 121 43)) (a (.virt 9806)) (a (.wire 129 55)) := by
  have c8451 := (publicBatchWrapper86_copies264 a h).2.2.2.1
  have c8452 := (publicBatchWrapper86_copies264 a h).2.2.2.2.1
  have c8453 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.1
  have c8454 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.1
  have c8455 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.1
  have c8456 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.1
  have e_130_8 := arithEq_of_rows h (row := 130) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_8
  simp only [← c8451, ← c8452, ← c8453] at e_130_8
  have e_130_9 := arithEq_of_rows h (row := 130) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_9
  simp only [← c8454, ← c8455, ← c8456] at e_130_9
  have hr := e_130_9
  simp only [e_130_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 47) = bselect (a (.wire 121 43)) (a (.virt 9807)) (a (.wire 129 63)) := by
  have c8457 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.1
  have c8458 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8459 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8460 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8461 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8462 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_130_10 := arithEq_of_rows h (row := 130) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_10
  simp only [← c8457, ← c8458, ← c8459] at e_130_10
  have e_130_11 := arithEq_of_rows h (row := 130) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_11
  simp only [← c8460, ← c8461, ← c8462] at e_130_11
  have hr := e_130_11
  simp only [e_130_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 55) = bselect (a (.wire 121 43)) (a (.virt 9808)) (a (.wire 129 71)) := by
  have c8463 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8464 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8465 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8466 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8467 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8468 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_130_12 := arithEq_of_rows h (row := 130) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_12
  simp only [← c8463, ← c8464, ← c8465] at e_130_12
  have e_130_13 := arithEq_of_rows h (row := 130) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_13
  simp only [← c8466, ← c8467, ← c8468] at e_130_13
  have hr := e_130_13
  simp only [e_130_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 63) = bselect (a (.wire 121 43)) (a (.virt 9809)) (a (.wire 129 79)) := by
  have c8469 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8470 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8471 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8472 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8473 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8474 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_130_14 := arithEq_of_rows h (row := 130) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_14
  simp only [← c8469, ← c8470, ← c8471] at e_130_14
  have e_130_15 := arithEq_of_rows h (row := 130) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_15
  simp only [← c8472, ← c8473, ← c8474] at e_130_15
  have hr := e_130_15
  simp only [e_130_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 71) = bselect (a (.wire 121 43)) (a (.virt 9803)) (a (.wire 130 7)) := by
  have c8475 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8476 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8477 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8478 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8479 := (publicBatchWrapper86_copies264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8480 := (publicBatchWrapper86_copies265 a h).1
  have e_130_16 := arithEq_of_rows h (row := 130) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_16
  simp only [← c8475, ← c8476, ← c8477] at e_130_16
  have e_130_17 := arithEq_of_rows h (row := 130) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_17
  simp only [← c8478, ← c8479, ← c8480] at e_130_17
  have hr := e_130_17
  simp only [e_130_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 130 79) = bselect (a (.wire 121 43)) (a (.virt 9804)) (a (.wire 130 15)) := by
  have c8481 := (publicBatchWrapper86_copies265 a h).2.1
  have c8482 := (publicBatchWrapper86_copies265 a h).2.2.1
  have c8483 := (publicBatchWrapper86_copies265 a h).2.2.2.1
  have c8484 := (publicBatchWrapper86_copies265 a h).2.2.2.2.1
  have c8485 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.1
  have c8486 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.1
  have e_130_18 := arithEq_of_rows h (row := 130) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_18
  simp only [← c8481, ← c8482, ← c8483] at e_130_18
  have e_130_19 := arithEq_of_rows h (row := 130) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_130_19
  simp only [← c8484, ← c8485, ← c8486] at e_130_19
  have hr := e_130_19
  simp only [e_130_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 67) = bor (a (.wire 116 63)) (a (.wire 130 19)) := by
  have c8487 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.1
  have c8488 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.1
  have c8489 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.1
  have c8490 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8491 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8492 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_16 := arithEq_of_rows h (row := 116) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_16
  simp only [← c8490, ← c8491, k3, ← c8492] at e_116_16
  have e_124_8 := arithEq_of_rows h (row := 124) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_8
  simp only [← c8487, ← c8488, ← c8489] at e_124_8
  have hr := e_116_16
  simp only [e_124_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 3) = bnot (a (.wire 49 67)) := by
  have c8493 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8494 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8495 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_131_0 := arithEq_of_rows h (row := 131) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_0
  simp only [← c8493, k3, ← c8494, k3, ← c8495] at e_131_0
  have hr := e_131_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 7) = bnot (a (.wire 116 67)) := by
  have c8496 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8497 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8498 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_131_1 := arithEq_of_rows h (row := 131) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_1
  simp only [← c8496, k3, ← c8497, k3, ← c8498] at e_131_1
  have hr := e_131_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 47) = band (a (.wire 131 3)) (a (.wire 131 7)) := by
  have c8499 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8500 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8501 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_11 := arithEq_of_rows h (row := 121) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_11
  simp only [← c8499, ← c8500, ← c8501] at e_121_11
  have hr := e_121_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 15) = bselect (a (.wire 121 47)) (a (.virt 10005)) (a (.wire 130 31)) := by
  have c8502 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8503 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8504 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8505 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8506 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8507 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_2 := arithEq_of_rows h (row := 131) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_2
  simp only [← c8502, ← c8503, ← c8504] at e_131_2
  have e_131_3 := arithEq_of_rows h (row := 131) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_3
  simp only [← c8505, ← c8506, ← c8507] at e_131_3
  have hr := e_131_3
  simp only [e_131_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 23) = bselect (a (.wire 121 47)) (a (.virt 10006)) (a (.wire 130 39)) := by
  have c8508 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8509 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8510 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8511 := (publicBatchWrapper86_copies265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8512 := (publicBatchWrapper86_copies266 a h).1
  have c8513 := (publicBatchWrapper86_copies266 a h).2.1
  have e_131_4 := arithEq_of_rows h (row := 131) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_4
  simp only [← c8508, ← c8509, ← c8510] at e_131_4
  have e_131_5 := arithEq_of_rows h (row := 131) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_5
  simp only [← c8511, ← c8512, ← c8513] at e_131_5
  have hr := e_131_5
  simp only [e_131_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 31) = bselect (a (.wire 121 47)) (a (.virt 10007)) (a (.wire 130 47)) := by
  have c8514 := (publicBatchWrapper86_copies266 a h).2.2.1
  have c8515 := (publicBatchWrapper86_copies266 a h).2.2.2.1
  have c8516 := (publicBatchWrapper86_copies266 a h).2.2.2.2.1
  have c8517 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.1
  have c8518 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.1
  have c8519 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.1
  have e_131_6 := arithEq_of_rows h (row := 131) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_6
  simp only [← c8514, ← c8515, ← c8516] at e_131_6
  have e_131_7 := arithEq_of_rows h (row := 131) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_7
  simp only [← c8517, ← c8518, ← c8519] at e_131_7
  have hr := e_131_7
  simp only [e_131_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 39) = bselect (a (.wire 121 47)) (a (.virt 10008)) (a (.wire 130 55)) := by
  have c8520 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.1
  have c8521 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.1
  have c8522 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8523 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8524 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8525 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_8 := arithEq_of_rows h (row := 131) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_8
  simp only [← c8520, ← c8521, ← c8522] at e_131_8
  have e_131_9 := arithEq_of_rows h (row := 131) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_9
  simp only [← c8523, ← c8524, ← c8525] at e_131_9
  have hr := e_131_9
  simp only [e_131_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 47) = bselect (a (.wire 121 47)) (a (.virt 10009)) (a (.wire 130 63)) := by
  have c8526 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8527 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8528 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8529 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8530 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8531 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_10 := arithEq_of_rows h (row := 131) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_10
  simp only [← c8526, ← c8527, ← c8528] at e_131_10
  have e_131_11 := arithEq_of_rows h (row := 131) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_11
  simp only [← c8529, ← c8530, ← c8531] at e_131_11
  have hr := e_131_11
  simp only [e_131_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 55) = bselect (a (.wire 121 47)) (a (.virt 10003)) (a (.wire 130 71)) := by
  have c8532 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8533 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8534 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8535 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8536 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8537 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_12 := arithEq_of_rows h (row := 131) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_12
  simp only [← c8532, ← c8533, ← c8534] at e_131_12
  have e_131_13 := arithEq_of_rows h (row := 131) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_13
  simp only [← c8535, ← c8536, ← c8537] at e_131_13
  have hr := e_131_13
  simp only [e_131_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 63) = bselect (a (.wire 121 47)) (a (.virt 10004)) (a (.wire 130 79)) := by
  have c8538 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8539 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8540 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8541 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8542 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8543 := (publicBatchWrapper86_copies266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_131_14 := arithEq_of_rows h (row := 131) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_14
  simp only [← c8538, ← c8539, ← c8540] at e_131_14
  have e_131_15 := arithEq_of_rows h (row := 131) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_15
  simp only [← c8541, ← c8542, ← c8543] at e_131_15
  have hr := e_131_15
  simp only [e_131_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 71) = bor (a (.wire 116 67)) (a (.wire 131 3)) := by
  have c8544 := (publicBatchWrapper86_copies267 a h).1
  have c8545 := (publicBatchWrapper86_copies267 a h).2.1
  have c8546 := (publicBatchWrapper86_copies267 a h).2.2.1
  have c8547 := (publicBatchWrapper86_copies267 a h).2.2.2.1
  have c8548 := (publicBatchWrapper86_copies267 a h).2.2.2.2.1
  have c8549 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_17 := arithEq_of_rows h (row := 116) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_17
  simp only [← c8547, ← c8548, k3, ← c8549] at e_116_17
  have e_124_9 := arithEq_of_rows h (row := 124) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_9
  simp only [← c8544, ← c8545, ← c8546] at e_124_9
  have hr := e_116_17
  simp only [e_124_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 67) = bnot (a (.wire 51 31)) := by
  have c8550 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.1
  have c8551 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.1
  have c8552 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_131_16 := arithEq_of_rows h (row := 131) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_16
  simp only [← c8550, k3, ← c8551, k3, ← c8552] at e_131_16
  have hr := e_131_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 71) = bnot (a (.wire 116 71)) := by
  have c8553 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.1
  have c8554 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8555 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_131_17 := arithEq_of_rows h (row := 131) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_17
  simp only [← c8553, k3, ← c8554, k3, ← c8555] at e_131_17
  have hr := e_131_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 51) = band (a (.wire 131 67)) (a (.wire 131 71)) := by
  have c8556 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8557 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8558 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_12 := arithEq_of_rows h (row := 121) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_12
  simp only [← c8556, ← c8557, ← c8558] at e_121_12
  have hr := e_121_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 131 79) = bselect (a (.wire 121 51)) (a (.virt 10205)) (a (.wire 131 15)) := by
  have c8559 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8560 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8561 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8562 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8563 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8564 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_131_18 := arithEq_of_rows h (row := 131) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_18
  simp only [← c8559, ← c8560, ← c8561] at e_131_18
  have e_131_19 := arithEq_of_rows h (row := 131) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_131_19
  simp only [← c8562, ← c8563, ← c8564] at e_131_19
  have hr := e_131_19
  simp only [e_131_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 7) = bselect (a (.wire 121 51)) (a (.virt 10206)) (a (.wire 131 23)) := by
  have c8565 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8566 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8567 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8568 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8569 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8570 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_0 := arithEq_of_rows h (row := 132) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_0
  simp only [← c8565, ← c8566, ← c8567] at e_132_0
  have e_132_1 := arithEq_of_rows h (row := 132) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_1
  simp only [← c8568, ← c8569, ← c8570] at e_132_1
  have hr := e_132_1
  simp only [e_132_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 15) = bselect (a (.wire 121 51)) (a (.virt 10207)) (a (.wire 131 31)) := by
  have c8571 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8572 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8573 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8574 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8575 := (publicBatchWrapper86_copies267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8576 := (publicBatchWrapper86_copies268 a h).1
  have e_132_2 := arithEq_of_rows h (row := 132) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_2
  simp only [← c8571, ← c8572, ← c8573] at e_132_2
  have e_132_3 := arithEq_of_rows h (row := 132) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_3
  simp only [← c8574, ← c8575, ← c8576] at e_132_3
  have hr := e_132_3
  simp only [e_132_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 23) = bselect (a (.wire 121 51)) (a (.virt 10208)) (a (.wire 131 39)) := by
  have c8577 := (publicBatchWrapper86_copies268 a h).2.1
  have c8578 := (publicBatchWrapper86_copies268 a h).2.2.1
  have c8579 := (publicBatchWrapper86_copies268 a h).2.2.2.1
  have c8580 := (publicBatchWrapper86_copies268 a h).2.2.2.2.1
  have c8581 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.1
  have c8582 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.1
  have e_132_4 := arithEq_of_rows h (row := 132) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_4
  simp only [← c8577, ← c8578, ← c8579] at e_132_4
  have e_132_5 := arithEq_of_rows h (row := 132) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_5
  simp only [← c8580, ← c8581, ← c8582] at e_132_5
  have hr := e_132_5
  simp only [e_132_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 31) = bselect (a (.wire 121 51)) (a (.virt 10209)) (a (.wire 131 47)) := by
  have c8583 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.1
  have c8584 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.1
  have c8585 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.1
  have c8586 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8587 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8588 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_6 := arithEq_of_rows h (row := 132) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_6
  simp only [← c8583, ← c8584, ← c8585] at e_132_6
  have e_132_7 := arithEq_of_rows h (row := 132) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_7
  simp only [← c8586, ← c8587, ← c8588] at e_132_7
  have hr := e_132_7
  simp only [e_132_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 39) = bselect (a (.wire 121 51)) (a (.virt 10203)) (a (.wire 131 55)) := by
  have c8589 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8590 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8591 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8592 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8593 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8594 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_8 := arithEq_of_rows h (row := 132) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_8
  simp only [← c8589, ← c8590, ← c8591] at e_132_8
  have e_132_9 := arithEq_of_rows h (row := 132) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_9
  simp only [← c8592, ← c8593, ← c8594] at e_132_9
  have hr := e_132_9
  simp only [e_132_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 47) = bselect (a (.wire 121 51)) (a (.virt 10204)) (a (.wire 131 63)) := by
  have c8595 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8596 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8597 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8598 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8599 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8600 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_10 := arithEq_of_rows h (row := 132) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_10
  simp only [← c8595, ← c8596, ← c8597] at e_132_10
  have e_132_11 := arithEq_of_rows h (row := 132) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_11
  simp only [← c8598, ← c8599, ← c8600] at e_132_11
  have hr := e_132_11
  simp only [e_132_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 75) = bor (a (.wire 116 71)) (a (.wire 131 67)) := by
  have c8601 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8602 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8603 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8604 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8605 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8606 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_18 := arithEq_of_rows h (row := 116) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_18
  simp only [← c8604, ← c8605, k3, ← c8606] at e_116_18
  have e_124_10 := arithEq_of_rows h (row := 124) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_10
  simp only [← c8601, ← c8602, ← c8603] at e_124_10
  have hr := e_116_18
  simp only [e_124_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 51) = bnot (a (.wire 51 75)) := by
  have c8607 := (publicBatchWrapper86_copies268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8608 := (publicBatchWrapper86_copies269 a h).1
  have c8609 := (publicBatchWrapper86_copies269 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_132_12 := arithEq_of_rows h (row := 132) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_12
  simp only [← c8607, k3, ← c8608, k3, ← c8609] at e_132_12
  have hr := e_132_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 55) = bnot (a (.wire 116 75)) := by
  have c8610 := (publicBatchWrapper86_copies269 a h).2.2.1
  have c8611 := (publicBatchWrapper86_copies269 a h).2.2.2.1
  have c8612 := (publicBatchWrapper86_copies269 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_132_13 := arithEq_of_rows h (row := 132) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_13
  simp only [← c8610, k3, ← c8611, k3, ← c8612] at e_132_13
  have hr := e_132_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 55) = band (a (.wire 132 51)) (a (.wire 132 55)) := by
  have c8613 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.1
  have c8614 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.1
  have c8615 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.1
  have e_121_13 := arithEq_of_rows h (row := 121) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_13
  simp only [← c8613, ← c8614, ← c8615] at e_121_13
  have hr := e_121_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 63) = bselect (a (.wire 121 55)) (a (.virt 10405)) (a (.wire 131 79)) := by
  have c8616 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.1
  have c8617 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.1
  have c8618 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8619 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8620 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8621 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_14 := arithEq_of_rows h (row := 132) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_14
  simp only [← c8616, ← c8617, ← c8618] at e_132_14
  have e_132_15 := arithEq_of_rows h (row := 132) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_15
  simp only [← c8619, ← c8620, ← c8621] at e_132_15
  have hr := e_132_15
  simp only [e_132_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 71) = bselect (a (.wire 121 55)) (a (.virt 10406)) (a (.wire 132 7)) := by
  have c8622 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8623 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8624 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8625 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8626 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8627 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_16 := arithEq_of_rows h (row := 132) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_16
  simp only [← c8622, ← c8623, ← c8624] at e_132_16
  have e_132_17 := arithEq_of_rows h (row := 132) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_17
  simp only [← c8625, ← c8626, ← c8627] at e_132_17
  have hr := e_132_17
  simp only [e_132_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 132 79) = bselect (a (.wire 121 55)) (a (.virt 10407)) (a (.wire 132 15)) := by
  have c8628 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8629 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8630 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8631 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8632 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8633 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_132_18 := arithEq_of_rows h (row := 132) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_18
  simp only [← c8628, ← c8629, ← c8630] at e_132_18
  have e_132_19 := arithEq_of_rows h (row := 132) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_132_19
  simp only [← c8631, ← c8632, ← c8633] at e_132_19
  have hr := e_132_19
  simp only [e_132_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 7) = bselect (a (.wire 121 55)) (a (.virt 10408)) (a (.wire 132 23)) := by
  have c8634 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8635 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8636 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8637 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8638 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8639 := (publicBatchWrapper86_copies269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_133_0 := arithEq_of_rows h (row := 133) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_0
  simp only [← c8634, ← c8635, ← c8636] at e_133_0
  have e_133_1 := arithEq_of_rows h (row := 133) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_1
  simp only [← c8637, ← c8638, ← c8639] at e_133_1
  have hr := e_133_1
  simp only [e_133_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 15) = bselect (a (.wire 121 55)) (a (.virt 10409)) (a (.wire 132 31)) := by
  have c8640 := (publicBatchWrapper86_copies270 a h).1
  have c8641 := (publicBatchWrapper86_copies270 a h).2.1
  have c8642 := (publicBatchWrapper86_copies270 a h).2.2.1
  have c8643 := (publicBatchWrapper86_copies270 a h).2.2.2.1
  have c8644 := (publicBatchWrapper86_copies270 a h).2.2.2.2.1
  have c8645 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.1
  have e_133_2 := arithEq_of_rows h (row := 133) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_2
  simp only [← c8640, ← c8641, ← c8642] at e_133_2
  have e_133_3 := arithEq_of_rows h (row := 133) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_3
  simp only [← c8643, ← c8644, ← c8645] at e_133_3
  have hr := e_133_3
  simp only [e_133_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 23) = bselect (a (.wire 121 55)) (a (.virt 10403)) (a (.wire 132 39)) := by
  have c8646 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.1
  have c8647 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.1
  have c8648 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.1
  have c8649 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.1
  have c8650 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8651 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_4 := arithEq_of_rows h (row := 133) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_4
  simp only [← c8646, ← c8647, ← c8648] at e_133_4
  have e_133_5 := arithEq_of_rows h (row := 133) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_5
  simp only [← c8649, ← c8650, ← c8651] at e_133_5
  have hr := e_133_5
  simp only [e_133_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 31) = bselect (a (.wire 121 55)) (a (.virt 10404)) (a (.wire 132 47)) := by
  have c8652 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8653 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8654 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8655 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8656 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8657 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_6 := arithEq_of_rows h (row := 133) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_6
  simp only [← c8652, ← c8653, ← c8654] at e_133_6
  have e_133_7 := arithEq_of_rows h (row := 133) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_7
  simp only [← c8655, ← c8656, ← c8657] at e_133_7
  have hr := e_133_7
  simp only [e_133_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 116 79) = bor (a (.wire 116 75)) (a (.wire 132 51)) := by
  have c8658 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8659 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8660 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8661 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8662 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8663 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_116_19 := arithEq_of_rows h (row := 116) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_116_19
  simp only [← c8661, ← c8662, k3, ← c8663] at e_116_19
  have e_124_11 := arithEq_of_rows h (row := 124) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_11
  simp only [← c8658, ← c8659, ← c8660] at e_124_11
  have hr := e_116_19
  simp only [e_124_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 35) = bnot (a (.wire 53 39)) := by
  have c8664 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8665 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8666 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_133_8 := arithEq_of_rows h (row := 133) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_8
  simp only [← c8664, k3, ← c8665, k3, ← c8666] at e_133_8
  have hr := e_133_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 39) = bnot (a (.wire 116 79)) := by
  have c8667 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8668 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8669 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_133_9 := arithEq_of_rows h (row := 133) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_9
  simp only [← c8667, k3, ← c8668, k3, ← c8669] at e_133_9
  have hr := e_133_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 59) = band (a (.wire 133 35)) (a (.wire 133 39)) := by
  have c8670 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8671 := (publicBatchWrapper86_copies270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8672 := (publicBatchWrapper86_copies271 a h).1
  have e_121_14 := arithEq_of_rows h (row := 121) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_14
  simp only [← c8670, ← c8671, ← c8672] at e_121_14
  have hr := e_121_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 47) = bselect (a (.wire 121 59)) (a (.virt 10605)) (a (.wire 132 63)) := by
  have c8673 := (publicBatchWrapper86_copies271 a h).2.1
  have c8674 := (publicBatchWrapper86_copies271 a h).2.2.1
  have c8675 := (publicBatchWrapper86_copies271 a h).2.2.2.1
  have c8676 := (publicBatchWrapper86_copies271 a h).2.2.2.2.1
  have c8677 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.1
  have c8678 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.1
  have e_133_10 := arithEq_of_rows h (row := 133) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_10
  simp only [← c8673, ← c8674, ← c8675] at e_133_10
  have e_133_11 := arithEq_of_rows h (row := 133) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_11
  simp only [← c8676, ← c8677, ← c8678] at e_133_11
  have hr := e_133_11
  simp only [e_133_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 55) = bselect (a (.wire 121 59)) (a (.virt 10606)) (a (.wire 132 71)) := by
  have c8679 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.1
  have c8680 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.1
  have c8681 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.1
  have c8682 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8683 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8684 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_12 := arithEq_of_rows h (row := 133) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_12
  simp only [← c8679, ← c8680, ← c8681] at e_133_12
  have e_133_13 := arithEq_of_rows h (row := 133) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_13
  simp only [← c8682, ← c8683, ← c8684] at e_133_13
  have hr := e_133_13
  simp only [e_133_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 63) = bselect (a (.wire 121 59)) (a (.virt 10607)) (a (.wire 132 79)) := by
  have c8685 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8686 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8687 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8688 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8689 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8690 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_14 := arithEq_of_rows h (row := 133) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_14
  simp only [← c8685, ← c8686, ← c8687] at e_133_14
  have e_133_15 := arithEq_of_rows h (row := 133) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_15
  simp only [← c8688, ← c8689, ← c8690] at e_133_15
  have hr := e_133_15
  simp only [e_133_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 71) = bselect (a (.wire 121 59)) (a (.virt 10608)) (a (.wire 133 7)) := by
  have c8691 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8692 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8693 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8694 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8695 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8696 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_16 := arithEq_of_rows h (row := 133) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_16
  simp only [← c8691, ← c8692, ← c8693] at e_133_16
  have e_133_17 := arithEq_of_rows h (row := 133) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_17
  simp only [← c8694, ← c8695, ← c8696] at e_133_17
  have hr := e_133_17
  simp only [e_133_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 133 79) = bselect (a (.wire 121 59)) (a (.virt 10609)) (a (.wire 133 15)) := by
  have c8697 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8698 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8699 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8700 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8701 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8702 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_133_18 := arithEq_of_rows h (row := 133) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_18
  simp only [← c8697, ← c8698, ← c8699] at e_133_18
  have e_133_19 := arithEq_of_rows h (row := 133) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_133_19
  simp only [← c8700, ← c8701, ← c8702] at e_133_19
  have hr := e_133_19
  simp only [e_133_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 7) = bselect (a (.wire 121 59)) (a (.virt 10603)) (a (.wire 133 23)) := by
  have c8703 := (publicBatchWrapper86_copies271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8704 := (publicBatchWrapper86_copies272 a h).1
  have c8705 := (publicBatchWrapper86_copies272 a h).2.1
  have c8706 := (publicBatchWrapper86_copies272 a h).2.2.1
  have c8707 := (publicBatchWrapper86_copies272 a h).2.2.2.1
  have c8708 := (publicBatchWrapper86_copies272 a h).2.2.2.2.1
  have e_134_0 := arithEq_of_rows h (row := 134) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_0
  simp only [← c8703, ← c8704, ← c8705] at e_134_0
  have e_134_1 := arithEq_of_rows h (row := 134) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_1
  simp only [← c8706, ← c8707, ← c8708] at e_134_1
  have hr := e_134_1
  simp only [e_134_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 15) = bselect (a (.wire 121 59)) (a (.virt 10604)) (a (.wire 133 31)) := by
  have c8709 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.1
  have c8710 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.1
  have c8711 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.1
  have c8712 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.1
  have c8713 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.1
  have c8714 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_134_2 := arithEq_of_rows h (row := 134) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_2
  simp only [← c8709, ← c8710, ← c8711] at e_134_2
  have e_134_3 := arithEq_of_rows h (row := 134) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_3
  simp only [← c8712, ← c8713, ← c8714] at e_134_3
  have hr := e_134_3
  simp only [e_134_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 3) = bor (a (.wire 116 79)) (a (.wire 133 35)) := by
  have c8715 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8716 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8717 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8718 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8719 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8720 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_124_12 := arithEq_of_rows h (row := 124) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_12
  simp only [← c8715, ← c8716, ← c8717] at e_124_12
  have e_135_0 := arithEq_of_rows h (row := 135) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_0
  simp only [← c8718, ← c8719, k3, ← c8720] at e_135_0
  have hr := e_135_0
  simp only [e_124_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 19) = bnot (a (.wire 54 3)) := by
  have c8721 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8722 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8723 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_134_4 := arithEq_of_rows h (row := 134) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_4
  simp only [← c8721, k3, ← c8722, k3, ← c8723] at e_134_4
  have hr := e_134_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 23) = bnot (a (.wire 135 3)) := by
  have c8724 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8725 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8726 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_134_5 := arithEq_of_rows h (row := 134) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_5
  simp only [← c8724, k3, ← c8725, k3, ← c8726] at e_134_5
  have hr := e_134_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 63) = band (a (.wire 134 19)) (a (.wire 134 23)) := by
  have c8727 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8728 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8729 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_15 := arithEq_of_rows h (row := 121) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_15
  simp only [← c8727, ← c8728, ← c8729] at e_121_15
  have hr := e_121_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 31) = bselect (a (.wire 121 63)) (a (.virt 10805)) (a (.wire 133 47)) := by
  have c8730 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8731 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8732 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8733 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8734 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8735 := (publicBatchWrapper86_copies272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_134_6 := arithEq_of_rows h (row := 134) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_6
  simp only [← c8730, ← c8731, ← c8732] at e_134_6
  have e_134_7 := arithEq_of_rows h (row := 134) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_7
  simp only [← c8733, ← c8734, ← c8735] at e_134_7
  have hr := e_134_7
  simp only [e_134_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 39) = bselect (a (.wire 121 63)) (a (.virt 10806)) (a (.wire 133 55)) := by
  have c8736 := (publicBatchWrapper86_copies273 a h).1
  have c8737 := (publicBatchWrapper86_copies273 a h).2.1
  have c8738 := (publicBatchWrapper86_copies273 a h).2.2.1
  have c8739 := (publicBatchWrapper86_copies273 a h).2.2.2.1
  have c8740 := (publicBatchWrapper86_copies273 a h).2.2.2.2.1
  have c8741 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.1
  have e_134_8 := arithEq_of_rows h (row := 134) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_8
  simp only [← c8736, ← c8737, ← c8738] at e_134_8
  have e_134_9 := arithEq_of_rows h (row := 134) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_9
  simp only [← c8739, ← c8740, ← c8741] at e_134_9
  have hr := e_134_9
  simp only [e_134_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 47) = bselect (a (.wire 121 63)) (a (.virt 10807)) (a (.wire 133 63)) := by
  have c8742 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.1
  have c8743 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.1
  have c8744 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.1
  have c8745 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.1
  have c8746 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8747 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_134_10 := arithEq_of_rows h (row := 134) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_10
  simp only [← c8742, ← c8743, ← c8744] at e_134_10
  have e_134_11 := arithEq_of_rows h (row := 134) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_11
  simp only [← c8745, ← c8746, ← c8747] at e_134_11
  have hr := e_134_11
  simp only [e_134_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 55) = bselect (a (.wire 121 63)) (a (.virt 10808)) (a (.wire 133 71)) := by
  have c8748 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8749 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8750 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8751 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8752 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8753 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_134_12 := arithEq_of_rows h (row := 134) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_12
  simp only [← c8748, ← c8749, ← c8750] at e_134_12
  have e_134_13 := arithEq_of_rows h (row := 134) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_13
  simp only [← c8751, ← c8752, ← c8753] at e_134_13
  have hr := e_134_13
  simp only [e_134_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 63) = bselect (a (.wire 121 63)) (a (.virt 10809)) (a (.wire 133 79)) := by
  have c8754 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8755 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8756 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8757 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8758 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8759 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_134_14 := arithEq_of_rows h (row := 134) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_14
  simp only [← c8754, ← c8755, ← c8756] at e_134_14
  have e_134_15 := arithEq_of_rows h (row := 134) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_15
  simp only [← c8757, ← c8758, ← c8759] at e_134_15
  have hr := e_134_15
  simp only [e_134_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 71) = bselect (a (.wire 121 63)) (a (.virt 10803)) (a (.wire 134 7)) := by
  have c8760 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8761 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8762 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8763 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8764 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8765 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_134_16 := arithEq_of_rows h (row := 134) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_16
  simp only [← c8760, ← c8761, ← c8762] at e_134_16
  have e_134_17 := arithEq_of_rows h (row := 134) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_17
  simp only [← c8763, ← c8764, ← c8765] at e_134_17
  have hr := e_134_17
  simp only [e_134_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 134 79) = bselect (a (.wire 121 63)) (a (.virt 10804)) (a (.wire 134 15)) := by
  have c8766 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8767 := (publicBatchWrapper86_copies273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8768 := (publicBatchWrapper86_copies274 a h).1
  have c8769 := (publicBatchWrapper86_copies274 a h).2.1
  have c8770 := (publicBatchWrapper86_copies274 a h).2.2.1
  have c8771 := (publicBatchWrapper86_copies274 a h).2.2.2.1
  have e_134_18 := arithEq_of_rows h (row := 134) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_18
  simp only [← c8766, ← c8767, ← c8768] at e_134_18
  have e_134_19 := arithEq_of_rows h (row := 134) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_134_19
  simp only [← c8769, ← c8770, ← c8771] at e_134_19
  have hr := e_134_19
  simp only [e_134_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 7) = bor (a (.wire 135 3)) (a (.wire 134 19)) := by
  have c8772 := (publicBatchWrapper86_copies274 a h).2.2.2.2.1
  have c8773 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.1
  have c8774 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.1
  have c8775 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.1
  have c8776 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.1
  have c8777 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_124_13 := arithEq_of_rows h (row := 124) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_13
  simp only [← c8772, ← c8773, ← c8774] at e_124_13
  have e_135_1 := arithEq_of_rows h (row := 135) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_1
  simp only [← c8775, ← c8776, k3, ← c8777] at e_135_1
  have hr := e_135_1
  simp only [e_124_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 3) = bnot (a (.wire 54 47)) := by
  have c8778 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8779 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8780 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_136_0 := arithEq_of_rows h (row := 136) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_0
  simp only [← c8778, k3, ← c8779, k3, ← c8780] at e_136_0
  have hr := e_136_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 7) = bnot (a (.wire 135 7)) := by
  have c8781 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8782 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8783 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_136_1 := arithEq_of_rows h (row := 136) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_1
  simp only [← c8781, k3, ← c8782, k3, ← c8783] at e_136_1
  have hr := e_136_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 67) = band (a (.wire 136 3)) (a (.wire 136 7)) := by
  have c8784 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8785 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8786 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_16 := arithEq_of_rows h (row := 121) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_16
  simp only [← c8784, ← c8785, ← c8786] at e_121_16
  have hr := e_121_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 15) = bselect (a (.wire 121 67)) (a (.virt 11005)) (a (.wire 134 31)) := by
  have c8787 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8788 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8789 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8790 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8791 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8792 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_2 := arithEq_of_rows h (row := 136) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_2
  simp only [← c8787, ← c8788, ← c8789] at e_136_2
  have e_136_3 := arithEq_of_rows h (row := 136) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_3
  simp only [← c8790, ← c8791, ← c8792] at e_136_3
  have hr := e_136_3
  simp only [e_136_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 23) = bselect (a (.wire 121 67)) (a (.virt 11006)) (a (.wire 134 39)) := by
  have c8793 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8794 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8795 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8796 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8797 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8798 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_4 := arithEq_of_rows h (row := 136) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_4
  simp only [← c8793, ← c8794, ← c8795] at e_136_4
  have e_136_5 := arithEq_of_rows h (row := 136) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_5
  simp only [← c8796, ← c8797, ← c8798] at e_136_5
  have hr := e_136_5
  simp only [e_136_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 31) = bselect (a (.wire 121 67)) (a (.virt 11007)) (a (.wire 134 47)) := by
  have c8799 := (publicBatchWrapper86_copies274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8800 := (publicBatchWrapper86_copies275 a h).1
  have c8801 := (publicBatchWrapper86_copies275 a h).2.1
  have c8802 := (publicBatchWrapper86_copies275 a h).2.2.1
  have c8803 := (publicBatchWrapper86_copies275 a h).2.2.2.1
  have c8804 := (publicBatchWrapper86_copies275 a h).2.2.2.2.1
  have e_136_6 := arithEq_of_rows h (row := 136) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_6
  simp only [← c8799, ← c8800, ← c8801] at e_136_6
  have e_136_7 := arithEq_of_rows h (row := 136) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_7
  simp only [← c8802, ← c8803, ← c8804] at e_136_7
  have hr := e_136_7
  simp only [e_136_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 39) = bselect (a (.wire 121 67)) (a (.virt 11008)) (a (.wire 134 55)) := by
  have c8805 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.1
  have c8806 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.1
  have c8807 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.1
  have c8808 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.1
  have c8809 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.1
  have c8810 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_136_8 := arithEq_of_rows h (row := 136) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_8
  simp only [← c8805, ← c8806, ← c8807] at e_136_8
  have e_136_9 := arithEq_of_rows h (row := 136) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_9
  simp only [← c8808, ← c8809, ← c8810] at e_136_9
  have hr := e_136_9
  simp only [e_136_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 47) = bselect (a (.wire 121 67)) (a (.virt 11009)) (a (.wire 134 63)) := by
  have c8811 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8812 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8813 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8814 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8815 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8816 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_10 := arithEq_of_rows h (row := 136) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_10
  simp only [← c8811, ← c8812, ← c8813] at e_136_10
  have e_136_11 := arithEq_of_rows h (row := 136) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_11
  simp only [← c8814, ← c8815, ← c8816] at e_136_11
  have hr := e_136_11
  simp only [e_136_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 55) = bselect (a (.wire 121 67)) (a (.virt 11003)) (a (.wire 134 71)) := by
  have c8817 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8818 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8819 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8820 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8821 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8822 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_12 := arithEq_of_rows h (row := 136) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_12
  simp only [← c8817, ← c8818, ← c8819] at e_136_12
  have e_136_13 := arithEq_of_rows h (row := 136) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_13
  simp only [← c8820, ← c8821, ← c8822] at e_136_13
  have hr := e_136_13
  simp only [e_136_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 63) = bselect (a (.wire 121 67)) (a (.virt 11004)) (a (.wire 134 79)) := by
  have c8823 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8824 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8825 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8826 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8827 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8828 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_14 := arithEq_of_rows h (row := 136) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_14
  simp only [← c8823, ← c8824, ← c8825] at e_136_14
  have e_136_15 := arithEq_of_rows h (row := 136) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_15
  simp only [← c8826, ← c8827, ← c8828] at e_136_15
  have hr := e_136_15
  simp only [e_136_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 11) = bor (a (.wire 135 7)) (a (.wire 136 3)) := by
  have c8829 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8830 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8831 := (publicBatchWrapper86_copies275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8832 := (publicBatchWrapper86_copies276 a h).1
  have c8833 := (publicBatchWrapper86_copies276 a h).2.1
  have c8834 := (publicBatchWrapper86_copies276 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_124_14 := arithEq_of_rows h (row := 124) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_14
  simp only [← c8829, ← c8830, ← c8831] at e_124_14
  have e_135_2 := arithEq_of_rows h (row := 135) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_2
  simp only [← c8832, ← c8833, k3, ← c8834] at e_135_2
  have hr := e_135_2
  simp only [e_124_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 67) = bnot (a (.wire 56 11)) := by
  have c8835 := (publicBatchWrapper86_copies276 a h).2.2.2.1
  have c8836 := (publicBatchWrapper86_copies276 a h).2.2.2.2.1
  have c8837 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_136_16 := arithEq_of_rows h (row := 136) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_16
  simp only [← c8835, k3, ← c8836, k3, ← c8837] at e_136_16
  have hr := e_136_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 71) = bnot (a (.wire 135 11)) := by
  have c8838 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.1
  have c8839 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.1
  have c8840 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_136_17 := arithEq_of_rows h (row := 136) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_17
  simp only [← c8838, k3, ← c8839, k3, ← c8840] at e_136_17
  have hr := e_136_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 71) = band (a (.wire 136 67)) (a (.wire 136 71)) := by
  have c8841 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.1
  have c8842 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8843 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_17 := arithEq_of_rows h (row := 121) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_17
  simp only [← c8841, ← c8842, ← c8843] at e_121_17
  have hr := e_121_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 136 79) = bselect (a (.wire 121 71)) (a (.virt 11205)) (a (.wire 136 15)) := by
  have c8844 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8845 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8846 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8847 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8848 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8849 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_136_18 := arithEq_of_rows h (row := 136) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_18
  simp only [← c8844, ← c8845, ← c8846] at e_136_18
  have e_136_19 := arithEq_of_rows h (row := 136) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_136_19
  simp only [← c8847, ← c8848, ← c8849] at e_136_19
  have hr := e_136_19
  simp only [e_136_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 7) = bselect (a (.wire 121 71)) (a (.virt 11206)) (a (.wire 136 23)) := by
  have c8850 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8851 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8852 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8853 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8854 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8855 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_0 := arithEq_of_rows h (row := 137) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_0
  simp only [← c8850, ← c8851, ← c8852] at e_137_0
  have e_137_1 := arithEq_of_rows h (row := 137) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_1
  simp only [← c8853, ← c8854, ← c8855] at e_137_1
  have hr := e_137_1
  simp only [e_137_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 15) = bselect (a (.wire 121 71)) (a (.virt 11207)) (a (.wire 136 31)) := by
  have c8856 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8857 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8858 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8859 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8860 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8861 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_2 := arithEq_of_rows h (row := 137) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_2
  simp only [← c8856, ← c8857, ← c8858] at e_137_2
  have e_137_3 := arithEq_of_rows h (row := 137) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_3
  simp only [← c8859, ← c8860, ← c8861] at e_137_3
  have hr := e_137_3
  simp only [e_137_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 23) = bselect (a (.wire 121 71)) (a (.virt 11208)) (a (.wire 136 39)) := by
  have c8862 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8863 := (publicBatchWrapper86_copies276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8864 := (publicBatchWrapper86_copies277 a h).1
  have c8865 := (publicBatchWrapper86_copies277 a h).2.1
  have c8866 := (publicBatchWrapper86_copies277 a h).2.2.1
  have c8867 := (publicBatchWrapper86_copies277 a h).2.2.2.1
  have e_137_4 := arithEq_of_rows h (row := 137) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_4
  simp only [← c8862, ← c8863, ← c8864] at e_137_4
  have e_137_5 := arithEq_of_rows h (row := 137) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_5
  simp only [← c8865, ← c8866, ← c8867] at e_137_5
  have hr := e_137_5
  simp only [e_137_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 31) = bselect (a (.wire 121 71)) (a (.virt 11209)) (a (.wire 136 47)) := by
  have c8868 := (publicBatchWrapper86_copies277 a h).2.2.2.2.1
  have c8869 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.1
  have c8870 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.1
  have c8871 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.1
  have c8872 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.1
  have c8873 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.1
  have e_137_6 := arithEq_of_rows h (row := 137) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_6
  simp only [← c8868, ← c8869, ← c8870] at e_137_6
  have e_137_7 := arithEq_of_rows h (row := 137) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_7
  simp only [← c8871, ← c8872, ← c8873] at e_137_7
  have hr := e_137_7
  simp only [e_137_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 39) = bselect (a (.wire 121 71)) (a (.virt 11203)) (a (.wire 136 55)) := by
  have c8874 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8875 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8876 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8877 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8878 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8879 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_8 := arithEq_of_rows h (row := 137) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_8
  simp only [← c8874, ← c8875, ← c8876] at e_137_8
  have e_137_9 := arithEq_of_rows h (row := 137) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_9
  simp only [← c8877, ← c8878, ← c8879] at e_137_9
  have hr := e_137_9
  simp only [e_137_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 47) = bselect (a (.wire 121 71)) (a (.virt 11204)) (a (.wire 136 63)) := by
  have c8880 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8881 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8882 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8883 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8884 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8885 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_10 := arithEq_of_rows h (row := 137) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_10
  simp only [← c8880, ← c8881, ← c8882] at e_137_10
  have e_137_11 := arithEq_of_rows h (row := 137) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_11
  simp only [← c8883, ← c8884, ← c8885] at e_137_11
  have hr := e_137_11
  simp only [e_137_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 15) = bor (a (.wire 135 11)) (a (.wire 136 67)) := by
  have c8886 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8887 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8888 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8889 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8890 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8891 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_124_15 := arithEq_of_rows h (row := 124) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_15
  simp only [← c8886, ← c8887, ← c8888] at e_124_15
  have e_135_3 := arithEq_of_rows h (row := 135) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_3
  simp only [← c8889, ← c8890, k3, ← c8891] at e_135_3
  have hr := e_135_3
  simp only [e_124_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 51) = bnot (a (.wire 56 55)) := by
  have c8892 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8893 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8894 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_137_12 := arithEq_of_rows h (row := 137) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_12
  simp only [← c8892, k3, ← c8893, k3, ← c8894] at e_137_12
  have hr := e_137_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 55) = bnot (a (.wire 135 15)) := by
  have c8895 := (publicBatchWrapper86_copies277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8896 := (publicBatchWrapper86_copies278 a h).1
  have c8897 := (publicBatchWrapper86_copies278 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_137_13 := arithEq_of_rows h (row := 137) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_13
  simp only [← c8895, k3, ← c8896, k3, ← c8897] at e_137_13
  have hr := e_137_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 75) = band (a (.wire 137 51)) (a (.wire 137 55)) := by
  have c8898 := (publicBatchWrapper86_copies278 a h).2.2.1
  have c8899 := (publicBatchWrapper86_copies278 a h).2.2.2.1
  have c8900 := (publicBatchWrapper86_copies278 a h).2.2.2.2.1
  have e_121_18 := arithEq_of_rows h (row := 121) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_18
  simp only [← c8898, ← c8899, ← c8900] at e_121_18
  have hr := e_121_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 63) = bselect (a (.wire 121 75)) (a (.virt 11405)) (a (.wire 136 79)) := by
  have c8901 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.1
  have c8902 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.1
  have c8903 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.1
  have c8904 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.1
  have c8905 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.1
  have c8906 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_137_14 := arithEq_of_rows h (row := 137) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_14
  simp only [← c8901, ← c8902, ← c8903] at e_137_14
  have e_137_15 := arithEq_of_rows h (row := 137) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_15
  simp only [← c8904, ← c8905, ← c8906] at e_137_15
  have hr := e_137_15
  simp only [e_137_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 71) = bselect (a (.wire 121 75)) (a (.virt 11406)) (a (.wire 137 7)) := by
  have c8907 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8908 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8909 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8910 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8911 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8912 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_16 := arithEq_of_rows h (row := 137) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_16
  simp only [← c8907, ← c8908, ← c8909] at e_137_16
  have e_137_17 := arithEq_of_rows h (row := 137) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_17
  simp only [← c8910, ← c8911, ← c8912] at e_137_17
  have hr := e_137_17
  simp only [e_137_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 137 79) = bselect (a (.wire 121 75)) (a (.virt 11407)) (a (.wire 137 15)) := by
  have c8913 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8914 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8915 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8916 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8917 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8918 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_137_18 := arithEq_of_rows h (row := 137) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_18
  simp only [← c8913, ← c8914, ← c8915] at e_137_18
  have e_137_19 := arithEq_of_rows h (row := 137) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_137_19
  simp only [← c8916, ← c8917, ← c8918] at e_137_19
  have hr := e_137_19
  simp only [e_137_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 7) = bselect (a (.wire 121 75)) (a (.virt 11408)) (a (.wire 137 23)) := by
  have c8919 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8920 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8921 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8922 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8923 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8924 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_0 := arithEq_of_rows h (row := 138) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_0
  simp only [← c8919, ← c8920, ← c8921] at e_138_0
  have e_138_1 := arithEq_of_rows h (row := 138) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_1
  simp only [← c8922, ← c8923, ← c8924] at e_138_1
  have hr := e_138_1
  simp only [e_138_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 15) = bselect (a (.wire 121 75)) (a (.virt 11409)) (a (.wire 137 31)) := by
  have c8925 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8926 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8927 := (publicBatchWrapper86_copies278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8928 := (publicBatchWrapper86_copies279 a h).1
  have c8929 := (publicBatchWrapper86_copies279 a h).2.1
  have c8930 := (publicBatchWrapper86_copies279 a h).2.2.1
  have e_138_2 := arithEq_of_rows h (row := 138) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_2
  simp only [← c8925, ← c8926, ← c8927] at e_138_2
  have e_138_3 := arithEq_of_rows h (row := 138) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_3
  simp only [← c8928, ← c8929, ← c8930] at e_138_3
  have hr := e_138_3
  simp only [e_138_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 23) = bselect (a (.wire 121 75)) (a (.virt 11403)) (a (.wire 137 39)) := by
  have c8931 := (publicBatchWrapper86_copies279 a h).2.2.2.1
  have c8932 := (publicBatchWrapper86_copies279 a h).2.2.2.2.1
  have c8933 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.1
  have c8934 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.1
  have c8935 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.1
  have c8936 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.1
  have e_138_4 := arithEq_of_rows h (row := 138) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_4
  simp only [← c8931, ← c8932, ← c8933] at e_138_4
  have e_138_5 := arithEq_of_rows h (row := 138) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_5
  simp only [← c8934, ← c8935, ← c8936] at e_138_5
  have hr := e_138_5
  simp only [e_138_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 31) = bselect (a (.wire 121 75)) (a (.virt 11404)) (a (.wire 137 47)) := by
  have c8937 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.1
  have c8938 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8939 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8940 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8941 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8942 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_6 := arithEq_of_rows h (row := 138) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_6
  simp only [← c8937, ← c8938, ← c8939] at e_138_6
  have e_138_7 := arithEq_of_rows h (row := 138) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_7
  simp only [← c8940, ← c8941, ← c8942] at e_138_7
  have hr := e_138_7
  simp only [e_138_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 19) = bor (a (.wire 135 15)) (a (.wire 137 51)) := by
  have c8943 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8944 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8945 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8946 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8947 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8948 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_124_16 := arithEq_of_rows h (row := 124) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_16
  simp only [← c8943, ← c8944, ← c8945] at e_124_16
  have e_135_4 := arithEq_of_rows h (row := 135) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_4
  simp only [← c8946, ← c8947, k3, ← c8948] at e_135_4
  have hr := e_135_4
  simp only [e_124_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 35) = bnot (a (.wire 58 19)) := by
  have c8949 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8950 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8951 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_138_8 := arithEq_of_rows h (row := 138) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_8
  simp only [← c8949, k3, ← c8950, k3, ← c8951] at e_138_8
  have hr := e_138_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 39) = bnot (a (.wire 135 19)) := by
  have c8952 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8953 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8954 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_138_9 := arithEq_of_rows h (row := 138) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_9
  simp only [← c8952, k3, ← c8953, k3, ← c8954] at e_138_9
  have hr := e_138_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 121 79) = band (a (.wire 138 35)) (a (.wire 138 39)) := by
  have c8955 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8956 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8957 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_121_19 := arithEq_of_rows h (row := 121) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_121_19
  simp only [← c8955, ← c8956, ← c8957] at e_121_19
  have hr := e_121_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 47) = bselect (a (.wire 121 79)) (a (.virt 11605)) (a (.wire 137 63)) := by
  have c8958 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8959 := (publicBatchWrapper86_copies279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8960 := (publicBatchWrapper86_copies280 a h).1
  have c8961 := (publicBatchWrapper86_copies280 a h).2.1
  have c8962 := (publicBatchWrapper86_copies280 a h).2.2.1
  have c8963 := (publicBatchWrapper86_copies280 a h).2.2.2.1
  have e_138_10 := arithEq_of_rows h (row := 138) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_10
  simp only [← c8958, ← c8959, ← c8960] at e_138_10
  have e_138_11 := arithEq_of_rows h (row := 138) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_11
  simp only [← c8961, ← c8962, ← c8963] at e_138_11
  have hr := e_138_11
  simp only [e_138_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 55) = bselect (a (.wire 121 79)) (a (.virt 11606)) (a (.wire 137 71)) := by
  have c8964 := (publicBatchWrapper86_copies280 a h).2.2.2.2.1
  have c8965 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.1
  have c8966 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.1
  have c8967 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.1
  have c8968 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.1
  have c8969 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.1
  have e_138_12 := arithEq_of_rows h (row := 138) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_12
  simp only [← c8964, ← c8965, ← c8966] at e_138_12
  have e_138_13 := arithEq_of_rows h (row := 138) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_13
  simp only [← c8967, ← c8968, ← c8969] at e_138_13
  have hr := e_138_13
  simp only [e_138_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 63) = bselect (a (.wire 121 79)) (a (.virt 11607)) (a (.wire 137 79)) := by
  have c8970 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.1
  have c8971 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c8972 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8973 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8974 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8975 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_14 := arithEq_of_rows h (row := 138) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_14
  simp only [← c8970, ← c8971, ← c8972] at e_138_14
  have e_138_15 := arithEq_of_rows h (row := 138) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_15
  simp only [← c8973, ← c8974, ← c8975] at e_138_15
  have hr := e_138_15
  simp only [e_138_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 71) = bselect (a (.wire 121 79)) (a (.virt 11608)) (a (.wire 138 7)) := by
  have c8976 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8977 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8978 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8979 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8980 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8981 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_16 := arithEq_of_rows h (row := 138) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_16
  simp only [← c8976, ← c8977, ← c8978] at e_138_16
  have e_138_17 := arithEq_of_rows h (row := 138) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_17
  simp only [← c8979, ← c8980, ← c8981] at e_138_17
  have hr := e_138_17
  simp only [e_138_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 138 79) = bselect (a (.wire 121 79)) (a (.virt 11609)) (a (.wire 138 15)) := by
  have c8982 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8983 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8984 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8985 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8986 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8987 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_138_18 := arithEq_of_rows h (row := 138) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_18
  simp only [← c8982, ← c8983, ← c8984] at e_138_18
  have e_138_19 := arithEq_of_rows h (row := 138) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_138_19
  simp only [← c8985, ← c8986, ← c8987] at e_138_19
  have hr := e_138_19
  simp only [e_138_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 7) = bselect (a (.wire 121 79)) (a (.virt 11603)) (a (.wire 138 23)) := by
  have c8988 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8989 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8990 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c8991 := (publicBatchWrapper86_copies280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c8992 := (publicBatchWrapper86_copies281 a h).1
  have c8993 := (publicBatchWrapper86_copies281 a h).2.1
  have e_139_0 := arithEq_of_rows h (row := 139) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_0
  simp only [← c8988, ← c8989, ← c8990] at e_139_0
  have e_139_1 := arithEq_of_rows h (row := 139) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_1
  simp only [← c8991, ← c8992, ← c8993] at e_139_1
  have hr := e_139_1
  simp only [e_139_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 15) = bselect (a (.wire 121 79)) (a (.virt 11604)) (a (.wire 138 31)) := by
  have c8994 := (publicBatchWrapper86_copies281 a h).2.2.1
  have c8995 := (publicBatchWrapper86_copies281 a h).2.2.2.1
  have c8996 := (publicBatchWrapper86_copies281 a h).2.2.2.2.1
  have c8997 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.1
  have c8998 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.1
  have c8999 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.1
  have e_139_2 := arithEq_of_rows h (row := 139) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_2
  simp only [← c8994, ← c8995, ← c8996] at e_139_2
  have e_139_3 := arithEq_of_rows h (row := 139) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_3
  simp only [← c8997, ← c8998, ← c8999] at e_139_3
  have hr := e_139_3
  simp only [e_139_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 23) = bor (a (.wire 135 19)) (a (.wire 138 35)) := by
  have c9000 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.1
  have c9001 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.1
  have c9002 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9003 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9004 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9005 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_124_17 := arithEq_of_rows h (row := 124) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_17
  simp only [← c9000, ← c9001, ← c9002] at e_124_17
  have e_135_5 := arithEq_of_rows h (row := 135) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_5
  simp only [← c9003, ← c9004, k3, ← c9005] at e_135_5
  have hr := e_135_5
  simp only [e_124_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 19) = bnot (a (.wire 58 63)) := by
  have c9006 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9007 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9008 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_139_4 := arithEq_of_rows h (row := 139) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_4
  simp only [← c9006, k3, ← c9007, k3, ← c9008] at e_139_4
  have hr := e_139_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 23) = bnot (a (.wire 135 23)) := by
  have c9009 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9010 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9011 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_139_5 := arithEq_of_rows h (row := 139) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_5
  simp only [← c9009, k3, ← c9010, k3, ← c9011] at e_139_5
  have hr := e_139_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 3) = band (a (.wire 139 19)) (a (.wire 139 23)) := by
  have c9012 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9013 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9014 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_0 := arithEq_of_rows h (row := 140) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_0
  simp only [← c9012, ← c9013, ← c9014] at e_140_0
  have hr := e_140_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 31) = bselect (a (.wire 140 3)) (a (.virt 11805)) (a (.wire 138 47)) := by
  have c9015 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9016 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9017 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9018 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9019 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9020 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_6 := arithEq_of_rows h (row := 139) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_6
  simp only [← c9015, ← c9016, ← c9017] at e_139_6
  have e_139_7 := arithEq_of_rows h (row := 139) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_7
  simp only [← c9018, ← c9019, ← c9020] at e_139_7
  have hr := e_139_7
  simp only [e_139_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 39) = bselect (a (.wire 140 3)) (a (.virt 11806)) (a (.wire 138 55)) := by
  have c9021 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9022 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9023 := (publicBatchWrapper86_copies281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9024 := (publicBatchWrapper86_copies282 a h).1
  have c9025 := (publicBatchWrapper86_copies282 a h).2.1
  have c9026 := (publicBatchWrapper86_copies282 a h).2.2.1
  have e_139_8 := arithEq_of_rows h (row := 139) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_8
  simp only [← c9021, ← c9022, ← c9023] at e_139_8
  have e_139_9 := arithEq_of_rows h (row := 139) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_9
  simp only [← c9024, ← c9025, ← c9026] at e_139_9
  have hr := e_139_9
  simp only [e_139_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 47) = bselect (a (.wire 140 3)) (a (.virt 11807)) (a (.wire 138 63)) := by
  have c9027 := (publicBatchWrapper86_copies282 a h).2.2.2.1
  have c9028 := (publicBatchWrapper86_copies282 a h).2.2.2.2.1
  have c9029 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.1
  have c9030 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.1
  have c9031 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.1
  have c9032 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.1
  have e_139_10 := arithEq_of_rows h (row := 139) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_10
  simp only [← c9027, ← c9028, ← c9029] at e_139_10
  have e_139_11 := arithEq_of_rows h (row := 139) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_11
  simp only [← c9030, ← c9031, ← c9032] at e_139_11
  have hr := e_139_11
  simp only [e_139_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 55) = bselect (a (.wire 140 3)) (a (.virt 11808)) (a (.wire 138 71)) := by
  have c9033 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.1
  have c9034 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9035 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9036 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9037 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9038 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_12 := arithEq_of_rows h (row := 139) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_12
  simp only [← c9033, ← c9034, ← c9035] at e_139_12
  have e_139_13 := arithEq_of_rows h (row := 139) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_13
  simp only [← c9036, ← c9037, ← c9038] at e_139_13
  have hr := e_139_13
  simp only [e_139_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 63) = bselect (a (.wire 140 3)) (a (.virt 11809)) (a (.wire 138 79)) := by
  have c9039 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9040 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9041 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9042 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9043 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9044 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_14 := arithEq_of_rows h (row := 139) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_14
  simp only [← c9039, ← c9040, ← c9041] at e_139_14
  have e_139_15 := arithEq_of_rows h (row := 139) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_15
  simp only [← c9042, ← c9043, ← c9044] at e_139_15
  have hr := e_139_15
  simp only [e_139_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 71) = bselect (a (.wire 140 3)) (a (.virt 11803)) (a (.wire 139 7)) := by
  have c9045 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9046 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9047 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9048 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9049 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9050 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_139_16 := arithEq_of_rows h (row := 139) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_16
  simp only [← c9045, ← c9046, ← c9047] at e_139_16
  have e_139_17 := arithEq_of_rows h (row := 139) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_17
  simp only [← c9048, ← c9049, ← c9050] at e_139_17
  have hr := e_139_17
  simp only [e_139_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 139 79) = bselect (a (.wire 140 3)) (a (.virt 11804)) (a (.wire 139 15)) := by
  have c9051 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9052 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9053 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9054 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9055 := (publicBatchWrapper86_copies282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9056 := (publicBatchWrapper86_copies283 a h).1
  have e_139_18 := arithEq_of_rows h (row := 139) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_18
  simp only [← c9051, ← c9052, ← c9053] at e_139_18
  have e_139_19 := arithEq_of_rows h (row := 139) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_139_19
  simp only [← c9054, ← c9055, ← c9056] at e_139_19
  have hr := e_139_19
  simp only [e_139_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 27) = bor (a (.wire 135 23)) (a (.wire 139 19)) := by
  have c9057 := (publicBatchWrapper86_copies283 a h).2.1
  have c9058 := (publicBatchWrapper86_copies283 a h).2.2.1
  have c9059 := (publicBatchWrapper86_copies283 a h).2.2.2.1
  have c9060 := (publicBatchWrapper86_copies283 a h).2.2.2.2.1
  have c9061 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.1
  have c9062 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_124_18 := arithEq_of_rows h (row := 124) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_18
  simp only [← c9057, ← c9058, ← c9059] at e_124_18
  have e_135_6 := arithEq_of_rows h (row := 135) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_6
  simp only [← c9060, ← c9061, k3, ← c9062] at e_135_6
  have hr := e_135_6
  simp only [e_124_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 3) = bnot (a (.wire 60 27)) := by
  have c9063 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.1
  have c9064 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.1
  have c9065 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_141_0 := arithEq_of_rows h (row := 141) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_0
  simp only [← c9063, k3, ← c9064, k3, ← c9065] at e_141_0
  have hr := e_141_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 7) = bnot (a (.wire 135 27)) := by
  have c9066 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9067 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9068 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_141_1 := arithEq_of_rows h (row := 141) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_1
  simp only [← c9066, k3, ← c9067, k3, ← c9068] at e_141_1
  have hr := e_141_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 7) = band (a (.wire 141 3)) (a (.wire 141 7)) := by
  have c9069 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9070 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9071 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_1 := arithEq_of_rows h (row := 140) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_1
  simp only [← c9069, ← c9070, ← c9071] at e_140_1
  have hr := e_140_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 15) = bselect (a (.wire 140 7)) (a (.virt 12005)) (a (.wire 139 31)) := by
  have c9072 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9073 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9074 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9075 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9076 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9077 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_2 := arithEq_of_rows h (row := 141) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_2
  simp only [← c9072, ← c9073, ← c9074] at e_141_2
  have e_141_3 := arithEq_of_rows h (row := 141) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_3
  simp only [← c9075, ← c9076, ← c9077] at e_141_3
  have hr := e_141_3
  simp only [e_141_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 23) = bselect (a (.wire 140 7)) (a (.virt 12006)) (a (.wire 139 39)) := by
  have c9078 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9079 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9080 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9081 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9082 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9083 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_4 := arithEq_of_rows h (row := 141) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_4
  simp only [← c9078, ← c9079, ← c9080] at e_141_4
  have e_141_5 := arithEq_of_rows h (row := 141) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_5
  simp only [← c9081, ← c9082, ← c9083] at e_141_5
  have hr := e_141_5
  simp only [e_141_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 31) = bselect (a (.wire 140 7)) (a (.virt 12007)) (a (.wire 139 47)) := by
  have c9084 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9085 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9086 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9087 := (publicBatchWrapper86_copies283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9088 := (publicBatchWrapper86_copies284 a h).1
  have c9089 := (publicBatchWrapper86_copies284 a h).2.1
  have e_141_6 := arithEq_of_rows h (row := 141) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_6
  simp only [← c9084, ← c9085, ← c9086] at e_141_6
  have e_141_7 := arithEq_of_rows h (row := 141) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_7
  simp only [← c9087, ← c9088, ← c9089] at e_141_7
  have hr := e_141_7
  simp only [e_141_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 39) = bselect (a (.wire 140 7)) (a (.virt 12008)) (a (.wire 139 55)) := by
  have c9090 := (publicBatchWrapper86_copies284 a h).2.2.1
  have c9091 := (publicBatchWrapper86_copies284 a h).2.2.2.1
  have c9092 := (publicBatchWrapper86_copies284 a h).2.2.2.2.1
  have c9093 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.1
  have c9094 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.1
  have c9095 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.1
  have e_141_8 := arithEq_of_rows h (row := 141) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_8
  simp only [← c9090, ← c9091, ← c9092] at e_141_8
  have e_141_9 := arithEq_of_rows h (row := 141) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_9
  simp only [← c9093, ← c9094, ← c9095] at e_141_9
  have hr := e_141_9
  simp only [e_141_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 47) = bselect (a (.wire 140 7)) (a (.virt 12009)) (a (.wire 139 63)) := by
  have c9096 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.1
  have c9097 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.1
  have c9098 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9099 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9100 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9101 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_10 := arithEq_of_rows h (row := 141) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_10
  simp only [← c9096, ← c9097, ← c9098] at e_141_10
  have e_141_11 := arithEq_of_rows h (row := 141) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_11
  simp only [← c9099, ← c9100, ← c9101] at e_141_11
  have hr := e_141_11
  simp only [e_141_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 55) = bselect (a (.wire 140 7)) (a (.virt 12003)) (a (.wire 139 71)) := by
  have c9102 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9103 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9104 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9105 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9106 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9107 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_12 := arithEq_of_rows h (row := 141) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_12
  simp only [← c9102, ← c9103, ← c9104] at e_141_12
  have e_141_13 := arithEq_of_rows h (row := 141) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_13
  simp only [← c9105, ← c9106, ← c9107] at e_141_13
  have hr := e_141_13
  simp only [e_141_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 63) = bselect (a (.wire 140 7)) (a (.virt 12004)) (a (.wire 139 79)) := by
  have c9108 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9109 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9110 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9111 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9112 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9113 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_14 := arithEq_of_rows h (row := 141) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_14
  simp only [← c9108, ← c9109, ← c9110] at e_141_14
  have e_141_15 := arithEq_of_rows h (row := 141) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_15
  simp only [← c9111, ← c9112, ← c9113] at e_141_15
  have hr := e_141_15
  simp only [e_141_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 31) = bor (a (.wire 135 27)) (a (.wire 141 3)) := by
  have c9114 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9115 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9116 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9117 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9118 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9119 := (publicBatchWrapper86_copies284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_124_19 := arithEq_of_rows h (row := 124) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_124_19
  simp only [← c9114, ← c9115, ← c9116] at e_124_19
  have e_135_7 := arithEq_of_rows h (row := 135) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_7
  simp only [← c9117, ← c9118, k3, ← c9119] at e_135_7
  have hr := e_135_7
  simp only [e_124_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 67) = bnot (a (.wire 60 71)) := by
  have c9120 := (publicBatchWrapper86_copies285 a h).1
  have c9121 := (publicBatchWrapper86_copies285 a h).2.1
  have c9122 := (publicBatchWrapper86_copies285 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_141_16 := arithEq_of_rows h (row := 141) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_16
  simp only [← c9120, k3, ← c9121, k3, ← c9122] at e_141_16
  have hr := e_141_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 71) = bnot (a (.wire 135 31)) := by
  have c9123 := (publicBatchWrapper86_copies285 a h).2.2.2.1
  have c9124 := (publicBatchWrapper86_copies285 a h).2.2.2.2.1
  have c9125 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_141_17 := arithEq_of_rows h (row := 141) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_17
  simp only [← c9123, k3, ← c9124, k3, ← c9125] at e_141_17
  have hr := e_141_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 11) = band (a (.wire 141 67)) (a (.wire 141 71)) := by
  have c9126 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.1
  have c9127 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.1
  have c9128 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.1
  have e_140_2 := arithEq_of_rows h (row := 140) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_2
  simp only [← c9126, ← c9127, ← c9128] at e_140_2
  have hr := e_140_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 141 79) = bselect (a (.wire 140 11)) (a (.virt 12205)) (a (.wire 141 15)) := by
  have c9129 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.1
  have c9130 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9131 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9132 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9133 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9134 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_141_18 := arithEq_of_rows h (row := 141) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_18
  simp only [← c9129, ← c9130, ← c9131] at e_141_18
  have e_141_19 := arithEq_of_rows h (row := 141) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_141_19
  simp only [← c9132, ← c9133, ← c9134] at e_141_19
  have hr := e_141_19
  simp only [e_141_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 7) = bselect (a (.wire 140 11)) (a (.virt 12206)) (a (.wire 141 23)) := by
  have c9135 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9136 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9137 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9138 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9139 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9140 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_0 := arithEq_of_rows h (row := 142) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_0
  simp only [← c9135, ← c9136, ← c9137] at e_142_0
  have e_142_1 := arithEq_of_rows h (row := 142) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_1
  simp only [← c9138, ← c9139, ← c9140] at e_142_1
  have hr := e_142_1
  simp only [e_142_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 15) = bselect (a (.wire 140 11)) (a (.virt 12207)) (a (.wire 141 31)) := by
  have c9141 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9142 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9143 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9144 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9145 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9146 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_2 := arithEq_of_rows h (row := 142) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_2
  simp only [← c9141, ← c9142, ← c9143] at e_142_2
  have e_142_3 := arithEq_of_rows h (row := 142) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_3
  simp only [← c9144, ← c9145, ← c9146] at e_142_3
  have hr := e_142_3
  simp only [e_142_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 23) = bselect (a (.wire 140 11)) (a (.virt 12208)) (a (.wire 141 39)) := by
  have c9147 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9148 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9149 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9150 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9151 := (publicBatchWrapper86_copies285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9152 := (publicBatchWrapper86_copies286 a h).1
  have e_142_4 := arithEq_of_rows h (row := 142) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_4
  simp only [← c9147, ← c9148, ← c9149] at e_142_4
  have e_142_5 := arithEq_of_rows h (row := 142) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_5
  simp only [← c9150, ← c9151, ← c9152] at e_142_5
  have hr := e_142_5
  simp only [e_142_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 31) = bselect (a (.wire 140 11)) (a (.virt 12209)) (a (.wire 141 47)) := by
  have c9153 := (publicBatchWrapper86_copies286 a h).2.1
  have c9154 := (publicBatchWrapper86_copies286 a h).2.2.1
  have c9155 := (publicBatchWrapper86_copies286 a h).2.2.2.1
  have c9156 := (publicBatchWrapper86_copies286 a h).2.2.2.2.1
  have c9157 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.1
  have c9158 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.1
  have e_142_6 := arithEq_of_rows h (row := 142) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_6
  simp only [← c9153, ← c9154, ← c9155] at e_142_6
  have e_142_7 := arithEq_of_rows h (row := 142) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_7
  simp only [← c9156, ← c9157, ← c9158] at e_142_7
  have hr := e_142_7
  simp only [e_142_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 39) = bselect (a (.wire 140 11)) (a (.virt 12203)) (a (.wire 141 55)) := by
  have c9159 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.1
  have c9160 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.1
  have c9161 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.1
  have c9162 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9163 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9164 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_8 := arithEq_of_rows h (row := 142) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_8
  simp only [← c9159, ← c9160, ← c9161] at e_142_8
  have e_142_9 := arithEq_of_rows h (row := 142) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_9
  simp only [← c9162, ← c9163, ← c9164] at e_142_9
  have hr := e_142_9
  simp only [e_142_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 47) = bselect (a (.wire 140 11)) (a (.virt 12204)) (a (.wire 141 63)) := by
  have c9165 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9166 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9167 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9168 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9169 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9170 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_10 := arithEq_of_rows h (row := 142) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_10
  simp only [← c9165, ← c9166, ← c9167] at e_142_10
  have e_142_11 := arithEq_of_rows h (row := 142) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_11
  simp only [← c9168, ← c9169, ← c9170] at e_142_11
  have hr := e_142_11
  simp only [e_142_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 35) = bor (a (.wire 135 31)) (a (.wire 141 67)) := by
  have c9171 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9172 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9173 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9174 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9175 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9176 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_8 := arithEq_of_rows h (row := 135) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_8
  simp only [← c9174, ← c9175, k3, ← c9176] at e_135_8
  have e_143_0 := arithEq_of_rows h (row := 143) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_0
  simp only [← c9171, ← c9172, ← c9173] at e_143_0
  have hr := e_135_8
  simp only [e_143_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 51) = bnot (a (.wire 61 35)) := by
  have c9177 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9178 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9179 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_142_12 := arithEq_of_rows h (row := 142) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_12
  simp only [← c9177, k3, ← c9178, k3, ← c9179] at e_142_12
  have hr := e_142_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 55) = bnot (a (.wire 135 35)) := by
  have c9180 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9181 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9182 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_142_13 := arithEq_of_rows h (row := 142) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_13
  simp only [← c9180, k3, ← c9181, k3, ← c9182] at e_142_13
  have hr := e_142_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 15) = band (a (.wire 142 51)) (a (.wire 142 55)) := by
  have c9183 := (publicBatchWrapper86_copies286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9184 := (publicBatchWrapper86_copies287 a h).1
  have c9185 := (publicBatchWrapper86_copies287 a h).2.1
  have e_140_3 := arithEq_of_rows h (row := 140) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_3
  simp only [← c9183, ← c9184, ← c9185] at e_140_3
  have hr := e_140_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 63) = bselect (a (.wire 140 15)) (a (.virt 12405)) (a (.wire 141 79)) := by
  have c9186 := (publicBatchWrapper86_copies287 a h).2.2.1
  have c9187 := (publicBatchWrapper86_copies287 a h).2.2.2.1
  have c9188 := (publicBatchWrapper86_copies287 a h).2.2.2.2.1
  have c9189 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.1
  have c9190 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.1
  have c9191 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.1
  have e_142_14 := arithEq_of_rows h (row := 142) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_14
  simp only [← c9186, ← c9187, ← c9188] at e_142_14
  have e_142_15 := arithEq_of_rows h (row := 142) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_15
  simp only [← c9189, ← c9190, ← c9191] at e_142_15
  have hr := e_142_15
  simp only [e_142_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 71) = bselect (a (.wire 140 15)) (a (.virt 12406)) (a (.wire 142 7)) := by
  have c9192 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.1
  have c9193 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.1
  have c9194 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9195 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9196 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9197 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_16 := arithEq_of_rows h (row := 142) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_16
  simp only [← c9192, ← c9193, ← c9194] at e_142_16
  have e_142_17 := arithEq_of_rows h (row := 142) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_17
  simp only [← c9195, ← c9196, ← c9197] at e_142_17
  have hr := e_142_17
  simp only [e_142_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 142 79) = bselect (a (.wire 140 15)) (a (.virt 12407)) (a (.wire 142 15)) := by
  have c9198 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9199 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9200 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9201 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9202 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9203 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_142_18 := arithEq_of_rows h (row := 142) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_18
  simp only [← c9198, ← c9199, ← c9200] at e_142_18
  have e_142_19 := arithEq_of_rows h (row := 142) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_142_19
  simp only [← c9201, ← c9202, ← c9203] at e_142_19
  have hr := e_142_19
  simp only [e_142_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 7) = bselect (a (.wire 140 15)) (a (.virt 12408)) (a (.wire 142 23)) := by
  have c9204 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9205 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9206 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9207 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9208 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9209 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_0 := arithEq_of_rows h (row := 144) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_0
  simp only [← c9204, ← c9205, ← c9206] at e_144_0
  have e_144_1 := arithEq_of_rows h (row := 144) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_1
  simp only [← c9207, ← c9208, ← c9209] at e_144_1
  have hr := e_144_1
  simp only [e_144_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 15) = bselect (a (.wire 140 15)) (a (.virt 12409)) (a (.wire 142 31)) := by
  have c9210 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9211 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9212 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9213 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9214 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9215 := (publicBatchWrapper86_copies287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_144_2 := arithEq_of_rows h (row := 144) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_2
  simp only [← c9210, ← c9211, ← c9212] at e_144_2
  have e_144_3 := arithEq_of_rows h (row := 144) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_3
  simp only [← c9213, ← c9214, ← c9215] at e_144_3
  have hr := e_144_3
  simp only [e_144_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 23) = bselect (a (.wire 140 15)) (a (.virt 12403)) (a (.wire 142 39)) := by
  have c9216 := (publicBatchWrapper86_copies288 a h).1
  have c9217 := (publicBatchWrapper86_copies288 a h).2.1
  have c9218 := (publicBatchWrapper86_copies288 a h).2.2.1
  have c9219 := (publicBatchWrapper86_copies288 a h).2.2.2.1
  have c9220 := (publicBatchWrapper86_copies288 a h).2.2.2.2.1
  have c9221 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.1
  have e_144_4 := arithEq_of_rows h (row := 144) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_4
  simp only [← c9216, ← c9217, ← c9218] at e_144_4
  have e_144_5 := arithEq_of_rows h (row := 144) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_5
  simp only [← c9219, ← c9220, ← c9221] at e_144_5
  have hr := e_144_5
  simp only [e_144_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 31) = bselect (a (.wire 140 15)) (a (.virt 12404)) (a (.wire 142 47)) := by
  have c9222 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.1
  have c9223 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.1
  have c9224 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.1
  have c9225 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.1
  have c9226 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9227 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_6 := arithEq_of_rows h (row := 144) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_6
  simp only [← c9222, ← c9223, ← c9224] at e_144_6
  have e_144_7 := arithEq_of_rows h (row := 144) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_7
  simp only [← c9225, ← c9226, ← c9227] at e_144_7
  have hr := e_144_7
  simp only [e_144_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 39) = bor (a (.wire 135 35)) (a (.wire 142 51)) := by
  have c9228 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9229 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9230 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9231 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9232 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9233 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_9 := arithEq_of_rows h (row := 135) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_9
  simp only [← c9231, ← c9232, k3, ← c9233] at e_135_9
  have e_143_1 := arithEq_of_rows h (row := 143) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_1
  simp only [← c9228, ← c9229, ← c9230] at e_143_1
  have hr := e_135_9
  simp only [e_143_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 35) = bnot (a (.wire 61 79)) := by
  have c9234 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9235 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9236 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_144_8 := arithEq_of_rows h (row := 144) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_8
  simp only [← c9234, k3, ← c9235, k3, ← c9236] at e_144_8
  have hr := e_144_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 39) = bnot (a (.wire 135 39)) := by
  have c9237 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9238 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9239 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_144_9 := arithEq_of_rows h (row := 144) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_9
  simp only [← c9237, k3, ← c9238, k3, ← c9239] at e_144_9
  have hr := e_144_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 19) = band (a (.wire 144 35)) (a (.wire 144 39)) := by
  have c9240 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9241 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9242 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_4 := arithEq_of_rows h (row := 140) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_4
  simp only [← c9240, ← c9241, ← c9242] at e_140_4
  have hr := e_140_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 47) = bselect (a (.wire 140 19)) (a (.virt 12605)) (a (.wire 142 63)) := by
  have c9243 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9244 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9245 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9246 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9247 := (publicBatchWrapper86_copies288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9248 := (publicBatchWrapper86_copies289 a h).1
  have e_144_10 := arithEq_of_rows h (row := 144) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_10
  simp only [← c9243, ← c9244, ← c9245] at e_144_10
  have e_144_11 := arithEq_of_rows h (row := 144) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_11
  simp only [← c9246, ← c9247, ← c9248] at e_144_11
  have hr := e_144_11
  simp only [e_144_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 55) = bselect (a (.wire 140 19)) (a (.virt 12606)) (a (.wire 142 71)) := by
  have c9249 := (publicBatchWrapper86_copies289 a h).2.1
  have c9250 := (publicBatchWrapper86_copies289 a h).2.2.1
  have c9251 := (publicBatchWrapper86_copies289 a h).2.2.2.1
  have c9252 := (publicBatchWrapper86_copies289 a h).2.2.2.2.1
  have c9253 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.1
  have c9254 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.1
  have e_144_12 := arithEq_of_rows h (row := 144) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_12
  simp only [← c9249, ← c9250, ← c9251] at e_144_12
  have e_144_13 := arithEq_of_rows h (row := 144) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_13
  simp only [← c9252, ← c9253, ← c9254] at e_144_13
  have hr := e_144_13
  simp only [e_144_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 63) = bselect (a (.wire 140 19)) (a (.virt 12607)) (a (.wire 142 79)) := by
  have c9255 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.1
  have c9256 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.1
  have c9257 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.1
  have c9258 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9259 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9260 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_14 := arithEq_of_rows h (row := 144) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_14
  simp only [← c9255, ← c9256, ← c9257] at e_144_14
  have e_144_15 := arithEq_of_rows h (row := 144) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_15
  simp only [← c9258, ← c9259, ← c9260] at e_144_15
  have hr := e_144_15
  simp only [e_144_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 71) = bselect (a (.wire 140 19)) (a (.virt 12608)) (a (.wire 144 7)) := by
  have c9261 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9262 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9263 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9264 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9265 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9266 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_16 := arithEq_of_rows h (row := 144) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_16
  simp only [← c9261, ← c9262, ← c9263] at e_144_16
  have e_144_17 := arithEq_of_rows h (row := 144) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_17
  simp only [← c9264, ← c9265, ← c9266] at e_144_17
  have hr := e_144_17
  simp only [e_144_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 144 79) = bselect (a (.wire 140 19)) (a (.virt 12609)) (a (.wire 144 15)) := by
  have c9267 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9268 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9269 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9270 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9271 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9272 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_144_18 := arithEq_of_rows h (row := 144) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_18
  simp only [← c9267, ← c9268, ← c9269] at e_144_18
  have e_144_19 := arithEq_of_rows h (row := 144) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_144_19
  simp only [← c9270, ← c9271, ← c9272] at e_144_19
  have hr := e_144_19
  simp only [e_144_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 7) = bselect (a (.wire 140 19)) (a (.virt 12603)) (a (.wire 144 23)) := by
  have c9273 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9274 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9275 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9276 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9277 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9278 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_145_0 := arithEq_of_rows h (row := 145) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_0
  simp only [← c9273, ← c9274, ← c9275] at e_145_0
  have e_145_1 := arithEq_of_rows h (row := 145) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_1
  simp only [← c9276, ← c9277, ← c9278] at e_145_1
  have hr := e_145_1
  simp only [e_145_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 15) = bselect (a (.wire 140 19)) (a (.virt 12604)) (a (.wire 144 31)) := by
  have c9279 := (publicBatchWrapper86_copies289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9280 := (publicBatchWrapper86_copies290 a h).1
  have c9281 := (publicBatchWrapper86_copies290 a h).2.1
  have c9282 := (publicBatchWrapper86_copies290 a h).2.2.1
  have c9283 := (publicBatchWrapper86_copies290 a h).2.2.2.1
  have c9284 := (publicBatchWrapper86_copies290 a h).2.2.2.2.1
  have e_145_2 := arithEq_of_rows h (row := 145) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_2
  simp only [← c9279, ← c9280, ← c9281] at e_145_2
  have e_145_3 := arithEq_of_rows h (row := 145) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_3
  simp only [← c9282, ← c9283, ← c9284] at e_145_3
  have hr := e_145_3
  simp only [e_145_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 43) = bor (a (.wire 135 39)) (a (.wire 144 35)) := by
  have c9285 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.1
  have c9286 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.1
  have c9287 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.1
  have c9288 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.1
  have c9289 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.1
  have c9290 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_10 := arithEq_of_rows h (row := 135) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_10
  simp only [← c9288, ← c9289, k3, ← c9290] at e_135_10
  have e_143_2 := arithEq_of_rows h (row := 143) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_2
  simp only [← c9285, ← c9286, ← c9287] at e_143_2
  have hr := e_135_10
  simp only [e_143_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 19) = bnot (a (.wire 63 43)) := by
  have c9291 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9292 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9293 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_145_4 := arithEq_of_rows h (row := 145) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_4
  simp only [← c9291, k3, ← c9292, k3, ← c9293] at e_145_4
  have hr := e_145_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 23) = bnot (a (.wire 135 43)) := by
  have c9294 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9295 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9296 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_145_5 := arithEq_of_rows h (row := 145) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_5
  simp only [← c9294, k3, ← c9295, k3, ← c9296] at e_145_5
  have hr := e_145_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 23) = band (a (.wire 145 19)) (a (.wire 145 23)) := by
  have c9297 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9298 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9299 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_5 := arithEq_of_rows h (row := 140) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_5
  simp only [← c9297, ← c9298, ← c9299] at e_140_5
  have hr := e_140_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 31) = bselect (a (.wire 140 23)) (a (.virt 12805)) (a (.wire 144 47)) := by
  have c9300 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9301 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9302 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9303 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9304 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9305 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_145_6 := arithEq_of_rows h (row := 145) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_6
  simp only [← c9300, ← c9301, ← c9302] at e_145_6
  have e_145_7 := arithEq_of_rows h (row := 145) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_7
  simp only [← c9303, ← c9304, ← c9305] at e_145_7
  have hr := e_145_7
  simp only [e_145_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 39) = bselect (a (.wire 140 23)) (a (.virt 12806)) (a (.wire 144 55)) := by
  have c9306 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9307 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9308 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9309 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9310 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9311 := (publicBatchWrapper86_copies290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_145_8 := arithEq_of_rows h (row := 145) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_8
  simp only [← c9306, ← c9307, ← c9308] at e_145_8
  have e_145_9 := arithEq_of_rows h (row := 145) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_9
  simp only [← c9309, ← c9310, ← c9311] at e_145_9
  have hr := e_145_9
  simp only [e_145_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 47) = bselect (a (.wire 140 23)) (a (.virt 12807)) (a (.wire 144 63)) := by
  have c9312 := (publicBatchWrapper86_copies291 a h).1
  have c9313 := (publicBatchWrapper86_copies291 a h).2.1
  have c9314 := (publicBatchWrapper86_copies291 a h).2.2.1
  have c9315 := (publicBatchWrapper86_copies291 a h).2.2.2.1
  have c9316 := (publicBatchWrapper86_copies291 a h).2.2.2.2.1
  have c9317 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.1
  have e_145_10 := arithEq_of_rows h (row := 145) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_10
  simp only [← c9312, ← c9313, ← c9314] at e_145_10
  have e_145_11 := arithEq_of_rows h (row := 145) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_11
  simp only [← c9315, ← c9316, ← c9317] at e_145_11
  have hr := e_145_11
  simp only [e_145_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 55) = bselect (a (.wire 140 23)) (a (.virt 12808)) (a (.wire 144 71)) := by
  have c9318 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.1
  have c9319 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.1
  have c9320 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.1
  have c9321 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.1
  have c9322 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9323 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_145_12 := arithEq_of_rows h (row := 145) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_12
  simp only [← c9318, ← c9319, ← c9320] at e_145_12
  have e_145_13 := arithEq_of_rows h (row := 145) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_13
  simp only [← c9321, ← c9322, ← c9323] at e_145_13
  have hr := e_145_13
  simp only [e_145_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 63) = bselect (a (.wire 140 23)) (a (.virt 12809)) (a (.wire 144 79)) := by
  have c9324 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9325 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9326 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9327 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9328 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9329 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_145_14 := arithEq_of_rows h (row := 145) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_14
  simp only [← c9324, ← c9325, ← c9326] at e_145_14
  have e_145_15 := arithEq_of_rows h (row := 145) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_15
  simp only [← c9327, ← c9328, ← c9329] at e_145_15
  have hr := e_145_15
  simp only [e_145_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 71) = bselect (a (.wire 140 23)) (a (.virt 12803)) (a (.wire 145 7)) := by
  have c9330 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9331 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9332 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9333 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9334 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9335 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_145_16 := arithEq_of_rows h (row := 145) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_16
  simp only [← c9330, ← c9331, ← c9332] at e_145_16
  have e_145_17 := arithEq_of_rows h (row := 145) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_17
  simp only [← c9333, ← c9334, ← c9335] at e_145_17
  have hr := e_145_17
  simp only [e_145_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 145 79) = bselect (a (.wire 140 23)) (a (.virt 12804)) (a (.wire 145 15)) := by
  have c9336 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9337 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9338 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9339 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9340 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9341 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_145_18 := arithEq_of_rows h (row := 145) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_18
  simp only [← c9336, ← c9337, ← c9338] at e_145_18
  have e_145_19 := arithEq_of_rows h (row := 145) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_145_19
  simp only [← c9339, ← c9340, ← c9341] at e_145_19
  have hr := e_145_19
  simp only [e_145_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 47) = bor (a (.wire 135 43)) (a (.wire 145 19)) := by
  have c9342 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9343 := (publicBatchWrapper86_copies291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9344 := (publicBatchWrapper86_copies292 a h).1
  have c9345 := (publicBatchWrapper86_copies292 a h).2.1
  have c9346 := (publicBatchWrapper86_copies292 a h).2.2.1
  have c9347 := (publicBatchWrapper86_copies292 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_11 := arithEq_of_rows h (row := 135) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_11
  simp only [← c9345, ← c9346, k3, ← c9347] at e_135_11
  have e_143_3 := arithEq_of_rows h (row := 143) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_3
  simp only [← c9342, ← c9343, ← c9344] at e_143_3
  have hr := e_135_11
  simp only [e_143_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 3) = bnot (a (.wire 65 7)) := by
  have c9348 := (publicBatchWrapper86_copies292 a h).2.2.2.2.1
  have c9349 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.1
  have c9350 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_146_0 := arithEq_of_rows h (row := 146) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_0
  simp only [← c9348, k3, ← c9349, k3, ← c9350] at e_146_0
  have hr := e_146_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 7) = bnot (a (.wire 135 47)) := by
  have c9351 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.1
  have c9352 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.1
  have c9353 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_146_1 := arithEq_of_rows h (row := 146) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_1
  simp only [← c9351, k3, ← c9352, k3, ← c9353] at e_146_1
  have hr := e_146_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 27) = band (a (.wire 146 3)) (a (.wire 146 7)) := by
  have c9354 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9355 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9356 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_6 := arithEq_of_rows h (row := 140) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_6
  simp only [← c9354, ← c9355, ← c9356] at e_140_6
  have hr := e_140_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 15) = bselect (a (.wire 140 27)) (a (.virt 13005)) (a (.wire 145 31)) := by
  have c9357 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9358 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9359 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9360 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9361 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9362 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_2 := arithEq_of_rows h (row := 146) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_2
  simp only [← c9357, ← c9358, ← c9359] at e_146_2
  have e_146_3 := arithEq_of_rows h (row := 146) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_3
  simp only [← c9360, ← c9361, ← c9362] at e_146_3
  have hr := e_146_3
  simp only [e_146_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 23) = bselect (a (.wire 140 27)) (a (.virt 13006)) (a (.wire 145 39)) := by
  have c9363 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9364 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9365 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9366 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9367 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9368 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_4 := arithEq_of_rows h (row := 146) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_4
  simp only [← c9363, ← c9364, ← c9365] at e_146_4
  have e_146_5 := arithEq_of_rows h (row := 146) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_5
  simp only [← c9366, ← c9367, ← c9368] at e_146_5
  have hr := e_146_5
  simp only [e_146_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 31) = bselect (a (.wire 140 27)) (a (.virt 13007)) (a (.wire 145 47)) := by
  have c9369 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9370 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9371 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9372 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9373 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9374 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_6 := arithEq_of_rows h (row := 146) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_6
  simp only [← c9369, ← c9370, ← c9371] at e_146_6
  have e_146_7 := arithEq_of_rows h (row := 146) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_7
  simp only [← c9372, ← c9373, ← c9374] at e_146_7
  have hr := e_146_7
  simp only [e_146_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 39) = bselect (a (.wire 140 27)) (a (.virt 13008)) (a (.wire 145 55)) := by
  have c9375 := (publicBatchWrapper86_copies292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9376 := (publicBatchWrapper86_copies293 a h).1
  have c9377 := (publicBatchWrapper86_copies293 a h).2.1
  have c9378 := (publicBatchWrapper86_copies293 a h).2.2.1
  have c9379 := (publicBatchWrapper86_copies293 a h).2.2.2.1
  have c9380 := (publicBatchWrapper86_copies293 a h).2.2.2.2.1
  have e_146_8 := arithEq_of_rows h (row := 146) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_8
  simp only [← c9375, ← c9376, ← c9377] at e_146_8
  have e_146_9 := arithEq_of_rows h (row := 146) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_9
  simp only [← c9378, ← c9379, ← c9380] at e_146_9
  have hr := e_146_9
  simp only [e_146_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 47) = bselect (a (.wire 140 27)) (a (.virt 13009)) (a (.wire 145 63)) := by
  have c9381 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.1
  have c9382 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.1
  have c9383 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.1
  have c9384 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.1
  have c9385 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.1
  have c9386 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_146_10 := arithEq_of_rows h (row := 146) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_10
  simp only [← c9381, ← c9382, ← c9383] at e_146_10
  have e_146_11 := arithEq_of_rows h (row := 146) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_11
  simp only [← c9384, ← c9385, ← c9386] at e_146_11
  have hr := e_146_11
  simp only [e_146_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 55) = bselect (a (.wire 140 27)) (a (.virt 13003)) (a (.wire 145 71)) := by
  have c9387 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9388 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9389 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9390 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9391 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9392 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_12 := arithEq_of_rows h (row := 146) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_12
  simp only [← c9387, ← c9388, ← c9389] at e_146_12
  have e_146_13 := arithEq_of_rows h (row := 146) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_13
  simp only [← c9390, ← c9391, ← c9392] at e_146_13
  have hr := e_146_13
  simp only [e_146_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 63) = bselect (a (.wire 140 27)) (a (.virt 13004)) (a (.wire 145 79)) := by
  have c9393 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9394 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9395 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9396 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9397 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9398 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_14 := arithEq_of_rows h (row := 146) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_14
  simp only [← c9393, ← c9394, ← c9395] at e_146_14
  have e_146_15 := arithEq_of_rows h (row := 146) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_15
  simp only [← c9396, ← c9397, ← c9398] at e_146_15
  have hr := e_146_15
  simp only [e_146_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 51) = bor (a (.wire 135 47)) (a (.wire 146 3)) := by
  have c9399 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9400 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9401 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9402 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9403 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9404 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_12 := arithEq_of_rows h (row := 135) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_12
  simp only [← c9402, ← c9403, k3, ← c9404] at e_135_12
  have e_143_4 := arithEq_of_rows h (row := 143) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_4
  simp only [← c9399, ← c9400, ← c9401] at e_143_4
  have hr := e_135_12
  simp only [e_143_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 67) = bnot (a (.wire 65 51)) := by
  have c9405 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9406 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9407 := (publicBatchWrapper86_copies293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_146_16 := arithEq_of_rows h (row := 146) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_16
  simp only [← c9405, k3, ← c9406, k3, ← c9407] at e_146_16
  have hr := e_146_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 71) = bnot (a (.wire 135 51)) := by
  have c9408 := (publicBatchWrapper86_copies294 a h).1
  have c9409 := (publicBatchWrapper86_copies294 a h).2.1
  have c9410 := (publicBatchWrapper86_copies294 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_146_17 := arithEq_of_rows h (row := 146) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_17
  simp only [← c9408, k3, ← c9409, k3, ← c9410] at e_146_17
  have hr := e_146_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 31) = band (a (.wire 146 67)) (a (.wire 146 71)) := by
  have c9411 := (publicBatchWrapper86_copies294 a h).2.2.2.1
  have c9412 := (publicBatchWrapper86_copies294 a h).2.2.2.2.1
  have c9413 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.1
  have e_140_7 := arithEq_of_rows h (row := 140) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_7
  simp only [← c9411, ← c9412, ← c9413] at e_140_7
  have hr := e_140_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 146 79) = bselect (a (.wire 140 31)) (a (.virt 13205)) (a (.wire 146 15)) := by
  have c9414 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.1
  have c9415 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.1
  have c9416 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.1
  have c9417 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.1
  have c9418 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9419 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_146_18 := arithEq_of_rows h (row := 146) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_18
  simp only [← c9414, ← c9415, ← c9416] at e_146_18
  have e_146_19 := arithEq_of_rows h (row := 146) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_146_19
  simp only [← c9417, ← c9418, ← c9419] at e_146_19
  have hr := e_146_19
  simp only [e_146_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 7) = bselect (a (.wire 140 31)) (a (.virt 13206)) (a (.wire 146 23)) := by
  have c9420 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9421 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9422 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9423 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9424 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9425 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_0 := arithEq_of_rows h (row := 147) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_0
  simp only [← c9420, ← c9421, ← c9422] at e_147_0
  have e_147_1 := arithEq_of_rows h (row := 147) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_1
  simp only [← c9423, ← c9424, ← c9425] at e_147_1
  have hr := e_147_1
  simp only [e_147_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 15) = bselect (a (.wire 140 31)) (a (.virt 13207)) (a (.wire 146 31)) := by
  have c9426 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9427 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9428 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9429 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9430 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9431 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_2 := arithEq_of_rows h (row := 147) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_2
  simp only [← c9426, ← c9427, ← c9428] at e_147_2
  have e_147_3 := arithEq_of_rows h (row := 147) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_3
  simp only [← c9429, ← c9430, ← c9431] at e_147_3
  have hr := e_147_3
  simp only [e_147_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 23) = bselect (a (.wire 140 31)) (a (.virt 13208)) (a (.wire 146 39)) := by
  have c9432 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9433 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9434 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9435 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9436 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9437 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_4 := arithEq_of_rows h (row := 147) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_4
  simp only [← c9432, ← c9433, ← c9434] at e_147_4
  have e_147_5 := arithEq_of_rows h (row := 147) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_5
  simp only [← c9435, ← c9436, ← c9437] at e_147_5
  have hr := e_147_5
  simp only [e_147_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 31) = bselect (a (.wire 140 31)) (a (.virt 13209)) (a (.wire 146 47)) := by
  have c9438 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9439 := (publicBatchWrapper86_copies294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9440 := (publicBatchWrapper86_copies295 a h).1
  have c9441 := (publicBatchWrapper86_copies295 a h).2.1
  have c9442 := (publicBatchWrapper86_copies295 a h).2.2.1
  have c9443 := (publicBatchWrapper86_copies295 a h).2.2.2.1
  have e_147_6 := arithEq_of_rows h (row := 147) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_6
  simp only [← c9438, ← c9439, ← c9440] at e_147_6
  have e_147_7 := arithEq_of_rows h (row := 147) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_7
  simp only [← c9441, ← c9442, ← c9443] at e_147_7
  have hr := e_147_7
  simp only [e_147_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 39) = bselect (a (.wire 140 31)) (a (.virt 13203)) (a (.wire 146 55)) := by
  have c9444 := (publicBatchWrapper86_copies295 a h).2.2.2.2.1
  have c9445 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.1
  have c9446 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.1
  have c9447 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.1
  have c9448 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.1
  have c9449 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.1
  have e_147_8 := arithEq_of_rows h (row := 147) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_8
  simp only [← c9444, ← c9445, ← c9446] at e_147_8
  have e_147_9 := arithEq_of_rows h (row := 147) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_9
  simp only [← c9447, ← c9448, ← c9449] at e_147_9
  have hr := e_147_9
  simp only [e_147_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 47) = bselect (a (.wire 140 31)) (a (.virt 13204)) (a (.wire 146 63)) := by
  have c9450 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9451 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9452 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9453 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9454 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9455 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_10 := arithEq_of_rows h (row := 147) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_10
  simp only [← c9450, ← c9451, ← c9452] at e_147_10
  have e_147_11 := arithEq_of_rows h (row := 147) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_11
  simp only [← c9453, ← c9454, ← c9455] at e_147_11
  have hr := e_147_11
  simp only [e_147_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 55) = bor (a (.wire 135 51)) (a (.wire 146 67)) := by
  have c9456 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9457 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9458 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9459 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9460 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9461 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_13 := arithEq_of_rows h (row := 135) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_13
  simp only [← c9459, ← c9460, k3, ← c9461] at e_135_13
  have e_143_5 := arithEq_of_rows h (row := 143) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_5
  simp only [← c9456, ← c9457, ← c9458] at e_143_5
  have hr := e_135_13
  simp only [e_143_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 51) = bnot (a (.wire 67 15)) := by
  have c9462 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9463 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9464 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_147_12 := arithEq_of_rows h (row := 147) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_12
  simp only [← c9462, k3, ← c9463, k3, ← c9464] at e_147_12
  have hr := e_147_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 55) = bnot (a (.wire 135 55)) := by
  have c9465 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9466 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9467 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_147_13 := arithEq_of_rows h (row := 147) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_13
  simp only [← c9465, k3, ← c9466, k3, ← c9467] at e_147_13
  have hr := e_147_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 35) = band (a (.wire 147 51)) (a (.wire 147 55)) := by
  have c9468 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9469 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9470 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_8 := arithEq_of_rows h (row := 140) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_8
  simp only [← c9468, ← c9469, ← c9470] at e_140_8
  have hr := e_140_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 63) = bselect (a (.wire 140 35)) (a (.virt 13405)) (a (.wire 146 79)) := by
  have c9471 := (publicBatchWrapper86_copies295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9472 := (publicBatchWrapper86_copies296 a h).1
  have c9473 := (publicBatchWrapper86_copies296 a h).2.1
  have c9474 := (publicBatchWrapper86_copies296 a h).2.2.1
  have c9475 := (publicBatchWrapper86_copies296 a h).2.2.2.1
  have c9476 := (publicBatchWrapper86_copies296 a h).2.2.2.2.1
  have e_147_14 := arithEq_of_rows h (row := 147) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_14
  simp only [← c9471, ← c9472, ← c9473] at e_147_14
  have e_147_15 := arithEq_of_rows h (row := 147) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_15
  simp only [← c9474, ← c9475, ← c9476] at e_147_15
  have hr := e_147_15
  simp only [e_147_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 71) = bselect (a (.wire 140 35)) (a (.virt 13406)) (a (.wire 147 7)) := by
  have c9477 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.1
  have c9478 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.1
  have c9479 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.1
  have c9480 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.1
  have c9481 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.1
  have c9482 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_147_16 := arithEq_of_rows h (row := 147) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_16
  simp only [← c9477, ← c9478, ← c9479] at e_147_16
  have e_147_17 := arithEq_of_rows h (row := 147) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_17
  simp only [← c9480, ← c9481, ← c9482] at e_147_17
  have hr := e_147_17
  simp only [e_147_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 147 79) = bselect (a (.wire 140 35)) (a (.virt 13407)) (a (.wire 147 15)) := by
  have c9483 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9484 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9485 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9486 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9487 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9488 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_147_18 := arithEq_of_rows h (row := 147) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_18
  simp only [← c9483, ← c9484, ← c9485] at e_147_18
  have e_147_19 := arithEq_of_rows h (row := 147) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_147_19
  simp only [← c9486, ← c9487, ← c9488] at e_147_19
  have hr := e_147_19
  simp only [e_147_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 7) = bselect (a (.wire 140 35)) (a (.virt 13408)) (a (.wire 147 23)) := by
  have c9489 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9490 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9491 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9492 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9493 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9494 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_0 := arithEq_of_rows h (row := 148) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_0
  simp only [← c9489, ← c9490, ← c9491] at e_148_0
  have e_148_1 := arithEq_of_rows h (row := 148) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_1
  simp only [← c9492, ← c9493, ← c9494] at e_148_1
  have hr := e_148_1
  simp only [e_148_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 15) = bselect (a (.wire 140 35)) (a (.virt 13409)) (a (.wire 147 31)) := by
  have c9495 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9496 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9497 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9498 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9499 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9500 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_2 := arithEq_of_rows h (row := 148) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_2
  simp only [← c9495, ← c9496, ← c9497] at e_148_2
  have e_148_3 := arithEq_of_rows h (row := 148) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_3
  simp only [← c9498, ← c9499, ← c9500] at e_148_3
  have hr := e_148_3
  simp only [e_148_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 23) = bselect (a (.wire 140 35)) (a (.virt 13403)) (a (.wire 147 39)) := by
  have c9501 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9502 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9503 := (publicBatchWrapper86_copies296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9504 := (publicBatchWrapper86_copies297 a h).1
  have c9505 := (publicBatchWrapper86_copies297 a h).2.1
  have c9506 := (publicBatchWrapper86_copies297 a h).2.2.1
  have e_148_4 := arithEq_of_rows h (row := 148) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_4
  simp only [← c9501, ← c9502, ← c9503] at e_148_4
  have e_148_5 := arithEq_of_rows h (row := 148) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_5
  simp only [← c9504, ← c9505, ← c9506] at e_148_5
  have hr := e_148_5
  simp only [e_148_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 31) = bselect (a (.wire 140 35)) (a (.virt 13404)) (a (.wire 147 47)) := by
  have c9507 := (publicBatchWrapper86_copies297 a h).2.2.2.1
  have c9508 := (publicBatchWrapper86_copies297 a h).2.2.2.2.1
  have c9509 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.1
  have c9510 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.1
  have c9511 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.1
  have c9512 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.1
  have e_148_6 := arithEq_of_rows h (row := 148) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_6
  simp only [← c9507, ← c9508, ← c9509] at e_148_6
  have e_148_7 := arithEq_of_rows h (row := 148) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_7
  simp only [← c9510, ← c9511, ← c9512] at e_148_7
  have hr := e_148_7
  simp only [e_148_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 59) = bor (a (.wire 135 55)) (a (.wire 147 51)) := by
  have c9513 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.1
  have c9514 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9515 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9516 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9517 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9518 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_14 := arithEq_of_rows h (row := 135) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_14
  simp only [← c9516, ← c9517, k3, ← c9518] at e_135_14
  have e_143_6 := arithEq_of_rows h (row := 143) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_6
  simp only [← c9513, ← c9514, ← c9515] at e_143_6
  have hr := e_135_14
  simp only [e_143_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 35) = bnot (a (.wire 67 59)) := by
  have c9519 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9520 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9521 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_148_8 := arithEq_of_rows h (row := 148) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_8
  simp only [← c9519, k3, ← c9520, k3, ← c9521] at e_148_8
  have hr := e_148_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 39) = bnot (a (.wire 135 59)) := by
  have c9522 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9523 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9524 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_148_9 := arithEq_of_rows h (row := 148) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_9
  simp only [← c9522, k3, ← c9523, k3, ← c9524] at e_148_9
  have hr := e_148_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 39) = band (a (.wire 148 35)) (a (.wire 148 39)) := by
  have c9525 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9526 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9527 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_9 := arithEq_of_rows h (row := 140) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_9
  simp only [← c9525, ← c9526, ← c9527] at e_140_9
  have hr := e_140_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 47) = bselect (a (.wire 140 39)) (a (.virt 13605)) (a (.wire 147 63)) := by
  have c9528 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9529 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9530 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9531 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9532 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9533 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_10 := arithEq_of_rows h (row := 148) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_10
  simp only [← c9528, ← c9529, ← c9530] at e_148_10
  have e_148_11 := arithEq_of_rows h (row := 148) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_11
  simp only [← c9531, ← c9532, ← c9533] at e_148_11
  have hr := e_148_11
  simp only [e_148_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 55) = bselect (a (.wire 140 39)) (a (.virt 13606)) (a (.wire 147 71)) := by
  have c9534 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9535 := (publicBatchWrapper86_copies297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9536 := (publicBatchWrapper86_copies298 a h).1
  have c9537 := (publicBatchWrapper86_copies298 a h).2.1
  have c9538 := (publicBatchWrapper86_copies298 a h).2.2.1
  have c9539 := (publicBatchWrapper86_copies298 a h).2.2.2.1
  have e_148_12 := arithEq_of_rows h (row := 148) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_12
  simp only [← c9534, ← c9535, ← c9536] at e_148_12
  have e_148_13 := arithEq_of_rows h (row := 148) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_13
  simp only [← c9537, ← c9538, ← c9539] at e_148_13
  have hr := e_148_13
  simp only [e_148_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 63) = bselect (a (.wire 140 39)) (a (.virt 13607)) (a (.wire 147 79)) := by
  have c9540 := (publicBatchWrapper86_copies298 a h).2.2.2.2.1
  have c9541 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.1
  have c9542 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.1
  have c9543 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.1
  have c9544 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.1
  have c9545 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.1
  have e_148_14 := arithEq_of_rows h (row := 148) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_14
  simp only [← c9540, ← c9541, ← c9542] at e_148_14
  have e_148_15 := arithEq_of_rows h (row := 148) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_15
  simp only [← c9543, ← c9544, ← c9545] at e_148_15
  have hr := e_148_15
  simp only [e_148_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 71) = bselect (a (.wire 140 39)) (a (.virt 13608)) (a (.wire 148 7)) := by
  have c9546 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9547 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9548 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9549 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9550 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9551 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_16 := arithEq_of_rows h (row := 148) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_16
  simp only [← c9546, ← c9547, ← c9548] at e_148_16
  have e_148_17 := arithEq_of_rows h (row := 148) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_17
  simp only [← c9549, ← c9550, ← c9551] at e_148_17
  have hr := e_148_17
  simp only [e_148_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 148 79) = bselect (a (.wire 140 39)) (a (.virt 13609)) (a (.wire 148 15)) := by
  have c9552 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9553 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9554 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9555 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9556 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9557 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_148_18 := arithEq_of_rows h (row := 148) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_18
  simp only [← c9552, ← c9553, ← c9554] at e_148_18
  have e_148_19 := arithEq_of_rows h (row := 148) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_148_19
  simp only [← c9555, ← c9556, ← c9557] at e_148_19
  have hr := e_148_19
  simp only [e_148_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 7) = bselect (a (.wire 140 39)) (a (.virt 13603)) (a (.wire 148 23)) := by
  have c9558 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9559 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9560 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9561 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9562 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9563 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_0 := arithEq_of_rows h (row := 149) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_0
  simp only [← c9558, ← c9559, ← c9560] at e_149_0
  have e_149_1 := arithEq_of_rows h (row := 149) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_1
  simp only [← c9561, ← c9562, ← c9563] at e_149_1
  have hr := e_149_1
  simp only [e_149_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 15) = bselect (a (.wire 140 39)) (a (.virt 13604)) (a (.wire 148 31)) := by
  have c9564 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9565 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9566 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9567 := (publicBatchWrapper86_copies298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9568 := (publicBatchWrapper86_copies299 a h).1
  have c9569 := (publicBatchWrapper86_copies299 a h).2.1
  have e_149_2 := arithEq_of_rows h (row := 149) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_2
  simp only [← c9564, ← c9565, ← c9566] at e_149_2
  have e_149_3 := arithEq_of_rows h (row := 149) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_3
  simp only [← c9567, ← c9568, ← c9569] at e_149_3
  have hr := e_149_3
  simp only [e_149_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 63) = bor (a (.wire 135 59)) (a (.wire 148 35)) := by
  have c9570 := (publicBatchWrapper86_copies299 a h).2.2.1
  have c9571 := (publicBatchWrapper86_copies299 a h).2.2.2.1
  have c9572 := (publicBatchWrapper86_copies299 a h).2.2.2.2.1
  have c9573 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.1
  have c9574 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.1
  have c9575 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_15 := arithEq_of_rows h (row := 135) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_15
  simp only [← c9573, ← c9574, k3, ← c9575] at e_135_15
  have e_143_7 := arithEq_of_rows h (row := 143) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_7
  simp only [← c9570, ← c9571, ← c9572] at e_143_7
  have hr := e_135_15
  simp only [e_143_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 19) = bnot (a (.wire 68 23)) := by
  have c9576 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.1
  have c9577 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.1
  have c9578 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_149_4 := arithEq_of_rows h (row := 149) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_4
  simp only [← c9576, k3, ← c9577, k3, ← c9578] at e_149_4
  have hr := e_149_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 23) = bnot (a (.wire 135 63)) := by
  have c9579 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9580 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9581 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_149_5 := arithEq_of_rows h (row := 149) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_5
  simp only [← c9579, k3, ← c9580, k3, ← c9581] at e_149_5
  have hr := e_149_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 43) = band (a (.wire 149 19)) (a (.wire 149 23)) := by
  have c9582 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9583 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9584 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_10 := arithEq_of_rows h (row := 140) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_10
  simp only [← c9582, ← c9583, ← c9584] at e_140_10
  have hr := e_140_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 31) = bselect (a (.wire 140 43)) (a (.virt 13805)) (a (.wire 148 47)) := by
  have c9585 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9586 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9587 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9588 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9589 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9590 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_6 := arithEq_of_rows h (row := 149) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_6
  simp only [← c9585, ← c9586, ← c9587] at e_149_6
  have e_149_7 := arithEq_of_rows h (row := 149) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_7
  simp only [← c9588, ← c9589, ← c9590] at e_149_7
  have hr := e_149_7
  simp only [e_149_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 39) = bselect (a (.wire 140 43)) (a (.virt 13806)) (a (.wire 148 55)) := by
  have c9591 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9592 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9593 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9594 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9595 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9596 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_8 := arithEq_of_rows h (row := 149) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_8
  simp only [← c9591, ← c9592, ← c9593] at e_149_8
  have e_149_9 := arithEq_of_rows h (row := 149) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_9
  simp only [← c9594, ← c9595, ← c9596] at e_149_9
  have hr := e_149_9
  simp only [e_149_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 47) = bselect (a (.wire 140 43)) (a (.virt 13807)) (a (.wire 148 63)) := by
  have c9597 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9598 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9599 := (publicBatchWrapper86_copies299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9600 := (publicBatchWrapper86_copies300 a h).1
  have c9601 := (publicBatchWrapper86_copies300 a h).2.1
  have c9602 := (publicBatchWrapper86_copies300 a h).2.2.1
  have e_149_10 := arithEq_of_rows h (row := 149) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_10
  simp only [← c9597, ← c9598, ← c9599] at e_149_10
  have e_149_11 := arithEq_of_rows h (row := 149) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_11
  simp only [← c9600, ← c9601, ← c9602] at e_149_11
  have hr := e_149_11
  simp only [e_149_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 55) = bselect (a (.wire 140 43)) (a (.virt 13808)) (a (.wire 148 71)) := by
  have c9603 := (publicBatchWrapper86_copies300 a h).2.2.2.1
  have c9604 := (publicBatchWrapper86_copies300 a h).2.2.2.2.1
  have c9605 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.1
  have c9606 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.1
  have c9607 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.1
  have c9608 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.1
  have e_149_12 := arithEq_of_rows h (row := 149) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_12
  simp only [← c9603, ← c9604, ← c9605] at e_149_12
  have e_149_13 := arithEq_of_rows h (row := 149) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_13
  simp only [← c9606, ← c9607, ← c9608] at e_149_13
  have hr := e_149_13
  simp only [e_149_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 63) = bselect (a (.wire 140 43)) (a (.virt 13809)) (a (.wire 148 79)) := by
  have c9609 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.1
  have c9610 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9611 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9612 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9613 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9614 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_14 := arithEq_of_rows h (row := 149) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_14
  simp only [← c9609, ← c9610, ← c9611] at e_149_14
  have e_149_15 := arithEq_of_rows h (row := 149) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_15
  simp only [← c9612, ← c9613, ← c9614] at e_149_15
  have hr := e_149_15
  simp only [e_149_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 71) = bselect (a (.wire 140 43)) (a (.virt 13803)) (a (.wire 149 7)) := by
  have c9615 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9616 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9617 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9618 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9619 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9620 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_16 := arithEq_of_rows h (row := 149) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_16
  simp only [← c9615, ← c9616, ← c9617] at e_149_16
  have e_149_17 := arithEq_of_rows h (row := 149) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_17
  simp only [← c9618, ← c9619, ← c9620] at e_149_17
  have hr := e_149_17
  simp only [e_149_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 149 79) = bselect (a (.wire 140 43)) (a (.virt 13804)) (a (.wire 149 15)) := by
  have c9621 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9622 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9623 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9624 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9625 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9626 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_149_18 := arithEq_of_rows h (row := 149) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_18
  simp only [← c9621, ← c9622, ← c9623] at e_149_18
  have e_149_19 := arithEq_of_rows h (row := 149) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_149_19
  simp only [← c9624, ← c9625, ← c9626] at e_149_19
  have hr := e_149_19
  simp only [e_149_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 67) = bor (a (.wire 135 63)) (a (.wire 149 19)) := by
  have c9627 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9628 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9629 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9630 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9631 := (publicBatchWrapper86_copies300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9632 := (publicBatchWrapper86_copies301 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_16 := arithEq_of_rows h (row := 135) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_16
  simp only [← c9630, ← c9631, k3, ← c9632] at e_135_16
  have e_143_8 := arithEq_of_rows h (row := 143) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_8
  simp only [← c9627, ← c9628, ← c9629] at e_143_8
  have hr := e_135_16
  simp only [e_143_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 3) = bnot (a (.wire 68 67)) := by
  have c9633 := (publicBatchWrapper86_copies301 a h).2.1
  have c9634 := (publicBatchWrapper86_copies301 a h).2.2.1
  have c9635 := (publicBatchWrapper86_copies301 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_150_0 := arithEq_of_rows h (row := 150) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_0
  simp only [← c9633, k3, ← c9634, k3, ← c9635] at e_150_0
  have hr := e_150_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 7) = bnot (a (.wire 135 67)) := by
  have c9636 := (publicBatchWrapper86_copies301 a h).2.2.2.2.1
  have c9637 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.1
  have c9638 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_150_1 := arithEq_of_rows h (row := 150) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_1
  simp only [← c9636, k3, ← c9637, k3, ← c9638] at e_150_1
  have hr := e_150_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 47) = band (a (.wire 150 3)) (a (.wire 150 7)) := by
  have c9639 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.1
  have c9640 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.1
  have c9641 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.1
  have e_140_11 := arithEq_of_rows h (row := 140) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_11
  simp only [← c9639, ← c9640, ← c9641] at e_140_11
  have hr := e_140_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 15) = bselect (a (.wire 140 47)) (a (.virt 14005)) (a (.wire 149 31)) := by
  have c9642 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9643 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9644 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9645 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9646 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9647 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_2 := arithEq_of_rows h (row := 150) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_2
  simp only [← c9642, ← c9643, ← c9644] at e_150_2
  have e_150_3 := arithEq_of_rows h (row := 150) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_3
  simp only [← c9645, ← c9646, ← c9647] at e_150_3
  have hr := e_150_3
  simp only [e_150_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 23) = bselect (a (.wire 140 47)) (a (.virt 14006)) (a (.wire 149 39)) := by
  have c9648 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9649 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9650 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9651 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9652 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9653 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_4 := arithEq_of_rows h (row := 150) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_4
  simp only [← c9648, ← c9649, ← c9650] at e_150_4
  have e_150_5 := arithEq_of_rows h (row := 150) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_5
  simp only [← c9651, ← c9652, ← c9653] at e_150_5
  have hr := e_150_5
  simp only [e_150_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 31) = bselect (a (.wire 140 47)) (a (.virt 14007)) (a (.wire 149 47)) := by
  have c9654 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9655 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9656 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9657 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9658 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9659 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_6 := arithEq_of_rows h (row := 150) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_6
  simp only [← c9654, ← c9655, ← c9656] at e_150_6
  have e_150_7 := arithEq_of_rows h (row := 150) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_7
  simp only [← c9657, ← c9658, ← c9659] at e_150_7
  have hr := e_150_7
  simp only [e_150_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 39) = bselect (a (.wire 140 47)) (a (.virt 14008)) (a (.wire 149 55)) := by
  have c9660 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9661 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9662 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9663 := (publicBatchWrapper86_copies301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9664 := (publicBatchWrapper86_copies302 a h).1
  have c9665 := (publicBatchWrapper86_copies302 a h).2.1
  have e_150_8 := arithEq_of_rows h (row := 150) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_8
  simp only [← c9660, ← c9661, ← c9662] at e_150_8
  have e_150_9 := arithEq_of_rows h (row := 150) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_9
  simp only [← c9663, ← c9664, ← c9665] at e_150_9
  have hr := e_150_9
  simp only [e_150_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 47) = bselect (a (.wire 140 47)) (a (.virt 14009)) (a (.wire 149 63)) := by
  have c9666 := (publicBatchWrapper86_copies302 a h).2.2.1
  have c9667 := (publicBatchWrapper86_copies302 a h).2.2.2.1
  have c9668 := (publicBatchWrapper86_copies302 a h).2.2.2.2.1
  have c9669 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.1
  have c9670 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.1
  have c9671 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.1
  have e_150_10 := arithEq_of_rows h (row := 150) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_10
  simp only [← c9666, ← c9667, ← c9668] at e_150_10
  have e_150_11 := arithEq_of_rows h (row := 150) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_11
  simp only [← c9669, ← c9670, ← c9671] at e_150_11
  have hr := e_150_11
  simp only [e_150_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 55) = bselect (a (.wire 140 47)) (a (.virt 14003)) (a (.wire 149 71)) := by
  have c9672 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.1
  have c9673 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.1
  have c9674 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9675 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9676 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9677 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_12 := arithEq_of_rows h (row := 150) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_12
  simp only [← c9672, ← c9673, ← c9674] at e_150_12
  have e_150_13 := arithEq_of_rows h (row := 150) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_13
  simp only [← c9675, ← c9676, ← c9677] at e_150_13
  have hr := e_150_13
  simp only [e_150_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 63) = bselect (a (.wire 140 47)) (a (.virt 14004)) (a (.wire 149 79)) := by
  have c9678 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9679 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9680 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9681 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9682 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9683 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_150_14 := arithEq_of_rows h (row := 150) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_14
  simp only [← c9678, ← c9679, ← c9680] at e_150_14
  have e_150_15 := arithEq_of_rows h (row := 150) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_15
  simp only [← c9681, ← c9682, ← c9683] at e_150_15
  have hr := e_150_15
  simp only [e_150_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 71) = bor (a (.wire 135 67)) (a (.wire 150 3)) := by
  have c9684 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9685 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9686 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9687 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9688 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9689 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_17 := arithEq_of_rows h (row := 135) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_17
  simp only [← c9687, ← c9688, k3, ← c9689] at e_135_17
  have e_143_9 := arithEq_of_rows h (row := 143) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_9
  simp only [← c9684, ← c9685, ← c9686] at e_143_9
  have hr := e_135_17
  simp only [e_143_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 67) = bnot (a (.wire 70 31)) := by
  have c9690 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9691 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9692 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_150_16 := arithEq_of_rows h (row := 150) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_16
  simp only [← c9690, k3, ← c9691, k3, ← c9692] at e_150_16
  have hr := e_150_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 71) = bnot (a (.wire 135 71)) := by
  have c9693 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9694 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9695 := (publicBatchWrapper86_copies302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_150_17 := arithEq_of_rows h (row := 150) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_17
  simp only [← c9693, k3, ← c9694, k3, ← c9695] at e_150_17
  have hr := e_150_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 51) = band (a (.wire 150 67)) (a (.wire 150 71)) := by
  have c9696 := (publicBatchWrapper86_copies303 a h).1
  have c9697 := (publicBatchWrapper86_copies303 a h).2.1
  have c9698 := (publicBatchWrapper86_copies303 a h).2.2.1
  have e_140_12 := arithEq_of_rows h (row := 140) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_12
  simp only [← c9696, ← c9697, ← c9698] at e_140_12
  have hr := e_140_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 150 79) = bselect (a (.wire 140 51)) (a (.virt 14205)) (a (.wire 150 15)) := by
  have c9699 := (publicBatchWrapper86_copies303 a h).2.2.2.1
  have c9700 := (publicBatchWrapper86_copies303 a h).2.2.2.2.1
  have c9701 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.1
  have c9702 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.1
  have c9703 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.1
  have c9704 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.1
  have e_150_18 := arithEq_of_rows h (row := 150) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_18
  simp only [← c9699, ← c9700, ← c9701] at e_150_18
  have e_150_19 := arithEq_of_rows h (row := 150) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_150_19
  simp only [← c9702, ← c9703, ← c9704] at e_150_19
  have hr := e_150_19
  simp only [e_150_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 7) = bselect (a (.wire 140 51)) (a (.virt 14206)) (a (.wire 150 23)) := by
  have c9705 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.1
  have c9706 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9707 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9708 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9709 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9710 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_0 := arithEq_of_rows h (row := 151) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_0
  simp only [← c9705, ← c9706, ← c9707] at e_151_0
  have e_151_1 := arithEq_of_rows h (row := 151) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_1
  simp only [← c9708, ← c9709, ← c9710] at e_151_1
  have hr := e_151_1
  simp only [e_151_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 15) = bselect (a (.wire 140 51)) (a (.virt 14207)) (a (.wire 150 31)) := by
  have c9711 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9712 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9713 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9714 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9715 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9716 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_2 := arithEq_of_rows h (row := 151) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_2
  simp only [← c9711, ← c9712, ← c9713] at e_151_2
  have e_151_3 := arithEq_of_rows h (row := 151) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_3
  simp only [← c9714, ← c9715, ← c9716] at e_151_3
  have hr := e_151_3
  simp only [e_151_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 23) = bselect (a (.wire 140 51)) (a (.virt 14208)) (a (.wire 150 39)) := by
  have c9717 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9718 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9719 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9720 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9721 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9722 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_4 := arithEq_of_rows h (row := 151) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_4
  simp only [← c9717, ← c9718, ← c9719] at e_151_4
  have e_151_5 := arithEq_of_rows h (row := 151) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_5
  simp only [← c9720, ← c9721, ← c9722] at e_151_5
  have hr := e_151_5
  simp only [e_151_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 31) = bselect (a (.wire 140 51)) (a (.virt 14209)) (a (.wire 150 47)) := by
  have c9723 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9724 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9725 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9726 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9727 := (publicBatchWrapper86_copies303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9728 := (publicBatchWrapper86_copies304 a h).1
  have e_151_6 := arithEq_of_rows h (row := 151) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_6
  simp only [← c9723, ← c9724, ← c9725] at e_151_6
  have e_151_7 := arithEq_of_rows h (row := 151) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_7
  simp only [← c9726, ← c9727, ← c9728] at e_151_7
  have hr := e_151_7
  simp only [e_151_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 39) = bselect (a (.wire 140 51)) (a (.virt 14203)) (a (.wire 150 55)) := by
  have c9729 := (publicBatchWrapper86_copies304 a h).2.1
  have c9730 := (publicBatchWrapper86_copies304 a h).2.2.1
  have c9731 := (publicBatchWrapper86_copies304 a h).2.2.2.1
  have c9732 := (publicBatchWrapper86_copies304 a h).2.2.2.2.1
  have c9733 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.1
  have c9734 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.1
  have e_151_8 := arithEq_of_rows h (row := 151) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_8
  simp only [← c9729, ← c9730, ← c9731] at e_151_8
  have e_151_9 := arithEq_of_rows h (row := 151) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_9
  simp only [← c9732, ← c9733, ← c9734] at e_151_9
  have hr := e_151_9
  simp only [e_151_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 47) = bselect (a (.wire 140 51)) (a (.virt 14204)) (a (.wire 150 63)) := by
  have c9735 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.1
  have c9736 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.1
  have c9737 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.1
  have c9738 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9739 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9740 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_10 := arithEq_of_rows h (row := 151) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_10
  simp only [← c9735, ← c9736, ← c9737] at e_151_10
  have e_151_11 := arithEq_of_rows h (row := 151) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_11
  simp only [← c9738, ← c9739, ← c9740] at e_151_11
  have hr := e_151_11
  simp only [e_151_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 75) = bor (a (.wire 135 71)) (a (.wire 150 67)) := by
  have c9741 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9742 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9743 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9744 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9745 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9746 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_18 := arithEq_of_rows h (row := 135) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_18
  simp only [← c9744, ← c9745, k3, ← c9746] at e_135_18
  have e_143_10 := arithEq_of_rows h (row := 143) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_10
  simp only [← c9741, ← c9742, ← c9743] at e_143_10
  have hr := e_135_18
  simp only [e_143_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 51) = bnot (a (.wire 70 75)) := by
  have c9747 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9748 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9749 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_151_12 := arithEq_of_rows h (row := 151) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_12
  simp only [← c9747, k3, ← c9748, k3, ← c9749] at e_151_12
  have hr := e_151_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 55) = bnot (a (.wire 135 75)) := by
  have c9750 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9751 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9752 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_151_13 := arithEq_of_rows h (row := 151) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_13
  simp only [← c9750, k3, ← c9751, k3, ← c9752] at e_151_13
  have hr := e_151_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 55) = band (a (.wire 151 51)) (a (.wire 151 55)) := by
  have c9753 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9754 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9755 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_13 := arithEq_of_rows h (row := 140) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_13
  simp only [← c9753, ← c9754, ← c9755] at e_140_13
  have hr := e_140_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 63) = bselect (a (.wire 140 55)) (a (.virt 14405)) (a (.wire 150 79)) := by
  have c9756 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9757 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9758 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9759 := (publicBatchWrapper86_copies304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9760 := (publicBatchWrapper86_copies305 a h).1
  have c9761 := (publicBatchWrapper86_copies305 a h).2.1
  have e_151_14 := arithEq_of_rows h (row := 151) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_14
  simp only [← c9756, ← c9757, ← c9758] at e_151_14
  have e_151_15 := arithEq_of_rows h (row := 151) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_15
  simp only [← c9759, ← c9760, ← c9761] at e_151_15
  have hr := e_151_15
  simp only [e_151_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 71) = bselect (a (.wire 140 55)) (a (.virt 14406)) (a (.wire 151 7)) := by
  have c9762 := (publicBatchWrapper86_copies305 a h).2.2.1
  have c9763 := (publicBatchWrapper86_copies305 a h).2.2.2.1
  have c9764 := (publicBatchWrapper86_copies305 a h).2.2.2.2.1
  have c9765 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.1
  have c9766 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.1
  have c9767 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.1
  have e_151_16 := arithEq_of_rows h (row := 151) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_16
  simp only [← c9762, ← c9763, ← c9764] at e_151_16
  have e_151_17 := arithEq_of_rows h (row := 151) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_17
  simp only [← c9765, ← c9766, ← c9767] at e_151_17
  have hr := e_151_17
  simp only [e_151_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 151 79) = bselect (a (.wire 140 55)) (a (.virt 14407)) (a (.wire 151 15)) := by
  have c9768 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.1
  have c9769 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.1
  have c9770 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9771 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9772 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9773 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_151_18 := arithEq_of_rows h (row := 151) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_18
  simp only [← c9768, ← c9769, ← c9770] at e_151_18
  have e_151_19 := arithEq_of_rows h (row := 151) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_151_19
  simp only [← c9771, ← c9772, ← c9773] at e_151_19
  have hr := e_151_19
  simp only [e_151_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 7) = bselect (a (.wire 140 55)) (a (.virt 14408)) (a (.wire 151 23)) := by
  have c9774 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9775 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9776 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9777 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9778 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9779 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_0 := arithEq_of_rows h (row := 152) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_0
  simp only [← c9774, ← c9775, ← c9776] at e_152_0
  have e_152_1 := arithEq_of_rows h (row := 152) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_1
  simp only [← c9777, ← c9778, ← c9779] at e_152_1
  have hr := e_152_1
  simp only [e_152_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 15) = bselect (a (.wire 140 55)) (a (.virt 14409)) (a (.wire 151 31)) := by
  have c9780 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9781 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9782 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9783 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9784 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9785 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_2 := arithEq_of_rows h (row := 152) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_2
  simp only [← c9780, ← c9781, ← c9782] at e_152_2
  have e_152_3 := arithEq_of_rows h (row := 152) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_3
  simp only [← c9783, ← c9784, ← c9785] at e_152_3
  have hr := e_152_3
  simp only [e_152_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 23) = bselect (a (.wire 140 55)) (a (.virt 14403)) (a (.wire 151 39)) := by
  have c9786 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9787 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9788 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9789 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9790 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9791 := (publicBatchWrapper86_copies305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_152_4 := arithEq_of_rows h (row := 152) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_4
  simp only [← c9786, ← c9787, ← c9788] at e_152_4
  have e_152_5 := arithEq_of_rows h (row := 152) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_5
  simp only [← c9789, ← c9790, ← c9791] at e_152_5
  have hr := e_152_5
  simp only [e_152_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 31) = bselect (a (.wire 140 55)) (a (.virt 14404)) (a (.wire 151 47)) := by
  have c9792 := (publicBatchWrapper86_copies306 a h).1
  have c9793 := (publicBatchWrapper86_copies306 a h).2.1
  have c9794 := (publicBatchWrapper86_copies306 a h).2.2.1
  have c9795 := (publicBatchWrapper86_copies306 a h).2.2.2.1
  have c9796 := (publicBatchWrapper86_copies306 a h).2.2.2.2.1
  have c9797 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.1
  have e_152_6 := arithEq_of_rows h (row := 152) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_6
  simp only [← c9792, ← c9793, ← c9794] at e_152_6
  have e_152_7 := arithEq_of_rows h (row := 152) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_7
  simp only [← c9795, ← c9796, ← c9797] at e_152_7
  have hr := e_152_7
  simp only [e_152_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 135 79) = bor (a (.wire 135 75)) (a (.wire 151 51)) := by
  have c9798 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.1
  have c9799 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.1
  have c9800 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.1
  have c9801 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.1
  have c9802 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9803 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_135_19 := arithEq_of_rows h (row := 135) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_135_19
  simp only [← c9801, ← c9802, k3, ← c9803] at e_135_19
  have e_143_11 := arithEq_of_rows h (row := 143) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_11
  simp only [← c9798, ← c9799, ← c9800] at e_143_11
  have hr := e_135_19
  simp only [e_143_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 35) = bnot (a (.wire 72 39)) := by
  have c9804 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9805 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9806 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_152_8 := arithEq_of_rows h (row := 152) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_8
  simp only [← c9804, k3, ← c9805, k3, ← c9806] at e_152_8
  have hr := e_152_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 39) = bnot (a (.wire 135 79)) := by
  have c9807 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9808 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9809 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_152_9 := arithEq_of_rows h (row := 152) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_9
  simp only [← c9807, k3, ← c9808, k3, ← c9809] at e_152_9
  have hr := e_152_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 59) = band (a (.wire 152 35)) (a (.wire 152 39)) := by
  have c9810 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9811 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9812 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_14 := arithEq_of_rows h (row := 140) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_14
  simp only [← c9810, ← c9811, ← c9812] at e_140_14
  have hr := e_140_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 47) = bselect (a (.wire 140 59)) (a (.virt 14605)) (a (.wire 151 63)) := by
  have c9813 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9814 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9815 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9816 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9817 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9818 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_10 := arithEq_of_rows h (row := 152) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_10
  simp only [← c9813, ← c9814, ← c9815] at e_152_10
  have e_152_11 := arithEq_of_rows h (row := 152) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_11
  simp only [← c9816, ← c9817, ← c9818] at e_152_11
  have hr := e_152_11
  simp only [e_152_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 55) = bselect (a (.wire 140 59)) (a (.virt 14606)) (a (.wire 151 71)) := by
  have c9819 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9820 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9821 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9822 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9823 := (publicBatchWrapper86_copies306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9824 := (publicBatchWrapper86_copies307 a h).1
  have e_152_12 := arithEq_of_rows h (row := 152) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_12
  simp only [← c9819, ← c9820, ← c9821] at e_152_12
  have e_152_13 := arithEq_of_rows h (row := 152) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_13
  simp only [← c9822, ← c9823, ← c9824] at e_152_13
  have hr := e_152_13
  simp only [e_152_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 63) = bselect (a (.wire 140 59)) (a (.virt 14607)) (a (.wire 151 79)) := by
  have c9825 := (publicBatchWrapper86_copies307 a h).2.1
  have c9826 := (publicBatchWrapper86_copies307 a h).2.2.1
  have c9827 := (publicBatchWrapper86_copies307 a h).2.2.2.1
  have c9828 := (publicBatchWrapper86_copies307 a h).2.2.2.2.1
  have c9829 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.1
  have c9830 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.1
  have e_152_14 := arithEq_of_rows h (row := 152) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_14
  simp only [← c9825, ← c9826, ← c9827] at e_152_14
  have e_152_15 := arithEq_of_rows h (row := 152) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_15
  simp only [← c9828, ← c9829, ← c9830] at e_152_15
  have hr := e_152_15
  simp only [e_152_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 71) = bselect (a (.wire 140 59)) (a (.virt 14608)) (a (.wire 152 7)) := by
  have c9831 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.1
  have c9832 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.1
  have c9833 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.1
  have c9834 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9835 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9836 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_16 := arithEq_of_rows h (row := 152) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_16
  simp only [← c9831, ← c9832, ← c9833] at e_152_16
  have e_152_17 := arithEq_of_rows h (row := 152) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_17
  simp only [← c9834, ← c9835, ← c9836] at e_152_17
  have hr := e_152_17
  simp only [e_152_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 152 79) = bselect (a (.wire 140 59)) (a (.virt 14609)) (a (.wire 152 15)) := by
  have c9837 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9838 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9839 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9840 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9841 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9842 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_152_18 := arithEq_of_rows h (row := 152) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_18
  simp only [← c9837, ← c9838, ← c9839] at e_152_18
  have e_152_19 := arithEq_of_rows h (row := 152) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_152_19
  simp only [← c9840, ← c9841, ← c9842] at e_152_19
  have hr := e_152_19
  simp only [e_152_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 7) = bselect (a (.wire 140 59)) (a (.virt 14603)) (a (.wire 152 23)) := by
  have c9843 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9844 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9845 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9846 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9847 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9848 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_153_0 := arithEq_of_rows h (row := 153) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_0
  simp only [← c9843, ← c9844, ← c9845] at e_153_0
  have e_153_1 := arithEq_of_rows h (row := 153) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_1
  simp only [← c9846, ← c9847, ← c9848] at e_153_1
  have hr := e_153_1
  simp only [e_153_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 15) = bselect (a (.wire 140 59)) (a (.virt 14604)) (a (.wire 152 31)) := by
  have c9849 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9850 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9851 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9852 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9853 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9854 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_153_2 := arithEq_of_rows h (row := 153) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_2
  simp only [← c9849, ← c9850, ← c9851] at e_153_2
  have e_153_3 := arithEq_of_rows h (row := 153) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_3
  simp only [← c9852, ← c9853, ← c9854] at e_153_3
  have hr := e_153_3
  simp only [e_153_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 3) = bor (a (.wire 135 79)) (a (.wire 152 35)) := by
  have c9855 := (publicBatchWrapper86_copies307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9856 := (publicBatchWrapper86_copies308 a h).1
  have c9857 := (publicBatchWrapper86_copies308 a h).2.1
  have c9858 := (publicBatchWrapper86_copies308 a h).2.2.1
  have c9859 := (publicBatchWrapper86_copies308 a h).2.2.2.1
  have c9860 := (publicBatchWrapper86_copies308 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_143_12 := arithEq_of_rows h (row := 143) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_12
  simp only [← c9855, ← c9856, ← c9857] at e_143_12
  have e_154_0 := arithEq_of_rows h (row := 154) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_0
  simp only [← c9858, ← c9859, k3, ← c9860] at e_154_0
  have hr := e_154_0
  simp only [e_143_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 19) = bnot (a (.wire 73 3)) := by
  have c9861 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.1
  have c9862 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.1
  have c9863 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_153_4 := arithEq_of_rows h (row := 153) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_4
  simp only [← c9861, k3, ← c9862, k3, ← c9863] at e_153_4
  have hr := e_153_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 23) = bnot (a (.wire 154 3)) := by
  have c9864 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.1
  have c9865 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.1
  have c9866 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_153_5 := arithEq_of_rows h (row := 153) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_5
  simp only [← c9864, k3, ← c9865, k3, ← c9866] at e_153_5
  have hr := e_153_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 63) = band (a (.wire 153 19)) (a (.wire 153 23)) := by
  have c9867 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9868 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9869 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_15 := arithEq_of_rows h (row := 140) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_15
  simp only [← c9867, ← c9868, ← c9869] at e_140_15
  have hr := e_140_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 31) = bselect (a (.wire 140 63)) (a (.virt 14805)) (a (.wire 152 47)) := by
  have c9870 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9871 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9872 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9873 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9874 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9875 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_153_6 := arithEq_of_rows h (row := 153) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_6
  simp only [← c9870, ← c9871, ← c9872] at e_153_6
  have e_153_7 := arithEq_of_rows h (row := 153) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_7
  simp only [← c9873, ← c9874, ← c9875] at e_153_7
  have hr := e_153_7
  simp only [e_153_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 39) = bselect (a (.wire 140 63)) (a (.virt 14806)) (a (.wire 152 55)) := by
  have c9876 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9877 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9878 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9879 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9880 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9881 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_153_8 := arithEq_of_rows h (row := 153) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_8
  simp only [← c9876, ← c9877, ← c9878] at e_153_8
  have e_153_9 := arithEq_of_rows h (row := 153) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_9
  simp only [← c9879, ← c9880, ← c9881] at e_153_9
  have hr := e_153_9
  simp only [e_153_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 47) = bselect (a (.wire 140 63)) (a (.virt 14807)) (a (.wire 152 63)) := by
  have c9882 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9883 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9884 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9885 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9886 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9887 := (publicBatchWrapper86_copies308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_153_10 := arithEq_of_rows h (row := 153) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_10
  simp only [← c9882, ← c9883, ← c9884] at e_153_10
  have e_153_11 := arithEq_of_rows h (row := 153) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_11
  simp only [← c9885, ← c9886, ← c9887] at e_153_11
  have hr := e_153_11
  simp only [e_153_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 55) = bselect (a (.wire 140 63)) (a (.virt 14808)) (a (.wire 152 71)) := by
  have c9888 := (publicBatchWrapper86_copies309 a h).1
  have c9889 := (publicBatchWrapper86_copies309 a h).2.1
  have c9890 := (publicBatchWrapper86_copies309 a h).2.2.1
  have c9891 := (publicBatchWrapper86_copies309 a h).2.2.2.1
  have c9892 := (publicBatchWrapper86_copies309 a h).2.2.2.2.1
  have c9893 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.1
  have e_153_12 := arithEq_of_rows h (row := 153) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_12
  simp only [← c9888, ← c9889, ← c9890] at e_153_12
  have e_153_13 := arithEq_of_rows h (row := 153) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_13
  simp only [← c9891, ← c9892, ← c9893] at e_153_13
  have hr := e_153_13
  simp only [e_153_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 63) = bselect (a (.wire 140 63)) (a (.virt 14809)) (a (.wire 152 79)) := by
  have c9894 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.1
  have c9895 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.1
  have c9896 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.1
  have c9897 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.1
  have c9898 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9899 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_153_14 := arithEq_of_rows h (row := 153) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_14
  simp only [← c9894, ← c9895, ← c9896] at e_153_14
  have e_153_15 := arithEq_of_rows h (row := 153) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_15
  simp only [← c9897, ← c9898, ← c9899] at e_153_15
  have hr := e_153_15
  simp only [e_153_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 71) = bselect (a (.wire 140 63)) (a (.virt 14803)) (a (.wire 153 7)) := by
  have c9900 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9901 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9902 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9903 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9904 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9905 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_153_16 := arithEq_of_rows h (row := 153) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_16
  simp only [← c9900, ← c9901, ← c9902] at e_153_16
  have e_153_17 := arithEq_of_rows h (row := 153) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_17
  simp only [← c9903, ← c9904, ← c9905] at e_153_17
  have hr := e_153_17
  simp only [e_153_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 153 79) = bselect (a (.wire 140 63)) (a (.virt 14804)) (a (.wire 153 15)) := by
  have c9906 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9907 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9908 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9909 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9910 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9911 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_153_18 := arithEq_of_rows h (row := 153) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_18
  simp only [← c9906, ← c9907, ← c9908] at e_153_18
  have e_153_19 := arithEq_of_rows h (row := 153) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_153_19
  simp only [← c9909, ← c9910, ← c9911] at e_153_19
  have hr := e_153_19
  simp only [e_153_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 7) = bor (a (.wire 154 3)) (a (.wire 153 19)) := by
  have c9912 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9913 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9914 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9915 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9916 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9917 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_143_13 := arithEq_of_rows h (row := 143) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_13
  simp only [← c9912, ← c9913, ← c9914] at e_143_13
  have e_154_1 := arithEq_of_rows h (row := 154) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_1
  simp only [← c9915, ← c9916, k3, ← c9917] at e_154_1
  have hr := e_154_1
  simp only [e_143_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 3) = bnot (a (.wire 73 47)) := by
  have c9918 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9919 := (publicBatchWrapper86_copies309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9920 := (publicBatchWrapper86_copies310 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_155_0 := arithEq_of_rows h (row := 155) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_0
  simp only [← c9918, k3, ← c9919, k3, ← c9920] at e_155_0
  have hr := e_155_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 7) = bnot (a (.wire 154 7)) := by
  have c9921 := (publicBatchWrapper86_copies310 a h).2.1
  have c9922 := (publicBatchWrapper86_copies310 a h).2.2.1
  have c9923 := (publicBatchWrapper86_copies310 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_155_1 := arithEq_of_rows h (row := 155) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_1
  simp only [← c9921, k3, ← c9922, k3, ← c9923] at e_155_1
  have hr := e_155_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 67) = band (a (.wire 155 3)) (a (.wire 155 7)) := by
  have c9924 := (publicBatchWrapper86_copies310 a h).2.2.2.2.1
  have c9925 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.1
  have c9926 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.1
  have e_140_16 := arithEq_of_rows h (row := 140) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_16
  simp only [← c9924, ← c9925, ← c9926] at e_140_16
  have hr := e_140_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 15) = bselect (a (.wire 140 67)) (a (.virt 15005)) (a (.wire 153 31)) := by
  have c9927 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.1
  have c9928 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.1
  have c9929 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.1
  have c9930 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9931 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9932 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_2 := arithEq_of_rows h (row := 155) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_2
  simp only [← c9927, ← c9928, ← c9929] at e_155_2
  have e_155_3 := arithEq_of_rows h (row := 155) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_3
  simp only [← c9930, ← c9931, ← c9932] at e_155_3
  have hr := e_155_3
  simp only [e_155_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 23) = bselect (a (.wire 140 67)) (a (.virt 15006)) (a (.wire 153 39)) := by
  have c9933 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9934 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9935 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9936 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9937 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9938 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_4 := arithEq_of_rows h (row := 155) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_4
  simp only [← c9933, ← c9934, ← c9935] at e_155_4
  have e_155_5 := arithEq_of_rows h (row := 155) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_5
  simp only [← c9936, ← c9937, ← c9938] at e_155_5
  have hr := e_155_5
  simp only [e_155_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 31) = bselect (a (.wire 140 67)) (a (.virt 15007)) (a (.wire 153 47)) := by
  have c9939 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9940 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9941 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9942 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9943 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9944 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_6 := arithEq_of_rows h (row := 155) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_6
  simp only [← c9939, ← c9940, ← c9941] at e_155_6
  have e_155_7 := arithEq_of_rows h (row := 155) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_7
  simp only [← c9942, ← c9943, ← c9944] at e_155_7
  have hr := e_155_7
  simp only [e_155_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 39) = bselect (a (.wire 140 67)) (a (.virt 15008)) (a (.wire 153 55)) := by
  have c9945 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9946 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9947 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9948 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9949 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9950 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_8 := arithEq_of_rows h (row := 155) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_8
  simp only [← c9945, ← c9946, ← c9947] at e_155_8
  have e_155_9 := arithEq_of_rows h (row := 155) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_9
  simp only [← c9948, ← c9949, ← c9950] at e_155_9
  have hr := e_155_9
  simp only [e_155_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 47) = bselect (a (.wire 140 67)) (a (.virt 15009)) (a (.wire 153 63)) := by
  have c9951 := (publicBatchWrapper86_copies310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c9952 := (publicBatchWrapper86_copies311 a h).1
  have c9953 := (publicBatchWrapper86_copies311 a h).2.1
  have c9954 := (publicBatchWrapper86_copies311 a h).2.2.1
  have c9955 := (publicBatchWrapper86_copies311 a h).2.2.2.1
  have c9956 := (publicBatchWrapper86_copies311 a h).2.2.2.2.1
  have e_155_10 := arithEq_of_rows h (row := 155) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_10
  simp only [← c9951, ← c9952, ← c9953] at e_155_10
  have e_155_11 := arithEq_of_rows h (row := 155) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_11
  simp only [← c9954, ← c9955, ← c9956] at e_155_11
  have hr := e_155_11
  simp only [e_155_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 55) = bselect (a (.wire 140 67)) (a (.virt 15003)) (a (.wire 153 71)) := by
  have c9957 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.1
  have c9958 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.1
  have c9959 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.1
  have c9960 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.1
  have c9961 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.1
  have c9962 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_155_12 := arithEq_of_rows h (row := 155) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_12
  simp only [← c9957, ← c9958, ← c9959] at e_155_12
  have e_155_13 := arithEq_of_rows h (row := 155) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_13
  simp only [← c9960, ← c9961, ← c9962] at e_155_13
  have hr := e_155_13
  simp only [e_155_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 63) = bselect (a (.wire 140 67)) (a (.virt 15004)) (a (.wire 153 79)) := by
  have c9963 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c9964 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9965 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9966 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9967 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9968 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_155_14 := arithEq_of_rows h (row := 155) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_14
  simp only [← c9963, ← c9964, ← c9965] at e_155_14
  have e_155_15 := arithEq_of_rows h (row := 155) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_15
  simp only [← c9966, ← c9967, ← c9968] at e_155_15
  have hr := e_155_15
  simp only [e_155_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 11) = bor (a (.wire 154 7)) (a (.wire 155 3)) := by
  have c9969 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9970 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9971 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9972 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9973 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9974 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_143_14 := arithEq_of_rows h (row := 143) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_14
  simp only [← c9969, ← c9970, ← c9971] at e_143_14
  have e_154_2 := arithEq_of_rows h (row := 154) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_2
  simp only [← c9972, ← c9973, k3, ← c9974] at e_154_2
  have hr := e_154_2
  simp only [e_143_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 67) = bnot (a (.wire 75 11)) := by
  have c9975 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9976 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9977 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_155_16 := arithEq_of_rows h (row := 155) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_16
  simp only [← c9975, k3, ← c9976, k3, ← c9977] at e_155_16
  have hr := e_155_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 71) = bnot (a (.wire 154 11)) := by
  have c9978 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9979 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9980 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_155_17 := arithEq_of_rows h (row := 155) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_17
  simp only [← c9978, k3, ← c9979, k3, ← c9980] at e_155_17
  have hr := e_155_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 71) = band (a (.wire 155 67)) (a (.wire 155 71)) := by
  have c9981 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9982 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9983 := (publicBatchWrapper86_copies311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_140_17 := arithEq_of_rows h (row := 140) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_17
  simp only [← c9981, ← c9982, ← c9983] at e_140_17
  have hr := e_140_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 155 79) = bselect (a (.wire 140 71)) (a (.virt 15205)) (a (.wire 155 15)) := by
  have c9984 := (publicBatchWrapper86_copies312 a h).1
  have c9985 := (publicBatchWrapper86_copies312 a h).2.1
  have c9986 := (publicBatchWrapper86_copies312 a h).2.2.1
  have c9987 := (publicBatchWrapper86_copies312 a h).2.2.2.1
  have c9988 := (publicBatchWrapper86_copies312 a h).2.2.2.2.1
  have c9989 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.1
  have e_155_18 := arithEq_of_rows h (row := 155) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_18
  simp only [← c9984, ← c9985, ← c9986] at e_155_18
  have e_155_19 := arithEq_of_rows h (row := 155) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_155_19
  simp only [← c9987, ← c9988, ← c9989] at e_155_19
  have hr := e_155_19
  simp only [e_155_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 7) = bselect (a (.wire 140 71)) (a (.virt 15206)) (a (.wire 155 23)) := by
  have c9990 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.1
  have c9991 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.1
  have c9992 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.1
  have c9993 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.1
  have c9994 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.1
  have c9995 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_0 := arithEq_of_rows h (row := 156) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_0
  simp only [← c9990, ← c9991, ← c9992] at e_156_0
  have e_156_1 := arithEq_of_rows h (row := 156) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_1
  simp only [← c9993, ← c9994, ← c9995] at e_156_1
  have hr := e_156_1
  simp only [e_156_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 15) = bselect (a (.wire 140 71)) (a (.virt 15207)) (a (.wire 155 31)) := by
  have c9996 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9997 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9998 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c9999 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10000 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10001 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_2 := arithEq_of_rows h (row := 156) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_2
  simp only [← c9996, ← c9997, ← c9998] at e_156_2
  have e_156_3 := arithEq_of_rows h (row := 156) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_3
  simp only [← c9999, ← c10000, ← c10001] at e_156_3
  have hr := e_156_3
  simp only [e_156_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 23) = bselect (a (.wire 140 71)) (a (.virt 15208)) (a (.wire 155 39)) := by
  have c10002 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10003 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10004 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10005 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10006 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10007 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_4 := arithEq_of_rows h (row := 156) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_4
  simp only [← c10002, ← c10003, ← c10004] at e_156_4
  have e_156_5 := arithEq_of_rows h (row := 156) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_5
  simp only [← c10005, ← c10006, ← c10007] at e_156_5
  have hr := e_156_5
  simp only [e_156_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 31) = bselect (a (.wire 140 71)) (a (.virt 15209)) (a (.wire 155 47)) := by
  have c10008 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10009 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10010 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10011 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10012 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10013 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_6 := arithEq_of_rows h (row := 156) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_6
  simp only [← c10008, ← c10009, ← c10010] at e_156_6
  have e_156_7 := arithEq_of_rows h (row := 156) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_7
  simp only [← c10011, ← c10012, ← c10013] at e_156_7
  have hr := e_156_7
  simp only [e_156_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 39) = bselect (a (.wire 140 71)) (a (.virt 15203)) (a (.wire 155 55)) := by
  have c10014 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10015 := (publicBatchWrapper86_copies312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10016 := (publicBatchWrapper86_copies313 a h).1
  have c10017 := (publicBatchWrapper86_copies313 a h).2.1
  have c10018 := (publicBatchWrapper86_copies313 a h).2.2.1
  have c10019 := (publicBatchWrapper86_copies313 a h).2.2.2.1
  have e_156_8 := arithEq_of_rows h (row := 156) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_8
  simp only [← c10014, ← c10015, ← c10016] at e_156_8
  have e_156_9 := arithEq_of_rows h (row := 156) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_9
  simp only [← c10017, ← c10018, ← c10019] at e_156_9
  have hr := e_156_9
  simp only [e_156_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 47) = bselect (a (.wire 140 71)) (a (.virt 15204)) (a (.wire 155 63)) := by
  have c10020 := (publicBatchWrapper86_copies313 a h).2.2.2.2.1
  have c10021 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.1
  have c10022 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.1
  have c10023 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.1
  have c10024 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.1
  have c10025 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.1
  have e_156_10 := arithEq_of_rows h (row := 156) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_10
  simp only [← c10020, ← c10021, ← c10022] at e_156_10
  have e_156_11 := arithEq_of_rows h (row := 156) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_11
  simp only [← c10023, ← c10024, ← c10025] at e_156_11
  have hr := e_156_11
  simp only [e_156_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 15) = bor (a (.wire 154 11)) (a (.wire 155 67)) := by
  have c10026 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10027 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10028 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10029 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10030 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10031 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_143_15 := arithEq_of_rows h (row := 143) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_15
  simp only [← c10026, ← c10027, ← c10028] at e_143_15
  have e_154_3 := arithEq_of_rows h (row := 154) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_3
  simp only [← c10029, ← c10030, k3, ← c10031] at e_154_3
  have hr := e_154_3
  simp only [e_143_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 51) = bnot (a (.wire 75 55)) := by
  have c10032 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10033 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10034 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_156_12 := arithEq_of_rows h (row := 156) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_12
  simp only [← c10032, k3, ← c10033, k3, ← c10034] at e_156_12
  have hr := e_156_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 55) = bnot (a (.wire 154 15)) := by
  have c10035 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10036 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10037 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_156_13 := arithEq_of_rows h (row := 156) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_13
  simp only [← c10035, k3, ← c10036, k3, ← c10037] at e_156_13
  have hr := e_156_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 75) = band (a (.wire 156 51)) (a (.wire 156 55)) := by
  have c10038 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10039 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10040 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_18 := arithEq_of_rows h (row := 140) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_18
  simp only [← c10038, ← c10039, ← c10040] at e_140_18
  have hr := e_140_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 63) = bselect (a (.wire 140 75)) (a (.virt 15405)) (a (.wire 155 79)) := by
  have c10041 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10042 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10043 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10044 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10045 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10046 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_156_14 := arithEq_of_rows h (row := 156) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_14
  simp only [← c10041, ← c10042, ← c10043] at e_156_14
  have e_156_15 := arithEq_of_rows h (row := 156) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_15
  simp only [← c10044, ← c10045, ← c10046] at e_156_15
  have hr := e_156_15
  simp only [e_156_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 71) = bselect (a (.wire 140 75)) (a (.virt 15406)) (a (.wire 156 7)) := by
  have c10047 := (publicBatchWrapper86_copies313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10048 := (publicBatchWrapper86_copies314 a h).1
  have c10049 := (publicBatchWrapper86_copies314 a h).2.1
  have c10050 := (publicBatchWrapper86_copies314 a h).2.2.1
  have c10051 := (publicBatchWrapper86_copies314 a h).2.2.2.1
  have c10052 := (publicBatchWrapper86_copies314 a h).2.2.2.2.1
  have e_156_16 := arithEq_of_rows h (row := 156) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_16
  simp only [← c10047, ← c10048, ← c10049] at e_156_16
  have e_156_17 := arithEq_of_rows h (row := 156) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_17
  simp only [← c10050, ← c10051, ← c10052] at e_156_17
  have hr := e_156_17
  simp only [e_156_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 156 79) = bselect (a (.wire 140 75)) (a (.virt 15407)) (a (.wire 156 15)) := by
  have c10053 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.1
  have c10054 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.1
  have c10055 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.1
  have c10056 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.1
  have c10057 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.1
  have c10058 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_156_18 := arithEq_of_rows h (row := 156) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_18
  simp only [← c10053, ← c10054, ← c10055] at e_156_18
  have e_156_19 := arithEq_of_rows h (row := 156) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_156_19
  simp only [← c10056, ← c10057, ← c10058] at e_156_19
  have hr := e_156_19
  simp only [e_156_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 7) = bselect (a (.wire 140 75)) (a (.virt 15408)) (a (.wire 156 23)) := by
  have c10059 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10060 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10061 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10062 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10063 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10064 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_0 := arithEq_of_rows h (row := 157) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_0
  simp only [← c10059, ← c10060, ← c10061] at e_157_0
  have e_157_1 := arithEq_of_rows h (row := 157) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_1
  simp only [← c10062, ← c10063, ← c10064] at e_157_1
  have hr := e_157_1
  simp only [e_157_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 15) = bselect (a (.wire 140 75)) (a (.virt 15409)) (a (.wire 156 31)) := by
  have c10065 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10066 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10067 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10068 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10069 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10070 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_2 := arithEq_of_rows h (row := 157) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_2
  simp only [← c10065, ← c10066, ← c10067] at e_157_2
  have e_157_3 := arithEq_of_rows h (row := 157) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_3
  simp only [← c10068, ← c10069, ← c10070] at e_157_3
  have hr := e_157_3
  simp only [e_157_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 23) = bselect (a (.wire 140 75)) (a (.virt 15403)) (a (.wire 156 39)) := by
  have c10071 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10072 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10073 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10074 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10075 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10076 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_4 := arithEq_of_rows h (row := 157) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_4
  simp only [← c10071, ← c10072, ← c10073] at e_157_4
  have e_157_5 := arithEq_of_rows h (row := 157) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_5
  simp only [← c10074, ← c10075, ← c10076] at e_157_5
  have hr := e_157_5
  simp only [e_157_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 31) = bselect (a (.wire 140 75)) (a (.virt 15404)) (a (.wire 156 47)) := by
  have c10077 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10078 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10079 := (publicBatchWrapper86_copies314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10080 := (publicBatchWrapper86_copies315 a h).1
  have c10081 := (publicBatchWrapper86_copies315 a h).2.1
  have c10082 := (publicBatchWrapper86_copies315 a h).2.2.1
  have e_157_6 := arithEq_of_rows h (row := 157) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_6
  simp only [← c10077, ← c10078, ← c10079] at e_157_6
  have e_157_7 := arithEq_of_rows h (row := 157) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_7
  simp only [← c10080, ← c10081, ← c10082] at e_157_7
  have hr := e_157_7
  simp only [e_157_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 19) = bor (a (.wire 154 15)) (a (.wire 156 51)) := by
  have c10083 := (publicBatchWrapper86_copies315 a h).2.2.2.1
  have c10084 := (publicBatchWrapper86_copies315 a h).2.2.2.2.1
  have c10085 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.1
  have c10086 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.1
  have c10087 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.1
  have c10088 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_143_16 := arithEq_of_rows h (row := 143) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_16
  simp only [← c10083, ← c10084, ← c10085] at e_143_16
  have e_154_4 := arithEq_of_rows h (row := 154) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_4
  simp only [← c10086, ← c10087, k3, ← c10088] at e_154_4
  have hr := e_154_4
  simp only [e_143_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 35) = bnot (a (.wire 77 19)) := by
  have c10089 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.1
  have c10090 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10091 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_157_8 := arithEq_of_rows h (row := 157) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_8
  simp only [← c10089, k3, ← c10090, k3, ← c10091] at e_157_8
  have hr := e_157_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 39) = bnot (a (.wire 154 19)) := by
  have c10092 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10093 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10094 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_157_9 := arithEq_of_rows h (row := 157) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_9
  simp only [← c10092, k3, ← c10093, k3, ← c10094] at e_157_9
  have hr := e_157_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 140 79) = band (a (.wire 157 35)) (a (.wire 157 39)) := by
  have c10095 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10096 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10097 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_140_19 := arithEq_of_rows h (row := 140) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_140_19
  simp only [← c10095, ← c10096, ← c10097] at e_140_19
  have hr := e_140_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 47) = bselect (a (.wire 140 79)) (a (.virt 15605)) (a (.wire 156 63)) := by
  have c10098 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10099 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10100 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10101 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10102 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10103 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_10 := arithEq_of_rows h (row := 157) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_10
  simp only [← c10098, ← c10099, ← c10100] at e_157_10
  have e_157_11 := arithEq_of_rows h (row := 157) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_11
  simp only [← c10101, ← c10102, ← c10103] at e_157_11
  have hr := e_157_11
  simp only [e_157_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 55) = bselect (a (.wire 140 79)) (a (.virt 15606)) (a (.wire 156 71)) := by
  have c10104 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10105 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10106 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10107 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10108 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10109 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_12 := arithEq_of_rows h (row := 157) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_12
  simp only [← c10104, ← c10105, ← c10106] at e_157_12
  have e_157_13 := arithEq_of_rows h (row := 157) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_13
  simp only [← c10107, ← c10108, ← c10109] at e_157_13
  have hr := e_157_13
  simp only [e_157_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 63) = bselect (a (.wire 140 79)) (a (.virt 15607)) (a (.wire 156 79)) := by
  have c10110 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10111 := (publicBatchWrapper86_copies315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10112 := (publicBatchWrapper86_copies316 a h).1
  have c10113 := (publicBatchWrapper86_copies316 a h).2.1
  have c10114 := (publicBatchWrapper86_copies316 a h).2.2.1
  have c10115 := (publicBatchWrapper86_copies316 a h).2.2.2.1
  have e_157_14 := arithEq_of_rows h (row := 157) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_14
  simp only [← c10110, ← c10111, ← c10112] at e_157_14
  have e_157_15 := arithEq_of_rows h (row := 157) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_15
  simp only [← c10113, ← c10114, ← c10115] at e_157_15
  have hr := e_157_15
  simp only [e_157_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 71) = bselect (a (.wire 140 79)) (a (.virt 15608)) (a (.wire 157 7)) := by
  have c10116 := (publicBatchWrapper86_copies316 a h).2.2.2.2.1
  have c10117 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.1
  have c10118 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.1
  have c10119 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.1
  have c10120 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.1
  have c10121 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.1
  have e_157_16 := arithEq_of_rows h (row := 157) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_16
  simp only [← c10116, ← c10117, ← c10118] at e_157_16
  have e_157_17 := arithEq_of_rows h (row := 157) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_17
  simp only [← c10119, ← c10120, ← c10121] at e_157_17
  have hr := e_157_17
  simp only [e_157_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 157 79) = bselect (a (.wire 140 79)) (a (.virt 15609)) (a (.wire 157 15)) := by
  have c10122 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10123 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10124 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10125 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10126 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10127 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_157_18 := arithEq_of_rows h (row := 157) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_18
  simp only [← c10122, ← c10123, ← c10124] at e_157_18
  have e_157_19 := arithEq_of_rows h (row := 157) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_157_19
  simp only [← c10125, ← c10126, ← c10127] at e_157_19
  have hr := e_157_19
  simp only [e_157_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 7) = bselect (a (.wire 140 79)) (a (.virt 15603)) (a (.wire 157 23)) := by
  have c10128 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10129 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10130 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10131 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10132 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10133 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_0 := arithEq_of_rows h (row := 158) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_0
  simp only [← c10128, ← c10129, ← c10130] at e_158_0
  have e_158_1 := arithEq_of_rows h (row := 158) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_1
  simp only [← c10131, ← c10132, ← c10133] at e_158_1
  have hr := e_158_1
  simp only [e_158_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 15) = bselect (a (.wire 140 79)) (a (.virt 15604)) (a (.wire 157 31)) := by
  have c10134 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10135 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10136 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10137 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10138 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10139 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_2 := arithEq_of_rows h (row := 158) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_2
  simp only [← c10134, ← c10135, ← c10136] at e_158_2
  have e_158_3 := arithEq_of_rows h (row := 158) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_3
  simp only [← c10137, ← c10138, ← c10139] at e_158_3
  have hr := e_158_3
  simp only [e_158_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 23) = bor (a (.wire 154 19)) (a (.wire 157 35)) := by
  have c10140 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10141 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10142 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10143 := (publicBatchWrapper86_copies316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10144 := (publicBatchWrapper86_copies317 a h).1
  have c10145 := (publicBatchWrapper86_copies317 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_143_17 := arithEq_of_rows h (row := 143) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_17
  simp only [← c10140, ← c10141, ← c10142] at e_143_17
  have e_154_5 := arithEq_of_rows h (row := 154) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_5
  simp only [← c10143, ← c10144, k3, ← c10145] at e_154_5
  have hr := e_154_5
  simp only [e_143_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 19) = bnot (a (.wire 77 63)) := by
  have c10146 := (publicBatchWrapper86_copies317 a h).2.2.1
  have c10147 := (publicBatchWrapper86_copies317 a h).2.2.2.1
  have c10148 := (publicBatchWrapper86_copies317 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_158_4 := arithEq_of_rows h (row := 158) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_4
  simp only [← c10146, k3, ← c10147, k3, ← c10148] at e_158_4
  have hr := e_158_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 23) = bnot (a (.wire 154 23)) := by
  have c10149 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.1
  have c10150 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.1
  have c10151 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_158_5 := arithEq_of_rows h (row := 158) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_5
  simp only [← c10149, k3, ← c10150, k3, ← c10151] at e_158_5
  have hr := e_158_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 159 3) = band (a (.wire 158 19)) (a (.wire 158 23)) := by
  have c10152 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.1
  have c10153 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.1
  have c10154 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_159_0 := arithEq_of_rows h (row := 159) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_0
  simp only [← c10152, ← c10153, ← c10154] at e_159_0
  have hr := e_159_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 31) = bselect (a (.wire 159 3)) (a (.virt 15805)) (a (.wire 157 47)) := by
  have c10155 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10156 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10157 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10158 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10159 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10160 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_6 := arithEq_of_rows h (row := 158) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_6
  simp only [← c10155, ← c10156, ← c10157] at e_158_6
  have e_158_7 := arithEq_of_rows h (row := 158) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_7
  simp only [← c10158, ← c10159, ← c10160] at e_158_7
  have hr := e_158_7
  simp only [e_158_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 39) = bselect (a (.wire 159 3)) (a (.virt 15806)) (a (.wire 157 55)) := by
  have c10161 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10162 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10163 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10164 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10165 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10166 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_8 := arithEq_of_rows h (row := 158) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_8
  simp only [← c10161, ← c10162, ← c10163] at e_158_8
  have e_158_9 := arithEq_of_rows h (row := 158) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_9
  simp only [← c10164, ← c10165, ← c10166] at e_158_9
  have hr := e_158_9
  simp only [e_158_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 47) = bselect (a (.wire 159 3)) (a (.virt 15807)) (a (.wire 157 63)) := by
  have c10167 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10168 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10169 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10170 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10171 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10172 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_10 := arithEq_of_rows h (row := 158) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_10
  simp only [← c10167, ← c10168, ← c10169] at e_158_10
  have e_158_11 := arithEq_of_rows h (row := 158) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_11
  simp only [← c10170, ← c10171, ← c10172] at e_158_11
  have hr := e_158_11
  simp only [e_158_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 55) = bselect (a (.wire 159 3)) (a (.virt 15808)) (a (.wire 157 71)) := by
  have c10173 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10174 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10175 := (publicBatchWrapper86_copies317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10176 := (publicBatchWrapper86_copies318 a h).1
  have c10177 := (publicBatchWrapper86_copies318 a h).2.1
  have c10178 := (publicBatchWrapper86_copies318 a h).2.2.1
  have e_158_12 := arithEq_of_rows h (row := 158) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_12
  simp only [← c10173, ← c10174, ← c10175] at e_158_12
  have e_158_13 := arithEq_of_rows h (row := 158) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_13
  simp only [← c10176, ← c10177, ← c10178] at e_158_13
  have hr := e_158_13
  simp only [e_158_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 63) = bselect (a (.wire 159 3)) (a (.virt 15809)) (a (.wire 157 79)) := by
  have c10179 := (publicBatchWrapper86_copies318 a h).2.2.2.1
  have c10180 := (publicBatchWrapper86_copies318 a h).2.2.2.2.1
  have c10181 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.1
  have c10182 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.1
  have c10183 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.1
  have c10184 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.1
  have e_158_14 := arithEq_of_rows h (row := 158) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_14
  simp only [← c10179, ← c10180, ← c10181] at e_158_14
  have e_158_15 := arithEq_of_rows h (row := 158) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_15
  simp only [← c10182, ← c10183, ← c10184] at e_158_15
  have hr := e_158_15
  simp only [e_158_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 71) = bselect (a (.wire 159 3)) (a (.virt 15803)) (a (.wire 158 7)) := by
  have c10185 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.1
  have c10186 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10187 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10188 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10189 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10190 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_16 := arithEq_of_rows h (row := 158) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_16
  simp only [← c10185, ← c10186, ← c10187] at e_158_16
  have e_158_17 := arithEq_of_rows h (row := 158) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_17
  simp only [← c10188, ← c10189, ← c10190] at e_158_17
  have hr := e_158_17
  simp only [e_158_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 158 79) = bselect (a (.wire 159 3)) (a (.virt 15804)) (a (.wire 158 15)) := by
  have c10191 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10192 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10193 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10194 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10195 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10196 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_158_18 := arithEq_of_rows h (row := 158) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_18
  simp only [← c10191, ← c10192, ← c10193] at e_158_18
  have e_158_19 := arithEq_of_rows h (row := 158) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_158_19
  simp only [← c10194, ← c10195, ← c10196] at e_158_19
  have hr := e_158_19
  simp only [e_158_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 27) = bor (a (.wire 154 23)) (a (.wire 158 19)) := by
  have c10197 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10198 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10199 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10200 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10201 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10202 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_143_18 := arithEq_of_rows h (row := 143) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_18
  simp only [← c10197, ← c10198, ← c10199] at e_143_18
  have e_154_6 := arithEq_of_rows h (row := 154) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_6
  simp only [← c10200, ← c10201, k3, ← c10202] at e_154_6
  have hr := e_154_6
  simp only [e_143_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 3) = bnot (a (.wire 79 27)) := by
  have c10203 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10204 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10205 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_160_0 := arithEq_of_rows h (row := 160) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_0
  simp only [← c10203, k3, ← c10204, k3, ← c10205] at e_160_0
  have hr := e_160_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 7) = bnot (a (.wire 154 27)) := by
  have c10206 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10207 := (publicBatchWrapper86_copies318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10208 := (publicBatchWrapper86_copies319 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_160_1 := arithEq_of_rows h (row := 160) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_1
  simp only [← c10206, k3, ← c10207, k3, ← c10208] at e_160_1
  have hr := e_160_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 159 7) = band (a (.wire 160 3)) (a (.wire 160 7)) := by
  have c10209 := (publicBatchWrapper86_copies319 a h).2.1
  have c10210 := (publicBatchWrapper86_copies319 a h).2.2.1
  have c10211 := (publicBatchWrapper86_copies319 a h).2.2.2.1
  have e_159_1 := arithEq_of_rows h (row := 159) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_1
  simp only [← c10209, ← c10210, ← c10211] at e_159_1
  have hr := e_159_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 15) = bselect (a (.wire 159 7)) (a (.virt 16005)) (a (.wire 158 31)) := by
  have c10212 := (publicBatchWrapper86_copies319 a h).2.2.2.2.1
  have c10213 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.1
  have c10214 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.1
  have c10215 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.1
  have c10216 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.1
  have c10217 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.1
  have e_160_2 := arithEq_of_rows h (row := 160) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_2
  simp only [← c10212, ← c10213, ← c10214] at e_160_2
  have e_160_3 := arithEq_of_rows h (row := 160) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_3
  simp only [← c10215, ← c10216, ← c10217] at e_160_3
  have hr := e_160_3
  simp only [e_160_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 23) = bselect (a (.wire 159 7)) (a (.virt 16006)) (a (.wire 158 39)) := by
  have c10218 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10219 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10220 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10221 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10222 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10223 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_4 := arithEq_of_rows h (row := 160) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_4
  simp only [← c10218, ← c10219, ← c10220] at e_160_4
  have e_160_5 := arithEq_of_rows h (row := 160) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_5
  simp only [← c10221, ← c10222, ← c10223] at e_160_5
  have hr := e_160_5
  simp only [e_160_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 31) = bselect (a (.wire 159 7)) (a (.virt 16007)) (a (.wire 158 47)) := by
  have c10224 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10225 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10226 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10227 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10228 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10229 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_6 := arithEq_of_rows h (row := 160) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_6
  simp only [← c10224, ← c10225, ← c10226] at e_160_6
  have e_160_7 := arithEq_of_rows h (row := 160) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_7
  simp only [← c10227, ← c10228, ← c10229] at e_160_7
  have hr := e_160_7
  simp only [e_160_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 39) = bselect (a (.wire 159 7)) (a (.virt 16008)) (a (.wire 158 55)) := by
  have c10230 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10231 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10232 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10233 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10234 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10235 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_8 := arithEq_of_rows h (row := 160) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_8
  simp only [← c10230, ← c10231, ← c10232] at e_160_8
  have e_160_9 := arithEq_of_rows h (row := 160) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_9
  simp only [← c10233, ← c10234, ← c10235] at e_160_9
  have hr := e_160_9
  simp only [e_160_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 47) = bselect (a (.wire 159 7)) (a (.virt 16009)) (a (.wire 158 63)) := by
  have c10236 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10237 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10238 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10239 := (publicBatchWrapper86_copies319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10240 := (publicBatchWrapper86_copies320 a h).1
  have c10241 := (publicBatchWrapper86_copies320 a h).2.1
  have e_160_10 := arithEq_of_rows h (row := 160) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_10
  simp only [← c10236, ← c10237, ← c10238] at e_160_10
  have e_160_11 := arithEq_of_rows h (row := 160) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_11
  simp only [← c10239, ← c10240, ← c10241] at e_160_11
  have hr := e_160_11
  simp only [e_160_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 55) = bselect (a (.wire 159 7)) (a (.virt 16003)) (a (.wire 158 71)) := by
  have c10242 := (publicBatchWrapper86_copies320 a h).2.2.1
  have c10243 := (publicBatchWrapper86_copies320 a h).2.2.2.1
  have c10244 := (publicBatchWrapper86_copies320 a h).2.2.2.2.1
  have c10245 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.1
  have c10246 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.1
  have c10247 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.1
  have e_160_12 := arithEq_of_rows h (row := 160) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_12
  simp only [← c10242, ← c10243, ← c10244] at e_160_12
  have e_160_13 := arithEq_of_rows h (row := 160) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_13
  simp only [← c10245, ← c10246, ← c10247] at e_160_13
  have hr := e_160_13
  simp only [e_160_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 63) = bselect (a (.wire 159 7)) (a (.virt 16004)) (a (.wire 158 79)) := by
  have c10248 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.1
  have c10249 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.1
  have c10250 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10251 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10252 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10253 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_160_14 := arithEq_of_rows h (row := 160) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_14
  simp only [← c10248, ← c10249, ← c10250] at e_160_14
  have e_160_15 := arithEq_of_rows h (row := 160) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_15
  simp only [← c10251, ← c10252, ← c10253] at e_160_15
  have hr := e_160_15
  simp only [e_160_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 31) = bor (a (.wire 154 27)) (a (.wire 160 3)) := by
  have c10254 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10255 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10256 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10257 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10258 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10259 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_143_19 := arithEq_of_rows h (row := 143) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_143_19
  simp only [← c10254, ← c10255, ← c10256] at e_143_19
  have e_154_7 := arithEq_of_rows h (row := 154) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_7
  simp only [← c10257, ← c10258, k3, ← c10259] at e_154_7
  have hr := e_154_7
  simp only [e_143_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 67) = bnot (a (.wire 79 71)) := by
  have c10260 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10261 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10262 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_160_16 := arithEq_of_rows h (row := 160) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_16
  simp only [← c10260, k3, ← c10261, k3, ← c10262] at e_160_16
  have hr := e_160_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 71) = bnot (a (.wire 154 31)) := by
  have c10263 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10264 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10265 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_160_17 := arithEq_of_rows h (row := 160) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_17
  simp only [← c10263, k3, ← c10264, k3, ← c10265] at e_160_17
  have hr := e_160_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 159 11) = band (a (.wire 160 67)) (a (.wire 160 71)) := by
  have c10266 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10267 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10268 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_159_2 := arithEq_of_rows h (row := 159) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_2
  simp only [← c10266, ← c10267, ← c10268] at e_159_2
  have hr := e_159_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 160 79) = bselect (a (.wire 159 11)) (a (.virt 16205)) (a (.wire 160 15)) := by
  have c10269 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10270 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10271 := (publicBatchWrapper86_copies320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10272 := (publicBatchWrapper86_copies321 a h).1
  have c10273 := (publicBatchWrapper86_copies321 a h).2.1
  have c10274 := (publicBatchWrapper86_copies321 a h).2.2.1
  have e_160_18 := arithEq_of_rows h (row := 160) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_18
  simp only [← c10269, ← c10270, ← c10271] at e_160_18
  have e_160_19 := arithEq_of_rows h (row := 160) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_160_19
  simp only [← c10272, ← c10273, ← c10274] at e_160_19
  have hr := e_160_19
  simp only [e_160_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 7) = bselect (a (.wire 159 11)) (a (.virt 16206)) (a (.wire 160 23)) := by
  have c10275 := (publicBatchWrapper86_copies321 a h).2.2.2.1
  have c10276 := (publicBatchWrapper86_copies321 a h).2.2.2.2.1
  have c10277 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.1
  have c10278 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.1
  have c10279 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.1
  have c10280 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.1
  have e_161_0 := arithEq_of_rows h (row := 161) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_0
  simp only [← c10275, ← c10276, ← c10277] at e_161_0
  have e_161_1 := arithEq_of_rows h (row := 161) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_1
  simp only [← c10278, ← c10279, ← c10280] at e_161_1
  have hr := e_161_1
  simp only [e_161_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 15) = bselect (a (.wire 159 11)) (a (.virt 16207)) (a (.wire 160 31)) := by
  have c10281 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.1
  have c10282 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10283 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10284 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10285 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10286 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_2 := arithEq_of_rows h (row := 161) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_2
  simp only [← c10281, ← c10282, ← c10283] at e_161_2
  have e_161_3 := arithEq_of_rows h (row := 161) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_3
  simp only [← c10284, ← c10285, ← c10286] at e_161_3
  have hr := e_161_3
  simp only [e_161_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 23) = bselect (a (.wire 159 11)) (a (.virt 16208)) (a (.wire 160 39)) := by
  have c10287 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10288 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10289 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10290 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10291 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10292 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_4 := arithEq_of_rows h (row := 161) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_4
  simp only [← c10287, ← c10288, ← c10289] at e_161_4
  have e_161_5 := arithEq_of_rows h (row := 161) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_5
  simp only [← c10290, ← c10291, ← c10292] at e_161_5
  have hr := e_161_5
  simp only [e_161_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 31) = bselect (a (.wire 159 11)) (a (.virt 16209)) (a (.wire 160 47)) := by
  have c10293 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10294 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10295 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10296 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10297 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10298 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_6 := arithEq_of_rows h (row := 161) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_6
  simp only [← c10293, ← c10294, ← c10295] at e_161_6
  have e_161_7 := arithEq_of_rows h (row := 161) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_7
  simp only [← c10296, ← c10297, ← c10298] at e_161_7
  have hr := e_161_7
  simp only [e_161_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 39) = bselect (a (.wire 159 11)) (a (.virt 16203)) (a (.wire 160 55)) := by
  have c10299 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10300 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10301 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10302 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10303 := (publicBatchWrapper86_copies321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10304 := (publicBatchWrapper86_copies322 a h).1
  have e_161_8 := arithEq_of_rows h (row := 161) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_8
  simp only [← c10299, ← c10300, ← c10301] at e_161_8
  have e_161_9 := arithEq_of_rows h (row := 161) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_9
  simp only [← c10302, ← c10303, ← c10304] at e_161_9
  have hr := e_161_9
  simp only [e_161_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 47) = bselect (a (.wire 159 11)) (a (.virt 16204)) (a (.wire 160 63)) := by
  have c10305 := (publicBatchWrapper86_copies322 a h).2.1
  have c10306 := (publicBatchWrapper86_copies322 a h).2.2.1
  have c10307 := (publicBatchWrapper86_copies322 a h).2.2.2.1
  have c10308 := (publicBatchWrapper86_copies322 a h).2.2.2.2.1
  have c10309 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.1
  have c10310 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.1
  have e_161_10 := arithEq_of_rows h (row := 161) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_10
  simp only [← c10305, ← c10306, ← c10307] at e_161_10
  have e_161_11 := arithEq_of_rows h (row := 161) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_11
  simp only [← c10308, ← c10309, ← c10310] at e_161_11
  have hr := e_161_11
  simp only [e_161_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 35) = bor (a (.wire 154 31)) (a (.wire 160 67)) := by
  have c10311 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.1
  have c10312 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.1
  have c10313 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.1
  have c10314 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10315 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10316 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_8 := arithEq_of_rows h (row := 154) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_8
  simp only [← c10314, ← c10315, k3, ← c10316] at e_154_8
  have e_162_0 := arithEq_of_rows h (row := 162) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_0
  simp only [← c10311, ← c10312, ← c10313] at e_162_0
  have hr := e_154_8
  simp only [e_162_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 51) = bnot (a (.wire 80 35)) := by
  have c10317 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10318 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10319 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_161_12 := arithEq_of_rows h (row := 161) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_12
  simp only [← c10317, k3, ← c10318, k3, ← c10319] at e_161_12
  have hr := e_161_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 55) = bnot (a (.wire 154 35)) := by
  have c10320 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10321 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10322 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_161_13 := arithEq_of_rows h (row := 161) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_13
  simp only [← c10320, k3, ← c10321, k3, ← c10322] at e_161_13
  have hr := e_161_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 159 15) = band (a (.wire 161 51)) (a (.wire 161 55)) := by
  have c10323 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10324 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10325 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_159_3 := arithEq_of_rows h (row := 159) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_3
  simp only [← c10323, ← c10324, ← c10325] at e_159_3
  have hr := e_159_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 63) = bselect (a (.wire 159 15)) (a (.virt 16405)) (a (.wire 160 79)) := by
  have c10326 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10327 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10328 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10329 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10330 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10331 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_161_14 := arithEq_of_rows h (row := 161) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_14
  simp only [← c10326, ← c10327, ← c10328] at e_161_14
  have e_161_15 := arithEq_of_rows h (row := 161) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_15
  simp only [← c10329, ← c10330, ← c10331] at e_161_15
  have hr := e_161_15
  simp only [e_161_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 71) = bselect (a (.wire 159 15)) (a (.virt 16406)) (a (.wire 161 7)) := by
  have c10332 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10333 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10334 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10335 := (publicBatchWrapper86_copies322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10336 := (publicBatchWrapper86_copies323 a h).1
  have c10337 := (publicBatchWrapper86_copies323 a h).2.1
  have e_161_16 := arithEq_of_rows h (row := 161) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_16
  simp only [← c10332, ← c10333, ← c10334] at e_161_16
  have e_161_17 := arithEq_of_rows h (row := 161) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_17
  simp only [← c10335, ← c10336, ← c10337] at e_161_17
  have hr := e_161_17
  simp only [e_161_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 161 79) = bselect (a (.wire 159 15)) (a (.virt 16407)) (a (.wire 161 15)) := by
  have c10338 := (publicBatchWrapper86_copies323 a h).2.2.1
  have c10339 := (publicBatchWrapper86_copies323 a h).2.2.2.1
  have c10340 := (publicBatchWrapper86_copies323 a h).2.2.2.2.1
  have c10341 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.1
  have c10342 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.1
  have c10343 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.1
  have e_161_18 := arithEq_of_rows h (row := 161) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_18
  simp only [← c10338, ← c10339, ← c10340] at e_161_18
  have e_161_19 := arithEq_of_rows h (row := 161) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_161_19
  simp only [← c10341, ← c10342, ← c10343] at e_161_19
  have hr := e_161_19
  simp only [e_161_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 7) = bselect (a (.wire 159 15)) (a (.virt 16408)) (a (.wire 161 23)) := by
  have c10344 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.1
  have c10345 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.1
  have c10346 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10347 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10348 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10349 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_0 := arithEq_of_rows h (row := 163) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_0
  simp only [← c10344, ← c10345, ← c10346] at e_163_0
  have e_163_1 := arithEq_of_rows h (row := 163) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_1
  simp only [← c10347, ← c10348, ← c10349] at e_163_1
  have hr := e_163_1
  simp only [e_163_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 15) = bselect (a (.wire 159 15)) (a (.virt 16409)) (a (.wire 161 31)) := by
  have c10350 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10351 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10352 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10353 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10354 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10355 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_2 := arithEq_of_rows h (row := 163) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_2
  simp only [← c10350, ← c10351, ← c10352] at e_163_2
  have e_163_3 := arithEq_of_rows h (row := 163) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_3
  simp only [← c10353, ← c10354, ← c10355] at e_163_3
  have hr := e_163_3
  simp only [e_163_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 23) = bselect (a (.wire 159 15)) (a (.virt 16403)) (a (.wire 161 39)) := by
  have c10356 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10357 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10358 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10359 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10360 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10361 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_4 := arithEq_of_rows h (row := 163) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_4
  simp only [← c10356, ← c10357, ← c10358] at e_163_4
  have e_163_5 := arithEq_of_rows h (row := 163) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_5
  simp only [← c10359, ← c10360, ← c10361] at e_163_5
  have hr := e_163_5
  simp only [e_163_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 31) = bselect (a (.wire 159 15)) (a (.virt 16404)) (a (.wire 161 47)) := by
  have c10362 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10363 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10364 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10365 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10366 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10367 := (publicBatchWrapper86_copies323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_163_6 := arithEq_of_rows h (row := 163) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_6
  simp only [← c10362, ← c10363, ← c10364] at e_163_6
  have e_163_7 := arithEq_of_rows h (row := 163) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_7
  simp only [← c10365, ← c10366, ← c10367] at e_163_7
  have hr := e_163_7
  simp only [e_163_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 39) = bor (a (.wire 154 35)) (a (.wire 161 51)) := by
  have c10368 := (publicBatchWrapper86_copies324 a h).1
  have c10369 := (publicBatchWrapper86_copies324 a h).2.1
  have c10370 := (publicBatchWrapper86_copies324 a h).2.2.1
  have c10371 := (publicBatchWrapper86_copies324 a h).2.2.2.1
  have c10372 := (publicBatchWrapper86_copies324 a h).2.2.2.2.1
  have c10373 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_9 := arithEq_of_rows h (row := 154) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_9
  simp only [← c10371, ← c10372, k3, ← c10373] at e_154_9
  have e_162_1 := arithEq_of_rows h (row := 162) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_1
  simp only [← c10368, ← c10369, ← c10370] at e_162_1
  have hr := e_154_9
  simp only [e_162_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 35) = bnot (a (.wire 80 79)) := by
  have c10374 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.1
  have c10375 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.1
  have c10376 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_163_8 := arithEq_of_rows h (row := 163) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_8
  simp only [← c10374, k3, ← c10375, k3, ← c10376] at e_163_8
  have hr := e_163_8
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 39) = bnot (a (.wire 154 39)) := by
  have c10377 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.1
  have c10378 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10379 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_163_9 := arithEq_of_rows h (row := 163) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_9
  simp only [← c10377, k3, ← c10378, k3, ← c10379] at e_163_9
  have hr := e_163_9
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 159 19) = band (a (.wire 163 35)) (a (.wire 163 39)) := by
  have c10380 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10381 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10382 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_159_4 := arithEq_of_rows h (row := 159) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_4
  simp only [← c10380, ← c10381, ← c10382] at e_159_4
  have hr := e_159_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 47) = bselect (a (.wire 159 19)) (a (.virt 16605)) (a (.wire 161 63)) := by
  have c10383 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10384 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10385 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10386 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10387 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10388 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_10 := arithEq_of_rows h (row := 163) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_10
  simp only [← c10383, ← c10384, ← c10385] at e_163_10
  have e_163_11 := arithEq_of_rows h (row := 163) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_11
  simp only [← c10386, ← c10387, ← c10388] at e_163_11
  have hr := e_163_11
  simp only [e_163_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 55) = bselect (a (.wire 159 19)) (a (.virt 16606)) (a (.wire 161 71)) := by
  have c10389 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10390 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10391 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10392 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10393 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10394 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_12 := arithEq_of_rows h (row := 163) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_12
  simp only [← c10389, ← c10390, ← c10391] at e_163_12
  have e_163_13 := arithEq_of_rows h (row := 163) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_13
  simp only [← c10392, ← c10393, ← c10394] at e_163_13
  have hr := e_163_13
  simp only [e_163_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 63) = bselect (a (.wire 159 19)) (a (.virt 16607)) (a (.wire 161 79)) := by
  have c10395 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10396 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10397 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10398 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10399 := (publicBatchWrapper86_copies324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10400 := (publicBatchWrapper86_copies325 a h).1
  have e_163_14 := arithEq_of_rows h (row := 163) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_14
  simp only [← c10395, ← c10396, ← c10397] at e_163_14
  have e_163_15 := arithEq_of_rows h (row := 163) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_15
  simp only [← c10398, ← c10399, ← c10400] at e_163_15
  have hr := e_163_15
  simp only [e_163_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 71) = bselect (a (.wire 159 19)) (a (.virt 16608)) (a (.wire 163 7)) := by
  have c10401 := (publicBatchWrapper86_copies325 a h).2.1
  have c10402 := (publicBatchWrapper86_copies325 a h).2.2.1
  have c10403 := (publicBatchWrapper86_copies325 a h).2.2.2.1
  have c10404 := (publicBatchWrapper86_copies325 a h).2.2.2.2.1
  have c10405 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.1
  have c10406 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.1
  have e_163_16 := arithEq_of_rows h (row := 163) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_16
  simp only [← c10401, ← c10402, ← c10403] at e_163_16
  have e_163_17 := arithEq_of_rows h (row := 163) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_17
  simp only [← c10404, ← c10405, ← c10406] at e_163_17
  have hr := e_163_17
  simp only [e_163_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 163 79) = bselect (a (.wire 159 19)) (a (.virt 16609)) (a (.wire 163 15)) := by
  have c10407 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.1
  have c10408 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.1
  have c10409 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.1
  have c10410 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10411 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10412 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_163_18 := arithEq_of_rows h (row := 163) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_18
  simp only [← c10407, ← c10408, ← c10409] at e_163_18
  have e_163_19 := arithEq_of_rows h (row := 163) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_163_19
  simp only [← c10410, ← c10411, ← c10412] at e_163_19
  have hr := e_163_19
  simp only [e_163_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 7) = bselect (a (.wire 159 19)) (a (.virt 16603)) (a (.wire 163 23)) := by
  have c10413 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10414 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10415 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10416 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10417 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10418 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_164_0 := arithEq_of_rows h (row := 164) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_0
  simp only [← c10413, ← c10414, ← c10415] at e_164_0
  have e_164_1 := arithEq_of_rows h (row := 164) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_1
  simp only [← c10416, ← c10417, ← c10418] at e_164_1
  have hr := e_164_1
  simp only [e_164_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 15) = bselect (a (.wire 159 19)) (a (.virt 16604)) (a (.wire 163 31)) := by
  have c10419 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10420 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10421 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10422 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10423 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10424 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_164_2 := arithEq_of_rows h (row := 164) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_2
  simp only [← c10419, ← c10420, ← c10421] at e_164_2
  have e_164_3 := arithEq_of_rows h (row := 164) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_3
  simp only [← c10422, ← c10423, ← c10424] at e_164_3
  have hr := e_164_3
  simp only [e_164_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 43) = bor (a (.wire 154 39)) (a (.wire 163 35)) := by
  have c10425 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10426 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10427 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10428 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10429 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10430 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_10 := arithEq_of_rows h (row := 154) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_10
  simp only [← c10428, ← c10429, k3, ← c10430] at e_154_10
  have e_162_2 := arithEq_of_rows h (row := 162) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_2
  simp only [← c10425, ← c10426, ← c10427] at e_162_2
  have hr := e_154_10
  simp only [e_162_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 19) = bnot (a (.wire 82 43)) := by
  have c10431 := (publicBatchWrapper86_copies325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10432 := (publicBatchWrapper86_copies326 a h).1
  have c10433 := (publicBatchWrapper86_copies326 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_164_4 := arithEq_of_rows h (row := 164) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_4
  simp only [← c10431, k3, ← c10432, k3, ← c10433] at e_164_4
  have hr := e_164_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 23) = bnot (a (.wire 154 43)) := by
  have c10434 := (publicBatchWrapper86_copies326 a h).2.2.1
  have c10435 := (publicBatchWrapper86_copies326 a h).2.2.2.1
  have c10436 := (publicBatchWrapper86_copies326 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_164_5 := arithEq_of_rows h (row := 164) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_5
  simp only [← c10434, k3, ← c10435, k3, ← c10436] at e_164_5
  have hr := e_164_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 159 23) = band (a (.wire 164 19)) (a (.wire 164 23)) := by
  have c10437 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.1
  have c10438 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.1
  have c10439 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.1
  have e_159_5 := arithEq_of_rows h (row := 159) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_5
  simp only [← c10437, ← c10438, ← c10439] at e_159_5
  have hr := e_159_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f1615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 31) = bselect (a (.wire 159 23)) (a (.virt 16805)) (a (.wire 163 47)) := by
  have c10440 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.1
  have c10441 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.1
  have c10442 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10443 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c10444 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10445 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_164_6 := arithEq_of_rows h (row := 164) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_6
  simp only [← c10440, ← c10441, ← c10442] at e_164_6
  have e_164_7 := arithEq_of_rows h (row := 164) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_7
  simp only [← c10443, ← c10444, ← c10445] at e_164_7
  have hr := e_164_7
  simp only [e_164_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 39) = bselect (a (.wire 159 23)) (a (.virt 16806)) (a (.wire 163 55)) := by
  have c10446 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10447 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10448 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10449 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10450 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10451 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_164_8 := arithEq_of_rows h (row := 164) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_8
  simp only [← c10446, ← c10447, ← c10448] at e_164_8
  have e_164_9 := arithEq_of_rows h (row := 164) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_9
  simp only [← c10449, ← c10450, ← c10451] at e_164_9
  have hr := e_164_9
  simp only [e_164_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 47) = bselect (a (.wire 159 23)) (a (.virt 16807)) (a (.wire 163 63)) := by
  have c10452 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10453 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10454 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10455 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10456 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10457 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_164_10 := arithEq_of_rows h (row := 164) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_10
  simp only [← c10452, ← c10453, ← c10454] at e_164_10
  have e_164_11 := arithEq_of_rows h (row := 164) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_11
  simp only [← c10455, ← c10456, ← c10457] at e_164_11
  have hr := e_164_11
  simp only [e_164_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 55) = bselect (a (.wire 159 23)) (a (.virt 16808)) (a (.wire 163 71)) := by
  have c10458 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10459 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10460 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10461 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10462 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10463 := (publicBatchWrapper86_copies326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_164_12 := arithEq_of_rows h (row := 164) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_12
  simp only [← c10458, ← c10459, ← c10460] at e_164_12
  have e_164_13 := arithEq_of_rows h (row := 164) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_13
  simp only [← c10461, ← c10462, ← c10463] at e_164_13
  have hr := e_164_13
  simp only [e_164_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 63) = bselect (a (.wire 159 23)) (a (.virt 16809)) (a (.wire 163 79)) := by
  have c10464 := (publicBatchWrapper86_copies327 a h).1
  have c10465 := (publicBatchWrapper86_copies327 a h).2.1
  have c10466 := (publicBatchWrapper86_copies327 a h).2.2.1
  have c10467 := (publicBatchWrapper86_copies327 a h).2.2.2.1
  have c10468 := (publicBatchWrapper86_copies327 a h).2.2.2.2.1
  have c10469 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.1
  have e_164_14 := arithEq_of_rows h (row := 164) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_14
  simp only [← c10464, ← c10465, ← c10466] at e_164_14
  have e_164_15 := arithEq_of_rows h (row := 164) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_15
  simp only [← c10467, ← c10468, ← c10469] at e_164_15
  have hr := e_164_15
  simp only [e_164_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 71) = bselect (a (.wire 159 23)) (a (.virt 16803)) (a (.wire 164 7)) := by
  have c10470 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.1
  have c10471 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.1
  have c10472 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.1
  have c10473 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.1
  have c10474 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.1
  have c10475 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_164_16 := arithEq_of_rows h (row := 164) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_16
  simp only [← c10470, ← c10471, ← c10472] at e_164_16
  have e_164_17 := arithEq_of_rows h (row := 164) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_17
  simp only [← c10473, ← c10474, ← c10475] at e_164_17
  have hr := e_164_17
  simp only [e_164_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 164 79) = bselect (a (.wire 159 23)) (a (.virt 16804)) (a (.wire 164 15)) := by
  have c10476 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10477 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10478 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10479 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10480 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10481 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_164_18 := arithEq_of_rows h (row := 164) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_18
  simp only [← c10476, ← c10477, ← c10478] at e_164_18
  have e_164_19 := arithEq_of_rows h (row := 164) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_164_19
  simp only [← c10479, ← c10480, ← c10481] at e_164_19
  have hr := e_164_19
  simp only [e_164_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper86_f1622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 154 47) = bor (a (.wire 154 43)) (a (.wire 164 19)) := by
  have c10482 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10483 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10484 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10485 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10486 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10487 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_154_11 := arithEq_of_rows h (row := 154) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_154_11
  simp only [← c10485, ← c10486, k3, ← c10487] at e_154_11
  have e_162_3 := arithEq_of_rows h (row := 162) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_162_3
  simp only [← c10482, ← c10483, ← c10484] at e_162_3
  have hr := e_154_11
  simp only [e_162_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f1623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 3) = bnot (a (.wire 84 7)) := by
  have c10488 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10489 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10490 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_165_0 := arithEq_of_rows h (row := 165) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_0
  simp only [← c10488, k3, ← c10489, k3, ← c10490] at e_165_0
  have hr := e_165_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 165 7) = bnot (a (.wire 154 47)) := by
  have c10491 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10492 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10493 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_165_1 := arithEq_of_rows h (row := 165) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_165_1
  simp only [← c10491, k3, ← c10492, k3, ← c10493] at e_165_1
  have hr := e_165_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper86_f1625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 159 27) = band (a (.wire 165 3)) (a (.wire 165 7)) := by
  have c10494 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c10495 := (publicBatchWrapper86_copies327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c10496 := (publicBatchWrapper86_copies328 a h).1
  have e_159_6 := arithEq_of_rows h (row := 159) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_159_6
  simp only [← c10494, ← c10495, ← c10496] at e_159_6
  have hr := e_159_6
  simp only [band]
  linear_combination hr

end Plonky2Spec.Generated
