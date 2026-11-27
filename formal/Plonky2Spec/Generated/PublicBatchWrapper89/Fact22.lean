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

theorem publicBatchWrapper89_f16704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17088)) := by
  have c104730 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104731 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104732 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104733 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104734 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104735 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_6
  simp only [← c104730, ← c104731, ← c104732] at e_1712_6
  have e_1712_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_7
  simp only [← c104733, ← c104734, k2, ← c104735] at e_1712_7
  have hr := e_1712_7
  simp only [e_1712_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17089)) := by
  have c104736 := (publicBatchWrapper89_copies3273 a h).1
  have c104737 := (publicBatchWrapper89_copies3273 a h).2.1
  have c104738 := (publicBatchWrapper89_copies3273 a h).2.2.1
  have c104739 := (publicBatchWrapper89_copies3273 a h).2.2.2.1
  have c104740 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.1
  have c104741 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_8
  simp only [← c104736, ← c104737, ← c104738] at e_1712_8
  have e_1712_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_9
  simp only [← c104739, ← c104740, k2, ← c104741] at e_1712_9
  have hr := e_1712_9
  simp only [e_1712_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17090)) := by
  have c104742 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.1
  have c104743 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.1
  have c104744 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.1
  have c104745 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.1
  have c104746 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104747 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_10
  simp only [← c104742, ← c104743, ← c104744] at e_1712_10
  have e_1712_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_11
  simp only [← c104745, ← c104746, k2, ← c104747] at e_1712_11
  have hr := e_1712_11
  simp only [e_1712_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17091)) := by
  have c104748 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104749 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104750 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104751 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104752 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104753 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_12
  simp only [← c104748, ← c104749, ← c104750] at e_1712_12
  have e_1712_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_13
  simp only [← c104751, ← c104752, k2, ← c104753] at e_1712_13
  have hr := e_1712_13
  simp only [e_1712_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17092)) := by
  have c104754 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104755 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104756 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104757 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104758 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104759 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_14
  simp only [← c104754, ← c104755, ← c104756] at e_1712_14
  have e_1712_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_15
  simp only [← c104757, ← c104758, k2, ← c104759] at e_1712_15
  have hr := e_1712_15
  simp only [e_1712_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17093)) := by
  have c104760 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104761 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104762 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104763 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104764 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104765 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_16
  simp only [← c104760, ← c104761, ← c104762] at e_1712_16
  have e_1712_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_17
  simp only [← c104763, ← c104764, k2, ← c104765] at e_1712_17
  have hr := e_1712_17
  simp only [e_1712_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17094)) := by
  have c104766 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104767 := (publicBatchWrapper89_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104768 := (publicBatchWrapper89_copies3274 a h).1
  have c104769 := (publicBatchWrapper89_copies3274 a h).2.1
  have c104770 := (publicBatchWrapper89_copies3274 a h).2.2.1
  have c104771 := (publicBatchWrapper89_copies3274 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_18
  simp only [← c104766, ← c104767, ← c104768] at e_1712_18
  have e_1712_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_19
  simp only [← c104769, ← c104770, k2, ← c104771] at e_1712_19
  have hr := e_1712_19
  simp only [e_1712_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17095)) := by
  have c104772 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.1
  have c104773 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.1
  have c104774 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.1
  have c104775 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.1
  have c104776 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.1
  have c104777 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_0
  simp only [← c104772, ← c104773, ← c104774] at e_1713_0
  have e_1713_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_1
  simp only [← c104775, ← c104776, k2, ← c104777] at e_1713_1
  have hr := e_1713_1
  simp only [e_1713_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17096)) := by
  have c104778 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104779 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104780 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104781 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104782 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104783 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_2
  simp only [← c104778, ← c104779, ← c104780] at e_1713_2
  have e_1713_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_3
  simp only [← c104781, ← c104782, k2, ← c104783] at e_1713_3
  have hr := e_1713_3
  simp only [e_1713_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17097)) := by
  have c104784 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104785 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104786 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104787 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104788 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104789 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_4
  simp only [← c104784, ← c104785, ← c104786] at e_1713_4
  have e_1713_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_5
  simp only [← c104787, ← c104788, k2, ← c104789] at e_1713_5
  have hr := e_1713_5
  simp only [e_1713_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17098)) := by
  have c104790 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104791 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104792 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104793 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104794 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104795 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_6
  simp only [← c104790, ← c104791, ← c104792] at e_1713_6
  have e_1713_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_7
  simp only [← c104793, ← c104794, k2, ← c104795] at e_1713_7
  have hr := e_1713_7
  simp only [e_1713_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17099)) := by
  have c104796 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104797 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104798 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104799 := (publicBatchWrapper89_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104800 := (publicBatchWrapper89_copies3275 a h).1
  have c104801 := (publicBatchWrapper89_copies3275 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_8
  simp only [← c104796, ← c104797, ← c104798] at e_1713_8
  have e_1713_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_9
  simp only [← c104799, ← c104800, k2, ← c104801] at e_1713_9
  have hr := e_1713_9
  simp only [e_1713_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17100)) := by
  have c104802 := (publicBatchWrapper89_copies3275 a h).2.2.1
  have c104803 := (publicBatchWrapper89_copies3275 a h).2.2.2.1
  have c104804 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.1
  have c104805 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.1
  have c104806 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.1
  have c104807 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_10
  simp only [← c104802, ← c104803, ← c104804] at e_1713_10
  have e_1713_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_11
  simp only [← c104805, ← c104806, k2, ← c104807] at e_1713_11
  have hr := e_1713_11
  simp only [e_1713_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17101)) := by
  have c104808 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.1
  have c104809 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.1
  have c104810 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104811 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104812 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104813 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_12
  simp only [← c104808, ← c104809, ← c104810] at e_1713_12
  have e_1713_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_13
  simp only [← c104811, ← c104812, k2, ← c104813] at e_1713_13
  have hr := e_1713_13
  simp only [e_1713_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17102)) := by
  have c104814 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104815 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104816 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104817 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104818 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104819 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_14
  simp only [← c104814, ← c104815, ← c104816] at e_1713_14
  have e_1713_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_15
  simp only [← c104817, ← c104818, k2, ← c104819] at e_1713_15
  have hr := e_1713_15
  simp only [e_1713_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17103)) := by
  have c104820 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104821 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104822 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104823 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104824 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104825 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_16
  simp only [← c104820, ← c104821, ← c104822] at e_1713_16
  have e_1713_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_17
  simp only [← c104823, ← c104824, k2, ← c104825] at e_1713_17
  have hr := e_1713_17
  simp only [e_1713_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1713 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17104)) := by
  have c104826 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104827 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104828 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104829 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104830 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104831 := (publicBatchWrapper89_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1713_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_18
  simp only [← c104826, ← c104827, ← c104828] at e_1713_18
  have e_1713_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1713 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_19
  simp only [← c104829, ← c104830, k2, ← c104831] at e_1713_19
  have hr := e_1713_19
  simp only [e_1713_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17105)) := by
  have c104832 := (publicBatchWrapper89_copies3276 a h).1
  have c104833 := (publicBatchWrapper89_copies3276 a h).2.1
  have c104834 := (publicBatchWrapper89_copies3276 a h).2.2.1
  have c104835 := (publicBatchWrapper89_copies3276 a h).2.2.2.1
  have c104836 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.1
  have c104837 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_0
  simp only [← c104832, ← c104833, ← c104834] at e_1714_0
  have e_1714_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_1
  simp only [← c104835, ← c104836, k2, ← c104837] at e_1714_1
  have hr := e_1714_1
  simp only [e_1714_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17106)) := by
  have c104838 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.1
  have c104839 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.1
  have c104840 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.1
  have c104841 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.1
  have c104842 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104843 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_2
  simp only [← c104838, ← c104839, ← c104840] at e_1714_2
  have e_1714_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_3
  simp only [← c104841, ← c104842, k2, ← c104843] at e_1714_3
  have hr := e_1714_3
  simp only [e_1714_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17107)) := by
  have c104844 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104845 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104846 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104847 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104848 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104849 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_4
  simp only [← c104844, ← c104845, ← c104846] at e_1714_4
  have e_1714_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_5
  simp only [← c104847, ← c104848, k2, ← c104849] at e_1714_5
  have hr := e_1714_5
  simp only [e_1714_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17108)) := by
  have c104850 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104851 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104852 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104853 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104854 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104855 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_6
  simp only [← c104850, ← c104851, ← c104852] at e_1714_6
  have e_1714_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_7
  simp only [← c104853, ← c104854, k2, ← c104855] at e_1714_7
  have hr := e_1714_7
  simp only [e_1714_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17109)) := by
  have c104856 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104857 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104858 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104859 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104860 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104861 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_8
  simp only [← c104856, ← c104857, ← c104858] at e_1714_8
  have e_1714_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_9
  simp only [← c104859, ← c104860, k2, ← c104861] at e_1714_9
  have hr := e_1714_9
  simp only [e_1714_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17110)) := by
  have c104862 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104863 := (publicBatchWrapper89_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104864 := (publicBatchWrapper89_copies3277 a h).1
  have c104865 := (publicBatchWrapper89_copies3277 a h).2.1
  have c104866 := (publicBatchWrapper89_copies3277 a h).2.2.1
  have c104867 := (publicBatchWrapper89_copies3277 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_10
  simp only [← c104862, ← c104863, ← c104864] at e_1714_10
  have e_1714_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_11
  simp only [← c104865, ← c104866, k2, ← c104867] at e_1714_11
  have hr := e_1714_11
  simp only [e_1714_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17111)) := by
  have c104868 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.1
  have c104869 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.1
  have c104870 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.1
  have c104871 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.1
  have c104872 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.1
  have c104873 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_12
  simp only [← c104868, ← c104869, ← c104870] at e_1714_12
  have e_1714_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_13
  simp only [← c104871, ← c104872, k2, ← c104873] at e_1714_13
  have hr := e_1714_13
  simp only [e_1714_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17112)) := by
  have c104874 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104875 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104876 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104877 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104878 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104879 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_14
  simp only [← c104874, ← c104875, ← c104876] at e_1714_14
  have e_1714_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_15
  simp only [← c104877, ← c104878, k2, ← c104879] at e_1714_15
  have hr := e_1714_15
  simp only [e_1714_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17113)) := by
  have c104880 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104881 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104882 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104883 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104884 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104885 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_16
  simp only [← c104880, ← c104881, ← c104882] at e_1714_16
  have e_1714_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_17
  simp only [← c104883, ← c104884, k2, ← c104885] at e_1714_17
  have hr := e_1714_17
  simp only [e_1714_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1714 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17114)) := by
  have c104886 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104887 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104888 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104889 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104890 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104891 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1714_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_18
  simp only [← c104886, ← c104887, ← c104888] at e_1714_18
  have e_1714_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1714 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_19
  simp only [← c104889, ← c104890, k2, ← c104891] at e_1714_19
  have hr := e_1714_19
  simp only [e_1714_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17115)) := by
  have c104892 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104893 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104894 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104895 := (publicBatchWrapper89_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104896 := (publicBatchWrapper89_copies3278 a h).1
  have c104897 := (publicBatchWrapper89_copies3278 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_0
  simp only [← c104892, ← c104893, ← c104894] at e_1715_0
  have e_1715_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_1
  simp only [← c104895, ← c104896, k2, ← c104897] at e_1715_1
  have hr := e_1715_1
  simp only [e_1715_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17116)) := by
  have c104898 := (publicBatchWrapper89_copies3278 a h).2.2.1
  have c104899 := (publicBatchWrapper89_copies3278 a h).2.2.2.1
  have c104900 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.1
  have c104901 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.1
  have c104902 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.1
  have c104903 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_2
  simp only [← c104898, ← c104899, ← c104900] at e_1715_2
  have e_1715_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_3
  simp only [← c104901, ← c104902, k2, ← c104903] at e_1715_3
  have hr := e_1715_3
  simp only [e_1715_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17117)) := by
  have c104904 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.1
  have c104905 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.1
  have c104906 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104907 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104908 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104909 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_4
  simp only [← c104904, ← c104905, ← c104906] at e_1715_4
  have e_1715_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_5
  simp only [← c104907, ← c104908, k2, ← c104909] at e_1715_5
  have hr := e_1715_5
  simp only [e_1715_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17118)) := by
  have c104910 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104911 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104912 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104913 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104914 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104915 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_6
  simp only [← c104910, ← c104911, ← c104912] at e_1715_6
  have e_1715_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_7
  simp only [← c104913, ← c104914, k2, ← c104915] at e_1715_7
  have hr := e_1715_7
  simp only [e_1715_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17119)) := by
  have c104916 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104917 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104918 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104919 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104920 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104921 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_8
  simp only [← c104916, ← c104917, ← c104918] at e_1715_8
  have e_1715_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_9
  simp only [← c104919, ← c104920, k2, ← c104921] at e_1715_9
  have hr := e_1715_9
  simp only [e_1715_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17120)) := by
  have c104922 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104923 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104924 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104925 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104926 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104927 := (publicBatchWrapper89_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_10
  simp only [← c104922, ← c104923, ← c104924] at e_1715_10
  have e_1715_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_11
  simp only [← c104925, ← c104926, k2, ← c104927] at e_1715_11
  have hr := e_1715_11
  simp only [e_1715_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17121)) := by
  have c104928 := (publicBatchWrapper89_copies3279 a h).1
  have c104929 := (publicBatchWrapper89_copies3279 a h).2.1
  have c104930 := (publicBatchWrapper89_copies3279 a h).2.2.1
  have c104931 := (publicBatchWrapper89_copies3279 a h).2.2.2.1
  have c104932 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.1
  have c104933 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_12
  simp only [← c104928, ← c104929, ← c104930] at e_1715_12
  have e_1715_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_13
  simp only [← c104931, ← c104932, k2, ← c104933] at e_1715_13
  have hr := e_1715_13
  simp only [e_1715_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17122)) := by
  have c104934 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.1
  have c104935 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.1
  have c104936 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.1
  have c104937 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.1
  have c104938 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104939 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_14
  simp only [← c104934, ← c104935, ← c104936] at e_1715_14
  have e_1715_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_15
  simp only [← c104937, ← c104938, k2, ← c104939] at e_1715_15
  have hr := e_1715_15
  simp only [e_1715_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17123)) := by
  have c104940 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104941 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104942 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104943 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104944 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104945 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_16
  simp only [← c104940, ← c104941, ← c104942] at e_1715_16
  have e_1715_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_17
  simp only [← c104943, ← c104944, k2, ← c104945] at e_1715_17
  have hr := e_1715_17
  simp only [e_1715_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1715 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17124)) := by
  have c104946 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104947 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104948 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104949 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104950 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104951 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1715_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_18
  simp only [← c104946, ← c104947, ← c104948] at e_1715_18
  have e_1715_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1715 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_19
  simp only [← c104949, ← c104950, k2, ← c104951] at e_1715_19
  have hr := e_1715_19
  simp only [e_1715_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17125)) := by
  have c104952 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104953 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104954 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104955 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104956 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104957 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_0
  simp only [← c104952, ← c104953, ← c104954] at e_1716_0
  have e_1716_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_1
  simp only [← c104955, ← c104956, k2, ← c104957] at e_1716_1
  have hr := e_1716_1
  simp only [e_1716_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17126)) := by
  have c104958 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104959 := (publicBatchWrapper89_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104960 := (publicBatchWrapper89_copies3280 a h).1
  have c104961 := (publicBatchWrapper89_copies3280 a h).2.1
  have c104962 := (publicBatchWrapper89_copies3280 a h).2.2.1
  have c104963 := (publicBatchWrapper89_copies3280 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_2
  simp only [← c104958, ← c104959, ← c104960] at e_1716_2
  have e_1716_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_3
  simp only [← c104961, ← c104962, k2, ← c104963] at e_1716_3
  have hr := e_1716_3
  simp only [e_1716_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17127)) := by
  have c104964 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.1
  have c104965 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.1
  have c104966 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.1
  have c104967 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.1
  have c104968 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.1
  have c104969 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_4
  simp only [← c104964, ← c104965, ← c104966] at e_1716_4
  have e_1716_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_5
  simp only [← c104967, ← c104968, k2, ← c104969] at e_1716_5
  have hr := e_1716_5
  simp only [e_1716_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17128)) := by
  have c104970 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104971 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104972 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104973 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104974 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104975 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_6
  simp only [← c104970, ← c104971, ← c104972] at e_1716_6
  have e_1716_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_7
  simp only [← c104973, ← c104974, k2, ← c104975] at e_1716_7
  have hr := e_1716_7
  simp only [e_1716_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17129)) := by
  have c104976 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104977 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104978 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104979 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104980 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104981 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_8
  simp only [← c104976, ← c104977, ← c104978] at e_1716_8
  have e_1716_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_9
  simp only [← c104979, ← c104980, k2, ← c104981] at e_1716_9
  have hr := e_1716_9
  simp only [e_1716_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17130)) := by
  have c104982 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104983 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104984 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104985 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104986 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104987 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_10
  simp only [← c104982, ← c104983, ← c104984] at e_1716_10
  have e_1716_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_11
  simp only [← c104985, ← c104986, k2, ← c104987] at e_1716_11
  have hr := e_1716_11
  simp only [e_1716_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17131)) := by
  have c104988 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104989 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104990 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104991 := (publicBatchWrapper89_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104992 := (publicBatchWrapper89_copies3281 a h).1
  have c104993 := (publicBatchWrapper89_copies3281 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_12
  simp only [← c104988, ← c104989, ← c104990] at e_1716_12
  have e_1716_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_13
  simp only [← c104991, ← c104992, k2, ← c104993] at e_1716_13
  have hr := e_1716_13
  simp only [e_1716_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17132)) := by
  have c104994 := (publicBatchWrapper89_copies3281 a h).2.2.1
  have c104995 := (publicBatchWrapper89_copies3281 a h).2.2.2.1
  have c104996 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.1
  have c104997 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.1
  have c104998 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.1
  have c104999 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_14
  simp only [← c104994, ← c104995, ← c104996] at e_1716_14
  have e_1716_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_15
  simp only [← c104997, ← c104998, k2, ← c104999] at e_1716_15
  have hr := e_1716_15
  simp only [e_1716_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17133)) := by
  have c105000 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.1
  have c105001 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.1
  have c105002 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105003 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105004 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105005 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_16
  simp only [← c105000, ← c105001, ← c105002] at e_1716_16
  have e_1716_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_17
  simp only [← c105003, ← c105004, k2, ← c105005] at e_1716_17
  have hr := e_1716_17
  simp only [e_1716_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1716 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17134)) := by
  have c105006 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105007 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105008 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105009 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105010 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105011 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1716_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_18
  simp only [← c105006, ← c105007, ← c105008] at e_1716_18
  have e_1716_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1716 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_19
  simp only [← c105009, ← c105010, k2, ← c105011] at e_1716_19
  have hr := e_1716_19
  simp only [e_1716_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17135)) := by
  have c105012 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105013 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105014 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105015 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105016 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105017 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_0
  simp only [← c105012, ← c105013, ← c105014] at e_1717_0
  have e_1717_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_1
  simp only [← c105015, ← c105016, k2, ← c105017] at e_1717_1
  have hr := e_1717_1
  simp only [e_1717_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17136)) := by
  have c105018 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105019 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105020 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105021 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105022 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105023 := (publicBatchWrapper89_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_2
  simp only [← c105018, ← c105019, ← c105020] at e_1717_2
  have e_1717_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_3
  simp only [← c105021, ← c105022, k2, ← c105023] at e_1717_3
  have hr := e_1717_3
  simp only [e_1717_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17137)) := by
  have c105024 := (publicBatchWrapper89_copies3282 a h).1
  have c105025 := (publicBatchWrapper89_copies3282 a h).2.1
  have c105026 := (publicBatchWrapper89_copies3282 a h).2.2.1
  have c105027 := (publicBatchWrapper89_copies3282 a h).2.2.2.1
  have c105028 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.1
  have c105029 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_4
  simp only [← c105024, ← c105025, ← c105026] at e_1717_4
  have e_1717_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_5
  simp only [← c105027, ← c105028, k2, ← c105029] at e_1717_5
  have hr := e_1717_5
  simp only [e_1717_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17138)) := by
  have c105030 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.1
  have c105031 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.1
  have c105032 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.1
  have c105033 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.1
  have c105034 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105035 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_6
  simp only [← c105030, ← c105031, ← c105032] at e_1717_6
  have e_1717_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_7
  simp only [← c105033, ← c105034, k2, ← c105035] at e_1717_7
  have hr := e_1717_7
  simp only [e_1717_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17139)) := by
  have c105036 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105037 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105038 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105039 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105040 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105041 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_8
  simp only [← c105036, ← c105037, ← c105038] at e_1717_8
  have e_1717_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_9
  simp only [← c105039, ← c105040, k2, ← c105041] at e_1717_9
  have hr := e_1717_9
  simp only [e_1717_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17140)) := by
  have c105042 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105043 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105044 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105045 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105046 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105047 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_10
  simp only [← c105042, ← c105043, ← c105044] at e_1717_10
  have e_1717_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_11
  simp only [← c105045, ← c105046, k2, ← c105047] at e_1717_11
  have hr := e_1717_11
  simp only [e_1717_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17141)) := by
  have c105048 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105049 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105050 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105051 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105052 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105053 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_12
  simp only [← c105048, ← c105049, ← c105050] at e_1717_12
  have e_1717_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_13
  simp only [← c105051, ← c105052, k2, ← c105053] at e_1717_13
  have hr := e_1717_13
  simp only [e_1717_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17142)) := by
  have c105054 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105055 := (publicBatchWrapper89_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105056 := (publicBatchWrapper89_copies3283 a h).1
  have c105057 := (publicBatchWrapper89_copies3283 a h).2.1
  have c105058 := (publicBatchWrapper89_copies3283 a h).2.2.1
  have c105059 := (publicBatchWrapper89_copies3283 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_14
  simp only [← c105054, ← c105055, ← c105056] at e_1717_14
  have e_1717_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_15
  simp only [← c105057, ← c105058, k2, ← c105059] at e_1717_15
  have hr := e_1717_15
  simp only [e_1717_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17143)) := by
  have c105060 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.1
  have c105061 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.1
  have c105062 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.1
  have c105063 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.1
  have c105064 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.1
  have c105065 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_16
  simp only [← c105060, ← c105061, ← c105062] at e_1717_16
  have e_1717_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_17
  simp only [← c105063, ← c105064, k2, ← c105065] at e_1717_17
  have hr := e_1717_17
  simp only [e_1717_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1717 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17144)) := by
  have c105066 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105067 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105068 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105069 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105070 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105071 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1717_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_18
  simp only [← c105066, ← c105067, ← c105068] at e_1717_18
  have e_1717_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1717 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_19
  simp only [← c105069, ← c105070, k2, ← c105071] at e_1717_19
  have hr := e_1717_19
  simp only [e_1717_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17145)) := by
  have c105072 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105073 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105074 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105075 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105076 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105077 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_0
  simp only [← c105072, ← c105073, ← c105074] at e_1718_0
  have e_1718_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_1
  simp only [← c105075, ← c105076, k2, ← c105077] at e_1718_1
  have hr := e_1718_1
  simp only [e_1718_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17146)) := by
  have c105078 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105079 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105080 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105081 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105082 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105083 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_2
  simp only [← c105078, ← c105079, ← c105080] at e_1718_2
  have e_1718_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_3
  simp only [← c105081, ← c105082, k2, ← c105083] at e_1718_3
  have hr := e_1718_3
  simp only [e_1718_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17147)) := by
  have c105084 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105085 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105086 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105087 := (publicBatchWrapper89_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105088 := (publicBatchWrapper89_copies3284 a h).1
  have c105089 := (publicBatchWrapper89_copies3284 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_4
  simp only [← c105084, ← c105085, ← c105086] at e_1718_4
  have e_1718_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_5
  simp only [← c105087, ← c105088, k2, ← c105089] at e_1718_5
  have hr := e_1718_5
  simp only [e_1718_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17148)) := by
  have c105090 := (publicBatchWrapper89_copies3284 a h).2.2.1
  have c105091 := (publicBatchWrapper89_copies3284 a h).2.2.2.1
  have c105092 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.1
  have c105093 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.1
  have c105094 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.1
  have c105095 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_6
  simp only [← c105090, ← c105091, ← c105092] at e_1718_6
  have e_1718_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_7
  simp only [← c105093, ← c105094, k2, ← c105095] at e_1718_7
  have hr := e_1718_7
  simp only [e_1718_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17149)) := by
  have c105096 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.1
  have c105097 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.1
  have c105098 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105099 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105100 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105101 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_8
  simp only [← c105096, ← c105097, ← c105098] at e_1718_8
  have e_1718_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_9
  simp only [← c105099, ← c105100, k2, ← c105101] at e_1718_9
  have hr := e_1718_9
  simp only [e_1718_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17150)) := by
  have c105102 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105103 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105104 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105105 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105106 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105107 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_10
  simp only [← c105102, ← c105103, ← c105104] at e_1718_10
  have e_1718_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_11
  simp only [← c105105, ← c105106, k2, ← c105107] at e_1718_11
  have hr := e_1718_11
  simp only [e_1718_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17151)) := by
  have c105108 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105109 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105110 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105111 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105112 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105113 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_12
  simp only [← c105108, ← c105109, ← c105110] at e_1718_12
  have e_1718_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_13
  simp only [← c105111, ← c105112, k2, ← c105113] at e_1718_13
  have hr := e_1718_13
  simp only [e_1718_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17152)) := by
  have c105114 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105115 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105116 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105117 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105118 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105119 := (publicBatchWrapper89_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_14
  simp only [← c105114, ← c105115, ← c105116] at e_1718_14
  have e_1718_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_15
  simp only [← c105117, ← c105118, k2, ← c105119] at e_1718_15
  have hr := e_1718_15
  simp only [e_1718_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17153)) := by
  have c105120 := (publicBatchWrapper89_copies3285 a h).1
  have c105121 := (publicBatchWrapper89_copies3285 a h).2.1
  have c105122 := (publicBatchWrapper89_copies3285 a h).2.2.1
  have c105123 := (publicBatchWrapper89_copies3285 a h).2.2.2.1
  have c105124 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.1
  have c105125 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_16
  simp only [← c105120, ← c105121, ← c105122] at e_1718_16
  have e_1718_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_17
  simp only [← c105123, ← c105124, k2, ← c105125] at e_1718_17
  have hr := e_1718_17
  simp only [e_1718_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1718 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17154)) := by
  have c105126 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.1
  have c105127 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.1
  have c105128 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.1
  have c105129 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.1
  have c105130 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105131 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1718_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_18
  simp only [← c105126, ← c105127, ← c105128] at e_1718_18
  have e_1718_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1718 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_19
  simp only [← c105129, ← c105130, k2, ← c105131] at e_1718_19
  have hr := e_1718_19
  simp only [e_1718_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17155)) := by
  have c105132 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105133 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105134 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105135 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105136 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105137 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_0
  simp only [← c105132, ← c105133, ← c105134] at e_1719_0
  have e_1719_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_1
  simp only [← c105135, ← c105136, k2, ← c105137] at e_1719_1
  have hr := e_1719_1
  simp only [e_1719_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17156)) := by
  have c105138 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105139 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105140 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105141 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105142 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105143 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_2
  simp only [← c105138, ← c105139, ← c105140] at e_1719_2
  have e_1719_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_3
  simp only [← c105141, ← c105142, k2, ← c105143] at e_1719_3
  have hr := e_1719_3
  simp only [e_1719_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17157)) := by
  have c105144 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105145 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105146 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105147 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105148 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105149 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_4
  simp only [← c105144, ← c105145, ← c105146] at e_1719_4
  have e_1719_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_5
  simp only [← c105147, ← c105148, k2, ← c105149] at e_1719_5
  have hr := e_1719_5
  simp only [e_1719_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17158)) := by
  have c105150 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105151 := (publicBatchWrapper89_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105152 := (publicBatchWrapper89_copies3286 a h).1
  have c105153 := (publicBatchWrapper89_copies3286 a h).2.1
  have c105154 := (publicBatchWrapper89_copies3286 a h).2.2.1
  have c105155 := (publicBatchWrapper89_copies3286 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_6
  simp only [← c105150, ← c105151, ← c105152] at e_1719_6
  have e_1719_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_7
  simp only [← c105153, ← c105154, k2, ← c105155] at e_1719_7
  have hr := e_1719_7
  simp only [e_1719_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17159)) := by
  have c105156 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.1
  have c105157 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.1
  have c105158 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.1
  have c105159 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.1
  have c105160 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.1
  have c105161 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_8
  simp only [← c105156, ← c105157, ← c105158] at e_1719_8
  have e_1719_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_9
  simp only [← c105159, ← c105160, k2, ← c105161] at e_1719_9
  have hr := e_1719_9
  simp only [e_1719_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17160)) := by
  have c105162 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105163 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105164 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105165 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105166 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105167 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_10
  simp only [← c105162, ← c105163, ← c105164] at e_1719_10
  have e_1719_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_11
  simp only [← c105165, ← c105166, k2, ← c105167] at e_1719_11
  have hr := e_1719_11
  simp only [e_1719_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17161)) := by
  have c105168 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105169 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105170 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105171 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105172 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105173 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_12
  simp only [← c105168, ← c105169, ← c105170] at e_1719_12
  have e_1719_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_13
  simp only [← c105171, ← c105172, k2, ← c105173] at e_1719_13
  have hr := e_1719_13
  simp only [e_1719_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17162)) := by
  have c105174 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105175 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105176 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105177 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105178 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105179 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_14
  simp only [← c105174, ← c105175, ← c105176] at e_1719_14
  have e_1719_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_15
  simp only [← c105177, ← c105178, k2, ← c105179] at e_1719_15
  have hr := e_1719_15
  simp only [e_1719_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17163)) := by
  have c105180 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105181 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105182 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105183 := (publicBatchWrapper89_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105184 := (publicBatchWrapper89_copies3287 a h).1
  have c105185 := (publicBatchWrapper89_copies3287 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_16
  simp only [← c105180, ← c105181, ← c105182] at e_1719_16
  have e_1719_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_17
  simp only [← c105183, ← c105184, k2, ← c105185] at e_1719_17
  have hr := e_1719_17
  simp only [e_1719_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1719 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17164)) := by
  have c105186 := (publicBatchWrapper89_copies3287 a h).2.2.1
  have c105187 := (publicBatchWrapper89_copies3287 a h).2.2.2.1
  have c105188 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.1
  have c105189 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.1
  have c105190 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.1
  have c105191 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1719_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_18
  simp only [← c105186, ← c105187, ← c105188] at e_1719_18
  have e_1719_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1719 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_19
  simp only [← c105189, ← c105190, k2, ← c105191] at e_1719_19
  have hr := e_1719_19
  simp only [e_1719_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17165)) := by
  have c105192 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.1
  have c105193 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.1
  have c105194 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105195 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105196 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105197 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_0
  simp only [← c105192, ← c105193, ← c105194] at e_1720_0
  have e_1720_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_1
  simp only [← c105195, ← c105196, k2, ← c105197] at e_1720_1
  have hr := e_1720_1
  simp only [e_1720_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17166)) := by
  have c105198 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105199 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105200 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105201 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105202 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105203 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_2
  simp only [← c105198, ← c105199, ← c105200] at e_1720_2
  have e_1720_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_3
  simp only [← c105201, ← c105202, k2, ← c105203] at e_1720_3
  have hr := e_1720_3
  simp only [e_1720_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17167)) := by
  have c105204 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105205 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105206 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105207 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105208 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105209 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_4
  simp only [← c105204, ← c105205, ← c105206] at e_1720_4
  have e_1720_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_5
  simp only [← c105207, ← c105208, k2, ← c105209] at e_1720_5
  have hr := e_1720_5
  simp only [e_1720_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17168)) := by
  have c105210 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105211 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105212 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105213 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105214 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105215 := (publicBatchWrapper89_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_6
  simp only [← c105210, ← c105211, ← c105212] at e_1720_6
  have e_1720_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_7
  simp only [← c105213, ← c105214, k2, ← c105215] at e_1720_7
  have hr := e_1720_7
  simp only [e_1720_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17169)) := by
  have c105216 := (publicBatchWrapper89_copies3288 a h).1
  have c105217 := (publicBatchWrapper89_copies3288 a h).2.1
  have c105218 := (publicBatchWrapper89_copies3288 a h).2.2.1
  have c105219 := (publicBatchWrapper89_copies3288 a h).2.2.2.1
  have c105220 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.1
  have c105221 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_8
  simp only [← c105216, ← c105217, ← c105218] at e_1720_8
  have e_1720_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_9
  simp only [← c105219, ← c105220, k2, ← c105221] at e_1720_9
  have hr := e_1720_9
  simp only [e_1720_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17210)) := by
  have c105222 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.1
  have c105223 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.1
  have c105224 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.1
  have c105225 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.1
  have c105226 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105227 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_10
  simp only [← c105222, ← c105223, ← c105224] at e_1720_10
  have e_1720_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_11
  simp only [← c105225, ← c105226, k2, ← c105227] at e_1720_11
  have hr := e_1720_11
  simp only [e_1720_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17211)) := by
  have c105228 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105229 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105230 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105231 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105232 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105233 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_12
  simp only [← c105228, ← c105229, ← c105230] at e_1720_12
  have e_1720_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_13
  simp only [← c105231, ← c105232, k2, ← c105233] at e_1720_13
  have hr := e_1720_13
  simp only [e_1720_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17212)) := by
  have c105234 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105235 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105236 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105237 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105238 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105239 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_14
  simp only [← c105234, ← c105235, ← c105236] at e_1720_14
  have e_1720_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_15
  simp only [← c105237, ← c105238, k2, ← c105239] at e_1720_15
  have hr := e_1720_15
  simp only [e_1720_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17213)) := by
  have c105240 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105241 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105242 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105243 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105244 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105245 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_16
  simp only [← c105240, ← c105241, ← c105242] at e_1720_16
  have e_1720_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_17
  simp only [← c105243, ← c105244, k2, ← c105245] at e_1720_17
  have hr := e_1720_17
  simp only [e_1720_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1720 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17214)) := by
  have c105246 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105247 := (publicBatchWrapper89_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105248 := (publicBatchWrapper89_copies3289 a h).1
  have c105249 := (publicBatchWrapper89_copies3289 a h).2.1
  have c105250 := (publicBatchWrapper89_copies3289 a h).2.2.1
  have c105251 := (publicBatchWrapper89_copies3289 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1720_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_18
  simp only [← c105246, ← c105247, ← c105248] at e_1720_18
  have e_1720_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1720 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_19
  simp only [← c105249, ← c105250, k2, ← c105251] at e_1720_19
  have hr := e_1720_19
  simp only [e_1720_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17215)) := by
  have c105252 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.1
  have c105253 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.1
  have c105254 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.1
  have c105255 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.1
  have c105256 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.1
  have c105257 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_0
  simp only [← c105252, ← c105253, ← c105254] at e_1721_0
  have e_1721_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_1
  simp only [← c105255, ← c105256, k2, ← c105257] at e_1721_1
  have hr := e_1721_1
  simp only [e_1721_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17216)) := by
  have c105258 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105259 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105260 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105261 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105262 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105263 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_2
  simp only [← c105258, ← c105259, ← c105260] at e_1721_2
  have e_1721_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_3
  simp only [← c105261, ← c105262, k2, ← c105263] at e_1721_3
  have hr := e_1721_3
  simp only [e_1721_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17217)) := by
  have c105264 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105265 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105266 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105267 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105268 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105269 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_4
  simp only [← c105264, ← c105265, ← c105266] at e_1721_4
  have e_1721_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_5
  simp only [← c105267, ← c105268, k2, ← c105269] at e_1721_5
  have hr := e_1721_5
  simp only [e_1721_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17218)) := by
  have c105270 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105271 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105272 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105273 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105274 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105275 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_6
  simp only [← c105270, ← c105271, ← c105272] at e_1721_6
  have e_1721_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_7
  simp only [← c105273, ← c105274, k2, ← c105275] at e_1721_7
  have hr := e_1721_7
  simp only [e_1721_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17219)) := by
  have c105276 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105277 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105278 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105279 := (publicBatchWrapper89_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105280 := (publicBatchWrapper89_copies3290 a h).1
  have c105281 := (publicBatchWrapper89_copies3290 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_8
  simp only [← c105276, ← c105277, ← c105278] at e_1721_8
  have e_1721_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_9
  simp only [← c105279, ← c105280, k2, ← c105281] at e_1721_9
  have hr := e_1721_9
  simp only [e_1721_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17220)) := by
  have c105282 := (publicBatchWrapper89_copies3290 a h).2.2.1
  have c105283 := (publicBatchWrapper89_copies3290 a h).2.2.2.1
  have c105284 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.1
  have c105285 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.1
  have c105286 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.1
  have c105287 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_10
  simp only [← c105282, ← c105283, ← c105284] at e_1721_10
  have e_1721_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_11
  simp only [← c105285, ← c105286, k2, ← c105287] at e_1721_11
  have hr := e_1721_11
  simp only [e_1721_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17221)) := by
  have c105288 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.1
  have c105289 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.1
  have c105290 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105291 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105292 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105293 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_12
  simp only [← c105288, ← c105289, ← c105290] at e_1721_12
  have e_1721_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_13
  simp only [← c105291, ← c105292, k2, ← c105293] at e_1721_13
  have hr := e_1721_13
  simp only [e_1721_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17222)) := by
  have c105294 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105295 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105296 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105297 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105298 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105299 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_14
  simp only [← c105294, ← c105295, ← c105296] at e_1721_14
  have e_1721_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_15
  simp only [← c105297, ← c105298, k2, ← c105299] at e_1721_15
  have hr := e_1721_15
  simp only [e_1721_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17223)) := by
  have c105300 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105301 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105302 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105303 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105304 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105305 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_16
  simp only [← c105300, ← c105301, ← c105302] at e_1721_16
  have e_1721_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_17
  simp only [← c105303, ← c105304, k2, ← c105305] at e_1721_17
  have hr := e_1721_17
  simp only [e_1721_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1721 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17224)) := by
  have c105306 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105307 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105308 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105309 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105310 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105311 := (publicBatchWrapper89_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1721_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_18
  simp only [← c105306, ← c105307, ← c105308] at e_1721_18
  have e_1721_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1721 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_19
  simp only [← c105309, ← c105310, k2, ← c105311] at e_1721_19
  have hr := e_1721_19
  simp only [e_1721_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17225)) := by
  have c105312 := (publicBatchWrapper89_copies3291 a h).1
  have c105313 := (publicBatchWrapper89_copies3291 a h).2.1
  have c105314 := (publicBatchWrapper89_copies3291 a h).2.2.1
  have c105315 := (publicBatchWrapper89_copies3291 a h).2.2.2.1
  have c105316 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.1
  have c105317 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_0
  simp only [← c105312, ← c105313, ← c105314] at e_1722_0
  have e_1722_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_1
  simp only [← c105315, ← c105316, k2, ← c105317] at e_1722_1
  have hr := e_1722_1
  simp only [e_1722_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17226)) := by
  have c105318 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.1
  have c105319 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.1
  have c105320 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.1
  have c105321 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.1
  have c105322 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105323 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_2
  simp only [← c105318, ← c105319, ← c105320] at e_1722_2
  have e_1722_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_3
  simp only [← c105321, ← c105322, k2, ← c105323] at e_1722_3
  have hr := e_1722_3
  simp only [e_1722_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17227)) := by
  have c105324 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105325 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105326 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105327 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105328 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105329 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_4
  simp only [← c105324, ← c105325, ← c105326] at e_1722_4
  have e_1722_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_5
  simp only [← c105327, ← c105328, k2, ← c105329] at e_1722_5
  have hr := e_1722_5
  simp only [e_1722_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17228)) := by
  have c105330 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105331 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105332 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105333 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105334 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105335 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_6
  simp only [← c105330, ← c105331, ← c105332] at e_1722_6
  have e_1722_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_7
  simp only [← c105333, ← c105334, k2, ← c105335] at e_1722_7
  have hr := e_1722_7
  simp only [e_1722_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17229)) := by
  have c105336 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105337 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105338 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105339 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105340 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105341 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_8
  simp only [← c105336, ← c105337, ← c105338] at e_1722_8
  have e_1722_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_9
  simp only [← c105339, ← c105340, k2, ← c105341] at e_1722_9
  have hr := e_1722_9
  simp only [e_1722_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17230)) := by
  have c105342 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105343 := (publicBatchWrapper89_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105344 := (publicBatchWrapper89_copies3292 a h).1
  have c105345 := (publicBatchWrapper89_copies3292 a h).2.1
  have c105346 := (publicBatchWrapper89_copies3292 a h).2.2.1
  have c105347 := (publicBatchWrapper89_copies3292 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_10
  simp only [← c105342, ← c105343, ← c105344] at e_1722_10
  have e_1722_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_11
  simp only [← c105345, ← c105346, k2, ← c105347] at e_1722_11
  have hr := e_1722_11
  simp only [e_1722_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17231)) := by
  have c105348 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.1
  have c105349 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.1
  have c105350 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.1
  have c105351 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.1
  have c105352 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.1
  have c105353 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_12
  simp only [← c105348, ← c105349, ← c105350] at e_1722_12
  have e_1722_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_13
  simp only [← c105351, ← c105352, k2, ← c105353] at e_1722_13
  have hr := e_1722_13
  simp only [e_1722_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17232)) := by
  have c105354 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105355 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105356 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105357 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105358 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105359 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_14
  simp only [← c105354, ← c105355, ← c105356] at e_1722_14
  have e_1722_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_15
  simp only [← c105357, ← c105358, k2, ← c105359] at e_1722_15
  have hr := e_1722_15
  simp only [e_1722_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17233)) := by
  have c105360 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105361 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105362 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105363 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105364 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105365 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_16
  simp only [← c105360, ← c105361, ← c105362] at e_1722_16
  have e_1722_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_17
  simp only [← c105363, ← c105364, k2, ← c105365] at e_1722_17
  have hr := e_1722_17
  simp only [e_1722_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1722 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17234)) := by
  have c105366 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105367 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105368 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105369 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105370 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105371 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1722_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_18
  simp only [← c105366, ← c105367, ← c105368] at e_1722_18
  have e_1722_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1722 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_19
  simp only [← c105369, ← c105370, k2, ← c105371] at e_1722_19
  have hr := e_1722_19
  simp only [e_1722_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17235)) := by
  have c105372 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105373 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105374 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105375 := (publicBatchWrapper89_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105376 := (publicBatchWrapper89_copies3293 a h).1
  have c105377 := (publicBatchWrapper89_copies3293 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_0
  simp only [← c105372, ← c105373, ← c105374] at e_1723_0
  have e_1723_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_1
  simp only [← c105375, ← c105376, k2, ← c105377] at e_1723_1
  have hr := e_1723_1
  simp only [e_1723_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17236)) := by
  have c105378 := (publicBatchWrapper89_copies3293 a h).2.2.1
  have c105379 := (publicBatchWrapper89_copies3293 a h).2.2.2.1
  have c105380 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.1
  have c105381 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.1
  have c105382 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.1
  have c105383 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_2
  simp only [← c105378, ← c105379, ← c105380] at e_1723_2
  have e_1723_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_3
  simp only [← c105381, ← c105382, k2, ← c105383] at e_1723_3
  have hr := e_1723_3
  simp only [e_1723_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17237)) := by
  have c105384 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.1
  have c105385 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.1
  have c105386 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105387 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105388 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105389 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_4
  simp only [← c105384, ← c105385, ← c105386] at e_1723_4
  have e_1723_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_5
  simp only [← c105387, ← c105388, k2, ← c105389] at e_1723_5
  have hr := e_1723_5
  simp only [e_1723_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17238)) := by
  have c105390 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105391 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105392 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105393 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105394 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105395 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_6
  simp only [← c105390, ← c105391, ← c105392] at e_1723_6
  have e_1723_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_7
  simp only [← c105393, ← c105394, k2, ← c105395] at e_1723_7
  have hr := e_1723_7
  simp only [e_1723_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17239)) := by
  have c105396 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105397 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105398 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105399 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105400 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105401 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_8
  simp only [← c105396, ← c105397, ← c105398] at e_1723_8
  have e_1723_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_9
  simp only [← c105399, ← c105400, k2, ← c105401] at e_1723_9
  have hr := e_1723_9
  simp only [e_1723_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17240)) := by
  have c105402 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105403 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105404 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105405 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105406 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105407 := (publicBatchWrapper89_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_10
  simp only [← c105402, ← c105403, ← c105404] at e_1723_10
  have e_1723_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_11
  simp only [← c105405, ← c105406, k2, ← c105407] at e_1723_11
  have hr := e_1723_11
  simp only [e_1723_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17241)) := by
  have c105408 := (publicBatchWrapper89_copies3294 a h).1
  have c105409 := (publicBatchWrapper89_copies3294 a h).2.1
  have c105410 := (publicBatchWrapper89_copies3294 a h).2.2.1
  have c105411 := (publicBatchWrapper89_copies3294 a h).2.2.2.1
  have c105412 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.1
  have c105413 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_12
  simp only [← c105408, ← c105409, ← c105410] at e_1723_12
  have e_1723_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_13
  simp only [← c105411, ← c105412, k2, ← c105413] at e_1723_13
  have hr := e_1723_13
  simp only [e_1723_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17242)) := by
  have c105414 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.1
  have c105415 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.1
  have c105416 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.1
  have c105417 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.1
  have c105418 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105419 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_14
  simp only [← c105414, ← c105415, ← c105416] at e_1723_14
  have e_1723_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_15
  simp only [← c105417, ← c105418, k2, ← c105419] at e_1723_15
  have hr := e_1723_15
  simp only [e_1723_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17243)) := by
  have c105420 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105421 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105422 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105423 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105424 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105425 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_16
  simp only [← c105420, ← c105421, ← c105422] at e_1723_16
  have e_1723_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_17
  simp only [← c105423, ← c105424, k2, ← c105425] at e_1723_17
  have hr := e_1723_17
  simp only [e_1723_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1723 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17244)) := by
  have c105426 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105427 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105428 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105429 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105430 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105431 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1723_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_18
  simp only [← c105426, ← c105427, ← c105428] at e_1723_18
  have e_1723_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1723 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_19
  simp only [← c105429, ← c105430, k2, ← c105431] at e_1723_19
  have hr := e_1723_19
  simp only [e_1723_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17245)) := by
  have c105432 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105433 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105434 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105435 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105436 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105437 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_0
  simp only [← c105432, ← c105433, ← c105434] at e_1724_0
  have e_1724_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_1
  simp only [← c105435, ← c105436, k2, ← c105437] at e_1724_1
  have hr := e_1724_1
  simp only [e_1724_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17246)) := by
  have c105438 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105439 := (publicBatchWrapper89_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105440 := (publicBatchWrapper89_copies3295 a h).1
  have c105441 := (publicBatchWrapper89_copies3295 a h).2.1
  have c105442 := (publicBatchWrapper89_copies3295 a h).2.2.1
  have c105443 := (publicBatchWrapper89_copies3295 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_2
  simp only [← c105438, ← c105439, ← c105440] at e_1724_2
  have e_1724_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_3
  simp only [← c105441, ← c105442, k2, ← c105443] at e_1724_3
  have hr := e_1724_3
  simp only [e_1724_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17247)) := by
  have c105444 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.1
  have c105445 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.1
  have c105446 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.1
  have c105447 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.1
  have c105448 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.1
  have c105449 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_4
  simp only [← c105444, ← c105445, ← c105446] at e_1724_4
  have e_1724_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_5
  simp only [← c105447, ← c105448, k2, ← c105449] at e_1724_5
  have hr := e_1724_5
  simp only [e_1724_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17248)) := by
  have c105450 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105451 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105452 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105453 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105454 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105455 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_6
  simp only [← c105450, ← c105451, ← c105452] at e_1724_6
  have e_1724_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_7
  simp only [← c105453, ← c105454, k2, ← c105455] at e_1724_7
  have hr := e_1724_7
  simp only [e_1724_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17249)) := by
  have c105456 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105457 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105458 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105459 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105460 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105461 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_8
  simp only [← c105456, ← c105457, ← c105458] at e_1724_8
  have e_1724_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_9
  simp only [← c105459, ← c105460, k2, ← c105461] at e_1724_9
  have hr := e_1724_9
  simp only [e_1724_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17250)) := by
  have c105462 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105463 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105464 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105465 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105466 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105467 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_10
  simp only [← c105462, ← c105463, ← c105464] at e_1724_10
  have e_1724_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_11
  simp only [← c105465, ← c105466, k2, ← c105467] at e_1724_11
  have hr := e_1724_11
  simp only [e_1724_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17251)) := by
  have c105468 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105469 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105470 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105471 := (publicBatchWrapper89_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105472 := (publicBatchWrapper89_copies3296 a h).1
  have c105473 := (publicBatchWrapper89_copies3296 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_12
  simp only [← c105468, ← c105469, ← c105470] at e_1724_12
  have e_1724_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_13
  simp only [← c105471, ← c105472, k2, ← c105473] at e_1724_13
  have hr := e_1724_13
  simp only [e_1724_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17252)) := by
  have c105474 := (publicBatchWrapper89_copies3296 a h).2.2.1
  have c105475 := (publicBatchWrapper89_copies3296 a h).2.2.2.1
  have c105476 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.1
  have c105477 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.1
  have c105478 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.1
  have c105479 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_14
  simp only [← c105474, ← c105475, ← c105476] at e_1724_14
  have e_1724_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_15
  simp only [← c105477, ← c105478, k2, ← c105479] at e_1724_15
  have hr := e_1724_15
  simp only [e_1724_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17253)) := by
  have c105480 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.1
  have c105481 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.1
  have c105482 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105483 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105484 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105485 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_16
  simp only [← c105480, ← c105481, ← c105482] at e_1724_16
  have e_1724_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_17
  simp only [← c105483, ← c105484, k2, ← c105485] at e_1724_17
  have hr := e_1724_17
  simp only [e_1724_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1724 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17254)) := by
  have c105486 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105487 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105488 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105489 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105490 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105491 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1724_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_18
  simp only [← c105486, ← c105487, ← c105488] at e_1724_18
  have e_1724_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1724 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_19
  simp only [← c105489, ← c105490, k2, ← c105491] at e_1724_19
  have hr := e_1724_19
  simp only [e_1724_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17255)) := by
  have c105492 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105493 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105494 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105495 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105496 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105497 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_0
  simp only [← c105492, ← c105493, ← c105494] at e_1725_0
  have e_1725_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_1
  simp only [← c105495, ← c105496, k2, ← c105497] at e_1725_1
  have hr := e_1725_1
  simp only [e_1725_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17256)) := by
  have c105498 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105499 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105500 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105501 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105502 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105503 := (publicBatchWrapper89_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_2
  simp only [← c105498, ← c105499, ← c105500] at e_1725_2
  have e_1725_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_3
  simp only [← c105501, ← c105502, k2, ← c105503] at e_1725_3
  have hr := e_1725_3
  simp only [e_1725_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17257)) := by
  have c105504 := (publicBatchWrapper89_copies3297 a h).1
  have c105505 := (publicBatchWrapper89_copies3297 a h).2.1
  have c105506 := (publicBatchWrapper89_copies3297 a h).2.2.1
  have c105507 := (publicBatchWrapper89_copies3297 a h).2.2.2.1
  have c105508 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.1
  have c105509 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_4
  simp only [← c105504, ← c105505, ← c105506] at e_1725_4
  have e_1725_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_5
  simp only [← c105507, ← c105508, k2, ← c105509] at e_1725_5
  have hr := e_1725_5
  simp only [e_1725_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17258)) := by
  have c105510 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.1
  have c105511 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.1
  have c105512 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.1
  have c105513 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.1
  have c105514 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105515 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_6
  simp only [← c105510, ← c105511, ← c105512] at e_1725_6
  have e_1725_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_7
  simp only [← c105513, ← c105514, k2, ← c105515] at e_1725_7
  have hr := e_1725_7
  simp only [e_1725_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17259)) := by
  have c105516 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105517 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105518 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105519 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105520 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105521 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_8
  simp only [← c105516, ← c105517, ← c105518] at e_1725_8
  have e_1725_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_9
  simp only [← c105519, ← c105520, k2, ← c105521] at e_1725_9
  have hr := e_1725_9
  simp only [e_1725_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17260)) := by
  have c105522 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105523 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105524 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105525 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105526 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105527 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_10
  simp only [← c105522, ← c105523, ← c105524] at e_1725_10
  have e_1725_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_11
  simp only [← c105525, ← c105526, k2, ← c105527] at e_1725_11
  have hr := e_1725_11
  simp only [e_1725_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17261)) := by
  have c105528 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105529 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105530 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105531 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105532 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105533 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_12
  simp only [← c105528, ← c105529, ← c105530] at e_1725_12
  have e_1725_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_13
  simp only [← c105531, ← c105532, k2, ← c105533] at e_1725_13
  have hr := e_1725_13
  simp only [e_1725_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17262)) := by
  have c105534 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105535 := (publicBatchWrapper89_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105536 := (publicBatchWrapper89_copies3298 a h).1
  have c105537 := (publicBatchWrapper89_copies3298 a h).2.1
  have c105538 := (publicBatchWrapper89_copies3298 a h).2.2.1
  have c105539 := (publicBatchWrapper89_copies3298 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_14
  simp only [← c105534, ← c105535, ← c105536] at e_1725_14
  have e_1725_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_15
  simp only [← c105537, ← c105538, k2, ← c105539] at e_1725_15
  have hr := e_1725_15
  simp only [e_1725_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17263)) := by
  have c105540 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.1
  have c105541 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.1
  have c105542 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.1
  have c105543 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.1
  have c105544 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.1
  have c105545 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_16
  simp only [← c105540, ← c105541, ← c105542] at e_1725_16
  have e_1725_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_17
  simp only [← c105543, ← c105544, k2, ← c105545] at e_1725_17
  have hr := e_1725_17
  simp only [e_1725_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1725 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17264)) := by
  have c105546 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105547 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105548 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105549 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105550 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105551 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1725_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_18
  simp only [← c105546, ← c105547, ← c105548] at e_1725_18
  have e_1725_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1725 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_19
  simp only [← c105549, ← c105550, k2, ← c105551] at e_1725_19
  have hr := e_1725_19
  simp only [e_1725_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17265)) := by
  have c105552 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105553 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105554 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105555 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105556 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105557 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_0
  simp only [← c105552, ← c105553, ← c105554] at e_1726_0
  have e_1726_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_1
  simp only [← c105555, ← c105556, k2, ← c105557] at e_1726_1
  have hr := e_1726_1
  simp only [e_1726_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17266)) := by
  have c105558 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105559 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105560 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105561 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105562 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105563 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_2
  simp only [← c105558, ← c105559, ← c105560] at e_1726_2
  have e_1726_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_3
  simp only [← c105561, ← c105562, k2, ← c105563] at e_1726_3
  have hr := e_1726_3
  simp only [e_1726_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17267)) := by
  have c105564 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105565 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105566 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105567 := (publicBatchWrapper89_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105568 := (publicBatchWrapper89_copies3299 a h).1
  have c105569 := (publicBatchWrapper89_copies3299 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_4
  simp only [← c105564, ← c105565, ← c105566] at e_1726_4
  have e_1726_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_5
  simp only [← c105567, ← c105568, k2, ← c105569] at e_1726_5
  have hr := e_1726_5
  simp only [e_1726_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17268)) := by
  have c105570 := (publicBatchWrapper89_copies3299 a h).2.2.1
  have c105571 := (publicBatchWrapper89_copies3299 a h).2.2.2.1
  have c105572 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.1
  have c105573 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.1
  have c105574 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.1
  have c105575 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_6
  simp only [← c105570, ← c105571, ← c105572] at e_1726_6
  have e_1726_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_7
  simp only [← c105573, ← c105574, k2, ← c105575] at e_1726_7
  have hr := e_1726_7
  simp only [e_1726_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17269)) := by
  have c105576 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.1
  have c105577 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.1
  have c105578 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105579 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105580 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105581 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_8
  simp only [← c105576, ← c105577, ← c105578] at e_1726_8
  have e_1726_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_9
  simp only [← c105579, ← c105580, k2, ← c105581] at e_1726_9
  have hr := e_1726_9
  simp only [e_1726_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17270)) := by
  have c105582 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105583 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105584 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105585 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105586 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105587 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_10
  simp only [← c105582, ← c105583, ← c105584] at e_1726_10
  have e_1726_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_11
  simp only [← c105585, ← c105586, k2, ← c105587] at e_1726_11
  have hr := e_1726_11
  simp only [e_1726_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17271)) := by
  have c105588 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105589 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105590 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105591 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105592 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105593 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_12
  simp only [← c105588, ← c105589, ← c105590] at e_1726_12
  have e_1726_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_13
  simp only [← c105591, ← c105592, k2, ← c105593] at e_1726_13
  have hr := e_1726_13
  simp only [e_1726_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17272)) := by
  have c105594 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105595 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105596 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105597 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105598 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105599 := (publicBatchWrapper89_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_14
  simp only [← c105594, ← c105595, ← c105596] at e_1726_14
  have e_1726_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_15
  simp only [← c105597, ← c105598, k2, ← c105599] at e_1726_15
  have hr := e_1726_15
  simp only [e_1726_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17273)) := by
  have c105600 := (publicBatchWrapper89_copies3300 a h).1
  have c105601 := (publicBatchWrapper89_copies3300 a h).2.1
  have c105602 := (publicBatchWrapper89_copies3300 a h).2.2.1
  have c105603 := (publicBatchWrapper89_copies3300 a h).2.2.2.1
  have c105604 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.1
  have c105605 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_16
  simp only [← c105600, ← c105601, ← c105602] at e_1726_16
  have e_1726_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_17
  simp only [← c105603, ← c105604, k2, ← c105605] at e_1726_17
  have hr := e_1726_17
  simp only [e_1726_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1726 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17274)) := by
  have c105606 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.1
  have c105607 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.1
  have c105608 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.1
  have c105609 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.1
  have c105610 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105611 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1726_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_18
  simp only [← c105606, ← c105607, ← c105608] at e_1726_18
  have e_1726_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1726 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_19
  simp only [← c105609, ← c105610, k2, ← c105611] at e_1726_19
  have hr := e_1726_19
  simp only [e_1726_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17275)) := by
  have c105612 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105613 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105614 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105615 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105616 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105617 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_0
  simp only [← c105612, ← c105613, ← c105614] at e_1727_0
  have e_1727_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_1
  simp only [← c105615, ← c105616, k2, ← c105617] at e_1727_1
  have hr := e_1727_1
  simp only [e_1727_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17276)) := by
  have c105618 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105619 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105620 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105621 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105622 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105623 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_2
  simp only [← c105618, ← c105619, ← c105620] at e_1727_2
  have e_1727_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_3
  simp only [← c105621, ← c105622, k2, ← c105623] at e_1727_3
  have hr := e_1727_3
  simp only [e_1727_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17277)) := by
  have c105624 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105625 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105626 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105627 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105628 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105629 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_4
  simp only [← c105624, ← c105625, ← c105626] at e_1727_4
  have e_1727_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_5
  simp only [← c105627, ← c105628, k2, ← c105629] at e_1727_5
  have hr := e_1727_5
  simp only [e_1727_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17278)) := by
  have c105630 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105631 := (publicBatchWrapper89_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105632 := (publicBatchWrapper89_copies3301 a h).1
  have c105633 := (publicBatchWrapper89_copies3301 a h).2.1
  have c105634 := (publicBatchWrapper89_copies3301 a h).2.2.1
  have c105635 := (publicBatchWrapper89_copies3301 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_6
  simp only [← c105630, ← c105631, ← c105632] at e_1727_6
  have e_1727_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_7
  simp only [← c105633, ← c105634, k2, ← c105635] at e_1727_7
  have hr := e_1727_7
  simp only [e_1727_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17279)) := by
  have c105636 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.1
  have c105637 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.1
  have c105638 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.1
  have c105639 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.1
  have c105640 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.1
  have c105641 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_8
  simp only [← c105636, ← c105637, ← c105638] at e_1727_8
  have e_1727_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_9
  simp only [← c105639, ← c105640, k2, ← c105641] at e_1727_9
  have hr := e_1727_9
  simp only [e_1727_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17280)) := by
  have c105642 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105643 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105644 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105645 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105646 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105647 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_10
  simp only [← c105642, ← c105643, ← c105644] at e_1727_10
  have e_1727_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_11
  simp only [← c105645, ← c105646, k2, ← c105647] at e_1727_11
  have hr := e_1727_11
  simp only [e_1727_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17281)) := by
  have c105648 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105649 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105650 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105651 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105652 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105653 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_12
  simp only [← c105648, ← c105649, ← c105650] at e_1727_12
  have e_1727_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_13
  simp only [← c105651, ← c105652, k2, ← c105653] at e_1727_13
  have hr := e_1727_13
  simp only [e_1727_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17282)) := by
  have c105654 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105655 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105656 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105657 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105658 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105659 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_14
  simp only [← c105654, ← c105655, ← c105656] at e_1727_14
  have e_1727_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_15
  simp only [← c105657, ← c105658, k2, ← c105659] at e_1727_15
  have hr := e_1727_15
  simp only [e_1727_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17283)) := by
  have c105660 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105661 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105662 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105663 := (publicBatchWrapper89_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105664 := (publicBatchWrapper89_copies3302 a h).1
  have c105665 := (publicBatchWrapper89_copies3302 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_16
  simp only [← c105660, ← c105661, ← c105662] at e_1727_16
  have e_1727_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_17
  simp only [← c105663, ← c105664, k2, ← c105665] at e_1727_17
  have hr := e_1727_17
  simp only [e_1727_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1727 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17284)) := by
  have c105666 := (publicBatchWrapper89_copies3302 a h).2.2.1
  have c105667 := (publicBatchWrapper89_copies3302 a h).2.2.2.1
  have c105668 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.1
  have c105669 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.1
  have c105670 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.1
  have c105671 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1727_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_18
  simp only [← c105666, ← c105667, ← c105668] at e_1727_18
  have e_1727_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1727 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_19
  simp only [← c105669, ← c105670, k2, ← c105671] at e_1727_19
  have hr := e_1727_19
  simp only [e_1727_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17285)) := by
  have c105672 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.1
  have c105673 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.1
  have c105674 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105675 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105676 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105677 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_0
  simp only [← c105672, ← c105673, ← c105674] at e_1728_0
  have e_1728_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_1
  simp only [← c105675, ← c105676, k2, ← c105677] at e_1728_1
  have hr := e_1728_1
  simp only [e_1728_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17286)) := by
  have c105678 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105679 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105680 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105681 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105682 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105683 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_2
  simp only [← c105678, ← c105679, ← c105680] at e_1728_2
  have e_1728_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_3
  simp only [← c105681, ← c105682, k2, ← c105683] at e_1728_3
  have hr := e_1728_3
  simp only [e_1728_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17287)) := by
  have c105684 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105685 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105686 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105687 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105688 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105689 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_4
  simp only [← c105684, ← c105685, ← c105686] at e_1728_4
  have e_1728_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_5
  simp only [← c105687, ← c105688, k2, ← c105689] at e_1728_5
  have hr := e_1728_5
  simp only [e_1728_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17288)) := by
  have c105690 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105691 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105692 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105693 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105694 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105695 := (publicBatchWrapper89_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_6
  simp only [← c105690, ← c105691, ← c105692] at e_1728_6
  have e_1728_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_7
  simp only [← c105693, ← c105694, k2, ← c105695] at e_1728_7
  have hr := e_1728_7
  simp only [e_1728_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17289)) := by
  have c105696 := (publicBatchWrapper89_copies3303 a h).1
  have c105697 := (publicBatchWrapper89_copies3303 a h).2.1
  have c105698 := (publicBatchWrapper89_copies3303 a h).2.2.1
  have c105699 := (publicBatchWrapper89_copies3303 a h).2.2.2.1
  have c105700 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.1
  have c105701 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_8
  simp only [← c105696, ← c105697, ← c105698] at e_1728_8
  have e_1728_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_9
  simp only [← c105699, ← c105700, k2, ← c105701] at e_1728_9
  have hr := e_1728_9
  simp only [e_1728_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17290)) := by
  have c105702 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.1
  have c105703 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.1
  have c105704 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.1
  have c105705 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.1
  have c105706 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105707 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_10
  simp only [← c105702, ← c105703, ← c105704] at e_1728_10
  have e_1728_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_11
  simp only [← c105705, ← c105706, k2, ← c105707] at e_1728_11
  have hr := e_1728_11
  simp only [e_1728_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17291)) := by
  have c105708 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105709 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105710 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105711 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105712 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105713 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_12
  simp only [← c105708, ← c105709, ← c105710] at e_1728_12
  have e_1728_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_13
  simp only [← c105711, ← c105712, k2, ← c105713] at e_1728_13
  have hr := e_1728_13
  simp only [e_1728_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17292)) := by
  have c105714 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105715 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105716 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105717 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105718 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105719 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_14
  simp only [← c105714, ← c105715, ← c105716] at e_1728_14
  have e_1728_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_15
  simp only [← c105717, ← c105718, k2, ← c105719] at e_1728_15
  have hr := e_1728_15
  simp only [e_1728_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17293)) := by
  have c105720 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105721 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105722 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105723 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105724 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105725 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_16
  simp only [← c105720, ← c105721, ← c105722] at e_1728_16
  have e_1728_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_17
  simp only [← c105723, ← c105724, k2, ← c105725] at e_1728_17
  have hr := e_1728_17
  simp only [e_1728_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1728 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17294)) := by
  have c105726 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105727 := (publicBatchWrapper89_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105728 := (publicBatchWrapper89_copies3304 a h).1
  have c105729 := (publicBatchWrapper89_copies3304 a h).2.1
  have c105730 := (publicBatchWrapper89_copies3304 a h).2.2.1
  have c105731 := (publicBatchWrapper89_copies3304 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1728_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_18
  simp only [← c105726, ← c105727, ← c105728] at e_1728_18
  have e_1728_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1728 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_19
  simp only [← c105729, ← c105730, k2, ← c105731] at e_1728_19
  have hr := e_1728_19
  simp only [e_1728_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17295)) := by
  have c105732 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.1
  have c105733 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.1
  have c105734 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.1
  have c105735 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.1
  have c105736 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.1
  have c105737 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_0
  simp only [← c105732, ← c105733, ← c105734] at e_1729_0
  have e_1729_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_1
  simp only [← c105735, ← c105736, k2, ← c105737] at e_1729_1
  have hr := e_1729_1
  simp only [e_1729_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17296)) := by
  have c105738 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105739 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105740 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105741 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105742 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105743 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_2
  simp only [← c105738, ← c105739, ← c105740] at e_1729_2
  have e_1729_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_3
  simp only [← c105741, ← c105742, k2, ← c105743] at e_1729_3
  have hr := e_1729_3
  simp only [e_1729_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17297)) := by
  have c105744 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105745 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105746 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105747 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105748 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105749 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_4
  simp only [← c105744, ← c105745, ← c105746] at e_1729_4
  have e_1729_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_5
  simp only [← c105747, ← c105748, k2, ← c105749] at e_1729_5
  have hr := e_1729_5
  simp only [e_1729_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17298)) := by
  have c105750 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105751 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105752 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105753 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105754 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105755 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_6
  simp only [← c105750, ← c105751, ← c105752] at e_1729_6
  have e_1729_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_7
  simp only [← c105753, ← c105754, k2, ← c105755] at e_1729_7
  have hr := e_1729_7
  simp only [e_1729_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17299)) := by
  have c105756 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105757 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105758 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105759 := (publicBatchWrapper89_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105760 := (publicBatchWrapper89_copies3305 a h).1
  have c105761 := (publicBatchWrapper89_copies3305 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_8
  simp only [← c105756, ← c105757, ← c105758] at e_1729_8
  have e_1729_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_9
  simp only [← c105759, ← c105760, k2, ← c105761] at e_1729_9
  have hr := e_1729_9
  simp only [e_1729_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17300)) := by
  have c105762 := (publicBatchWrapper89_copies3305 a h).2.2.1
  have c105763 := (publicBatchWrapper89_copies3305 a h).2.2.2.1
  have c105764 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.1
  have c105765 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.1
  have c105766 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.1
  have c105767 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_10
  simp only [← c105762, ← c105763, ← c105764] at e_1729_10
  have e_1729_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_11
  simp only [← c105765, ← c105766, k2, ← c105767] at e_1729_11
  have hr := e_1729_11
  simp only [e_1729_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17301)) := by
  have c105768 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.1
  have c105769 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.1
  have c105770 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105771 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105772 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105773 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_12
  simp only [← c105768, ← c105769, ← c105770] at e_1729_12
  have e_1729_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_13
  simp only [← c105771, ← c105772, k2, ← c105773] at e_1729_13
  have hr := e_1729_13
  simp only [e_1729_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17302)) := by
  have c105774 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105775 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105776 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105777 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105778 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105779 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_14
  simp only [← c105774, ← c105775, ← c105776] at e_1729_14
  have e_1729_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_15
  simp only [← c105777, ← c105778, k2, ← c105779] at e_1729_15
  have hr := e_1729_15
  simp only [e_1729_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17303)) := by
  have c105780 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105781 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105782 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105783 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105784 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105785 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_16
  simp only [← c105780, ← c105781, ← c105782] at e_1729_16
  have e_1729_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_17
  simp only [← c105783, ← c105784, k2, ← c105785] at e_1729_17
  have hr := e_1729_17
  simp only [e_1729_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1729 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17304)) := by
  have c105786 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105787 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105788 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105789 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105790 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105791 := (publicBatchWrapper89_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1729_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_18
  simp only [← c105786, ← c105787, ← c105788] at e_1729_18
  have e_1729_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1729 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_19
  simp only [← c105789, ← c105790, k2, ← c105791] at e_1729_19
  have hr := e_1729_19
  simp only [e_1729_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17305)) := by
  have c105792 := (publicBatchWrapper89_copies3306 a h).1
  have c105793 := (publicBatchWrapper89_copies3306 a h).2.1
  have c105794 := (publicBatchWrapper89_copies3306 a h).2.2.1
  have c105795 := (publicBatchWrapper89_copies3306 a h).2.2.2.1
  have c105796 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.1
  have c105797 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_0
  simp only [← c105792, ← c105793, ← c105794] at e_1730_0
  have e_1730_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_1
  simp only [← c105795, ← c105796, k2, ← c105797] at e_1730_1
  have hr := e_1730_1
  simp only [e_1730_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17306)) := by
  have c105798 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.1
  have c105799 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.1
  have c105800 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.1
  have c105801 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.1
  have c105802 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105803 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_2
  simp only [← c105798, ← c105799, ← c105800] at e_1730_2
  have e_1730_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_3
  simp only [← c105801, ← c105802, k2, ← c105803] at e_1730_3
  have hr := e_1730_3
  simp only [e_1730_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17307)) := by
  have c105804 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105805 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105806 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105807 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105808 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105809 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_4
  simp only [← c105804, ← c105805, ← c105806] at e_1730_4
  have e_1730_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_5
  simp only [← c105807, ← c105808, k2, ← c105809] at e_1730_5
  have hr := e_1730_5
  simp only [e_1730_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17308)) := by
  have c105810 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105811 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105812 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105813 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105814 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105815 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_6
  simp only [← c105810, ← c105811, ← c105812] at e_1730_6
  have e_1730_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_7
  simp only [← c105813, ← c105814, k2, ← c105815] at e_1730_7
  have hr := e_1730_7
  simp only [e_1730_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17309)) := by
  have c105816 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105817 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105818 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105819 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105820 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105821 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_8
  simp only [← c105816, ← c105817, ← c105818] at e_1730_8
  have e_1730_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_9
  simp only [← c105819, ← c105820, k2, ← c105821] at e_1730_9
  have hr := e_1730_9
  simp only [e_1730_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17310)) := by
  have c105822 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105823 := (publicBatchWrapper89_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105824 := (publicBatchWrapper89_copies3307 a h).1
  have c105825 := (publicBatchWrapper89_copies3307 a h).2.1
  have c105826 := (publicBatchWrapper89_copies3307 a h).2.2.1
  have c105827 := (publicBatchWrapper89_copies3307 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_10
  simp only [← c105822, ← c105823, ← c105824] at e_1730_10
  have e_1730_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_11
  simp only [← c105825, ← c105826, k2, ← c105827] at e_1730_11
  have hr := e_1730_11
  simp only [e_1730_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17311)) := by
  have c105828 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.1
  have c105829 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.1
  have c105830 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.1
  have c105831 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.1
  have c105832 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.1
  have c105833 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_12
  simp only [← c105828, ← c105829, ← c105830] at e_1730_12
  have e_1730_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_13
  simp only [← c105831, ← c105832, k2, ← c105833] at e_1730_13
  have hr := e_1730_13
  simp only [e_1730_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17312)) := by
  have c105834 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105835 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105836 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105837 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105838 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105839 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_14
  simp only [← c105834, ← c105835, ← c105836] at e_1730_14
  have e_1730_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_15
  simp only [← c105837, ← c105838, k2, ← c105839] at e_1730_15
  have hr := e_1730_15
  simp only [e_1730_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17313)) := by
  have c105840 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105841 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105842 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105843 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105844 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105845 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_16
  simp only [← c105840, ← c105841, ← c105842] at e_1730_16
  have e_1730_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_17
  simp only [← c105843, ← c105844, k2, ← c105845] at e_1730_17
  have hr := e_1730_17
  simp only [e_1730_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1730 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17314)) := by
  have c105846 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105847 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105848 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105849 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105850 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105851 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1730_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_18
  simp only [← c105846, ← c105847, ← c105848] at e_1730_18
  have e_1730_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1730 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_19
  simp only [← c105849, ← c105850, k2, ← c105851] at e_1730_19
  have hr := e_1730_19
  simp only [e_1730_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17315)) := by
  have c105852 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105853 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105854 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105855 := (publicBatchWrapper89_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105856 := (publicBatchWrapper89_copies3308 a h).1
  have c105857 := (publicBatchWrapper89_copies3308 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_0
  simp only [← c105852, ← c105853, ← c105854] at e_1731_0
  have e_1731_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_1
  simp only [← c105855, ← c105856, k2, ← c105857] at e_1731_1
  have hr := e_1731_1
  simp only [e_1731_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17316)) := by
  have c105858 := (publicBatchWrapper89_copies3308 a h).2.2.1
  have c105859 := (publicBatchWrapper89_copies3308 a h).2.2.2.1
  have c105860 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.1
  have c105861 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.1
  have c105862 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.1
  have c105863 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_2
  simp only [← c105858, ← c105859, ← c105860] at e_1731_2
  have e_1731_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_3
  simp only [← c105861, ← c105862, k2, ← c105863] at e_1731_3
  have hr := e_1731_3
  simp only [e_1731_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17317)) := by
  have c105864 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.1
  have c105865 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.1
  have c105866 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105867 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105868 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105869 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_4
  simp only [← c105864, ← c105865, ← c105866] at e_1731_4
  have e_1731_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_5
  simp only [← c105867, ← c105868, k2, ← c105869] at e_1731_5
  have hr := e_1731_5
  simp only [e_1731_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17318)) := by
  have c105870 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105871 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105872 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105873 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105874 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105875 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_6
  simp only [← c105870, ← c105871, ← c105872] at e_1731_6
  have e_1731_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_7
  simp only [← c105873, ← c105874, k2, ← c105875] at e_1731_7
  have hr := e_1731_7
  simp only [e_1731_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17319)) := by
  have c105876 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105877 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105878 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105879 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105880 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105881 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_8
  simp only [← c105876, ← c105877, ← c105878] at e_1731_8
  have e_1731_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_9
  simp only [← c105879, ← c105880, k2, ← c105881] at e_1731_9
  have hr := e_1731_9
  simp only [e_1731_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17320)) := by
  have c105882 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105883 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105884 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105885 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105886 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105887 := (publicBatchWrapper89_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_10
  simp only [← c105882, ← c105883, ← c105884] at e_1731_10
  have e_1731_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_11
  simp only [← c105885, ← c105886, k2, ← c105887] at e_1731_11
  have hr := e_1731_11
  simp only [e_1731_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17321)) := by
  have c105888 := (publicBatchWrapper89_copies3309 a h).1
  have c105889 := (publicBatchWrapper89_copies3309 a h).2.1
  have c105890 := (publicBatchWrapper89_copies3309 a h).2.2.1
  have c105891 := (publicBatchWrapper89_copies3309 a h).2.2.2.1
  have c105892 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.1
  have c105893 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_12
  simp only [← c105888, ← c105889, ← c105890] at e_1731_12
  have e_1731_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_13
  simp only [← c105891, ← c105892, k2, ← c105893] at e_1731_13
  have hr := e_1731_13
  simp only [e_1731_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17322)) := by
  have c105894 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.1
  have c105895 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.1
  have c105896 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.1
  have c105897 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.1
  have c105898 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105899 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_14
  simp only [← c105894, ← c105895, ← c105896] at e_1731_14
  have e_1731_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_15
  simp only [← c105897, ← c105898, k2, ← c105899] at e_1731_15
  have hr := e_1731_15
  simp only [e_1731_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17323)) := by
  have c105900 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105901 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105902 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105903 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105904 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105905 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_16
  simp only [← c105900, ← c105901, ← c105902] at e_1731_16
  have e_1731_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_17
  simp only [← c105903, ← c105904, k2, ← c105905] at e_1731_17
  have hr := e_1731_17
  simp only [e_1731_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1731 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17324)) := by
  have c105906 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105907 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105908 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105909 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105910 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105911 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1731_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_18
  simp only [← c105906, ← c105907, ← c105908] at e_1731_18
  have e_1731_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1731 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_19
  simp only [← c105909, ← c105910, k2, ← c105911] at e_1731_19
  have hr := e_1731_19
  simp only [e_1731_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17325)) := by
  have c105912 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105913 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105914 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105915 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105916 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105917 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_0
  simp only [← c105912, ← c105913, ← c105914] at e_1732_0
  have e_1732_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_1
  simp only [← c105915, ← c105916, k2, ← c105917] at e_1732_1
  have hr := e_1732_1
  simp only [e_1732_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17326)) := by
  have c105918 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105919 := (publicBatchWrapper89_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105920 := (publicBatchWrapper89_copies3310 a h).1
  have c105921 := (publicBatchWrapper89_copies3310 a h).2.1
  have c105922 := (publicBatchWrapper89_copies3310 a h).2.2.1
  have c105923 := (publicBatchWrapper89_copies3310 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_2
  simp only [← c105918, ← c105919, ← c105920] at e_1732_2
  have e_1732_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_3
  simp only [← c105921, ← c105922, k2, ← c105923] at e_1732_3
  have hr := e_1732_3
  simp only [e_1732_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17327)) := by
  have c105924 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.1
  have c105925 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.1
  have c105926 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.1
  have c105927 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.1
  have c105928 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.1
  have c105929 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_4
  simp only [← c105924, ← c105925, ← c105926] at e_1732_4
  have e_1732_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_5
  simp only [← c105927, ← c105928, k2, ← c105929] at e_1732_5
  have hr := e_1732_5
  simp only [e_1732_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17328)) := by
  have c105930 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105931 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105932 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105933 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105934 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105935 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_6
  simp only [← c105930, ← c105931, ← c105932] at e_1732_6
  have e_1732_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_7
  simp only [← c105933, ← c105934, k2, ← c105935] at e_1732_7
  have hr := e_1732_7
  simp only [e_1732_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17329)) := by
  have c105936 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105937 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105938 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105939 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105940 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105941 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_8
  simp only [← c105936, ← c105937, ← c105938] at e_1732_8
  have e_1732_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_9
  simp only [← c105939, ← c105940, k2, ← c105941] at e_1732_9
  have hr := e_1732_9
  simp only [e_1732_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17330)) := by
  have c105942 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105943 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105944 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105945 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105946 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105947 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_10
  simp only [← c105942, ← c105943, ← c105944] at e_1732_10
  have e_1732_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_11
  simp only [← c105945, ← c105946, k2, ← c105947] at e_1732_11
  have hr := e_1732_11
  simp only [e_1732_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17331)) := by
  have c105948 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105949 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105950 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105951 := (publicBatchWrapper89_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105952 := (publicBatchWrapper89_copies3311 a h).1
  have c105953 := (publicBatchWrapper89_copies3311 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_12
  simp only [← c105948, ← c105949, ← c105950] at e_1732_12
  have e_1732_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_13
  simp only [← c105951, ← c105952, k2, ← c105953] at e_1732_13
  have hr := e_1732_13
  simp only [e_1732_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17332)) := by
  have c105954 := (publicBatchWrapper89_copies3311 a h).2.2.1
  have c105955 := (publicBatchWrapper89_copies3311 a h).2.2.2.1
  have c105956 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.1
  have c105957 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.1
  have c105958 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.1
  have c105959 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_14
  simp only [← c105954, ← c105955, ← c105956] at e_1732_14
  have e_1732_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_15
  simp only [← c105957, ← c105958, k2, ← c105959] at e_1732_15
  have hr := e_1732_15
  simp only [e_1732_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17333)) := by
  have c105960 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.1
  have c105961 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.1
  have c105962 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105963 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105964 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105965 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_16
  simp only [← c105960, ← c105961, ← c105962] at e_1732_16
  have e_1732_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_17
  simp only [← c105963, ← c105964, k2, ← c105965] at e_1732_17
  have hr := e_1732_17
  simp only [e_1732_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1732 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17334)) := by
  have c105966 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105967 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105968 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105969 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105970 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105971 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1732_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_18
  simp only [← c105966, ← c105967, ← c105968] at e_1732_18
  have e_1732_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1732 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_19
  simp only [← c105969, ← c105970, k2, ← c105971] at e_1732_19
  have hr := e_1732_19
  simp only [e_1732_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17335)) := by
  have c105972 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105973 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105974 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105975 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105976 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105977 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_0
  simp only [← c105972, ← c105973, ← c105974] at e_1733_0
  have e_1733_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_1
  simp only [← c105975, ← c105976, k2, ← c105977] at e_1733_1
  have hr := e_1733_1
  simp only [e_1733_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17336)) := by
  have c105978 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105979 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105980 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105981 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105982 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105983 := (publicBatchWrapper89_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_2
  simp only [← c105978, ← c105979, ← c105980] at e_1733_2
  have e_1733_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_3
  simp only [← c105981, ← c105982, k2, ← c105983] at e_1733_3
  have hr := e_1733_3
  simp only [e_1733_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17337)) := by
  have c105984 := (publicBatchWrapper89_copies3312 a h).1
  have c105985 := (publicBatchWrapper89_copies3312 a h).2.1
  have c105986 := (publicBatchWrapper89_copies3312 a h).2.2.1
  have c105987 := (publicBatchWrapper89_copies3312 a h).2.2.2.1
  have c105988 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.1
  have c105989 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_4
  simp only [← c105984, ← c105985, ← c105986] at e_1733_4
  have e_1733_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_5
  simp only [← c105987, ← c105988, k2, ← c105989] at e_1733_5
  have hr := e_1733_5
  simp only [e_1733_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17338)) := by
  have c105990 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.1
  have c105991 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.1
  have c105992 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.1
  have c105993 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.1
  have c105994 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105995 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_6
  simp only [← c105990, ← c105991, ← c105992] at e_1733_6
  have e_1733_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_7
  simp only [← c105993, ← c105994, k2, ← c105995] at e_1733_7
  have hr := e_1733_7
  simp only [e_1733_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17339)) := by
  have c105996 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105997 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105998 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105999 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106000 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106001 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_8
  simp only [← c105996, ← c105997, ← c105998] at e_1733_8
  have e_1733_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_9
  simp only [← c105999, ← c106000, k2, ← c106001] at e_1733_9
  have hr := e_1733_9
  simp only [e_1733_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17340)) := by
  have c106002 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106003 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106004 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106005 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106006 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106007 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_10
  simp only [← c106002, ← c106003, ← c106004] at e_1733_10
  have e_1733_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_11
  simp only [← c106005, ← c106006, k2, ← c106007] at e_1733_11
  have hr := e_1733_11
  simp only [e_1733_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17341)) := by
  have c106008 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106009 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106010 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106011 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106012 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106013 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_12
  simp only [← c106008, ← c106009, ← c106010] at e_1733_12
  have e_1733_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_13
  simp only [← c106011, ← c106012, k2, ← c106013] at e_1733_13
  have hr := e_1733_13
  simp only [e_1733_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17342)) := by
  have c106014 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106015 := (publicBatchWrapper89_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106016 := (publicBatchWrapper89_copies3313 a h).1
  have c106017 := (publicBatchWrapper89_copies3313 a h).2.1
  have c106018 := (publicBatchWrapper89_copies3313 a h).2.2.1
  have c106019 := (publicBatchWrapper89_copies3313 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_14
  simp only [← c106014, ← c106015, ← c106016] at e_1733_14
  have e_1733_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_15
  simp only [← c106017, ← c106018, k2, ← c106019] at e_1733_15
  have hr := e_1733_15
  simp only [e_1733_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17343)) := by
  have c106020 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.1
  have c106021 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.1
  have c106022 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.1
  have c106023 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.1
  have c106024 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.1
  have c106025 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_16
  simp only [← c106020, ← c106021, ← c106022] at e_1733_16
  have e_1733_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_17
  simp only [← c106023, ← c106024, k2, ← c106025] at e_1733_17
  have hr := e_1733_17
  simp only [e_1733_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1733 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17344)) := by
  have c106026 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106027 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106028 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106029 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106030 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106031 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1733_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_18
  simp only [← c106026, ← c106027, ← c106028] at e_1733_18
  have e_1733_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1733 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_19
  simp only [← c106029, ← c106030, k2, ← c106031] at e_1733_19
  have hr := e_1733_19
  simp only [e_1733_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17345)) := by
  have c106032 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106033 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106034 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106035 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106036 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106037 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_0
  simp only [← c106032, ← c106033, ← c106034] at e_1734_0
  have e_1734_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_1
  simp only [← c106035, ← c106036, k2, ← c106037] at e_1734_1
  have hr := e_1734_1
  simp only [e_1734_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17346)) := by
  have c106038 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106039 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106040 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106041 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106042 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106043 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_2
  simp only [← c106038, ← c106039, ← c106040] at e_1734_2
  have e_1734_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_3
  simp only [← c106041, ← c106042, k2, ← c106043] at e_1734_3
  have hr := e_1734_3
  simp only [e_1734_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17347)) := by
  have c106044 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106045 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106046 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106047 := (publicBatchWrapper89_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106048 := (publicBatchWrapper89_copies3314 a h).1
  have c106049 := (publicBatchWrapper89_copies3314 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_4
  simp only [← c106044, ← c106045, ← c106046] at e_1734_4
  have e_1734_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_5
  simp only [← c106047, ← c106048, k2, ← c106049] at e_1734_5
  have hr := e_1734_5
  simp only [e_1734_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17348)) := by
  have c106050 := (publicBatchWrapper89_copies3314 a h).2.2.1
  have c106051 := (publicBatchWrapper89_copies3314 a h).2.2.2.1
  have c106052 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.1
  have c106053 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.1
  have c106054 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.1
  have c106055 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_6
  simp only [← c106050, ← c106051, ← c106052] at e_1734_6
  have e_1734_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_7
  simp only [← c106053, ← c106054, k2, ← c106055] at e_1734_7
  have hr := e_1734_7
  simp only [e_1734_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17349)) := by
  have c106056 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.1
  have c106057 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.1
  have c106058 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106059 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106060 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106061 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_8
  simp only [← c106056, ← c106057, ← c106058] at e_1734_8
  have e_1734_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_9
  simp only [← c106059, ← c106060, k2, ← c106061] at e_1734_9
  have hr := e_1734_9
  simp only [e_1734_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17350)) := by
  have c106062 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106063 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106064 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106065 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106066 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106067 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_10
  simp only [← c106062, ← c106063, ← c106064] at e_1734_10
  have e_1734_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_11
  simp only [← c106065, ← c106066, k2, ← c106067] at e_1734_11
  have hr := e_1734_11
  simp only [e_1734_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17351)) := by
  have c106068 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106069 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106070 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106071 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106072 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106073 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_12
  simp only [← c106068, ← c106069, ← c106070] at e_1734_12
  have e_1734_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_13
  simp only [← c106071, ← c106072, k2, ← c106073] at e_1734_13
  have hr := e_1734_13
  simp only [e_1734_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17352)) := by
  have c106074 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106075 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106076 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106077 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106078 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106079 := (publicBatchWrapper89_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_14
  simp only [← c106074, ← c106075, ← c106076] at e_1734_14
  have e_1734_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_15
  simp only [← c106077, ← c106078, k2, ← c106079] at e_1734_15
  have hr := e_1734_15
  simp only [e_1734_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17353)) := by
  have c106080 := (publicBatchWrapper89_copies3315 a h).1
  have c106081 := (publicBatchWrapper89_copies3315 a h).2.1
  have c106082 := (publicBatchWrapper89_copies3315 a h).2.2.1
  have c106083 := (publicBatchWrapper89_copies3315 a h).2.2.2.1
  have c106084 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.1
  have c106085 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_16
  simp only [← c106080, ← c106081, ← c106082] at e_1734_16
  have e_1734_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_17
  simp only [← c106083, ← c106084, k2, ← c106085] at e_1734_17
  have hr := e_1734_17
  simp only [e_1734_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1734 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17354)) := by
  have c106086 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.1
  have c106087 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.1
  have c106088 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.1
  have c106089 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.1
  have c106090 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106091 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1734_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_18
  simp only [← c106086, ← c106087, ← c106088] at e_1734_18
  have e_1734_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1734 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_19
  simp only [← c106089, ← c106090, k2, ← c106091] at e_1734_19
  have hr := e_1734_19
  simp only [e_1734_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17355)) := by
  have c106092 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106093 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106094 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106095 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106096 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106097 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_0
  simp only [← c106092, ← c106093, ← c106094] at e_1735_0
  have e_1735_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_1
  simp only [← c106095, ← c106096, k2, ← c106097] at e_1735_1
  have hr := e_1735_1
  simp only [e_1735_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17356)) := by
  have c106098 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106099 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106100 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106101 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106102 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106103 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_2
  simp only [← c106098, ← c106099, ← c106100] at e_1735_2
  have e_1735_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_3
  simp only [← c106101, ← c106102, k2, ← c106103] at e_1735_3
  have hr := e_1735_3
  simp only [e_1735_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17357)) := by
  have c106104 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106105 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106106 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106107 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106108 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106109 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_4
  simp only [← c106104, ← c106105, ← c106106] at e_1735_4
  have e_1735_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_5
  simp only [← c106107, ← c106108, k2, ← c106109] at e_1735_5
  have hr := e_1735_5
  simp only [e_1735_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17358)) := by
  have c106110 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106111 := (publicBatchWrapper89_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106112 := (publicBatchWrapper89_copies3316 a h).1
  have c106113 := (publicBatchWrapper89_copies3316 a h).2.1
  have c106114 := (publicBatchWrapper89_copies3316 a h).2.2.1
  have c106115 := (publicBatchWrapper89_copies3316 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_6
  simp only [← c106110, ← c106111, ← c106112] at e_1735_6
  have e_1735_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_7
  simp only [← c106113, ← c106114, k2, ← c106115] at e_1735_7
  have hr := e_1735_7
  simp only [e_1735_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17359)) := by
  have c106116 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.1
  have c106117 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.1
  have c106118 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.1
  have c106119 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.1
  have c106120 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.1
  have c106121 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_8
  simp only [← c106116, ← c106117, ← c106118] at e_1735_8
  have e_1735_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_9
  simp only [← c106119, ← c106120, k2, ← c106121] at e_1735_9
  have hr := e_1735_9
  simp only [e_1735_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17360)) := by
  have c106122 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106123 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106124 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106125 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106126 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106127 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_10
  simp only [← c106122, ← c106123, ← c106124] at e_1735_10
  have e_1735_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_11
  simp only [← c106125, ← c106126, k2, ← c106127] at e_1735_11
  have hr := e_1735_11
  simp only [e_1735_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17361)) := by
  have c106128 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106129 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106130 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106131 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106132 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106133 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_12
  simp only [← c106128, ← c106129, ← c106130] at e_1735_12
  have e_1735_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_13
  simp only [← c106131, ← c106132, k2, ← c106133] at e_1735_13
  have hr := e_1735_13
  simp only [e_1735_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17362)) := by
  have c106134 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106135 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106136 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106137 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106138 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106139 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_14
  simp only [← c106134, ← c106135, ← c106136] at e_1735_14
  have e_1735_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_15
  simp only [← c106137, ← c106138, k2, ← c106139] at e_1735_15
  have hr := e_1735_15
  simp only [e_1735_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17363)) := by
  have c106140 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106141 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106142 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106143 := (publicBatchWrapper89_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106144 := (publicBatchWrapper89_copies3317 a h).1
  have c106145 := (publicBatchWrapper89_copies3317 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_16
  simp only [← c106140, ← c106141, ← c106142] at e_1735_16
  have e_1735_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_17
  simp only [← c106143, ← c106144, k2, ← c106145] at e_1735_17
  have hr := e_1735_17
  simp only [e_1735_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1735 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17364)) := by
  have c106146 := (publicBatchWrapper89_copies3317 a h).2.2.1
  have c106147 := (publicBatchWrapper89_copies3317 a h).2.2.2.1
  have c106148 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.1
  have c106149 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.1
  have c106150 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.1
  have c106151 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1735_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_18
  simp only [← c106146, ← c106147, ← c106148] at e_1735_18
  have e_1735_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1735 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_19
  simp only [← c106149, ← c106150, k2, ← c106151] at e_1735_19
  have hr := e_1735_19
  simp only [e_1735_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17365)) := by
  have c106152 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.1
  have c106153 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.1
  have c106154 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106155 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106156 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106157 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_0
  simp only [← c106152, ← c106153, ← c106154] at e_1736_0
  have e_1736_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_1
  simp only [← c106155, ← c106156, k2, ← c106157] at e_1736_1
  have hr := e_1736_1
  simp only [e_1736_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17366)) := by
  have c106158 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106159 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106160 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106161 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106162 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106163 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_2
  simp only [← c106158, ← c106159, ← c106160] at e_1736_2
  have e_1736_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_3
  simp only [← c106161, ← c106162, k2, ← c106163] at e_1736_3
  have hr := e_1736_3
  simp only [e_1736_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17367)) := by
  have c106164 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106165 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106166 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106167 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106168 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106169 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_4
  simp only [← c106164, ← c106165, ← c106166] at e_1736_4
  have e_1736_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_5
  simp only [← c106167, ← c106168, k2, ← c106169] at e_1736_5
  have hr := e_1736_5
  simp only [e_1736_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17368)) := by
  have c106170 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106171 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106172 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106173 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106174 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106175 := (publicBatchWrapper89_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_6
  simp only [← c106170, ← c106171, ← c106172] at e_1736_6
  have e_1736_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_7
  simp only [← c106173, ← c106174, k2, ← c106175] at e_1736_7
  have hr := e_1736_7
  simp only [e_1736_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17369)) := by
  have c106176 := (publicBatchWrapper89_copies3318 a h).1
  have c106177 := (publicBatchWrapper89_copies3318 a h).2.1
  have c106178 := (publicBatchWrapper89_copies3318 a h).2.2.1
  have c106179 := (publicBatchWrapper89_copies3318 a h).2.2.2.1
  have c106180 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.1
  have c106181 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_8
  simp only [← c106176, ← c106177, ← c106178] at e_1736_8
  have e_1736_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_9
  simp only [← c106179, ← c106180, k2, ← c106181] at e_1736_9
  have hr := e_1736_9
  simp only [e_1736_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17410)) := by
  have c106182 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.1
  have c106183 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.1
  have c106184 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.1
  have c106185 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.1
  have c106186 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106187 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_10
  simp only [← c106182, ← c106183, ← c106184] at e_1736_10
  have e_1736_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_11
  simp only [← c106185, ← c106186, k2, ← c106187] at e_1736_11
  have hr := e_1736_11
  simp only [e_1736_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17411)) := by
  have c106188 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106189 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106190 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106191 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106192 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106193 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_12
  simp only [← c106188, ← c106189, ← c106190] at e_1736_12
  have e_1736_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_13
  simp only [← c106191, ← c106192, k2, ← c106193] at e_1736_13
  have hr := e_1736_13
  simp only [e_1736_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17412)) := by
  have c106194 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106195 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106196 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106197 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106198 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106199 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_14
  simp only [← c106194, ← c106195, ← c106196] at e_1736_14
  have e_1736_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_15
  simp only [← c106197, ← c106198, k2, ← c106199] at e_1736_15
  have hr := e_1736_15
  simp only [e_1736_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17413)) := by
  have c106200 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106201 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106202 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106203 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106204 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106205 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_16
  simp only [← c106200, ← c106201, ← c106202] at e_1736_16
  have e_1736_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_17
  simp only [← c106203, ← c106204, k2, ← c106205] at e_1736_17
  have hr := e_1736_17
  simp only [e_1736_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1736 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17414)) := by
  have c106206 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106207 := (publicBatchWrapper89_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106208 := (publicBatchWrapper89_copies3319 a h).1
  have c106209 := (publicBatchWrapper89_copies3319 a h).2.1
  have c106210 := (publicBatchWrapper89_copies3319 a h).2.2.1
  have c106211 := (publicBatchWrapper89_copies3319 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1736_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_18
  simp only [← c106206, ← c106207, ← c106208] at e_1736_18
  have e_1736_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1736 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_19
  simp only [← c106209, ← c106210, k2, ← c106211] at e_1736_19
  have hr := e_1736_19
  simp only [e_1736_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17415)) := by
  have c106212 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.1
  have c106213 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.1
  have c106214 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.1
  have c106215 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.1
  have c106216 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.1
  have c106217 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_0
  simp only [← c106212, ← c106213, ← c106214] at e_1737_0
  have e_1737_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_1
  simp only [← c106215, ← c106216, k2, ← c106217] at e_1737_1
  have hr := e_1737_1
  simp only [e_1737_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17416)) := by
  have c106218 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106219 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106220 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106221 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106222 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106223 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_2
  simp only [← c106218, ← c106219, ← c106220] at e_1737_2
  have e_1737_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_3
  simp only [← c106221, ← c106222, k2, ← c106223] at e_1737_3
  have hr := e_1737_3
  simp only [e_1737_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17417)) := by
  have c106224 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106225 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106226 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106227 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106228 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106229 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_4
  simp only [← c106224, ← c106225, ← c106226] at e_1737_4
  have e_1737_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_5
  simp only [← c106227, ← c106228, k2, ← c106229] at e_1737_5
  have hr := e_1737_5
  simp only [e_1737_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17418)) := by
  have c106230 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106231 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106232 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106233 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106234 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106235 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_6
  simp only [← c106230, ← c106231, ← c106232] at e_1737_6
  have e_1737_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_7
  simp only [← c106233, ← c106234, k2, ← c106235] at e_1737_7
  have hr := e_1737_7
  simp only [e_1737_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17419)) := by
  have c106236 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106237 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106238 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106239 := (publicBatchWrapper89_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106240 := (publicBatchWrapper89_copies3320 a h).1
  have c106241 := (publicBatchWrapper89_copies3320 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_8
  simp only [← c106236, ← c106237, ← c106238] at e_1737_8
  have e_1737_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_9
  simp only [← c106239, ← c106240, k2, ← c106241] at e_1737_9
  have hr := e_1737_9
  simp only [e_1737_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17420)) := by
  have c106242 := (publicBatchWrapper89_copies3320 a h).2.2.1
  have c106243 := (publicBatchWrapper89_copies3320 a h).2.2.2.1
  have c106244 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.1
  have c106245 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.1
  have c106246 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.1
  have c106247 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_10
  simp only [← c106242, ← c106243, ← c106244] at e_1737_10
  have e_1737_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_11
  simp only [← c106245, ← c106246, k2, ← c106247] at e_1737_11
  have hr := e_1737_11
  simp only [e_1737_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17421)) := by
  have c106248 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.1
  have c106249 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.1
  have c106250 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106251 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106252 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106253 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_12
  simp only [← c106248, ← c106249, ← c106250] at e_1737_12
  have e_1737_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_13
  simp only [← c106251, ← c106252, k2, ← c106253] at e_1737_13
  have hr := e_1737_13
  simp only [e_1737_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17422)) := by
  have c106254 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106255 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106256 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106257 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106258 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106259 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_14
  simp only [← c106254, ← c106255, ← c106256] at e_1737_14
  have e_1737_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_15
  simp only [← c106257, ← c106258, k2, ← c106259] at e_1737_15
  have hr := e_1737_15
  simp only [e_1737_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17423)) := by
  have c106260 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106261 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106262 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106263 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106264 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106265 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_16
  simp only [← c106260, ← c106261, ← c106262] at e_1737_16
  have e_1737_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_17
  simp only [← c106263, ← c106264, k2, ← c106265] at e_1737_17
  have hr := e_1737_17
  simp only [e_1737_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1737 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17424)) := by
  have c106266 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106267 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106268 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106269 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106270 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106271 := (publicBatchWrapper89_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1737_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_18
  simp only [← c106266, ← c106267, ← c106268] at e_1737_18
  have e_1737_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1737 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_19
  simp only [← c106269, ← c106270, k2, ← c106271] at e_1737_19
  have hr := e_1737_19
  simp only [e_1737_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17425)) := by
  have c106272 := (publicBatchWrapper89_copies3321 a h).1
  have c106273 := (publicBatchWrapper89_copies3321 a h).2.1
  have c106274 := (publicBatchWrapper89_copies3321 a h).2.2.1
  have c106275 := (publicBatchWrapper89_copies3321 a h).2.2.2.1
  have c106276 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.1
  have c106277 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_0
  simp only [← c106272, ← c106273, ← c106274] at e_1738_0
  have e_1738_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_1
  simp only [← c106275, ← c106276, k2, ← c106277] at e_1738_1
  have hr := e_1738_1
  simp only [e_1738_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17426)) := by
  have c106278 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.1
  have c106279 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.1
  have c106280 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.1
  have c106281 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.1
  have c106282 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106283 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_2
  simp only [← c106278, ← c106279, ← c106280] at e_1738_2
  have e_1738_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_3
  simp only [← c106281, ← c106282, k2, ← c106283] at e_1738_3
  have hr := e_1738_3
  simp only [e_1738_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17427)) := by
  have c106284 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106285 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106286 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106287 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106288 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106289 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_4
  simp only [← c106284, ← c106285, ← c106286] at e_1738_4
  have e_1738_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_5
  simp only [← c106287, ← c106288, k2, ← c106289] at e_1738_5
  have hr := e_1738_5
  simp only [e_1738_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17428)) := by
  have c106290 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106291 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106292 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106293 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106294 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106295 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_6
  simp only [← c106290, ← c106291, ← c106292] at e_1738_6
  have e_1738_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_7
  simp only [← c106293, ← c106294, k2, ← c106295] at e_1738_7
  have hr := e_1738_7
  simp only [e_1738_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17429)) := by
  have c106296 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106297 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106298 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106299 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106300 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106301 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_8
  simp only [← c106296, ← c106297, ← c106298] at e_1738_8
  have e_1738_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_9
  simp only [← c106299, ← c106300, k2, ← c106301] at e_1738_9
  have hr := e_1738_9
  simp only [e_1738_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17430)) := by
  have c106302 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106303 := (publicBatchWrapper89_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106304 := (publicBatchWrapper89_copies3322 a h).1
  have c106305 := (publicBatchWrapper89_copies3322 a h).2.1
  have c106306 := (publicBatchWrapper89_copies3322 a h).2.2.1
  have c106307 := (publicBatchWrapper89_copies3322 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_10
  simp only [← c106302, ← c106303, ← c106304] at e_1738_10
  have e_1738_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_11
  simp only [← c106305, ← c106306, k2, ← c106307] at e_1738_11
  have hr := e_1738_11
  simp only [e_1738_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17431)) := by
  have c106308 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.1
  have c106309 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.1
  have c106310 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.1
  have c106311 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.1
  have c106312 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.1
  have c106313 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_12
  simp only [← c106308, ← c106309, ← c106310] at e_1738_12
  have e_1738_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_13
  simp only [← c106311, ← c106312, k2, ← c106313] at e_1738_13
  have hr := e_1738_13
  simp only [e_1738_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17432)) := by
  have c106314 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106315 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106316 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106317 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106318 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106319 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_14
  simp only [← c106314, ← c106315, ← c106316] at e_1738_14
  have e_1738_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_15
  simp only [← c106317, ← c106318, k2, ← c106319] at e_1738_15
  have hr := e_1738_15
  simp only [e_1738_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17433)) := by
  have c106320 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106321 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106322 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106323 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106324 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106325 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_16
  simp only [← c106320, ← c106321, ← c106322] at e_1738_16
  have e_1738_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_17
  simp only [← c106323, ← c106324, k2, ← c106325] at e_1738_17
  have hr := e_1738_17
  simp only [e_1738_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1738 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17434)) := by
  have c106326 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106327 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106328 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106329 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106330 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106331 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1738_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_18
  simp only [← c106326, ← c106327, ← c106328] at e_1738_18
  have e_1738_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1738 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_19
  simp only [← c106329, ← c106330, k2, ← c106331] at e_1738_19
  have hr := e_1738_19
  simp only [e_1738_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17435)) := by
  have c106332 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106333 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106334 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106335 := (publicBatchWrapper89_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106336 := (publicBatchWrapper89_copies3323 a h).1
  have c106337 := (publicBatchWrapper89_copies3323 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_0
  simp only [← c106332, ← c106333, ← c106334] at e_1739_0
  have e_1739_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_1
  simp only [← c106335, ← c106336, k2, ← c106337] at e_1739_1
  have hr := e_1739_1
  simp only [e_1739_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17436)) := by
  have c106338 := (publicBatchWrapper89_copies3323 a h).2.2.1
  have c106339 := (publicBatchWrapper89_copies3323 a h).2.2.2.1
  have c106340 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.1
  have c106341 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.1
  have c106342 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.1
  have c106343 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_2
  simp only [← c106338, ← c106339, ← c106340] at e_1739_2
  have e_1739_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_3
  simp only [← c106341, ← c106342, k2, ← c106343] at e_1739_3
  have hr := e_1739_3
  simp only [e_1739_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17437)) := by
  have c106344 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.1
  have c106345 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.1
  have c106346 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106347 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106348 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106349 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_4
  simp only [← c106344, ← c106345, ← c106346] at e_1739_4
  have e_1739_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_5
  simp only [← c106347, ← c106348, k2, ← c106349] at e_1739_5
  have hr := e_1739_5
  simp only [e_1739_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17438)) := by
  have c106350 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106351 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106352 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106353 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106354 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106355 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_6
  simp only [← c106350, ← c106351, ← c106352] at e_1739_6
  have e_1739_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_7
  simp only [← c106353, ← c106354, k2, ← c106355] at e_1739_7
  have hr := e_1739_7
  simp only [e_1739_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17439)) := by
  have c106356 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106357 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106358 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106359 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106360 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106361 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_8
  simp only [← c106356, ← c106357, ← c106358] at e_1739_8
  have e_1739_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_9
  simp only [← c106359, ← c106360, k2, ← c106361] at e_1739_9
  have hr := e_1739_9
  simp only [e_1739_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17440)) := by
  have c106362 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106363 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106364 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106365 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106366 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106367 := (publicBatchWrapper89_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_10
  simp only [← c106362, ← c106363, ← c106364] at e_1739_10
  have e_1739_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_11
  simp only [← c106365, ← c106366, k2, ← c106367] at e_1739_11
  have hr := e_1739_11
  simp only [e_1739_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17441)) := by
  have c106368 := (publicBatchWrapper89_copies3324 a h).1
  have c106369 := (publicBatchWrapper89_copies3324 a h).2.1
  have c106370 := (publicBatchWrapper89_copies3324 a h).2.2.1
  have c106371 := (publicBatchWrapper89_copies3324 a h).2.2.2.1
  have c106372 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.1
  have c106373 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_12
  simp only [← c106368, ← c106369, ← c106370] at e_1739_12
  have e_1739_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_13
  simp only [← c106371, ← c106372, k2, ← c106373] at e_1739_13
  have hr := e_1739_13
  simp only [e_1739_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17442)) := by
  have c106374 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.1
  have c106375 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.1
  have c106376 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.1
  have c106377 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.1
  have c106378 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106379 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_14
  simp only [← c106374, ← c106375, ← c106376] at e_1739_14
  have e_1739_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_15
  simp only [← c106377, ← c106378, k2, ← c106379] at e_1739_15
  have hr := e_1739_15
  simp only [e_1739_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17443)) := by
  have c106380 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106381 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106382 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106383 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106384 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106385 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_16
  simp only [← c106380, ← c106381, ← c106382] at e_1739_16
  have e_1739_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_17
  simp only [← c106383, ← c106384, k2, ← c106385] at e_1739_17
  have hr := e_1739_17
  simp only [e_1739_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1739 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17444)) := by
  have c106386 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106387 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106388 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106389 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106390 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106391 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1739_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_18
  simp only [← c106386, ← c106387, ← c106388] at e_1739_18
  have e_1739_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1739 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_19
  simp only [← c106389, ← c106390, k2, ← c106391] at e_1739_19
  have hr := e_1739_19
  simp only [e_1739_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17445)) := by
  have c106392 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106393 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106394 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106395 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106396 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106397 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_0
  simp only [← c106392, ← c106393, ← c106394] at e_1740_0
  have e_1740_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_1
  simp only [← c106395, ← c106396, k2, ← c106397] at e_1740_1
  have hr := e_1740_1
  simp only [e_1740_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17446)) := by
  have c106398 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106399 := (publicBatchWrapper89_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106400 := (publicBatchWrapper89_copies3325 a h).1
  have c106401 := (publicBatchWrapper89_copies3325 a h).2.1
  have c106402 := (publicBatchWrapper89_copies3325 a h).2.2.1
  have c106403 := (publicBatchWrapper89_copies3325 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_2
  simp only [← c106398, ← c106399, ← c106400] at e_1740_2
  have e_1740_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_3
  simp only [← c106401, ← c106402, k2, ← c106403] at e_1740_3
  have hr := e_1740_3
  simp only [e_1740_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17447)) := by
  have c106404 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.1
  have c106405 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.1
  have c106406 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.1
  have c106407 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.1
  have c106408 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.1
  have c106409 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_4
  simp only [← c106404, ← c106405, ← c106406] at e_1740_4
  have e_1740_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_5
  simp only [← c106407, ← c106408, k2, ← c106409] at e_1740_5
  have hr := e_1740_5
  simp only [e_1740_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17448)) := by
  have c106410 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106411 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106412 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106413 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106414 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106415 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_6
  simp only [← c106410, ← c106411, ← c106412] at e_1740_6
  have e_1740_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_7
  simp only [← c106413, ← c106414, k2, ← c106415] at e_1740_7
  have hr := e_1740_7
  simp only [e_1740_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17449)) := by
  have c106416 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106417 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106418 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106419 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106420 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106421 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_8
  simp only [← c106416, ← c106417, ← c106418] at e_1740_8
  have e_1740_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_9
  simp only [← c106419, ← c106420, k2, ← c106421] at e_1740_9
  have hr := e_1740_9
  simp only [e_1740_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17450)) := by
  have c106422 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106423 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106424 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106425 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106426 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106427 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_10
  simp only [← c106422, ← c106423, ← c106424] at e_1740_10
  have e_1740_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_11
  simp only [← c106425, ← c106426, k2, ← c106427] at e_1740_11
  have hr := e_1740_11
  simp only [e_1740_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17451)) := by
  have c106428 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106429 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106430 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106431 := (publicBatchWrapper89_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106432 := (publicBatchWrapper89_copies3326 a h).1
  have c106433 := (publicBatchWrapper89_copies3326 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_12
  simp only [← c106428, ← c106429, ← c106430] at e_1740_12
  have e_1740_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_13
  simp only [← c106431, ← c106432, k2, ← c106433] at e_1740_13
  have hr := e_1740_13
  simp only [e_1740_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17452)) := by
  have c106434 := (publicBatchWrapper89_copies3326 a h).2.2.1
  have c106435 := (publicBatchWrapper89_copies3326 a h).2.2.2.1
  have c106436 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.1
  have c106437 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.1
  have c106438 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.1
  have c106439 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_14
  simp only [← c106434, ← c106435, ← c106436] at e_1740_14
  have e_1740_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_15
  simp only [← c106437, ← c106438, k2, ← c106439] at e_1740_15
  have hr := e_1740_15
  simp only [e_1740_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17453)) := by
  have c106440 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.1
  have c106441 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.1
  have c106442 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106443 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106444 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106445 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_16
  simp only [← c106440, ← c106441, ← c106442] at e_1740_16
  have e_1740_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_17
  simp only [← c106443, ← c106444, k2, ← c106445] at e_1740_17
  have hr := e_1740_17
  simp only [e_1740_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1740 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17454)) := by
  have c106446 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106447 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106448 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106449 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106450 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106451 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1740_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_18
  simp only [← c106446, ← c106447, ← c106448] at e_1740_18
  have e_1740_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1740 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_19
  simp only [← c106449, ← c106450, k2, ← c106451] at e_1740_19
  have hr := e_1740_19
  simp only [e_1740_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17455)) := by
  have c106452 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106453 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106454 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106455 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106456 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106457 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_0
  simp only [← c106452, ← c106453, ← c106454] at e_1741_0
  have e_1741_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1741 (by decide)
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
  have e_1741_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_2
  simp only [← c106458, ← c106459, ← c106460] at e_1741_2
  have e_1741_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1741 (by decide)
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
  have e_1741_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_4
  simp only [← c106464, ← c106465, ← c106466] at e_1741_4
  have e_1741_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1741 (by decide)
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
  have e_1741_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_6
  simp only [← c106470, ← c106471, ← c106472] at e_1741_6
  have e_1741_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1741 (by decide)
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
  have e_1741_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_8
  simp only [← c106476, ← c106477, ← c106478] at e_1741_8
  have e_1741_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1741 (by decide)
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
  have e_1741_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_10
  simp only [← c106482, ← c106483, ← c106484] at e_1741_10
  have e_1741_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1741 (by decide)
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
  have e_1741_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_12
  simp only [← c106488, ← c106489, ← c106490] at e_1741_12
  have e_1741_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1741 (by decide)
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
  have e_1741_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_14
  simp only [← c106494, ← c106495, ← c106496] at e_1741_14
  have e_1741_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1741 (by decide)
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
  have e_1741_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_16
  simp only [← c106500, ← c106501, ← c106502] at e_1741_16
  have e_1741_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1741 (by decide)
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
  have e_1741_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1741 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_18
  simp only [← c106506, ← c106507, ← c106508] at e_1741_18
  have e_1741_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1741 (by decide)
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
  have e_1742_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_0
  simp only [← c106512, ← c106513, ← c106514] at e_1742_0
  have e_1742_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1742 (by decide)
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
  have e_1742_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_2
  simp only [← c106518, ← c106519, ← c106520] at e_1742_2
  have e_1742_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1742 (by decide)
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
  have e_1742_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_4
  simp only [← c106524, ← c106525, ← c106526] at e_1742_4
  have e_1742_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1742 (by decide)
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
  have e_1742_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_6
  simp only [← c106530, ← c106531, ← c106532] at e_1742_6
  have e_1742_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1742 (by decide)
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
  have e_1742_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_8
  simp only [← c106536, ← c106537, ← c106538] at e_1742_8
  have e_1742_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1742 (by decide)
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
  have e_1742_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_10
  simp only [← c106542, ← c106543, ← c106544] at e_1742_10
  have e_1742_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1742 (by decide)
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
  have e_1742_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_12
  simp only [← c106548, ← c106549, ← c106550] at e_1742_12
  have e_1742_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1742 (by decide)
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
  have e_1742_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_14
  simp only [← c106554, ← c106555, ← c106556] at e_1742_14
  have e_1742_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1742 (by decide)
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
  have e_1742_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_16
  simp only [← c106560, ← c106561, ← c106562] at e_1742_16
  have e_1742_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1742 (by decide)
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
  have e_1742_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1742 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_18
  simp only [← c106566, ← c106567, ← c106568] at e_1742_18
  have e_1742_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1742 (by decide)
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
  have e_1743_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_0
  simp only [← c106572, ← c106573, ← c106574] at e_1743_0
  have e_1743_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1743 (by decide)
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
  have e_1743_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_2
  simp only [← c106578, ← c106579, ← c106580] at e_1743_2
  have e_1743_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1743 (by decide)
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
  have e_1743_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_4
  simp only [← c106584, ← c106585, ← c106586] at e_1743_4
  have e_1743_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1743 (by decide)
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
  have e_1743_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_6
  simp only [← c106590, ← c106591, ← c106592] at e_1743_6
  have e_1743_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1743 (by decide)
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
  have e_1743_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_8
  simp only [← c106596, ← c106597, ← c106598] at e_1743_8
  have e_1743_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1743 (by decide)
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
  have e_1743_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_10
  simp only [← c106602, ← c106603, ← c106604] at e_1743_10
  have e_1743_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1743 (by decide)
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
  have e_1743_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_12
  simp only [← c106608, ← c106609, ← c106610] at e_1743_12
  have e_1743_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1743 (by decide)
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
  have e_1743_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_14
  simp only [← c106614, ← c106615, ← c106616] at e_1743_14
  have e_1743_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1743 (by decide)
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
  have e_1743_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_16
  simp only [← c106620, ← c106621, ← c106622] at e_1743_16
  have e_1743_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1743 (by decide)
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
  have e_1743_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1743 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_18
  simp only [← c106626, ← c106627, ← c106628] at e_1743_18
  have e_1743_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1743 (by decide)
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
  have e_1744_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_0
  simp only [← c106632, ← c106633, ← c106634] at e_1744_0
  have e_1744_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1744 (by decide)
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
  have e_1744_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_2
  simp only [← c106638, ← c106639, ← c106640] at e_1744_2
  have e_1744_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1744 (by decide)
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
  have e_1744_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_4
  simp only [← c106644, ← c106645, ← c106646] at e_1744_4
  have e_1744_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1744 (by decide)
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
  have e_1744_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_6
  simp only [← c106650, ← c106651, ← c106652] at e_1744_6
  have e_1744_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1744 (by decide)
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
  have e_1744_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_8
  simp only [← c106656, ← c106657, ← c106658] at e_1744_8
  have e_1744_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1744 (by decide)
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
  have e_1744_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_10
  simp only [← c106662, ← c106663, ← c106664] at e_1744_10
  have e_1744_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1744 (by decide)
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
  have e_1744_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_12
  simp only [← c106668, ← c106669, ← c106670] at e_1744_12
  have e_1744_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1744 (by decide)
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
  have e_1744_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_14
  simp only [← c106674, ← c106675, ← c106676] at e_1744_14
  have e_1744_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1744 (by decide)
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
  have e_1744_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_16
  simp only [← c106680, ← c106681, ← c106682] at e_1744_16
  have e_1744_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1744 (by decide)
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
  have e_1744_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1744 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_18
  simp only [← c106686, ← c106687, ← c106688] at e_1744_18
  have e_1744_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1744 (by decide)
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
  have e_1745_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_0
  simp only [← c106692, ← c106693, ← c106694] at e_1745_0
  have e_1745_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1745 (by decide)
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
  have e_1745_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_2
  simp only [← c106698, ← c106699, ← c106700] at e_1745_2
  have e_1745_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1745 (by decide)
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
  have e_1745_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_4
  simp only [← c106704, ← c106705, ← c106706] at e_1745_4
  have e_1745_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1745 (by decide)
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
  have e_1745_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_6
  simp only [← c106710, ← c106711, ← c106712] at e_1745_6
  have e_1745_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1745 (by decide)
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
  have e_1745_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_8
  simp only [← c106716, ← c106717, ← c106718] at e_1745_8
  have e_1745_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1745 (by decide)
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
  have e_1745_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_10
  simp only [← c106722, ← c106723, ← c106724] at e_1745_10
  have e_1745_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1745 (by decide)
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
  have e_1745_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_12
  simp only [← c106728, ← c106729, ← c106730] at e_1745_12
  have e_1745_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1745 (by decide)
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
  have e_1745_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_14
  simp only [← c106734, ← c106735, ← c106736] at e_1745_14
  have e_1745_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1745 (by decide)
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
  have e_1745_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_16
  simp only [← c106740, ← c106741, ← c106742] at e_1745_16
  have e_1745_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1745 (by decide)
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
  have e_1745_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1745 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_18
  simp only [← c106746, ← c106747, ← c106748] at e_1745_18
  have e_1745_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1745 (by decide)
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
  have e_1746_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_0
  simp only [← c106752, ← c106753, ← c106754] at e_1746_0
  have e_1746_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1746 (by decide)
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
  have e_1746_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_2
  simp only [← c106758, ← c106759, ← c106760] at e_1746_2
  have e_1746_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1746 (by decide)
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
  have e_1746_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_4
  simp only [← c106764, ← c106765, ← c106766] at e_1746_4
  have e_1746_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1746 (by decide)
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
  have e_1746_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_6
  simp only [← c106770, ← c106771, ← c106772] at e_1746_6
  have e_1746_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1746 (by decide)
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
  have e_1746_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_8
  simp only [← c106776, ← c106777, ← c106778] at e_1746_8
  have e_1746_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1746 (by decide)
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
  have e_1746_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_10
  simp only [← c106782, ← c106783, ← c106784] at e_1746_10
  have e_1746_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1746 (by decide)
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
  have e_1746_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_12
  simp only [← c106788, ← c106789, ← c106790] at e_1746_12
  have e_1746_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1746 (by decide)
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
  have e_1746_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_14
  simp only [← c106794, ← c106795, ← c106796] at e_1746_14
  have e_1746_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1746 (by decide)
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
  have e_1746_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_16
  simp only [← c106800, ← c106801, ← c106802] at e_1746_16
  have e_1746_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1746 (by decide)
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
  have e_1746_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1746 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_18
  simp only [← c106806, ← c106807, ← c106808] at e_1746_18
  have e_1746_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1746 (by decide)
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
  have e_1747_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_0
  simp only [← c106812, ← c106813, ← c106814] at e_1747_0
  have e_1747_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1747 (by decide)
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
  have e_1747_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_2
  simp only [← c106818, ← c106819, ← c106820] at e_1747_2
  have e_1747_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1747 (by decide)
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
  have e_1747_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_4
  simp only [← c106824, ← c106825, ← c106826] at e_1747_4
  have e_1747_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1747 (by decide)
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
  have e_1747_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_6
  simp only [← c106830, ← c106831, ← c106832] at e_1747_6
  have e_1747_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1747 (by decide)
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
  have e_1747_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_8
  simp only [← c106836, ← c106837, ← c106838] at e_1747_8
  have e_1747_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1747 (by decide)
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
  have e_1747_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_10
  simp only [← c106842, ← c106843, ← c106844] at e_1747_10
  have e_1747_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1747 (by decide)
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
  have e_1747_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_12
  simp only [← c106848, ← c106849, ← c106850] at e_1747_12
  have e_1747_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1747 (by decide)
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
  have e_1747_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_14
  simp only [← c106854, ← c106855, ← c106856] at e_1747_14
  have e_1747_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1747 (by decide)
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
  have e_1747_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_16
  simp only [← c106860, ← c106861, ← c106862] at e_1747_16
  have e_1747_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1747 (by decide)
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
  have e_1747_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1747 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_18
  simp only [← c106866, ← c106867, ← c106868] at e_1747_18
  have e_1747_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1747 (by decide)
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
  have e_1748_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_0
  simp only [← c106872, ← c106873, ← c106874] at e_1748_0
  have e_1748_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1748 (by decide)
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
  have e_1748_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_2
  simp only [← c106878, ← c106879, ← c106880] at e_1748_2
  have e_1748_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1748 (by decide)
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
  have e_1748_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_4
  simp only [← c106884, ← c106885, ← c106886] at e_1748_4
  have e_1748_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1748 (by decide)
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
  have e_1748_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_6
  simp only [← c106890, ← c106891, ← c106892] at e_1748_6
  have e_1748_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1748 (by decide)
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
  have e_1748_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_8
  simp only [← c106896, ← c106897, ← c106898] at e_1748_8
  have e_1748_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1748 (by decide)
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
  have e_1748_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_10
  simp only [← c106902, ← c106903, ← c106904] at e_1748_10
  have e_1748_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1748 (by decide)
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
  have e_1748_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_12
  simp only [← c106908, ← c106909, ← c106910] at e_1748_12
  have e_1748_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1748 (by decide)
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
  have e_1748_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_14
  simp only [← c106914, ← c106915, ← c106916] at e_1748_14
  have e_1748_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1748 (by decide)
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
  have e_1748_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_16
  simp only [← c106920, ← c106921, ← c106922] at e_1748_16
  have e_1748_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1748 (by decide)
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
  have e_1748_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1748 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_18
  simp only [← c106926, ← c106927, ← c106928] at e_1748_18
  have e_1748_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1748 (by decide)
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
  have e_1749_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_0
  simp only [← c106932, ← c106933, ← c106934] at e_1749_0
  have e_1749_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1749 (by decide)
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
  have e_1749_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_2
  simp only [← c106938, ← c106939, ← c106940] at e_1749_2
  have e_1749_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1749 (by decide)
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
  have e_1749_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_4
  simp only [← c106944, ← c106945, ← c106946] at e_1749_4
  have e_1749_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1749 (by decide)
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
  have e_1749_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_6
  simp only [← c106950, ← c106951, ← c106952] at e_1749_6
  have e_1749_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1749 (by decide)
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
  have e_1749_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_8
  simp only [← c106956, ← c106957, ← c106958] at e_1749_8
  have e_1749_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1749 (by decide)
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
  have e_1749_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_10
  simp only [← c106962, ← c106963, ← c106964] at e_1749_10
  have e_1749_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1749 (by decide)
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
  have e_1749_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_12
  simp only [← c106968, ← c106969, ← c106970] at e_1749_12
  have e_1749_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1749 (by decide)
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
  have e_1749_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_14
  simp only [← c106974, ← c106975, ← c106976] at e_1749_14
  have e_1749_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1749 (by decide)
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
  have e_1749_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_16
  simp only [← c106980, ← c106981, ← c106982] at e_1749_16
  have e_1749_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1749 (by decide)
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
  have e_1749_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1749 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_18
  simp only [← c106986, ← c106987, ← c106988] at e_1749_18
  have e_1749_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1749 (by decide)
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
  have e_1750_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_0
  simp only [← c106992, ← c106993, ← c106994] at e_1750_0
  have e_1750_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1750 (by decide)
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
  have e_1750_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_2
  simp only [← c106998, ← c106999, ← c107000] at e_1750_2
  have e_1750_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1750 (by decide)
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
  have e_1750_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_4
  simp only [← c107004, ← c107005, ← c107006] at e_1750_4
  have e_1750_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1750 (by decide)
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
  have e_1750_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_6
  simp only [← c107010, ← c107011, ← c107012] at e_1750_6
  have e_1750_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1750 (by decide)
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
  have e_1750_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_8
  simp only [← c107016, ← c107017, ← c107018] at e_1750_8
  have e_1750_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1750 (by decide)
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
  have e_1750_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_10
  simp only [← c107022, ← c107023, ← c107024] at e_1750_10
  have e_1750_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1750 (by decide)
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
  have e_1750_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_12
  simp only [← c107028, ← c107029, ← c107030] at e_1750_12
  have e_1750_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1750 (by decide)
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
  have e_1750_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_14
  simp only [← c107034, ← c107035, ← c107036] at e_1750_14
  have e_1750_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1750 (by decide)
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
  have e_1750_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_16
  simp only [← c107040, ← c107041, ← c107042] at e_1750_16
  have e_1750_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1750 (by decide)
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
  have e_1750_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1750 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_18
  simp only [← c107046, ← c107047, ← c107048] at e_1750_18
  have e_1750_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1750 (by decide)
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
  have e_1751_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_0
  simp only [← c107052, ← c107053, ← c107054] at e_1751_0
  have e_1751_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1751 (by decide)
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
  have e_1751_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_2
  simp only [← c107058, ← c107059, ← c107060] at e_1751_2
  have e_1751_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1751 (by decide)
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
  have e_1751_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_4
  simp only [← c107064, ← c107065, ← c107066] at e_1751_4
  have e_1751_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1751 (by decide)
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
  have e_1751_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_6
  simp only [← c107070, ← c107071, ← c107072] at e_1751_6
  have e_1751_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1751 (by decide)
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
  have e_1751_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_8
  simp only [← c107076, ← c107077, ← c107078] at e_1751_8
  have e_1751_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1751 (by decide)
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
  have e_1751_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_10
  simp only [← c107082, ← c107083, ← c107084] at e_1751_10
  have e_1751_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1751 (by decide)
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
  have e_1751_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_12
  simp only [← c107088, ← c107089, ← c107090] at e_1751_12
  have e_1751_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1751 (by decide)
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
  have e_1751_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_14
  simp only [← c107094, ← c107095, ← c107096] at e_1751_14
  have e_1751_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1751 (by decide)
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
  have e_1751_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_16
  simp only [← c107100, ← c107101, ← c107102] at e_1751_16
  have e_1751_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1751 (by decide)
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
  have e_1751_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1751 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_18
  simp only [← c107106, ← c107107, ← c107108] at e_1751_18
  have e_1751_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1751 (by decide)
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
  have e_1752_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_0
  simp only [← c107112, ← c107113, ← c107114] at e_1752_0
  have e_1752_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1752 (by decide)
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
  have e_1752_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_2
  simp only [← c107118, ← c107119, ← c107120] at e_1752_2
  have e_1752_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1752 (by decide)
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
  have e_1752_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_4
  simp only [← c107124, ← c107125, ← c107126] at e_1752_4
  have e_1752_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1752 (by decide)
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
  have e_1752_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_6
  simp only [← c107130, ← c107131, ← c107132] at e_1752_6
  have e_1752_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1752 (by decide)
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
  have e_1752_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_8
  simp only [← c107136, ← c107137, ← c107138] at e_1752_8
  have e_1752_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1752 (by decide)
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
  have e_1752_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_10
  simp only [← c107142, ← c107143, ← c107144] at e_1752_10
  have e_1752_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1752 (by decide)
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
  have e_1752_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_12
  simp only [← c107148, ← c107149, ← c107150] at e_1752_12
  have e_1752_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1752 (by decide)
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
  have e_1752_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_14
  simp only [← c107154, ← c107155, ← c107156] at e_1752_14
  have e_1752_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1752 (by decide)
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
  have e_1752_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_16
  simp only [← c107160, ← c107161, ← c107162] at e_1752_16
  have e_1752_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1752 (by decide)
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
  have e_1752_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1752 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_18
  simp only [← c107166, ← c107167, ← c107168] at e_1752_18
  have e_1752_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1752 (by decide)
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
  have e_1753_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_0
  simp only [← c107172, ← c107173, ← c107174] at e_1753_0
  have e_1753_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1753 (by decide)
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
  have e_1753_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_2
  simp only [← c107178, ← c107179, ← c107180] at e_1753_2
  have e_1753_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1753 (by decide)
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
  have e_1753_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_4
  simp only [← c107184, ← c107185, ← c107186] at e_1753_4
  have e_1753_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1753 (by decide)
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
  have e_1753_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_6
  simp only [← c107190, ← c107191, ← c107192] at e_1753_6
  have e_1753_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1753 (by decide)
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
  have e_1753_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_8
  simp only [← c107196, ← c107197, ← c107198] at e_1753_8
  have e_1753_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1753 (by decide)
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
  have e_1753_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_10
  simp only [← c107202, ← c107203, ← c107204] at e_1753_10
  have e_1753_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1753 (by decide)
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
  have e_1753_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_12
  simp only [← c107208, ← c107209, ← c107210] at e_1753_12
  have e_1753_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1753 (by decide)
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
  have e_1753_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_14
  simp only [← c107214, ← c107215, ← c107216] at e_1753_14
  have e_1753_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1753 (by decide)
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
  have e_1753_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_16
  simp only [← c107220, ← c107221, ← c107222] at e_1753_16
  have e_1753_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1753 (by decide)
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
  have e_1753_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1753 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_18
  simp only [← c107226, ← c107227, ← c107228] at e_1753_18
  have e_1753_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1753 (by decide)
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
  have e_1754_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_0
  simp only [← c107232, ← c107233, ← c107234] at e_1754_0
  have e_1754_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1754 (by decide)
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
  have e_1754_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_2
  simp only [← c107238, ← c107239, ← c107240] at e_1754_2
  have e_1754_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1754 (by decide)
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
  have e_1754_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_4
  simp only [← c107244, ← c107245, ← c107246] at e_1754_4
  have e_1754_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1754 (by decide)
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
  have e_1754_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_6
  simp only [← c107250, ← c107251, ← c107252] at e_1754_6
  have e_1754_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1754 (by decide)
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
  have e_1754_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_8
  simp only [← c107256, ← c107257, ← c107258] at e_1754_8
  have e_1754_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1754 (by decide)
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
  have e_1754_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_10
  simp only [← c107262, ← c107263, ← c107264] at e_1754_10
  have e_1754_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1754 (by decide)
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
  have e_1754_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_12
  simp only [← c107268, ← c107269, ← c107270] at e_1754_12
  have e_1754_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1754 (by decide)
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
  have e_1754_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_14
  simp only [← c107274, ← c107275, ← c107276] at e_1754_14
  have e_1754_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1754 (by decide)
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
  have e_1754_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_16
  simp only [← c107280, ← c107281, ← c107282] at e_1754_16
  have e_1754_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1754 (by decide)
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
  have e_1754_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1754 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_18
  simp only [← c107286, ← c107287, ← c107288] at e_1754_18
  have e_1754_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1754 (by decide)
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
  have e_1755_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_0
  simp only [← c107292, ← c107293, ← c107294] at e_1755_0
  have e_1755_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1755 (by decide)
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
  have e_1755_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_2
  simp only [← c107298, ← c107299, ← c107300] at e_1755_2
  have e_1755_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1755 (by decide)
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
  have e_1755_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_4
  simp only [← c107304, ← c107305, ← c107306] at e_1755_4
  have e_1755_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1755 (by decide)
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
  have e_1755_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_6
  simp only [← c107310, ← c107311, ← c107312] at e_1755_6
  have e_1755_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1755 (by decide)
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
  have e_1755_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_8
  simp only [← c107316, ← c107317, ← c107318] at e_1755_8
  have e_1755_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1755 (by decide)
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
  have e_1755_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_10
  simp only [← c107322, ← c107323, ← c107324] at e_1755_10
  have e_1755_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1755 (by decide)
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
  have e_1755_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_12
  simp only [← c107328, ← c107329, ← c107330] at e_1755_12
  have e_1755_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1755 (by decide)
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
  have e_1755_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_14
  simp only [← c107334, ← c107335, ← c107336] at e_1755_14
  have e_1755_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1755 (by decide)
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
  have e_1755_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_16
  simp only [← c107340, ← c107341, ← c107342] at e_1755_16
  have e_1755_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1755 (by decide)
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
  have e_1755_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1755 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_18
  simp only [← c107346, ← c107347, ← c107348] at e_1755_18
  have e_1755_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1755 (by decide)
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
  have e_1756_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_0
  simp only [← c107352, ← c107353, ← c107354] at e_1756_0
  have e_1756_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1756 (by decide)
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
  have e_1756_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_2
  simp only [← c107358, ← c107359, ← c107360] at e_1756_2
  have e_1756_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1756 (by decide)
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
  have e_1756_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_4
  simp only [← c107364, ← c107365, ← c107366] at e_1756_4
  have e_1756_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1756 (by decide)
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
  have e_1756_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_6
  simp only [← c107370, ← c107371, ← c107372] at e_1756_6
  have e_1756_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1756 (by decide)
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
  have e_1756_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_8
  simp only [← c107376, ← c107377, ← c107378] at e_1756_8
  have e_1756_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1756 (by decide)
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
  have e_1756_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_10
  simp only [← c107382, ← c107383, ← c107384] at e_1756_10
  have e_1756_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1756 (by decide)
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
  have e_1756_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_12
  simp only [← c107388, ← c107389, ← c107390] at e_1756_12
  have e_1756_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1756 (by decide)
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
  have e_1756_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_14
  simp only [← c107394, ← c107395, ← c107396] at e_1756_14
  have e_1756_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1756 (by decide)
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
  have e_1756_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_16
  simp only [← c107400, ← c107401, ← c107402] at e_1756_16
  have e_1756_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1756 (by decide)
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
  have e_1756_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1756 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_18
  simp only [← c107406, ← c107407, ← c107408] at e_1756_18
  have e_1756_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1756 (by decide)
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
  have e_1757_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_0
  simp only [← c107412, ← c107413, ← c107414] at e_1757_0
  have e_1757_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1757 (by decide)
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
  have e_1757_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_2
  simp only [← c107418, ← c107419, ← c107420] at e_1757_2
  have e_1757_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1757 (by decide)
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
  have e_1757_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_4
  simp only [← c107424, ← c107425, ← c107426] at e_1757_4
  have e_1757_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1757 (by decide)
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
  have e_1757_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_6
  simp only [← c107430, ← c107431, ← c107432] at e_1757_6
  have e_1757_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1757 (by decide)
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
  have e_1757_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_8
  simp only [← c107436, ← c107437, ← c107438] at e_1757_8
  have e_1757_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1757 (by decide)
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
  have e_1757_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_10
  simp only [← c107442, ← c107443, ← c107444] at e_1757_10
  have e_1757_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1757 (by decide)
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
  have e_1757_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_12
  simp only [← c107448, ← c107449, ← c107450] at e_1757_12
  have e_1757_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1757 (by decide)
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
  have e_1757_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_14
  simp only [← c107454, ← c107455, ← c107456] at e_1757_14
  have e_1757_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1757 (by decide)
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
  have e_1757_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_16
  simp only [← c107460, ← c107461, ← c107462] at e_1757_16
  have e_1757_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1757 (by decide)
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
  have e_1757_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1757 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_18
  simp only [← c107466, ← c107467, ← c107468] at e_1757_18
  have e_1757_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1757 (by decide)
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
  have e_1758_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_0
  simp only [← c107472, ← c107473, ← c107474] at e_1758_0
  have e_1758_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1758 (by decide)
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
  have e_1758_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_2
  simp only [← c107478, ← c107479, ← c107480] at e_1758_2
  have e_1758_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1758 (by decide)
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
  have e_1758_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_4
  simp only [← c107484, ← c107485, ← c107486] at e_1758_4
  have e_1758_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1758 (by decide)
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
  have e_1758_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_6
  simp only [← c107490, ← c107491, ← c107492] at e_1758_6
  have e_1758_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1758 (by decide)
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
  have e_1758_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_8
  simp only [← c107496, ← c107497, ← c107498] at e_1758_8
  have e_1758_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1758 (by decide)
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
  have e_1758_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_10
  simp only [← c107502, ← c107503, ← c107504] at e_1758_10
  have e_1758_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1758 (by decide)
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
  have e_1758_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_12
  simp only [← c107508, ← c107509, ← c107510] at e_1758_12
  have e_1758_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1758 (by decide)
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
  have e_1758_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_14
  simp only [← c107514, ← c107515, ← c107516] at e_1758_14
  have e_1758_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1758 (by decide)
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
  have e_1758_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_16
  simp only [← c107520, ← c107521, ← c107522] at e_1758_16
  have e_1758_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1758 (by decide)
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
  have e_1758_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1758 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_18
  simp only [← c107526, ← c107527, ← c107528] at e_1758_18
  have e_1758_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1758 (by decide)
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
  have e_1759_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_0
  simp only [← c107532, ← c107533, ← c107534] at e_1759_0
  have e_1759_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1759 (by decide)
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
  have e_1759_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_2
  simp only [← c107538, ← c107539, ← c107540] at e_1759_2
  have e_1759_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1759 (by decide)
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
  have e_1759_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_4
  simp only [← c107544, ← c107545, ← c107546] at e_1759_4
  have e_1759_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1759 (by decide)
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
  have e_1759_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_6
  simp only [← c107550, ← c107551, ← c107552] at e_1759_6
  have e_1759_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1759 (by decide)
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
  have e_1759_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_8
  simp only [← c107556, ← c107557, ← c107558] at e_1759_8
  have e_1759_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1759 (by decide)
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
  have e_1759_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_10
  simp only [← c107562, ← c107563, ← c107564] at e_1759_10
  have e_1759_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1759 (by decide)
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
  have e_1759_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_12
  simp only [← c107568, ← c107569, ← c107570] at e_1759_12
  have e_1759_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1759 (by decide)
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
  have e_1759_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_14
  simp only [← c107574, ← c107575, ← c107576] at e_1759_14
  have e_1759_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1759 (by decide)
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
  have e_1759_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_16
  simp only [← c107580, ← c107581, ← c107582] at e_1759_16
  have e_1759_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1759 (by decide)
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
  have e_1759_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1759 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_18
  simp only [← c107586, ← c107587, ← c107588] at e_1759_18
  have e_1759_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1759 (by decide)
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
  have e_1760_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_0
  simp only [← c107592, ← c107593, ← c107594] at e_1760_0
  have e_1760_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1760 (by decide)
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
  have e_1760_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_2
  simp only [← c107598, ← c107599, ← c107600] at e_1760_2
  have e_1760_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1760 (by decide)
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
  have e_1760_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_4
  simp only [← c107604, ← c107605, ← c107606] at e_1760_4
  have e_1760_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1760 (by decide)
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
  have e_1760_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_6
  simp only [← c107610, ← c107611, ← c107612] at e_1760_6
  have e_1760_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1760 (by decide)
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
  have e_1760_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_8
  simp only [← c107616, ← c107617, ← c107618] at e_1760_8
  have e_1760_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1760 (by decide)
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
  have e_1760_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_10
  simp only [← c107622, ← c107623, ← c107624] at e_1760_10
  have e_1760_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1760 (by decide)
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
  have e_1760_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_12
  simp only [← c107628, ← c107629, ← c107630] at e_1760_12
  have e_1760_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1760 (by decide)
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
  have e_1760_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_14
  simp only [← c107634, ← c107635, ← c107636] at e_1760_14
  have e_1760_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1760 (by decide)
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
  have e_1760_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_16
  simp only [← c107640, ← c107641, ← c107642] at e_1760_16
  have e_1760_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1760 (by decide)
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
  have e_1760_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1760 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_18
  simp only [← c107646, ← c107647, ← c107648] at e_1760_18
  have e_1760_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1760 (by decide)
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
  have e_1761_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_0
  simp only [← c107652, ← c107653, ← c107654] at e_1761_0
  have e_1761_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1761 (by decide)
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
  have e_1761_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_2
  simp only [← c107658, ← c107659, ← c107660] at e_1761_2
  have e_1761_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1761 (by decide)
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
  have e_1761_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_4
  simp only [← c107664, ← c107665, ← c107666] at e_1761_4
  have e_1761_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1761 (by decide)
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
  have e_1761_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_6
  simp only [← c107670, ← c107671, ← c107672] at e_1761_6
  have e_1761_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1761 (by decide)
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
  have e_1761_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_8
  simp only [← c107676, ← c107677, ← c107678] at e_1761_8
  have e_1761_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1761 (by decide)
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
  have e_1761_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_10
  simp only [← c107682, ← c107683, ← c107684] at e_1761_10
  have e_1761_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1761 (by decide)
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
  have e_1761_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_12
  simp only [← c107688, ← c107689, ← c107690] at e_1761_12
  have e_1761_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1761 (by decide)
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
  have e_1761_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_14
  simp only [← c107694, ← c107695, ← c107696] at e_1761_14
  have e_1761_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1761 (by decide)
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
  have e_1761_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_16
  simp only [← c107700, ← c107701, ← c107702] at e_1761_16
  have e_1761_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1761 (by decide)
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
  have e_1761_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1761 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_18
  simp only [← c107706, ← c107707, ← c107708] at e_1761_18
  have e_1761_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1761 (by decide)
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
  have e_1762_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_0
  simp only [← c107712, ← c107713, ← c107714] at e_1762_0
  have e_1762_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1762 (by decide)
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
  have e_1762_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_2
  simp only [← c107718, ← c107719, ← c107720] at e_1762_2
  have e_1762_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1762 (by decide)
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
  have e_1762_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_4
  simp only [← c107724, ← c107725, ← c107726] at e_1762_4
  have e_1762_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1762 (by decide)
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
  have e_1762_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_6
  simp only [← c107730, ← c107731, ← c107732] at e_1762_6
  have e_1762_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1762 (by decide)
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
  have e_1762_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_8
  simp only [← c107736, ← c107737, ← c107738] at e_1762_8
  have e_1762_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1762 (by decide)
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
  have e_1762_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_10
  simp only [← c107742, ← c107743, ← c107744] at e_1762_10
  have e_1762_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1762 (by decide)
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
  have e_1762_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_12
  simp only [← c107748, ← c107749, ← c107750] at e_1762_12
  have e_1762_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1762 (by decide)
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
  have e_1762_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_14
  simp only [← c107754, ← c107755, ← c107756] at e_1762_14
  have e_1762_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1762 (by decide)
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
  have e_1762_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_16
  simp only [← c107760, ← c107761, ← c107762] at e_1762_16
  have e_1762_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1762 (by decide)
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
  have e_1762_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1762 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_18
  simp only [← c107766, ← c107767, ← c107768] at e_1762_18
  have e_1762_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1762 (by decide)
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
  have e_1763_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_0
  simp only [← c107772, ← c107773, ← c107774] at e_1763_0
  have e_1763_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1763 (by decide)
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
  have e_1763_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_2
  simp only [← c107778, ← c107779, ← c107780] at e_1763_2
  have e_1763_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1763 (by decide)
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
  have e_1763_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_4
  simp only [← c107784, ← c107785, ← c107786] at e_1763_4
  have e_1763_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1763 (by decide)
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
  have e_1763_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_6
  simp only [← c107790, ← c107791, ← c107792] at e_1763_6
  have e_1763_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1763 (by decide)
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
  have e_1763_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_8
  simp only [← c107796, ← c107797, ← c107798] at e_1763_8
  have e_1763_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1763 (by decide)
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
  have e_1763_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_10
  simp only [← c107802, ← c107803, ← c107804] at e_1763_10
  have e_1763_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1763 (by decide)
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
  have e_1763_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_12
  simp only [← c107808, ← c107809, ← c107810] at e_1763_12
  have e_1763_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1763 (by decide)
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
  have e_1763_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_14
  simp only [← c107814, ← c107815, ← c107816] at e_1763_14
  have e_1763_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1763 (by decide)
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
  have e_1763_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_16
  simp only [← c107820, ← c107821, ← c107822] at e_1763_16
  have e_1763_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1763 (by decide)
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
  have e_1763_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1763 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_18
  simp only [← c107826, ← c107827, ← c107828] at e_1763_18
  have e_1763_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1763 (by decide)
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
  have e_1764_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_0
  simp only [← c107832, ← c107833, ← c107834] at e_1764_0
  have e_1764_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1764 (by decide)
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
  have e_1764_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_2
  simp only [← c107838, ← c107839, ← c107840] at e_1764_2
  have e_1764_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1764 (by decide)
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
  have e_1764_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_4
  simp only [← c107844, ← c107845, ← c107846] at e_1764_4
  have e_1764_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1764 (by decide)
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
  have e_1764_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_6
  simp only [← c107850, ← c107851, ← c107852] at e_1764_6
  have e_1764_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1764 (by decide)
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
  have e_1764_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_8
  simp only [← c107856, ← c107857, ← c107858] at e_1764_8
  have e_1764_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1764 (by decide)
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
  have e_1764_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_10
  simp only [← c107862, ← c107863, ← c107864] at e_1764_10
  have e_1764_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1764 (by decide)
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
  have e_1764_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_12
  simp only [← c107868, ← c107869, ← c107870] at e_1764_12
  have e_1764_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1764 (by decide)
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
  have e_1764_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_14
  simp only [← c107874, ← c107875, ← c107876] at e_1764_14
  have e_1764_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1764 (by decide)
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
  have e_1764_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_16
  simp only [← c107880, ← c107881, ← c107882] at e_1764_16
  have e_1764_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1764 (by decide)
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
  have e_1764_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1764 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_18
  simp only [← c107886, ← c107887, ← c107888] at e_1764_18
  have e_1764_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1764 (by decide)
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
  have e_1765_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_0
  simp only [← c107892, ← c107893, ← c107894] at e_1765_0
  have e_1765_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1765 (by decide)
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
  have e_1765_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_2
  simp only [← c107898, ← c107899, ← c107900] at e_1765_2
  have e_1765_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1765 (by decide)
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
  have e_1765_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_4
  simp only [← c107904, ← c107905, ← c107906] at e_1765_4
  have e_1765_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1765 (by decide)
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
  have e_1765_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_6
  simp only [← c107910, ← c107911, ← c107912] at e_1765_6
  have e_1765_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1765 (by decide)
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
  have e_1765_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_8
  simp only [← c107916, ← c107917, ← c107918] at e_1765_8
  have e_1765_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1765 (by decide)
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
  have e_1765_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_10
  simp only [← c107922, ← c107923, ← c107924] at e_1765_10
  have e_1765_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1765 (by decide)
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
  have e_1765_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_12
  simp only [← c107928, ← c107929, ← c107930] at e_1765_12
  have e_1765_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1765 (by decide)
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
  have e_1765_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_14
  simp only [← c107934, ← c107935, ← c107936] at e_1765_14
  have e_1765_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1765 (by decide)
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
  have e_1765_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_16
  simp only [← c107940, ← c107941, ← c107942] at e_1765_16
  have e_1765_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1765 (by decide)
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
  have e_1765_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1765 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_18
  simp only [← c107946, ← c107947, ← c107948] at e_1765_18
  have e_1765_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1765 (by decide)
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
  have e_1766_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_0
  simp only [← c107952, ← c107953, ← c107954] at e_1766_0
  have e_1766_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1766 (by decide)
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
  have e_1766_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_2
  simp only [← c107958, ← c107959, ← c107960] at e_1766_2
  have e_1766_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1766 (by decide)
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
  have e_1766_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_4
  simp only [← c107964, ← c107965, ← c107966] at e_1766_4
  have e_1766_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1766 (by decide)
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
  have e_1766_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_6
  simp only [← c107970, ← c107971, ← c107972] at e_1766_6
  have e_1766_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1766 (by decide)
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
  have e_1766_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_8
  simp only [← c107976, ← c107977, ← c107978] at e_1766_8
  have e_1766_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1766 (by decide)
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
  have e_1766_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_10
  simp only [← c107982, ← c107983, ← c107984] at e_1766_10
  have e_1766_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1766 (by decide)
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
  have e_1766_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_12
  simp only [← c107988, ← c107989, ← c107990] at e_1766_12
  have e_1766_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1766 (by decide)
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
  have e_1766_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_14
  simp only [← c107994, ← c107995, ← c107996] at e_1766_14
  have e_1766_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1766 (by decide)
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
  have e_1766_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_16
  simp only [← c108000, ← c108001, ← c108002] at e_1766_16
  have e_1766_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1766 (by decide)
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
  have e_1766_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1766 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_18
  simp only [← c108006, ← c108007, ← c108008] at e_1766_18
  have e_1766_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1766 (by decide)
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
  have e_1767_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_0
  simp only [← c108012, ← c108013, ← c108014] at e_1767_0
  have e_1767_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1767 (by decide)
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
  have e_1767_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_2
  simp only [← c108018, ← c108019, ← c108020] at e_1767_2
  have e_1767_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1767 (by decide)
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
  have e_1767_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_4
  simp only [← c108024, ← c108025, ← c108026] at e_1767_4
  have e_1767_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1767 (by decide)
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
  have e_1767_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_6
  simp only [← c108030, ← c108031, ← c108032] at e_1767_6
  have e_1767_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1767 (by decide)
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
  have e_1767_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_8
  simp only [← c108036, ← c108037, ← c108038] at e_1767_8
  have e_1767_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1767 (by decide)
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
  have e_1767_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_10
  simp only [← c108042, ← c108043, ← c108044] at e_1767_10
  have e_1767_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1767 (by decide)
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
  have e_1767_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_12
  simp only [← c108048, ← c108049, ← c108050] at e_1767_12
  have e_1767_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1767 (by decide)
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
  have e_1767_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_14
  simp only [← c108054, ← c108055, ← c108056] at e_1767_14
  have e_1767_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1767 (by decide)
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
  have e_1767_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_16
  simp only [← c108060, ← c108061, ← c108062] at e_1767_16
  have e_1767_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1767 (by decide)
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
  have e_1767_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1767 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_18
  simp only [← c108066, ← c108067, ← c108068] at e_1767_18
  have e_1767_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1767 (by decide)
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
  have e_1768_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_0
  simp only [← c108072, ← c108073, ← c108074] at e_1768_0
  have e_1768_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1768 (by decide)
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
  have e_1768_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_2
  simp only [← c108078, ← c108079, ← c108080] at e_1768_2
  have e_1768_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1768 (by decide)
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
  have e_1768_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_4
  simp only [← c108084, ← c108085, ← c108086] at e_1768_4
  have e_1768_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1768 (by decide)
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
  have e_1768_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_6
  simp only [← c108090, ← c108091, ← c108092] at e_1768_6
  have e_1768_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1768 (by decide)
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
  have e_1768_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_8
  simp only [← c108096, ← c108097, ← c108098] at e_1768_8
  have e_1768_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1768 (by decide)
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
  have e_1768_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_10
  simp only [← c108102, ← c108103, ← c108104] at e_1768_10
  have e_1768_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1768 (by decide)
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
  have e_1768_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_12
  simp only [← c108108, ← c108109, ← c108110] at e_1768_12
  have e_1768_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1768 (by decide)
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
  have e_1768_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_14
  simp only [← c108114, ← c108115, ← c108116] at e_1768_14
  have e_1768_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1768 (by decide)
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
  have e_1768_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_16
  simp only [← c108120, ← c108121, ← c108122] at e_1768_16
  have e_1768_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1768 (by decide)
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
  have e_1768_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1768 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_18
  simp only [← c108126, ← c108127, ← c108128] at e_1768_18
  have e_1768_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1768 (by decide)
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
  have e_1769_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_0
  simp only [← c108132, ← c108133, ← c108134] at e_1769_0
  have e_1769_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1769 (by decide)
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
  have e_1769_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_2
  simp only [← c108138, ← c108139, ← c108140] at e_1769_2
  have e_1769_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1769 (by decide)
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
  have e_1769_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_4
  simp only [← c108144, ← c108145, ← c108146] at e_1769_4
  have e_1769_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1769 (by decide)
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
  have e_1769_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_6
  simp only [← c108150, ← c108151, ← c108152] at e_1769_6
  have e_1769_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1769 (by decide)
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
  have e_1769_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_8
  simp only [← c108156, ← c108157, ← c108158] at e_1769_8
  have e_1769_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1769 (by decide)
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
  have e_1769_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_10
  simp only [← c108162, ← c108163, ← c108164] at e_1769_10
  have e_1769_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1769 (by decide)
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
  have e_1769_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_12
  simp only [← c108168, ← c108169, ← c108170] at e_1769_12
  have e_1769_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1769 (by decide)
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
  have e_1769_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_14
  simp only [← c108174, ← c108175, ← c108176] at e_1769_14
  have e_1769_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1769 (by decide)
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
  have e_1769_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_16
  simp only [← c108180, ← c108181, ← c108182] at e_1769_16
  have e_1769_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1769 (by decide)
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
  have e_1769_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1769 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_18
  simp only [← c108186, ← c108187, ← c108188] at e_1769_18
  have e_1769_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1769 (by decide)
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
  have e_1770_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_0
  simp only [← c108192, ← c108193, ← c108194] at e_1770_0
  have e_1770_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1770 (by decide)
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
  have e_1770_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_2
  simp only [← c108198, ← c108199, ← c108200] at e_1770_2
  have e_1770_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1770 (by decide)
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
  have e_1770_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_4
  simp only [← c108204, ← c108205, ← c108206] at e_1770_4
  have e_1770_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1770 (by decide)
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
  have e_1770_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_6
  simp only [← c108210, ← c108211, ← c108212] at e_1770_6
  have e_1770_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1770 (by decide)
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
  have e_1770_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_8
  simp only [← c108216, ← c108217, ← c108218] at e_1770_8
  have e_1770_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1770 (by decide)
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
  have e_1770_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_10
  simp only [← c108222, ← c108223, ← c108224] at e_1770_10
  have e_1770_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1770 (by decide)
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
  have e_1770_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_12
  simp only [← c108228, ← c108229, ← c108230] at e_1770_12
  have e_1770_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1770 (by decide)
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
  have e_1770_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_14
  simp only [← c108234, ← c108235, ← c108236] at e_1770_14
  have e_1770_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1770 (by decide)
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
  have e_1770_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_16
  simp only [← c108240, ← c108241, ← c108242] at e_1770_16
  have e_1770_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1770 (by decide)
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
  have e_1770_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1770 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_18
  simp only [← c108246, ← c108247, ← c108248] at e_1770_18
  have e_1770_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1770 (by decide)
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
  have e_1771_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_0
  simp only [← c108252, ← c108253, ← c108254] at e_1771_0
  have e_1771_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1771 (by decide)
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
  have e_1771_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_2
  simp only [← c108258, ← c108259, ← c108260] at e_1771_2
  have e_1771_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1771 (by decide)
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
  have e_1771_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_4
  simp only [← c108264, ← c108265, ← c108266] at e_1771_4
  have e_1771_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1771 (by decide)
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
  have e_1771_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_6
  simp only [← c108270, ← c108271, ← c108272] at e_1771_6
  have e_1771_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1771 (by decide)
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
  have e_1771_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_8
  simp only [← c108276, ← c108277, ← c108278] at e_1771_8
  have e_1771_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1771 (by decide)
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
  have e_1771_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_10
  simp only [← c108282, ← c108283, ← c108284] at e_1771_10
  have e_1771_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1771 (by decide)
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
  have e_1771_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_12
  simp only [← c108288, ← c108289, ← c108290] at e_1771_12
  have e_1771_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1771 (by decide)
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
  have e_1771_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_14
  simp only [← c108294, ← c108295, ← c108296] at e_1771_14
  have e_1771_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1771 (by decide)
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
  have e_1771_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_16
  simp only [← c108300, ← c108301, ← c108302] at e_1771_16
  have e_1771_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1771 (by decide)
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
  have e_1771_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1771 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_18
  simp only [← c108306, ← c108307, ← c108308] at e_1771_18
  have e_1771_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1771 (by decide)
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
  have e_1772_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_0
  simp only [← c108312, ← c108313, ← c108314] at e_1772_0
  have e_1772_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1772 (by decide)
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
  have e_1772_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_2
  simp only [← c108318, ← c108319, ← c108320] at e_1772_2
  have e_1772_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1772 (by decide)
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
  have e_1772_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_4
  simp only [← c108324, ← c108325, ← c108326] at e_1772_4
  have e_1772_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1772 (by decide)
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
  have e_1772_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_6
  simp only [← c108330, ← c108331, ← c108332] at e_1772_6
  have e_1772_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1772 (by decide)
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
  have e_1772_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_8
  simp only [← c108336, ← c108337, ← c108338] at e_1772_8
  have e_1772_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1772 (by decide)
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
  have e_1772_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_10
  simp only [← c108342, ← c108343, ← c108344] at e_1772_10
  have e_1772_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1772 (by decide)
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
  have e_1772_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_12
  simp only [← c108348, ← c108349, ← c108350] at e_1772_12
  have e_1772_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1772 (by decide)
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
  have e_1772_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_14
  simp only [← c108354, ← c108355, ← c108356] at e_1772_14
  have e_1772_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1772 (by decide)
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
  have e_1772_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_16
  simp only [← c108360, ← c108361, ← c108362] at e_1772_16
  have e_1772_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1772 (by decide)
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
  have e_1772_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1772 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_18
  simp only [← c108366, ← c108367, ← c108368] at e_1772_18
  have e_1772_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1772 (by decide)
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
  have e_1773_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_0
  simp only [← c108372, ← c108373, ← c108374] at e_1773_0
  have e_1773_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1773 (by decide)
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
  have e_1773_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_2
  simp only [← c108378, ← c108379, ← c108380] at e_1773_2
  have e_1773_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1773 (by decide)
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
  have e_1773_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_4
  simp only [← c108384, ← c108385, ← c108386] at e_1773_4
  have e_1773_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1773 (by decide)
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
  have e_1773_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_6
  simp only [← c108390, ← c108391, ← c108392] at e_1773_6
  have e_1773_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1773 (by decide)
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
  have e_1773_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_8
  simp only [← c108396, ← c108397, ← c108398] at e_1773_8
  have e_1773_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1773 (by decide)
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
  have e_1773_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_10
  simp only [← c108402, ← c108403, ← c108404] at e_1773_10
  have e_1773_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1773 (by decide)
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
  have e_1773_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_12
  simp only [← c108408, ← c108409, ← c108410] at e_1773_12
  have e_1773_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1773 (by decide)
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
  have e_1773_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_14
  simp only [← c108414, ← c108415, ← c108416] at e_1773_14
  have e_1773_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1773 (by decide)
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
  have e_1773_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_16
  simp only [← c108420, ← c108421, ← c108422] at e_1773_16
  have e_1773_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1773 (by decide)
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
  have e_1773_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1773 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_18
  simp only [← c108426, ← c108427, ← c108428] at e_1773_18
  have e_1773_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1773 (by decide)
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
  have e_1774_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_0
  simp only [← c108432, ← c108433, ← c108434] at e_1774_0
  have e_1774_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1774 (by decide)
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
  have e_1774_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_2
  simp only [← c108438, ← c108439, ← c108440] at e_1774_2
  have e_1774_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1774 (by decide)
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
  have e_1774_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_4
  simp only [← c108444, ← c108445, ← c108446] at e_1774_4
  have e_1774_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1774 (by decide)
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
  have e_1774_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_6
  simp only [← c108450, ← c108451, ← c108452] at e_1774_6
  have e_1774_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1774 (by decide)
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
  have e_1774_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_8
  simp only [← c108456, ← c108457, ← c108458] at e_1774_8
  have e_1774_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1774 (by decide)
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
  have e_1774_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_10
  simp only [← c108462, ← c108463, ← c108464] at e_1774_10
  have e_1774_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1774 (by decide)
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
  have e_1774_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_12
  simp only [← c108468, ← c108469, ← c108470] at e_1774_12
  have e_1774_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1774 (by decide)
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
  have e_1774_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_14
  simp only [← c108474, ← c108475, ← c108476] at e_1774_14
  have e_1774_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1774 (by decide)
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
  have e_1774_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_16
  simp only [← c108480, ← c108481, ← c108482] at e_1774_16
  have e_1774_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1774 (by decide)
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
  have e_1774_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1774 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_18
  simp only [← c108486, ← c108487, ← c108488] at e_1774_18
  have e_1774_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1774 (by decide)
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
  have e_1775_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_0
  simp only [← c108492, ← c108493, ← c108494] at e_1775_0
  have e_1775_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1775 (by decide)
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
  have e_1775_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_2
  simp only [← c108498, ← c108499, ← c108500] at e_1775_2
  have e_1775_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1775 (by decide)
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
  have e_1775_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_4
  simp only [← c108504, ← c108505, ← c108506] at e_1775_4
  have e_1775_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1775 (by decide)
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
  have e_1775_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_6
  simp only [← c108510, ← c108511, ← c108512] at e_1775_6
  have e_1775_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1775 (by decide)
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
  have e_1775_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_8
  simp only [← c108516, ← c108517, ← c108518] at e_1775_8
  have e_1775_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1775 (by decide)
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
  have e_1775_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_10
  simp only [← c108522, ← c108523, ← c108524] at e_1775_10
  have e_1775_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1775 (by decide)
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
  have e_1775_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_12
  simp only [← c108528, ← c108529, ← c108530] at e_1775_12
  have e_1775_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1775 (by decide)
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
  have e_1775_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_14
  simp only [← c108534, ← c108535, ← c108536] at e_1775_14
  have e_1775_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1775 (by decide)
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
  have e_1775_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_16
  simp only [← c108540, ← c108541, ← c108542] at e_1775_16
  have e_1775_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1775 (by decide)
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
  have e_1775_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1775 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_18
  simp only [← c108546, ← c108547, ← c108548] at e_1775_18
  have e_1775_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1775 (by decide)
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
  have e_1776_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_0
  simp only [← c108552, ← c108553, ← c108554] at e_1776_0
  have e_1776_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1776 (by decide)
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
  have e_1776_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_2
  simp only [← c108558, ← c108559, ← c108560] at e_1776_2
  have e_1776_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1776 (by decide)
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
  have e_1776_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_4
  simp only [← c108564, ← c108565, ← c108566] at e_1776_4
  have e_1776_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1776 (by decide)
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
  have e_1776_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_6
  simp only [← c108570, ← c108571, ← c108572] at e_1776_6
  have e_1776_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1776 (by decide)
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
  have e_1776_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_8
  simp only [← c108576, ← c108577, ← c108578] at e_1776_8
  have e_1776_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1776 (by decide)
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
  have e_1776_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_10
  simp only [← c108582, ← c108583, ← c108584] at e_1776_10
  have e_1776_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1776 (by decide)
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
  have e_1776_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_12
  simp only [← c108588, ← c108589, ← c108590] at e_1776_12
  have e_1776_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1776 (by decide)
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
  have e_1776_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_14
  simp only [← c108594, ← c108595, ← c108596] at e_1776_14
  have e_1776_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1776 (by decide)
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
  have e_1776_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_16
  simp only [← c108600, ← c108601, ← c108602] at e_1776_16
  have e_1776_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1776 (by decide)
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
  have e_1776_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1776 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_18
  simp only [← c108606, ← c108607, ← c108608] at e_1776_18
  have e_1776_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1776 (by decide)
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
  have e_1777_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_0
  simp only [← c108612, ← c108613, ← c108614] at e_1777_0
  have e_1777_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1777 (by decide)
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
  have e_1777_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_2
  simp only [← c108618, ← c108619, ← c108620] at e_1777_2
  have e_1777_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1777 (by decide)
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
  have e_1777_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_4
  simp only [← c108624, ← c108625, ← c108626] at e_1777_4
  have e_1777_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1777 (by decide)
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
  have e_1777_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_6
  simp only [← c108630, ← c108631, ← c108632] at e_1777_6
  have e_1777_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1777 (by decide)
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
  have e_1777_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_8
  simp only [← c108636, ← c108637, ← c108638] at e_1777_8
  have e_1777_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1777 (by decide)
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
  have e_1777_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_10
  simp only [← c108642, ← c108643, ← c108644] at e_1777_10
  have e_1777_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1777 (by decide)
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
  have e_1777_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_12
  simp only [← c108648, ← c108649, ← c108650] at e_1777_12
  have e_1777_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1777 (by decide)
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
  have e_1777_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_14
  simp only [← c108654, ← c108655, ← c108656] at e_1777_14
  have e_1777_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1777 (by decide)
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
  have e_1777_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_16
  simp only [← c108660, ← c108661, ← c108662] at e_1777_16
  have e_1777_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1777 (by decide)
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
  have e_1777_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1777 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_18
  simp only [← c108666, ← c108667, ← c108668] at e_1777_18
  have e_1777_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1777 (by decide)
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
  have e_1778_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_0
  simp only [← c108672, ← c108673, ← c108674] at e_1778_0
  have e_1778_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1778 (by decide)
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
  have e_1778_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_2
  simp only [← c108678, ← c108679, ← c108680] at e_1778_2
  have e_1778_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1778 (by decide)
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
  have e_1778_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_4
  simp only [← c108684, ← c108685, ← c108686] at e_1778_4
  have e_1778_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1778 (by decide)
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
  have e_1778_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_6
  simp only [← c108690, ← c108691, ← c108692] at e_1778_6
  have e_1778_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1778 (by decide)
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
  have e_1778_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_8
  simp only [← c108696, ← c108697, ← c108698] at e_1778_8
  have e_1778_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1778 (by decide)
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
  have e_1778_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_10
  simp only [← c108702, ← c108703, ← c108704] at e_1778_10
  have e_1778_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1778 (by decide)
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
  have e_1778_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_12
  simp only [← c108708, ← c108709, ← c108710] at e_1778_12
  have e_1778_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1778 (by decide)
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
  have e_1778_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_14
  simp only [← c108714, ← c108715, ← c108716] at e_1778_14
  have e_1778_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1778 (by decide)
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
  have e_1778_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_16
  simp only [← c108720, ← c108721, ← c108722] at e_1778_16
  have e_1778_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1778 (by decide)
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
  have e_1778_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1778 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_18
  simp only [← c108726, ← c108727, ← c108728] at e_1778_18
  have e_1778_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1778 (by decide)
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
  have e_1779_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_0
  simp only [← c108732, ← c108733, ← c108734] at e_1779_0
  have e_1779_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1779 (by decide)
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
  have e_1779_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_2
  simp only [← c108738, ← c108739, ← c108740] at e_1779_2
  have e_1779_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1779 (by decide)
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
  have e_1779_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_4
  simp only [← c108744, ← c108745, ← c108746] at e_1779_4
  have e_1779_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1779 (by decide)
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
  have e_1779_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_6
  simp only [← c108750, ← c108751, ← c108752] at e_1779_6
  have e_1779_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1779 (by decide)
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
  have e_1779_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_8
  simp only [← c108756, ← c108757, ← c108758] at e_1779_8
  have e_1779_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1779 (by decide)
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
  have e_1779_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_10
  simp only [← c108762, ← c108763, ← c108764] at e_1779_10
  have e_1779_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1779 (by decide)
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
  have e_1779_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_12
  simp only [← c108768, ← c108769, ← c108770] at e_1779_12
  have e_1779_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1779 (by decide)
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
  have e_1779_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_14
  simp only [← c108774, ← c108775, ← c108776] at e_1779_14
  have e_1779_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1779 (by decide)
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
  have e_1779_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_16
  simp only [← c108780, ← c108781, ← c108782] at e_1779_16
  have e_1779_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1779 (by decide)
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
  have e_1779_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1779 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_18
  simp only [← c108786, ← c108787, ← c108788] at e_1779_18
  have e_1779_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1779 (by decide)
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
  have e_1780_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_0
  simp only [← c108792, ← c108793, ← c108794] at e_1780_0
  have e_1780_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1780 (by decide)
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
  have e_1780_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_2
  simp only [← c108798, ← c108799, ← c108800] at e_1780_2
  have e_1780_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1780 (by decide)
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
  have e_1780_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_4
  simp only [← c108804, ← c108805, ← c108806] at e_1780_4
  have e_1780_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1780 (by decide)
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
  have e_1780_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_6
  simp only [← c108810, ← c108811, ← c108812] at e_1780_6
  have e_1780_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1780 (by decide)
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
  have e_1780_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_8
  simp only [← c108816, ← c108817, ← c108818] at e_1780_8
  have e_1780_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1780 (by decide)
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
  have e_1780_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_10
  simp only [← c108822, ← c108823, ← c108824] at e_1780_10
  have e_1780_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1780 (by decide)
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
  have e_1780_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_12
  simp only [← c108828, ← c108829, ← c108830] at e_1780_12
  have e_1780_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1780 (by decide)
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
  have e_1780_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_14
  simp only [← c108834, ← c108835, ← c108836] at e_1780_14
  have e_1780_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1780 (by decide)
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
  have e_1780_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_16
  simp only [← c108840, ← c108841, ← c108842] at e_1780_16
  have e_1780_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1780 (by decide)
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
  have e_1780_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1780 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_18
  simp only [← c108846, ← c108847, ← c108848] at e_1780_18
  have e_1780_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1780 (by decide)
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
  have e_1781_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_0
  simp only [← c108852, ← c108853, ← c108854] at e_1781_0
  have e_1781_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1781 (by decide)
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
  have e_1781_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_2
  simp only [← c108858, ← c108859, ← c108860] at e_1781_2
  have e_1781_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1781 (by decide)
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
  have e_1781_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_4
  simp only [← c108864, ← c108865, ← c108866] at e_1781_4
  have e_1781_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1781 (by decide)
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
  have e_1781_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_6
  simp only [← c108870, ← c108871, ← c108872] at e_1781_6
  have e_1781_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1781 (by decide)
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
  have e_1781_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_8
  simp only [← c108876, ← c108877, ← c108878] at e_1781_8
  have e_1781_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1781 (by decide)
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
  have e_1781_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_10
  simp only [← c108882, ← c108883, ← c108884] at e_1781_10
  have e_1781_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1781 (by decide)
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
  have e_1781_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_12
  simp only [← c108888, ← c108889, ← c108890] at e_1781_12
  have e_1781_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1781 (by decide)
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
  have e_1781_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_14
  simp only [← c108894, ← c108895, ← c108896] at e_1781_14
  have e_1781_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1781 (by decide)
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
  have e_1781_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_16
  simp only [← c108900, ← c108901, ← c108902] at e_1781_16
  have e_1781_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1781 (by decide)
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
  have e_1781_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1781 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_18
  simp only [← c108906, ← c108907, ← c108908] at e_1781_18
  have e_1781_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1781 (by decide)
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
  have e_1782_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_0
  simp only [← c108912, ← c108913, ← c108914] at e_1782_0
  have e_1782_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1782 (by decide)
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
  have e_1782_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_2
  simp only [← c108918, ← c108919, ← c108920] at e_1782_2
  have e_1782_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1782 (by decide)
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
  have e_1782_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_4
  simp only [← c108924, ← c108925, ← c108926] at e_1782_4
  have e_1782_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1782 (by decide)
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
  have e_1782_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_6
  simp only [← c108930, ← c108931, ← c108932] at e_1782_6
  have e_1782_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1782 (by decide)
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
  have e_1782_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_8
  simp only [← c108936, ← c108937, ← c108938] at e_1782_8
  have e_1782_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1782 (by decide)
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
  have e_1782_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_10
  simp only [← c108942, ← c108943, ← c108944] at e_1782_10
  have e_1782_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1782 (by decide)
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
  have e_1782_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_12
  simp only [← c108948, ← c108949, ← c108950] at e_1782_12
  have e_1782_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1782 (by decide)
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
  have e_1782_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_14
  simp only [← c108954, ← c108955, ← c108956] at e_1782_14
  have e_1782_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1782 (by decide)
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
  have e_1782_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_16
  simp only [← c108960, ← c108961, ← c108962] at e_1782_16
  have e_1782_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1782 (by decide)
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
  have e_1782_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1782 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_18
  simp only [← c108966, ← c108967, ← c108968] at e_1782_18
  have e_1782_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1782 (by decide)
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
  have e_1783_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_0
  simp only [← c108972, ← c108973, ← c108974] at e_1783_0
  have e_1783_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1783 (by decide)
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
  have e_1783_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_2
  simp only [← c108978, ← c108979, ← c108980] at e_1783_2
  have e_1783_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1783 (by decide)
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
  have e_1783_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_4
  simp only [← c108984, ← c108985, ← c108986] at e_1783_4
  have e_1783_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1783 (by decide)
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
  have e_1783_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_6
  simp only [← c108990, ← c108991, ← c108992] at e_1783_6
  have e_1783_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1783 (by decide)
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
  have e_1783_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_8
  simp only [← c108996, ← c108997, ← c108998] at e_1783_8
  have e_1783_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1783 (by decide)
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
  have e_1783_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_10
  simp only [← c109002, ← c109003, ← c109004] at e_1783_10
  have e_1783_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1783 (by decide)
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
  have e_1783_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_12
  simp only [← c109008, ← c109009, ← c109010] at e_1783_12
  have e_1783_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1783 (by decide)
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
  have e_1783_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_14
  simp only [← c109014, ← c109015, ← c109016] at e_1783_14
  have e_1783_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1783 (by decide)
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
  have e_1783_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_16
  simp only [← c109020, ← c109021, ← c109022] at e_1783_16
  have e_1783_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1783 (by decide)
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
  have e_1783_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1783 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_18
  simp only [← c109026, ← c109027, ← c109028] at e_1783_18
  have e_1783_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1783 (by decide)
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
  have e_1784_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_0
  simp only [← c109032, ← c109033, ← c109034] at e_1784_0
  have e_1784_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1784 (by decide)
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
  have e_1784_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_2
  simp only [← c109038, ← c109039, ← c109040] at e_1784_2
  have e_1784_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1784 (by decide)
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
  have e_1784_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_4
  simp only [← c109044, ← c109045, ← c109046] at e_1784_4
  have e_1784_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1784 (by decide)
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
  have e_1784_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_6
  simp only [← c109050, ← c109051, ← c109052] at e_1784_6
  have e_1784_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1784 (by decide)
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
  have e_1784_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_8
  simp only [← c109056, ← c109057, ← c109058] at e_1784_8
  have e_1784_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1784 (by decide)
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
  have e_1784_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_10
  simp only [← c109062, ← c109063, ← c109064] at e_1784_10
  have e_1784_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1784 (by decide)
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
  have e_1784_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_12
  simp only [← c109068, ← c109069, ← c109070] at e_1784_12
  have e_1784_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1784 (by decide)
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
  have e_1784_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_14
  simp only [← c109074, ← c109075, ← c109076] at e_1784_14
  have e_1784_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1784 (by decide)
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
  have e_1784_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_16
  simp only [← c109080, ← c109081, ← c109082] at e_1784_16
  have e_1784_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1784 (by decide)
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
  have e_1784_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1784 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_18
  simp only [← c109086, ← c109087, ← c109088] at e_1784_18
  have e_1784_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1784 (by decide)
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
  have e_1785_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_0
  simp only [← c109092, ← c109093, ← c109094] at e_1785_0
  have e_1785_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1785 (by decide)
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
  have e_1785_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_2
  simp only [← c109098, ← c109099, ← c109100] at e_1785_2
  have e_1785_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1785 (by decide)
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
  have e_1785_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_4
  simp only [← c109104, ← c109105, ← c109106] at e_1785_4
  have e_1785_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1785 (by decide)
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
  have e_1785_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_6
  simp only [← c109110, ← c109111, ← c109112] at e_1785_6
  have e_1785_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1785 (by decide)
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
  have e_1785_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_8
  simp only [← c109116, ← c109117, ← c109118] at e_1785_8
  have e_1785_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1785 (by decide)
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
  have e_1785_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_10
  simp only [← c109122, ← c109123, ← c109124] at e_1785_10
  have e_1785_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1785 (by decide)
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
  have e_1785_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_12
  simp only [← c109128, ← c109129, ← c109130] at e_1785_12
  have e_1785_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1785 (by decide)
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
  have e_1785_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_14
  simp only [← c109134, ← c109135, ← c109136] at e_1785_14
  have e_1785_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1785 (by decide)
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
  have e_1785_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_16
  simp only [← c109140, ← c109141, ← c109142] at e_1785_16
  have e_1785_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1785 (by decide)
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
  have e_1785_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1785 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_18
  simp only [← c109146, ← c109147, ← c109148] at e_1785_18
  have e_1785_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1785 (by decide)
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
  have e_1786_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_0
  simp only [← c109152, ← c109153, ← c109154] at e_1786_0
  have e_1786_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1786 (by decide)
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
  have e_1786_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_2
  simp only [← c109158, ← c109159, ← c109160] at e_1786_2
  have e_1786_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1786 (by decide)
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
  have e_1786_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_4
  simp only [← c109164, ← c109165, ← c109166] at e_1786_4
  have e_1786_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1786 (by decide)
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
  have e_1786_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_6
  simp only [← c109170, ← c109171, ← c109172] at e_1786_6
  have e_1786_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1786 (by decide)
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
  have e_1786_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_8
  simp only [← c109176, ← c109177, ← c109178] at e_1786_8
  have e_1786_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1786 (by decide)
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
  have e_1786_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_10
  simp only [← c109182, ← c109183, ← c109184] at e_1786_10
  have e_1786_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1786 (by decide)
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
  have e_1786_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_12
  simp only [← c109188, ← c109189, ← c109190] at e_1786_12
  have e_1786_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1786 (by decide)
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
  have e_1786_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_14
  simp only [← c109194, ← c109195, ← c109196] at e_1786_14
  have e_1786_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1786 (by decide)
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
  have e_1786_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_16
  simp only [← c109200, ← c109201, ← c109202] at e_1786_16
  have e_1786_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1786 (by decide)
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
  have e_1786_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1786 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_18
  simp only [← c109206, ← c109207, ← c109208] at e_1786_18
  have e_1786_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1786 (by decide)
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
  have e_1787_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_0
  simp only [← c109212, ← c109213, ← c109214] at e_1787_0
  have e_1787_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1787 (by decide)
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
  have e_1787_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_2
  simp only [← c109218, ← c109219, ← c109220] at e_1787_2
  have e_1787_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1787 (by decide)
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
  have e_1787_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_4
  simp only [← c109224, ← c109225, ← c109226] at e_1787_4
  have e_1787_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1787 (by decide)
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
  have e_1787_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_6
  simp only [← c109230, ← c109231, ← c109232] at e_1787_6
  have e_1787_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1787 (by decide)
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
  have e_1787_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_8
  simp only [← c109236, ← c109237, ← c109238] at e_1787_8
  have e_1787_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1787 (by decide)
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
  have e_1787_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_10
  simp only [← c109242, ← c109243, ← c109244] at e_1787_10
  have e_1787_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1787 (by decide)
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
  have e_1787_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_12
  simp only [← c109248, ← c109249, ← c109250] at e_1787_12
  have e_1787_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1787 (by decide)
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
  have e_1787_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_14
  simp only [← c109254, ← c109255, ← c109256] at e_1787_14
  have e_1787_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1787 (by decide)
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
  have e_1787_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_16
  simp only [← c109260, ← c109261, ← c109262] at e_1787_16
  have e_1787_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1787 (by decide)
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
  have e_1787_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1787 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_18
  simp only [← c109266, ← c109267, ← c109268] at e_1787_18
  have e_1787_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1787 (by decide)
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
  have e_1788_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_0
  simp only [← c109272, ← c109273, ← c109274] at e_1788_0
  have e_1788_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1788 (by decide)
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
  have e_1788_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_2
  simp only [← c109278, ← c109279, ← c109280] at e_1788_2
  have e_1788_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1788 (by decide)
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
  have e_1788_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_4
  simp only [← c109284, ← c109285, ← c109286] at e_1788_4
  have e_1788_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1788 (by decide)
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
  have e_1788_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_6
  simp only [← c109290, ← c109291, ← c109292] at e_1788_6
  have e_1788_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1788 (by decide)
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
  have e_1788_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_8
  simp only [← c109296, ← c109297, ← c109298] at e_1788_8
  have e_1788_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_9
  simp only [← c109299, ← c109300, k2, ← c109301] at e_1788_9
  have hr := e_1788_9
  simp only [e_1788_8] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
