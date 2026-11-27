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

theorem publicBatchWrapper89_f12866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12290)) := by
  have c81702 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.1
  have c81703 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.1
  have c81704 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.1
  have c81705 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.1
  have c81706 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81707 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_10
  simp only [← c81702, ← c81703, ← c81704] at e_1328_10
  have e_1328_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_11
  simp only [← c81705, ← c81706, k2, ← c81707] at e_1328_11
  have hr := e_1328_11
  simp only [e_1328_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12291)) := by
  have c81708 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81709 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81710 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81711 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81712 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81713 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_12
  simp only [← c81708, ← c81709, ← c81710] at e_1328_12
  have e_1328_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_13
  simp only [← c81711, ← c81712, k2, ← c81713] at e_1328_13
  have hr := e_1328_13
  simp only [e_1328_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12292)) := by
  have c81714 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81715 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81716 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81717 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81718 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81719 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_14
  simp only [← c81714, ← c81715, ← c81716] at e_1328_14
  have e_1328_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_15
  simp only [← c81717, ← c81718, k2, ← c81719] at e_1328_15
  have hr := e_1328_15
  simp only [e_1328_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12293)) := by
  have c81720 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81721 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81722 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81723 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81724 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81725 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_16
  simp only [← c81720, ← c81721, ← c81722] at e_1328_16
  have e_1328_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_17
  simp only [← c81723, ← c81724, k2, ← c81725] at e_1328_17
  have hr := e_1328_17
  simp only [e_1328_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12294)) := by
  have c81726 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81727 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81728 := (publicBatchWrapper89_copies2554 a h).1
  have c81729 := (publicBatchWrapper89_copies2554 a h).2.1
  have c81730 := (publicBatchWrapper89_copies2554 a h).2.2.1
  have c81731 := (publicBatchWrapper89_copies2554 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_18
  simp only [← c81726, ← c81727, ← c81728] at e_1328_18
  have e_1328_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1328 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_19
  simp only [← c81729, ← c81730, k2, ← c81731] at e_1328_19
  have hr := e_1328_19
  simp only [e_1328_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12295)) := by
  have c81732 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.1
  have c81733 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.1
  have c81734 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.1
  have c81735 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.1
  have c81736 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.1
  have c81737 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_0
  simp only [← c81732, ← c81733, ← c81734] at e_1329_0
  have e_1329_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_1
  simp only [← c81735, ← c81736, k2, ← c81737] at e_1329_1
  have hr := e_1329_1
  simp only [e_1329_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12296)) := by
  have c81738 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81739 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81740 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81741 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81742 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81743 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_2
  simp only [← c81738, ← c81739, ← c81740] at e_1329_2
  have e_1329_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_3
  simp only [← c81741, ← c81742, k2, ← c81743] at e_1329_3
  have hr := e_1329_3
  simp only [e_1329_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12297)) := by
  have c81744 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81745 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81746 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81747 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81748 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81749 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_4
  simp only [← c81744, ← c81745, ← c81746] at e_1329_4
  have e_1329_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_5
  simp only [← c81747, ← c81748, k2, ← c81749] at e_1329_5
  have hr := e_1329_5
  simp only [e_1329_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12298)) := by
  have c81750 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81751 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81752 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81753 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81754 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81755 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_6
  simp only [← c81750, ← c81751, ← c81752] at e_1329_6
  have e_1329_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_7
  simp only [← c81753, ← c81754, k2, ← c81755] at e_1329_7
  have hr := e_1329_7
  simp only [e_1329_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12299)) := by
  have c81756 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81757 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81758 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81759 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81760 := (publicBatchWrapper89_copies2555 a h).1
  have c81761 := (publicBatchWrapper89_copies2555 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_8
  simp only [← c81756, ← c81757, ← c81758] at e_1329_8
  have e_1329_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_9
  simp only [← c81759, ← c81760, k2, ← c81761] at e_1329_9
  have hr := e_1329_9
  simp only [e_1329_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12300)) := by
  have c81762 := (publicBatchWrapper89_copies2555 a h).2.2.1
  have c81763 := (publicBatchWrapper89_copies2555 a h).2.2.2.1
  have c81764 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.1
  have c81765 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.1
  have c81766 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.1
  have c81767 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_10
  simp only [← c81762, ← c81763, ← c81764] at e_1329_10
  have e_1329_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_11
  simp only [← c81765, ← c81766, k2, ← c81767] at e_1329_11
  have hr := e_1329_11
  simp only [e_1329_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12301)) := by
  have c81768 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.1
  have c81769 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.1
  have c81770 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81771 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81772 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81773 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_12
  simp only [← c81768, ← c81769, ← c81770] at e_1329_12
  have e_1329_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_13
  simp only [← c81771, ← c81772, k2, ← c81773] at e_1329_13
  have hr := e_1329_13
  simp only [e_1329_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12302)) := by
  have c81774 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81775 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81776 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81777 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81778 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81779 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_14
  simp only [← c81774, ← c81775, ← c81776] at e_1329_14
  have e_1329_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_15
  simp only [← c81777, ← c81778, k2, ← c81779] at e_1329_15
  have hr := e_1329_15
  simp only [e_1329_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12303)) := by
  have c81780 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81781 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81782 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81783 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81784 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81785 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_16
  simp only [← c81780, ← c81781, ← c81782] at e_1329_16
  have e_1329_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_17
  simp only [← c81783, ← c81784, k2, ← c81785] at e_1329_17
  have hr := e_1329_17
  simp only [e_1329_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12304)) := by
  have c81786 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81787 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81788 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81789 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81790 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81791 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_18
  simp only [← c81786, ← c81787, ← c81788] at e_1329_18
  have e_1329_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1329 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_19
  simp only [← c81789, ← c81790, k2, ← c81791] at e_1329_19
  have hr := e_1329_19
  simp only [e_1329_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12305)) := by
  have c81792 := (publicBatchWrapper89_copies2556 a h).1
  have c81793 := (publicBatchWrapper89_copies2556 a h).2.1
  have c81794 := (publicBatchWrapper89_copies2556 a h).2.2.1
  have c81795 := (publicBatchWrapper89_copies2556 a h).2.2.2.1
  have c81796 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.1
  have c81797 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_0
  simp only [← c81792, ← c81793, ← c81794] at e_1330_0
  have e_1330_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_1
  simp only [← c81795, ← c81796, k2, ← c81797] at e_1330_1
  have hr := e_1330_1
  simp only [e_1330_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12306)) := by
  have c81798 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.1
  have c81799 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.1
  have c81800 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.1
  have c81801 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.1
  have c81802 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81803 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_2
  simp only [← c81798, ← c81799, ← c81800] at e_1330_2
  have e_1330_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_3
  simp only [← c81801, ← c81802, k2, ← c81803] at e_1330_3
  have hr := e_1330_3
  simp only [e_1330_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12307)) := by
  have c81804 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81805 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81806 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81807 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81808 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81809 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_4
  simp only [← c81804, ← c81805, ← c81806] at e_1330_4
  have e_1330_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_5
  simp only [← c81807, ← c81808, k2, ← c81809] at e_1330_5
  have hr := e_1330_5
  simp only [e_1330_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12308)) := by
  have c81810 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81811 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81812 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81813 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81814 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81815 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_6
  simp only [← c81810, ← c81811, ← c81812] at e_1330_6
  have e_1330_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_7
  simp only [← c81813, ← c81814, k2, ← c81815] at e_1330_7
  have hr := e_1330_7
  simp only [e_1330_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12309)) := by
  have c81816 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81817 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81818 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81819 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81820 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81821 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_8
  simp only [← c81816, ← c81817, ← c81818] at e_1330_8
  have e_1330_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_9
  simp only [← c81819, ← c81820, k2, ← c81821] at e_1330_9
  have hr := e_1330_9
  simp only [e_1330_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12310)) := by
  have c81822 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81823 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81824 := (publicBatchWrapper89_copies2557 a h).1
  have c81825 := (publicBatchWrapper89_copies2557 a h).2.1
  have c81826 := (publicBatchWrapper89_copies2557 a h).2.2.1
  have c81827 := (publicBatchWrapper89_copies2557 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_10
  simp only [← c81822, ← c81823, ← c81824] at e_1330_10
  have e_1330_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_11
  simp only [← c81825, ← c81826, k2, ← c81827] at e_1330_11
  have hr := e_1330_11
  simp only [e_1330_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12311)) := by
  have c81828 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.1
  have c81829 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.1
  have c81830 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.1
  have c81831 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.1
  have c81832 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.1
  have c81833 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_12
  simp only [← c81828, ← c81829, ← c81830] at e_1330_12
  have e_1330_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_13
  simp only [← c81831, ← c81832, k2, ← c81833] at e_1330_13
  have hr := e_1330_13
  simp only [e_1330_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12312)) := by
  have c81834 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81835 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81836 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81837 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81838 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81839 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_14
  simp only [← c81834, ← c81835, ← c81836] at e_1330_14
  have e_1330_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_15
  simp only [← c81837, ← c81838, k2, ← c81839] at e_1330_15
  have hr := e_1330_15
  simp only [e_1330_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12313)) := by
  have c81840 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81841 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81842 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81843 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81844 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81845 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_16
  simp only [← c81840, ← c81841, ← c81842] at e_1330_16
  have e_1330_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_17
  simp only [← c81843, ← c81844, k2, ← c81845] at e_1330_17
  have hr := e_1330_17
  simp only [e_1330_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12314)) := by
  have c81846 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81847 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81848 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81849 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81850 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81851 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_18
  simp only [← c81846, ← c81847, ← c81848] at e_1330_18
  have e_1330_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1330 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_19
  simp only [← c81849, ← c81850, k2, ← c81851] at e_1330_19
  have hr := e_1330_19
  simp only [e_1330_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12315)) := by
  have c81852 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81853 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81854 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81855 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81856 := (publicBatchWrapper89_copies2558 a h).1
  have c81857 := (publicBatchWrapper89_copies2558 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_0
  simp only [← c81852, ← c81853, ← c81854] at e_1331_0
  have e_1331_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_1
  simp only [← c81855, ← c81856, k2, ← c81857] at e_1331_1
  have hr := e_1331_1
  simp only [e_1331_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12316)) := by
  have c81858 := (publicBatchWrapper89_copies2558 a h).2.2.1
  have c81859 := (publicBatchWrapper89_copies2558 a h).2.2.2.1
  have c81860 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.1
  have c81861 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.1
  have c81862 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.1
  have c81863 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_2
  simp only [← c81858, ← c81859, ← c81860] at e_1331_2
  have e_1331_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_3
  simp only [← c81861, ← c81862, k2, ← c81863] at e_1331_3
  have hr := e_1331_3
  simp only [e_1331_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12317)) := by
  have c81864 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.1
  have c81865 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.1
  have c81866 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81867 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81868 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81869 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_4
  simp only [← c81864, ← c81865, ← c81866] at e_1331_4
  have e_1331_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_5
  simp only [← c81867, ← c81868, k2, ← c81869] at e_1331_5
  have hr := e_1331_5
  simp only [e_1331_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12318)) := by
  have c81870 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81871 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81872 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81873 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81874 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81875 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_6
  simp only [← c81870, ← c81871, ← c81872] at e_1331_6
  have e_1331_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_7
  simp only [← c81873, ← c81874, k2, ← c81875] at e_1331_7
  have hr := e_1331_7
  simp only [e_1331_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12319)) := by
  have c81876 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81877 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81878 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81879 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81880 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81881 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_8
  simp only [← c81876, ← c81877, ← c81878] at e_1331_8
  have e_1331_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_9
  simp only [← c81879, ← c81880, k2, ← c81881] at e_1331_9
  have hr := e_1331_9
  simp only [e_1331_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12320)) := by
  have c81882 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81883 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81884 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81885 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81886 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81887 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_10
  simp only [← c81882, ← c81883, ← c81884] at e_1331_10
  have e_1331_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_11
  simp only [← c81885, ← c81886, k2, ← c81887] at e_1331_11
  have hr := e_1331_11
  simp only [e_1331_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12321)) := by
  have c81888 := (publicBatchWrapper89_copies2559 a h).1
  have c81889 := (publicBatchWrapper89_copies2559 a h).2.1
  have c81890 := (publicBatchWrapper89_copies2559 a h).2.2.1
  have c81891 := (publicBatchWrapper89_copies2559 a h).2.2.2.1
  have c81892 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.1
  have c81893 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_12
  simp only [← c81888, ← c81889, ← c81890] at e_1331_12
  have e_1331_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_13
  simp only [← c81891, ← c81892, k2, ← c81893] at e_1331_13
  have hr := e_1331_13
  simp only [e_1331_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12322)) := by
  have c81894 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.1
  have c81895 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.1
  have c81896 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.1
  have c81897 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.1
  have c81898 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81899 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_14
  simp only [← c81894, ← c81895, ← c81896] at e_1331_14
  have e_1331_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_15
  simp only [← c81897, ← c81898, k2, ← c81899] at e_1331_15
  have hr := e_1331_15
  simp only [e_1331_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12323)) := by
  have c81900 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81901 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81902 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81903 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81904 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81905 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_16
  simp only [← c81900, ← c81901, ← c81902] at e_1331_16
  have e_1331_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_17
  simp only [← c81903, ← c81904, k2, ← c81905] at e_1331_17
  have hr := e_1331_17
  simp only [e_1331_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12324)) := by
  have c81906 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81907 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81908 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81909 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81910 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81911 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_18
  simp only [← c81906, ← c81907, ← c81908] at e_1331_18
  have e_1331_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1331 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_19
  simp only [← c81909, ← c81910, k2, ← c81911] at e_1331_19
  have hr := e_1331_19
  simp only [e_1331_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12325)) := by
  have c81912 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81913 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81914 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81915 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81916 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81917 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_0
  simp only [← c81912, ← c81913, ← c81914] at e_1332_0
  have e_1332_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_1
  simp only [← c81915, ← c81916, k2, ← c81917] at e_1332_1
  have hr := e_1332_1
  simp only [e_1332_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12326)) := by
  have c81918 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81919 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81920 := (publicBatchWrapper89_copies2560 a h).1
  have c81921 := (publicBatchWrapper89_copies2560 a h).2.1
  have c81922 := (publicBatchWrapper89_copies2560 a h).2.2.1
  have c81923 := (publicBatchWrapper89_copies2560 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_2
  simp only [← c81918, ← c81919, ← c81920] at e_1332_2
  have e_1332_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_3
  simp only [← c81921, ← c81922, k2, ← c81923] at e_1332_3
  have hr := e_1332_3
  simp only [e_1332_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12327)) := by
  have c81924 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.1
  have c81925 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.1
  have c81926 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.1
  have c81927 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.1
  have c81928 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.1
  have c81929 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_4
  simp only [← c81924, ← c81925, ← c81926] at e_1332_4
  have e_1332_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_5
  simp only [← c81927, ← c81928, k2, ← c81929] at e_1332_5
  have hr := e_1332_5
  simp only [e_1332_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12328)) := by
  have c81930 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81931 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81932 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81933 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81934 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81935 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_6
  simp only [← c81930, ← c81931, ← c81932] at e_1332_6
  have e_1332_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_7
  simp only [← c81933, ← c81934, k2, ← c81935] at e_1332_7
  have hr := e_1332_7
  simp only [e_1332_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12329)) := by
  have c81936 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81937 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81938 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81939 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81940 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81941 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_8
  simp only [← c81936, ← c81937, ← c81938] at e_1332_8
  have e_1332_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_9
  simp only [← c81939, ← c81940, k2, ← c81941] at e_1332_9
  have hr := e_1332_9
  simp only [e_1332_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12330)) := by
  have c81942 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81943 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81944 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81945 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81946 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81947 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_10
  simp only [← c81942, ← c81943, ← c81944] at e_1332_10
  have e_1332_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_11
  simp only [← c81945, ← c81946, k2, ← c81947] at e_1332_11
  have hr := e_1332_11
  simp only [e_1332_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12331)) := by
  have c81948 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81949 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81950 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81951 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81952 := (publicBatchWrapper89_copies2561 a h).1
  have c81953 := (publicBatchWrapper89_copies2561 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_12
  simp only [← c81948, ← c81949, ← c81950] at e_1332_12
  have e_1332_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_13
  simp only [← c81951, ← c81952, k2, ← c81953] at e_1332_13
  have hr := e_1332_13
  simp only [e_1332_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12332)) := by
  have c81954 := (publicBatchWrapper89_copies2561 a h).2.2.1
  have c81955 := (publicBatchWrapper89_copies2561 a h).2.2.2.1
  have c81956 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.1
  have c81957 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.1
  have c81958 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.1
  have c81959 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_14
  simp only [← c81954, ← c81955, ← c81956] at e_1332_14
  have e_1332_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_15
  simp only [← c81957, ← c81958, k2, ← c81959] at e_1332_15
  have hr := e_1332_15
  simp only [e_1332_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12333)) := by
  have c81960 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.1
  have c81961 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.1
  have c81962 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81963 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81964 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81965 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_16
  simp only [← c81960, ← c81961, ← c81962] at e_1332_16
  have e_1332_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_17
  simp only [← c81963, ← c81964, k2, ← c81965] at e_1332_17
  have hr := e_1332_17
  simp only [e_1332_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12334)) := by
  have c81966 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81967 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81968 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81969 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81970 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81971 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_18
  simp only [← c81966, ← c81967, ← c81968] at e_1332_18
  have e_1332_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1332 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_19
  simp only [← c81969, ← c81970, k2, ← c81971] at e_1332_19
  have hr := e_1332_19
  simp only [e_1332_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12335)) := by
  have c81972 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81973 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81974 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81975 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81976 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81977 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_0
  simp only [← c81972, ← c81973, ← c81974] at e_1333_0
  have e_1333_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_1
  simp only [← c81975, ← c81976, k2, ← c81977] at e_1333_1
  have hr := e_1333_1
  simp only [e_1333_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12336)) := by
  have c81978 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81979 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81980 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81981 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81982 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81983 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_2
  simp only [← c81978, ← c81979, ← c81980] at e_1333_2
  have e_1333_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_3
  simp only [← c81981, ← c81982, k2, ← c81983] at e_1333_3
  have hr := e_1333_3
  simp only [e_1333_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12337)) := by
  have c81984 := (publicBatchWrapper89_copies2562 a h).1
  have c81985 := (publicBatchWrapper89_copies2562 a h).2.1
  have c81986 := (publicBatchWrapper89_copies2562 a h).2.2.1
  have c81987 := (publicBatchWrapper89_copies2562 a h).2.2.2.1
  have c81988 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.1
  have c81989 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_4
  simp only [← c81984, ← c81985, ← c81986] at e_1333_4
  have e_1333_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_5
  simp only [← c81987, ← c81988, k2, ← c81989] at e_1333_5
  have hr := e_1333_5
  simp only [e_1333_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12338)) := by
  have c81990 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.1
  have c81991 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.1
  have c81992 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.1
  have c81993 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.1
  have c81994 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81995 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_6
  simp only [← c81990, ← c81991, ← c81992] at e_1333_6
  have e_1333_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_7
  simp only [← c81993, ← c81994, k2, ← c81995] at e_1333_7
  have hr := e_1333_7
  simp only [e_1333_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12339)) := by
  have c81996 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81997 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81998 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81999 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82000 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82001 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_8
  simp only [← c81996, ← c81997, ← c81998] at e_1333_8
  have e_1333_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_9
  simp only [← c81999, ← c82000, k2, ← c82001] at e_1333_9
  have hr := e_1333_9
  simp only [e_1333_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12340)) := by
  have c82002 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82003 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82004 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82005 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82006 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82007 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_10
  simp only [← c82002, ← c82003, ← c82004] at e_1333_10
  have e_1333_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_11
  simp only [← c82005, ← c82006, k2, ← c82007] at e_1333_11
  have hr := e_1333_11
  simp only [e_1333_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12341)) := by
  have c82008 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82009 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82010 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82011 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82012 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82013 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_12
  simp only [← c82008, ← c82009, ← c82010] at e_1333_12
  have e_1333_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_13
  simp only [← c82011, ← c82012, k2, ← c82013] at e_1333_13
  have hr := e_1333_13
  simp only [e_1333_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12342)) := by
  have c82014 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82015 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82016 := (publicBatchWrapper89_copies2563 a h).1
  have c82017 := (publicBatchWrapper89_copies2563 a h).2.1
  have c82018 := (publicBatchWrapper89_copies2563 a h).2.2.1
  have c82019 := (publicBatchWrapper89_copies2563 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_14
  simp only [← c82014, ← c82015, ← c82016] at e_1333_14
  have e_1333_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_15
  simp only [← c82017, ← c82018, k2, ← c82019] at e_1333_15
  have hr := e_1333_15
  simp only [e_1333_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12343)) := by
  have c82020 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.1
  have c82021 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.1
  have c82022 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.1
  have c82023 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.1
  have c82024 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.1
  have c82025 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_16
  simp only [← c82020, ← c82021, ← c82022] at e_1333_16
  have e_1333_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_17
  simp only [← c82023, ← c82024, k2, ← c82025] at e_1333_17
  have hr := e_1333_17
  simp only [e_1333_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12344)) := by
  have c82026 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82027 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82028 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82029 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82030 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82031 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_18
  simp only [← c82026, ← c82027, ← c82028] at e_1333_18
  have e_1333_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1333 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_19
  simp only [← c82029, ← c82030, k2, ← c82031] at e_1333_19
  have hr := e_1333_19
  simp only [e_1333_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12345)) := by
  have c82032 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82033 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82034 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82035 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82036 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82037 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_0
  simp only [← c82032, ← c82033, ← c82034] at e_1334_0
  have e_1334_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_1
  simp only [← c82035, ← c82036, k2, ← c82037] at e_1334_1
  have hr := e_1334_1
  simp only [e_1334_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12346)) := by
  have c82038 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82039 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82040 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82041 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82042 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82043 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_2
  simp only [← c82038, ← c82039, ← c82040] at e_1334_2
  have e_1334_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_3
  simp only [← c82041, ← c82042, k2, ← c82043] at e_1334_3
  have hr := e_1334_3
  simp only [e_1334_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12347)) := by
  have c82044 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82045 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82046 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82047 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82048 := (publicBatchWrapper89_copies2564 a h).1
  have c82049 := (publicBatchWrapper89_copies2564 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_4
  simp only [← c82044, ← c82045, ← c82046] at e_1334_4
  have e_1334_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_5
  simp only [← c82047, ← c82048, k2, ← c82049] at e_1334_5
  have hr := e_1334_5
  simp only [e_1334_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12348)) := by
  have c82050 := (publicBatchWrapper89_copies2564 a h).2.2.1
  have c82051 := (publicBatchWrapper89_copies2564 a h).2.2.2.1
  have c82052 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.1
  have c82053 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.1
  have c82054 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.1
  have c82055 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_6
  simp only [← c82050, ← c82051, ← c82052] at e_1334_6
  have e_1334_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_7
  simp only [← c82053, ← c82054, k2, ← c82055] at e_1334_7
  have hr := e_1334_7
  simp only [e_1334_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12349)) := by
  have c82056 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.1
  have c82057 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.1
  have c82058 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82059 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82060 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82061 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_8
  simp only [← c82056, ← c82057, ← c82058] at e_1334_8
  have e_1334_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_9
  simp only [← c82059, ← c82060, k2, ← c82061] at e_1334_9
  have hr := e_1334_9
  simp only [e_1334_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12350)) := by
  have c82062 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82063 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82064 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82065 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82066 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82067 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_10
  simp only [← c82062, ← c82063, ← c82064] at e_1334_10
  have e_1334_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_11
  simp only [← c82065, ← c82066, k2, ← c82067] at e_1334_11
  have hr := e_1334_11
  simp only [e_1334_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12351)) := by
  have c82068 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82069 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82070 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82071 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82072 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82073 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_12
  simp only [← c82068, ← c82069, ← c82070] at e_1334_12
  have e_1334_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_13
  simp only [← c82071, ← c82072, k2, ← c82073] at e_1334_13
  have hr := e_1334_13
  simp only [e_1334_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12352)) := by
  have c82074 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82075 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82076 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82077 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82078 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82079 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_14
  simp only [← c82074, ← c82075, ← c82076] at e_1334_14
  have e_1334_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_15
  simp only [← c82077, ← c82078, k2, ← c82079] at e_1334_15
  have hr := e_1334_15
  simp only [e_1334_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12353)) := by
  have c82080 := (publicBatchWrapper89_copies2565 a h).1
  have c82081 := (publicBatchWrapper89_copies2565 a h).2.1
  have c82082 := (publicBatchWrapper89_copies2565 a h).2.2.1
  have c82083 := (publicBatchWrapper89_copies2565 a h).2.2.2.1
  have c82084 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.1
  have c82085 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_16
  simp only [← c82080, ← c82081, ← c82082] at e_1334_16
  have e_1334_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_17
  simp only [← c82083, ← c82084, k2, ← c82085] at e_1334_17
  have hr := e_1334_17
  simp only [e_1334_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12354)) := by
  have c82086 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.1
  have c82087 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.1
  have c82088 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.1
  have c82089 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.1
  have c82090 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82091 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_18
  simp only [← c82086, ← c82087, ← c82088] at e_1334_18
  have e_1334_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1334 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_19
  simp only [← c82089, ← c82090, k2, ← c82091] at e_1334_19
  have hr := e_1334_19
  simp only [e_1334_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12355)) := by
  have c82092 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82093 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82094 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82095 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82096 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82097 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_0
  simp only [← c82092, ← c82093, ← c82094] at e_1335_0
  have e_1335_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_1
  simp only [← c82095, ← c82096, k2, ← c82097] at e_1335_1
  have hr := e_1335_1
  simp only [e_1335_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12356)) := by
  have c82098 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82099 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82100 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82101 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82102 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82103 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_2
  simp only [← c82098, ← c82099, ← c82100] at e_1335_2
  have e_1335_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_3
  simp only [← c82101, ← c82102, k2, ← c82103] at e_1335_3
  have hr := e_1335_3
  simp only [e_1335_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12357)) := by
  have c82104 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82105 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82106 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82107 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82108 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82109 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_4
  simp only [← c82104, ← c82105, ← c82106] at e_1335_4
  have e_1335_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_5
  simp only [← c82107, ← c82108, k2, ← c82109] at e_1335_5
  have hr := e_1335_5
  simp only [e_1335_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12358)) := by
  have c82110 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82111 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82112 := (publicBatchWrapper89_copies2566 a h).1
  have c82113 := (publicBatchWrapper89_copies2566 a h).2.1
  have c82114 := (publicBatchWrapper89_copies2566 a h).2.2.1
  have c82115 := (publicBatchWrapper89_copies2566 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_6
  simp only [← c82110, ← c82111, ← c82112] at e_1335_6
  have e_1335_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_7
  simp only [← c82113, ← c82114, k2, ← c82115] at e_1335_7
  have hr := e_1335_7
  simp only [e_1335_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12359)) := by
  have c82116 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.1
  have c82117 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.1
  have c82118 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.1
  have c82119 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.1
  have c82120 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.1
  have c82121 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_8
  simp only [← c82116, ← c82117, ← c82118] at e_1335_8
  have e_1335_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_9
  simp only [← c82119, ← c82120, k2, ← c82121] at e_1335_9
  have hr := e_1335_9
  simp only [e_1335_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12360)) := by
  have c82122 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82123 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82124 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82125 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82126 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82127 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_10
  simp only [← c82122, ← c82123, ← c82124] at e_1335_10
  have e_1335_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_11
  simp only [← c82125, ← c82126, k2, ← c82127] at e_1335_11
  have hr := e_1335_11
  simp only [e_1335_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12361)) := by
  have c82128 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82129 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82130 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82131 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82132 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82133 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_12
  simp only [← c82128, ← c82129, ← c82130] at e_1335_12
  have e_1335_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_13
  simp only [← c82131, ← c82132, k2, ← c82133] at e_1335_13
  have hr := e_1335_13
  simp only [e_1335_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12362)) := by
  have c82134 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82135 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82136 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82137 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82138 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82139 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_14
  simp only [← c82134, ← c82135, ← c82136] at e_1335_14
  have e_1335_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_15
  simp only [← c82137, ← c82138, k2, ← c82139] at e_1335_15
  have hr := e_1335_15
  simp only [e_1335_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12363)) := by
  have c82140 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82141 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82142 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82143 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82144 := (publicBatchWrapper89_copies2567 a h).1
  have c82145 := (publicBatchWrapper89_copies2567 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_16
  simp only [← c82140, ← c82141, ← c82142] at e_1335_16
  have e_1335_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_17
  simp only [← c82143, ← c82144, k2, ← c82145] at e_1335_17
  have hr := e_1335_17
  simp only [e_1335_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12364)) := by
  have c82146 := (publicBatchWrapper89_copies2567 a h).2.2.1
  have c82147 := (publicBatchWrapper89_copies2567 a h).2.2.2.1
  have c82148 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.1
  have c82149 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.1
  have c82150 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.1
  have c82151 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_18
  simp only [← c82146, ← c82147, ← c82148] at e_1335_18
  have e_1335_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1335 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_19
  simp only [← c82149, ← c82150, k2, ← c82151] at e_1335_19
  have hr := e_1335_19
  simp only [e_1335_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12365)) := by
  have c82152 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.1
  have c82153 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.1
  have c82154 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82155 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82156 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82157 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_0
  simp only [← c82152, ← c82153, ← c82154] at e_1336_0
  have e_1336_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_1
  simp only [← c82155, ← c82156, k2, ← c82157] at e_1336_1
  have hr := e_1336_1
  simp only [e_1336_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12366)) := by
  have c82158 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82159 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82160 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82161 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82162 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82163 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_2
  simp only [← c82158, ← c82159, ← c82160] at e_1336_2
  have e_1336_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_3
  simp only [← c82161, ← c82162, k2, ← c82163] at e_1336_3
  have hr := e_1336_3
  simp only [e_1336_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12367)) := by
  have c82164 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82165 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82166 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82167 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82168 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82169 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_4
  simp only [← c82164, ← c82165, ← c82166] at e_1336_4
  have e_1336_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_5
  simp only [← c82167, ← c82168, k2, ← c82169] at e_1336_5
  have hr := e_1336_5
  simp only [e_1336_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12368)) := by
  have c82170 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82171 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82172 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82173 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82174 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82175 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_6
  simp only [← c82170, ← c82171, ← c82172] at e_1336_6
  have e_1336_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_7
  simp only [← c82173, ← c82174, k2, ← c82175] at e_1336_7
  have hr := e_1336_7
  simp only [e_1336_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12369)) := by
  have c82176 := (publicBatchWrapper89_copies2568 a h).1
  have c82177 := (publicBatchWrapper89_copies2568 a h).2.1
  have c82178 := (publicBatchWrapper89_copies2568 a h).2.2.1
  have c82179 := (publicBatchWrapper89_copies2568 a h).2.2.2.1
  have c82180 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.1
  have c82181 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_8
  simp only [← c82176, ← c82177, ← c82178] at e_1336_8
  have e_1336_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_9
  simp only [← c82179, ← c82180, k2, ← c82181] at e_1336_9
  have hr := e_1336_9
  simp only [e_1336_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12410)) := by
  have c82182 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.1
  have c82183 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.1
  have c82184 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.1
  have c82185 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.1
  have c82186 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82187 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_10
  simp only [← c82182, ← c82183, ← c82184] at e_1336_10
  have e_1336_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_11
  simp only [← c82185, ← c82186, k2, ← c82187] at e_1336_11
  have hr := e_1336_11
  simp only [e_1336_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12411)) := by
  have c82188 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82189 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82190 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82191 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82192 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82193 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_12
  simp only [← c82188, ← c82189, ← c82190] at e_1336_12
  have e_1336_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_13
  simp only [← c82191, ← c82192, k2, ← c82193] at e_1336_13
  have hr := e_1336_13
  simp only [e_1336_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12412)) := by
  have c82194 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82195 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82196 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82197 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82198 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82199 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_14
  simp only [← c82194, ← c82195, ← c82196] at e_1336_14
  have e_1336_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_15
  simp only [← c82197, ← c82198, k2, ← c82199] at e_1336_15
  have hr := e_1336_15
  simp only [e_1336_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12413)) := by
  have c82200 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82201 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82202 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82203 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82204 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82205 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_16
  simp only [← c82200, ← c82201, ← c82202] at e_1336_16
  have e_1336_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_17
  simp only [← c82203, ← c82204, k2, ← c82205] at e_1336_17
  have hr := e_1336_17
  simp only [e_1336_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12414)) := by
  have c82206 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82207 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82208 := (publicBatchWrapper89_copies2569 a h).1
  have c82209 := (publicBatchWrapper89_copies2569 a h).2.1
  have c82210 := (publicBatchWrapper89_copies2569 a h).2.2.1
  have c82211 := (publicBatchWrapper89_copies2569 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_18
  simp only [← c82206, ← c82207, ← c82208] at e_1336_18
  have e_1336_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1336 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_19
  simp only [← c82209, ← c82210, k2, ← c82211] at e_1336_19
  have hr := e_1336_19
  simp only [e_1336_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12415)) := by
  have c82212 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.1
  have c82213 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.1
  have c82214 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.1
  have c82215 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.1
  have c82216 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.1
  have c82217 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_0
  simp only [← c82212, ← c82213, ← c82214] at e_1337_0
  have e_1337_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_1
  simp only [← c82215, ← c82216, k2, ← c82217] at e_1337_1
  have hr := e_1337_1
  simp only [e_1337_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12416)) := by
  have c82218 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82219 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82220 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82221 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82222 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82223 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_2
  simp only [← c82218, ← c82219, ← c82220] at e_1337_2
  have e_1337_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_3
  simp only [← c82221, ← c82222, k2, ← c82223] at e_1337_3
  have hr := e_1337_3
  simp only [e_1337_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12417)) := by
  have c82224 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82225 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82226 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82227 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82228 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82229 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_4
  simp only [← c82224, ← c82225, ← c82226] at e_1337_4
  have e_1337_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_5
  simp only [← c82227, ← c82228, k2, ← c82229] at e_1337_5
  have hr := e_1337_5
  simp only [e_1337_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12418)) := by
  have c82230 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82231 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82232 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82233 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82234 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82235 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_6
  simp only [← c82230, ← c82231, ← c82232] at e_1337_6
  have e_1337_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_7
  simp only [← c82233, ← c82234, k2, ← c82235] at e_1337_7
  have hr := e_1337_7
  simp only [e_1337_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12419)) := by
  have c82236 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82237 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82238 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82239 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82240 := (publicBatchWrapper89_copies2570 a h).1
  have c82241 := (publicBatchWrapper89_copies2570 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_8
  simp only [← c82236, ← c82237, ← c82238] at e_1337_8
  have e_1337_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_9
  simp only [← c82239, ← c82240, k2, ← c82241] at e_1337_9
  have hr := e_1337_9
  simp only [e_1337_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12420)) := by
  have c82242 := (publicBatchWrapper89_copies2570 a h).2.2.1
  have c82243 := (publicBatchWrapper89_copies2570 a h).2.2.2.1
  have c82244 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.1
  have c82245 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.1
  have c82246 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.1
  have c82247 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_10
  simp only [← c82242, ← c82243, ← c82244] at e_1337_10
  have e_1337_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_11
  simp only [← c82245, ← c82246, k2, ← c82247] at e_1337_11
  have hr := e_1337_11
  simp only [e_1337_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12421)) := by
  have c82248 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.1
  have c82249 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.1
  have c82250 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82251 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82252 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82253 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_12
  simp only [← c82248, ← c82249, ← c82250] at e_1337_12
  have e_1337_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_13
  simp only [← c82251, ← c82252, k2, ← c82253] at e_1337_13
  have hr := e_1337_13
  simp only [e_1337_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12422)) := by
  have c82254 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82255 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82256 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82257 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82258 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82259 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_14
  simp only [← c82254, ← c82255, ← c82256] at e_1337_14
  have e_1337_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_15
  simp only [← c82257, ← c82258, k2, ← c82259] at e_1337_15
  have hr := e_1337_15
  simp only [e_1337_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12423)) := by
  have c82260 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82261 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82262 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82263 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82264 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82265 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_16
  simp only [← c82260, ← c82261, ← c82262] at e_1337_16
  have e_1337_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_17
  simp only [← c82263, ← c82264, k2, ← c82265] at e_1337_17
  have hr := e_1337_17
  simp only [e_1337_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12424)) := by
  have c82266 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82267 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82268 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82269 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82270 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82271 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_18
  simp only [← c82266, ← c82267, ← c82268] at e_1337_18
  have e_1337_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1337 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_19
  simp only [← c82269, ← c82270, k2, ← c82271] at e_1337_19
  have hr := e_1337_19
  simp only [e_1337_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12425)) := by
  have c82272 := (publicBatchWrapper89_copies2571 a h).1
  have c82273 := (publicBatchWrapper89_copies2571 a h).2.1
  have c82274 := (publicBatchWrapper89_copies2571 a h).2.2.1
  have c82275 := (publicBatchWrapper89_copies2571 a h).2.2.2.1
  have c82276 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.1
  have c82277 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_0
  simp only [← c82272, ← c82273, ← c82274] at e_1338_0
  have e_1338_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_1
  simp only [← c82275, ← c82276, k2, ← c82277] at e_1338_1
  have hr := e_1338_1
  simp only [e_1338_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12426)) := by
  have c82278 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.1
  have c82279 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.1
  have c82280 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.1
  have c82281 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.1
  have c82282 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82283 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_2
  simp only [← c82278, ← c82279, ← c82280] at e_1338_2
  have e_1338_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_3
  simp only [← c82281, ← c82282, k2, ← c82283] at e_1338_3
  have hr := e_1338_3
  simp only [e_1338_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12427)) := by
  have c82284 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82285 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82286 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82287 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82288 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82289 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_4
  simp only [← c82284, ← c82285, ← c82286] at e_1338_4
  have e_1338_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_5
  simp only [← c82287, ← c82288, k2, ← c82289] at e_1338_5
  have hr := e_1338_5
  simp only [e_1338_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12428)) := by
  have c82290 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82291 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82292 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82293 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82294 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82295 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_6
  simp only [← c82290, ← c82291, ← c82292] at e_1338_6
  have e_1338_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_7
  simp only [← c82293, ← c82294, k2, ← c82295] at e_1338_7
  have hr := e_1338_7
  simp only [e_1338_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12429)) := by
  have c82296 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82297 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82298 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82299 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82300 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82301 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_8
  simp only [← c82296, ← c82297, ← c82298] at e_1338_8
  have e_1338_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_9
  simp only [← c82299, ← c82300, k2, ← c82301] at e_1338_9
  have hr := e_1338_9
  simp only [e_1338_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12430)) := by
  have c82302 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82303 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82304 := (publicBatchWrapper89_copies2572 a h).1
  have c82305 := (publicBatchWrapper89_copies2572 a h).2.1
  have c82306 := (publicBatchWrapper89_copies2572 a h).2.2.1
  have c82307 := (publicBatchWrapper89_copies2572 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_10
  simp only [← c82302, ← c82303, ← c82304] at e_1338_10
  have e_1338_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_11
  simp only [← c82305, ← c82306, k2, ← c82307] at e_1338_11
  have hr := e_1338_11
  simp only [e_1338_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12431)) := by
  have c82308 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.1
  have c82309 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.1
  have c82310 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.1
  have c82311 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.1
  have c82312 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.1
  have c82313 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_12
  simp only [← c82308, ← c82309, ← c82310] at e_1338_12
  have e_1338_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_13
  simp only [← c82311, ← c82312, k2, ← c82313] at e_1338_13
  have hr := e_1338_13
  simp only [e_1338_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12432)) := by
  have c82314 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82315 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82316 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82317 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82318 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82319 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_14
  simp only [← c82314, ← c82315, ← c82316] at e_1338_14
  have e_1338_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_15
  simp only [← c82317, ← c82318, k2, ← c82319] at e_1338_15
  have hr := e_1338_15
  simp only [e_1338_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12433)) := by
  have c82320 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82321 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82322 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82323 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82324 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82325 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_16
  simp only [← c82320, ← c82321, ← c82322] at e_1338_16
  have e_1338_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_17
  simp only [← c82323, ← c82324, k2, ← c82325] at e_1338_17
  have hr := e_1338_17
  simp only [e_1338_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12434)) := by
  have c82326 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82327 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82328 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82329 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82330 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82331 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_18
  simp only [← c82326, ← c82327, ← c82328] at e_1338_18
  have e_1338_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1338 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_19
  simp only [← c82329, ← c82330, k2, ← c82331] at e_1338_19
  have hr := e_1338_19
  simp only [e_1338_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12435)) := by
  have c82332 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82333 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82334 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82335 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82336 := (publicBatchWrapper89_copies2573 a h).1
  have c82337 := (publicBatchWrapper89_copies2573 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_0
  simp only [← c82332, ← c82333, ← c82334] at e_1339_0
  have e_1339_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_1
  simp only [← c82335, ← c82336, k2, ← c82337] at e_1339_1
  have hr := e_1339_1
  simp only [e_1339_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12436)) := by
  have c82338 := (publicBatchWrapper89_copies2573 a h).2.2.1
  have c82339 := (publicBatchWrapper89_copies2573 a h).2.2.2.1
  have c82340 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.1
  have c82341 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.1
  have c82342 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.1
  have c82343 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_2
  simp only [← c82338, ← c82339, ← c82340] at e_1339_2
  have e_1339_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_3
  simp only [← c82341, ← c82342, k2, ← c82343] at e_1339_3
  have hr := e_1339_3
  simp only [e_1339_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12437)) := by
  have c82344 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.1
  have c82345 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.1
  have c82346 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82347 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82348 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82349 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_4
  simp only [← c82344, ← c82345, ← c82346] at e_1339_4
  have e_1339_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_5
  simp only [← c82347, ← c82348, k2, ← c82349] at e_1339_5
  have hr := e_1339_5
  simp only [e_1339_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12438)) := by
  have c82350 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82351 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82352 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82353 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82354 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82355 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_6
  simp only [← c82350, ← c82351, ← c82352] at e_1339_6
  have e_1339_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_7
  simp only [← c82353, ← c82354, k2, ← c82355] at e_1339_7
  have hr := e_1339_7
  simp only [e_1339_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12439)) := by
  have c82356 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82357 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82358 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82359 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82360 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82361 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_8
  simp only [← c82356, ← c82357, ← c82358] at e_1339_8
  have e_1339_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_9
  simp only [← c82359, ← c82360, k2, ← c82361] at e_1339_9
  have hr := e_1339_9
  simp only [e_1339_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12440)) := by
  have c82362 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82363 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82364 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82365 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82366 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82367 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_10
  simp only [← c82362, ← c82363, ← c82364] at e_1339_10
  have e_1339_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_11
  simp only [← c82365, ← c82366, k2, ← c82367] at e_1339_11
  have hr := e_1339_11
  simp only [e_1339_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12441)) := by
  have c82368 := (publicBatchWrapper89_copies2574 a h).1
  have c82369 := (publicBatchWrapper89_copies2574 a h).2.1
  have c82370 := (publicBatchWrapper89_copies2574 a h).2.2.1
  have c82371 := (publicBatchWrapper89_copies2574 a h).2.2.2.1
  have c82372 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.1
  have c82373 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_12
  simp only [← c82368, ← c82369, ← c82370] at e_1339_12
  have e_1339_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_13
  simp only [← c82371, ← c82372, k2, ← c82373] at e_1339_13
  have hr := e_1339_13
  simp only [e_1339_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12442)) := by
  have c82374 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.1
  have c82375 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.1
  have c82376 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.1
  have c82377 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.1
  have c82378 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82379 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_14
  simp only [← c82374, ← c82375, ← c82376] at e_1339_14
  have e_1339_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_15
  simp only [← c82377, ← c82378, k2, ← c82379] at e_1339_15
  have hr := e_1339_15
  simp only [e_1339_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12443)) := by
  have c82380 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82381 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82382 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82383 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82384 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82385 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_16
  simp only [← c82380, ← c82381, ← c82382] at e_1339_16
  have e_1339_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_17
  simp only [← c82383, ← c82384, k2, ← c82385] at e_1339_17
  have hr := e_1339_17
  simp only [e_1339_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12444)) := by
  have c82386 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82387 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82388 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82389 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82390 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82391 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_18
  simp only [← c82386, ← c82387, ← c82388] at e_1339_18
  have e_1339_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1339 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_19
  simp only [← c82389, ← c82390, k2, ← c82391] at e_1339_19
  have hr := e_1339_19
  simp only [e_1339_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12445)) := by
  have c82392 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82393 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82394 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82395 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82396 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82397 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_0
  simp only [← c82392, ← c82393, ← c82394] at e_1340_0
  have e_1340_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_1
  simp only [← c82395, ← c82396, k2, ← c82397] at e_1340_1
  have hr := e_1340_1
  simp only [e_1340_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12446)) := by
  have c82398 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82399 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82400 := (publicBatchWrapper89_copies2575 a h).1
  have c82401 := (publicBatchWrapper89_copies2575 a h).2.1
  have c82402 := (publicBatchWrapper89_copies2575 a h).2.2.1
  have c82403 := (publicBatchWrapper89_copies2575 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_2
  simp only [← c82398, ← c82399, ← c82400] at e_1340_2
  have e_1340_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_3
  simp only [← c82401, ← c82402, k2, ← c82403] at e_1340_3
  have hr := e_1340_3
  simp only [e_1340_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12447)) := by
  have c82404 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.1
  have c82405 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.1
  have c82406 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.1
  have c82407 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.1
  have c82408 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.1
  have c82409 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_4
  simp only [← c82404, ← c82405, ← c82406] at e_1340_4
  have e_1340_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_5
  simp only [← c82407, ← c82408, k2, ← c82409] at e_1340_5
  have hr := e_1340_5
  simp only [e_1340_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12448)) := by
  have c82410 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82411 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82412 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82413 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82414 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82415 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_6
  simp only [← c82410, ← c82411, ← c82412] at e_1340_6
  have e_1340_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_7
  simp only [← c82413, ← c82414, k2, ← c82415] at e_1340_7
  have hr := e_1340_7
  simp only [e_1340_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12449)) := by
  have c82416 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82417 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82418 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82419 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82420 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82421 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_8
  simp only [← c82416, ← c82417, ← c82418] at e_1340_8
  have e_1340_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_9
  simp only [← c82419, ← c82420, k2, ← c82421] at e_1340_9
  have hr := e_1340_9
  simp only [e_1340_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12450)) := by
  have c82422 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82423 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82424 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82425 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82426 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82427 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_10
  simp only [← c82422, ← c82423, ← c82424] at e_1340_10
  have e_1340_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_11
  simp only [← c82425, ← c82426, k2, ← c82427] at e_1340_11
  have hr := e_1340_11
  simp only [e_1340_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12451)) := by
  have c82428 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82429 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82430 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82431 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82432 := (publicBatchWrapper89_copies2576 a h).1
  have c82433 := (publicBatchWrapper89_copies2576 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_12
  simp only [← c82428, ← c82429, ← c82430] at e_1340_12
  have e_1340_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_13
  simp only [← c82431, ← c82432, k2, ← c82433] at e_1340_13
  have hr := e_1340_13
  simp only [e_1340_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12452)) := by
  have c82434 := (publicBatchWrapper89_copies2576 a h).2.2.1
  have c82435 := (publicBatchWrapper89_copies2576 a h).2.2.2.1
  have c82436 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.1
  have c82437 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.1
  have c82438 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.1
  have c82439 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_14
  simp only [← c82434, ← c82435, ← c82436] at e_1340_14
  have e_1340_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_15
  simp only [← c82437, ← c82438, k2, ← c82439] at e_1340_15
  have hr := e_1340_15
  simp only [e_1340_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12453)) := by
  have c82440 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.1
  have c82441 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.1
  have c82442 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82443 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82444 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82445 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_16
  simp only [← c82440, ← c82441, ← c82442] at e_1340_16
  have e_1340_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_17
  simp only [← c82443, ← c82444, k2, ← c82445] at e_1340_17
  have hr := e_1340_17
  simp only [e_1340_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12454)) := by
  have c82446 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82447 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82448 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82449 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82450 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82451 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_18
  simp only [← c82446, ← c82447, ← c82448] at e_1340_18
  have e_1340_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1340 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_19
  simp only [← c82449, ← c82450, k2, ← c82451] at e_1340_19
  have hr := e_1340_19
  simp only [e_1340_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12455)) := by
  have c82452 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82453 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82454 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82455 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82456 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82457 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_0
  simp only [← c82452, ← c82453, ← c82454] at e_1341_0
  have e_1341_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_1
  simp only [← c82455, ← c82456, k2, ← c82457] at e_1341_1
  have hr := e_1341_1
  simp only [e_1341_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12456)) := by
  have c82458 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82459 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82460 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82461 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82462 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82463 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_2
  simp only [← c82458, ← c82459, ← c82460] at e_1341_2
  have e_1341_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_3
  simp only [← c82461, ← c82462, k2, ← c82463] at e_1341_3
  have hr := e_1341_3
  simp only [e_1341_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12457)) := by
  have c82464 := (publicBatchWrapper89_copies2577 a h).1
  have c82465 := (publicBatchWrapper89_copies2577 a h).2.1
  have c82466 := (publicBatchWrapper89_copies2577 a h).2.2.1
  have c82467 := (publicBatchWrapper89_copies2577 a h).2.2.2.1
  have c82468 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.1
  have c82469 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_4
  simp only [← c82464, ← c82465, ← c82466] at e_1341_4
  have e_1341_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_5
  simp only [← c82467, ← c82468, k2, ← c82469] at e_1341_5
  have hr := e_1341_5
  simp only [e_1341_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12458)) := by
  have c82470 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.1
  have c82471 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.1
  have c82472 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.1
  have c82473 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.1
  have c82474 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82475 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_6
  simp only [← c82470, ← c82471, ← c82472] at e_1341_6
  have e_1341_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_7
  simp only [← c82473, ← c82474, k2, ← c82475] at e_1341_7
  have hr := e_1341_7
  simp only [e_1341_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12459)) := by
  have c82476 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82477 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82478 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82479 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82480 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82481 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_8
  simp only [← c82476, ← c82477, ← c82478] at e_1341_8
  have e_1341_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_9
  simp only [← c82479, ← c82480, k2, ← c82481] at e_1341_9
  have hr := e_1341_9
  simp only [e_1341_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12460)) := by
  have c82482 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82483 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82484 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82485 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82486 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82487 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_10
  simp only [← c82482, ← c82483, ← c82484] at e_1341_10
  have e_1341_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_11
  simp only [← c82485, ← c82486, k2, ← c82487] at e_1341_11
  have hr := e_1341_11
  simp only [e_1341_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12461)) := by
  have c82488 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82489 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82490 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82491 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82492 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82493 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_12
  simp only [← c82488, ← c82489, ← c82490] at e_1341_12
  have e_1341_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_13
  simp only [← c82491, ← c82492, k2, ← c82493] at e_1341_13
  have hr := e_1341_13
  simp only [e_1341_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12462)) := by
  have c82494 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82495 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82496 := (publicBatchWrapper89_copies2578 a h).1
  have c82497 := (publicBatchWrapper89_copies2578 a h).2.1
  have c82498 := (publicBatchWrapper89_copies2578 a h).2.2.1
  have c82499 := (publicBatchWrapper89_copies2578 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_14
  simp only [← c82494, ← c82495, ← c82496] at e_1341_14
  have e_1341_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_15
  simp only [← c82497, ← c82498, k2, ← c82499] at e_1341_15
  have hr := e_1341_15
  simp only [e_1341_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12463)) := by
  have c82500 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.1
  have c82501 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.1
  have c82502 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.1
  have c82503 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.1
  have c82504 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.1
  have c82505 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_16
  simp only [← c82500, ← c82501, ← c82502] at e_1341_16
  have e_1341_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_17
  simp only [← c82503, ← c82504, k2, ← c82505] at e_1341_17
  have hr := e_1341_17
  simp only [e_1341_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12464)) := by
  have c82506 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82507 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82508 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82509 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82510 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82511 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_18
  simp only [← c82506, ← c82507, ← c82508] at e_1341_18
  have e_1341_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1341 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_19
  simp only [← c82509, ← c82510, k2, ← c82511] at e_1341_19
  have hr := e_1341_19
  simp only [e_1341_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12465)) := by
  have c82512 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82513 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82514 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82515 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82516 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82517 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_0
  simp only [← c82512, ← c82513, ← c82514] at e_1342_0
  have e_1342_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_1
  simp only [← c82515, ← c82516, k2, ← c82517] at e_1342_1
  have hr := e_1342_1
  simp only [e_1342_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12466)) := by
  have c82518 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82519 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82520 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82521 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82522 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82523 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_2
  simp only [← c82518, ← c82519, ← c82520] at e_1342_2
  have e_1342_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_3
  simp only [← c82521, ← c82522, k2, ← c82523] at e_1342_3
  have hr := e_1342_3
  simp only [e_1342_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12467)) := by
  have c82524 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82525 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82526 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82527 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82528 := (publicBatchWrapper89_copies2579 a h).1
  have c82529 := (publicBatchWrapper89_copies2579 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_4
  simp only [← c82524, ← c82525, ← c82526] at e_1342_4
  have e_1342_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_5
  simp only [← c82527, ← c82528, k2, ← c82529] at e_1342_5
  have hr := e_1342_5
  simp only [e_1342_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12468)) := by
  have c82530 := (publicBatchWrapper89_copies2579 a h).2.2.1
  have c82531 := (publicBatchWrapper89_copies2579 a h).2.2.2.1
  have c82532 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.1
  have c82533 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.1
  have c82534 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.1
  have c82535 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_6
  simp only [← c82530, ← c82531, ← c82532] at e_1342_6
  have e_1342_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_7
  simp only [← c82533, ← c82534, k2, ← c82535] at e_1342_7
  have hr := e_1342_7
  simp only [e_1342_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12469)) := by
  have c82536 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.1
  have c82537 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.1
  have c82538 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82539 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82540 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82541 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_8
  simp only [← c82536, ← c82537, ← c82538] at e_1342_8
  have e_1342_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_9
  simp only [← c82539, ← c82540, k2, ← c82541] at e_1342_9
  have hr := e_1342_9
  simp only [e_1342_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12470)) := by
  have c82542 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82543 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82544 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82545 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82546 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82547 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_10
  simp only [← c82542, ← c82543, ← c82544] at e_1342_10
  have e_1342_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_11
  simp only [← c82545, ← c82546, k2, ← c82547] at e_1342_11
  have hr := e_1342_11
  simp only [e_1342_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12471)) := by
  have c82548 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82549 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82550 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82551 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82552 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82553 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_12
  simp only [← c82548, ← c82549, ← c82550] at e_1342_12
  have e_1342_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_13
  simp only [← c82551, ← c82552, k2, ← c82553] at e_1342_13
  have hr := e_1342_13
  simp only [e_1342_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12472)) := by
  have c82554 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82555 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82556 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82557 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82558 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82559 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_14
  simp only [← c82554, ← c82555, ← c82556] at e_1342_14
  have e_1342_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_15
  simp only [← c82557, ← c82558, k2, ← c82559] at e_1342_15
  have hr := e_1342_15
  simp only [e_1342_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12473)) := by
  have c82560 := (publicBatchWrapper89_copies2580 a h).1
  have c82561 := (publicBatchWrapper89_copies2580 a h).2.1
  have c82562 := (publicBatchWrapper89_copies2580 a h).2.2.1
  have c82563 := (publicBatchWrapper89_copies2580 a h).2.2.2.1
  have c82564 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.1
  have c82565 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_16
  simp only [← c82560, ← c82561, ← c82562] at e_1342_16
  have e_1342_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_17
  simp only [← c82563, ← c82564, k2, ← c82565] at e_1342_17
  have hr := e_1342_17
  simp only [e_1342_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12474)) := by
  have c82566 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.1
  have c82567 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.1
  have c82568 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.1
  have c82569 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.1
  have c82570 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82571 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_18
  simp only [← c82566, ← c82567, ← c82568] at e_1342_18
  have e_1342_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1342 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_19
  simp only [← c82569, ← c82570, k2, ← c82571] at e_1342_19
  have hr := e_1342_19
  simp only [e_1342_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12475)) := by
  have c82572 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82573 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82574 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82575 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82576 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82577 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_0
  simp only [← c82572, ← c82573, ← c82574] at e_1343_0
  have e_1343_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_1
  simp only [← c82575, ← c82576, k2, ← c82577] at e_1343_1
  have hr := e_1343_1
  simp only [e_1343_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12476)) := by
  have c82578 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82579 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82580 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82581 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82582 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82583 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_2
  simp only [← c82578, ← c82579, ← c82580] at e_1343_2
  have e_1343_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_3
  simp only [← c82581, ← c82582, k2, ← c82583] at e_1343_3
  have hr := e_1343_3
  simp only [e_1343_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12477)) := by
  have c82584 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82585 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82586 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82587 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82588 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82589 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_4
  simp only [← c82584, ← c82585, ← c82586] at e_1343_4
  have e_1343_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_5
  simp only [← c82587, ← c82588, k2, ← c82589] at e_1343_5
  have hr := e_1343_5
  simp only [e_1343_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12478)) := by
  have c82590 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82591 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82592 := (publicBatchWrapper89_copies2581 a h).1
  have c82593 := (publicBatchWrapper89_copies2581 a h).2.1
  have c82594 := (publicBatchWrapper89_copies2581 a h).2.2.1
  have c82595 := (publicBatchWrapper89_copies2581 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_6
  simp only [← c82590, ← c82591, ← c82592] at e_1343_6
  have e_1343_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_7
  simp only [← c82593, ← c82594, k2, ← c82595] at e_1343_7
  have hr := e_1343_7
  simp only [e_1343_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12479)) := by
  have c82596 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.1
  have c82597 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.1
  have c82598 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.1
  have c82599 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.1
  have c82600 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.1
  have c82601 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_8
  simp only [← c82596, ← c82597, ← c82598] at e_1343_8
  have e_1343_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_9
  simp only [← c82599, ← c82600, k2, ← c82601] at e_1343_9
  have hr := e_1343_9
  simp only [e_1343_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12480)) := by
  have c82602 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82603 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82604 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82605 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82606 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82607 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_10
  simp only [← c82602, ← c82603, ← c82604] at e_1343_10
  have e_1343_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_11
  simp only [← c82605, ← c82606, k2, ← c82607] at e_1343_11
  have hr := e_1343_11
  simp only [e_1343_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12481)) := by
  have c82608 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82609 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82610 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82611 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82612 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82613 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_12
  simp only [← c82608, ← c82609, ← c82610] at e_1343_12
  have e_1343_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_13
  simp only [← c82611, ← c82612, k2, ← c82613] at e_1343_13
  have hr := e_1343_13
  simp only [e_1343_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12482)) := by
  have c82614 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82615 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82616 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82617 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82618 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82619 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_14
  simp only [← c82614, ← c82615, ← c82616] at e_1343_14
  have e_1343_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_15
  simp only [← c82617, ← c82618, k2, ← c82619] at e_1343_15
  have hr := e_1343_15
  simp only [e_1343_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12483)) := by
  have c82620 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82621 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82622 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82623 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82624 := (publicBatchWrapper89_copies2582 a h).1
  have c82625 := (publicBatchWrapper89_copies2582 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_16
  simp only [← c82620, ← c82621, ← c82622] at e_1343_16
  have e_1343_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_17
  simp only [← c82623, ← c82624, k2, ← c82625] at e_1343_17
  have hr := e_1343_17
  simp only [e_1343_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12484)) := by
  have c82626 := (publicBatchWrapper89_copies2582 a h).2.2.1
  have c82627 := (publicBatchWrapper89_copies2582 a h).2.2.2.1
  have c82628 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.1
  have c82629 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.1
  have c82630 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.1
  have c82631 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_18
  simp only [← c82626, ← c82627, ← c82628] at e_1343_18
  have e_1343_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1343 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_19
  simp only [← c82629, ← c82630, k2, ← c82631] at e_1343_19
  have hr := e_1343_19
  simp only [e_1343_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12485)) := by
  have c82632 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.1
  have c82633 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.1
  have c82634 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82635 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82636 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82637 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_0
  simp only [← c82632, ← c82633, ← c82634] at e_1344_0
  have e_1344_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_1
  simp only [← c82635, ← c82636, k2, ← c82637] at e_1344_1
  have hr := e_1344_1
  simp only [e_1344_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12486)) := by
  have c82638 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82639 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82640 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82641 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82642 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82643 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_2
  simp only [← c82638, ← c82639, ← c82640] at e_1344_2
  have e_1344_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_3
  simp only [← c82641, ← c82642, k2, ← c82643] at e_1344_3
  have hr := e_1344_3
  simp only [e_1344_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12487)) := by
  have c82644 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82645 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82646 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82647 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82648 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82649 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_4
  simp only [← c82644, ← c82645, ← c82646] at e_1344_4
  have e_1344_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_5
  simp only [← c82647, ← c82648, k2, ← c82649] at e_1344_5
  have hr := e_1344_5
  simp only [e_1344_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12488)) := by
  have c82650 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82651 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82652 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82653 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82654 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82655 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_6
  simp only [← c82650, ← c82651, ← c82652] at e_1344_6
  have e_1344_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_7
  simp only [← c82653, ← c82654, k2, ← c82655] at e_1344_7
  have hr := e_1344_7
  simp only [e_1344_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12489)) := by
  have c82656 := (publicBatchWrapper89_copies2583 a h).1
  have c82657 := (publicBatchWrapper89_copies2583 a h).2.1
  have c82658 := (publicBatchWrapper89_copies2583 a h).2.2.1
  have c82659 := (publicBatchWrapper89_copies2583 a h).2.2.2.1
  have c82660 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.1
  have c82661 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_8
  simp only [← c82656, ← c82657, ← c82658] at e_1344_8
  have e_1344_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_9
  simp only [← c82659, ← c82660, k2, ← c82661] at e_1344_9
  have hr := e_1344_9
  simp only [e_1344_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12490)) := by
  have c82662 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.1
  have c82663 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.1
  have c82664 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.1
  have c82665 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.1
  have c82666 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82667 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_10
  simp only [← c82662, ← c82663, ← c82664] at e_1344_10
  have e_1344_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_11
  simp only [← c82665, ← c82666, k2, ← c82667] at e_1344_11
  have hr := e_1344_11
  simp only [e_1344_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12491)) := by
  have c82668 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82669 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82670 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82671 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82672 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82673 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_12
  simp only [← c82668, ← c82669, ← c82670] at e_1344_12
  have e_1344_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_13
  simp only [← c82671, ← c82672, k2, ← c82673] at e_1344_13
  have hr := e_1344_13
  simp only [e_1344_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12492)) := by
  have c82674 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82675 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82676 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82677 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82678 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82679 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_14
  simp only [← c82674, ← c82675, ← c82676] at e_1344_14
  have e_1344_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_15
  simp only [← c82677, ← c82678, k2, ← c82679] at e_1344_15
  have hr := e_1344_15
  simp only [e_1344_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12493)) := by
  have c82680 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82681 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82682 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82683 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82684 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82685 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_16
  simp only [← c82680, ← c82681, ← c82682] at e_1344_16
  have e_1344_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_17
  simp only [← c82683, ← c82684, k2, ← c82685] at e_1344_17
  have hr := e_1344_17
  simp only [e_1344_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12494)) := by
  have c82686 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82687 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82688 := (publicBatchWrapper89_copies2584 a h).1
  have c82689 := (publicBatchWrapper89_copies2584 a h).2.1
  have c82690 := (publicBatchWrapper89_copies2584 a h).2.2.1
  have c82691 := (publicBatchWrapper89_copies2584 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_18
  simp only [← c82686, ← c82687, ← c82688] at e_1344_18
  have e_1344_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1344 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_19
  simp only [← c82689, ← c82690, k2, ← c82691] at e_1344_19
  have hr := e_1344_19
  simp only [e_1344_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12495)) := by
  have c82692 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.1
  have c82693 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.1
  have c82694 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.1
  have c82695 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.1
  have c82696 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.1
  have c82697 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_0
  simp only [← c82692, ← c82693, ← c82694] at e_1345_0
  have e_1345_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_1
  simp only [← c82695, ← c82696, k2, ← c82697] at e_1345_1
  have hr := e_1345_1
  simp only [e_1345_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12496)) := by
  have c82698 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82699 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82700 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82701 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82702 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82703 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_2
  simp only [← c82698, ← c82699, ← c82700] at e_1345_2
  have e_1345_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_3
  simp only [← c82701, ← c82702, k2, ← c82703] at e_1345_3
  have hr := e_1345_3
  simp only [e_1345_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12497)) := by
  have c82704 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82705 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82706 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82707 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82708 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82709 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_4
  simp only [← c82704, ← c82705, ← c82706] at e_1345_4
  have e_1345_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_5
  simp only [← c82707, ← c82708, k2, ← c82709] at e_1345_5
  have hr := e_1345_5
  simp only [e_1345_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12498)) := by
  have c82710 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82711 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82712 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82713 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82714 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82715 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_6
  simp only [← c82710, ← c82711, ← c82712] at e_1345_6
  have e_1345_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_7
  simp only [← c82713, ← c82714, k2, ← c82715] at e_1345_7
  have hr := e_1345_7
  simp only [e_1345_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12499)) := by
  have c82716 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82717 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82718 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82719 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82720 := (publicBatchWrapper89_copies2585 a h).1
  have c82721 := (publicBatchWrapper89_copies2585 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_8
  simp only [← c82716, ← c82717, ← c82718] at e_1345_8
  have e_1345_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_9
  simp only [← c82719, ← c82720, k2, ← c82721] at e_1345_9
  have hr := e_1345_9
  simp only [e_1345_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12500)) := by
  have c82722 := (publicBatchWrapper89_copies2585 a h).2.2.1
  have c82723 := (publicBatchWrapper89_copies2585 a h).2.2.2.1
  have c82724 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.1
  have c82725 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.1
  have c82726 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.1
  have c82727 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_10
  simp only [← c82722, ← c82723, ← c82724] at e_1345_10
  have e_1345_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_11
  simp only [← c82725, ← c82726, k2, ← c82727] at e_1345_11
  have hr := e_1345_11
  simp only [e_1345_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12501)) := by
  have c82728 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.1
  have c82729 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.1
  have c82730 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82731 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82732 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82733 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_12
  simp only [← c82728, ← c82729, ← c82730] at e_1345_12
  have e_1345_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_13
  simp only [← c82731, ← c82732, k2, ← c82733] at e_1345_13
  have hr := e_1345_13
  simp only [e_1345_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12502)) := by
  have c82734 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82735 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82736 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82737 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82738 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82739 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_14
  simp only [← c82734, ← c82735, ← c82736] at e_1345_14
  have e_1345_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_15
  simp only [← c82737, ← c82738, k2, ← c82739] at e_1345_15
  have hr := e_1345_15
  simp only [e_1345_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12503)) := by
  have c82740 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82741 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82742 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82743 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82744 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82745 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_16
  simp only [← c82740, ← c82741, ← c82742] at e_1345_16
  have e_1345_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_17
  simp only [← c82743, ← c82744, k2, ← c82745] at e_1345_17
  have hr := e_1345_17
  simp only [e_1345_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12504)) := by
  have c82746 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82747 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82748 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82749 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82750 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82751 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_18
  simp only [← c82746, ← c82747, ← c82748] at e_1345_18
  have e_1345_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1345 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_19
  simp only [← c82749, ← c82750, k2, ← c82751] at e_1345_19
  have hr := e_1345_19
  simp only [e_1345_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12505)) := by
  have c82752 := (publicBatchWrapper89_copies2586 a h).1
  have c82753 := (publicBatchWrapper89_copies2586 a h).2.1
  have c82754 := (publicBatchWrapper89_copies2586 a h).2.2.1
  have c82755 := (publicBatchWrapper89_copies2586 a h).2.2.2.1
  have c82756 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.1
  have c82757 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_0
  simp only [← c82752, ← c82753, ← c82754] at e_1346_0
  have e_1346_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_1
  simp only [← c82755, ← c82756, k2, ← c82757] at e_1346_1
  have hr := e_1346_1
  simp only [e_1346_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12506)) := by
  have c82758 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.1
  have c82759 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.1
  have c82760 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.1
  have c82761 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.1
  have c82762 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82763 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_2
  simp only [← c82758, ← c82759, ← c82760] at e_1346_2
  have e_1346_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_3
  simp only [← c82761, ← c82762, k2, ← c82763] at e_1346_3
  have hr := e_1346_3
  simp only [e_1346_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12507)) := by
  have c82764 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82765 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82766 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82767 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82768 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82769 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_4
  simp only [← c82764, ← c82765, ← c82766] at e_1346_4
  have e_1346_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_5
  simp only [← c82767, ← c82768, k2, ← c82769] at e_1346_5
  have hr := e_1346_5
  simp only [e_1346_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12508)) := by
  have c82770 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82771 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82772 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82773 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82774 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82775 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_6
  simp only [← c82770, ← c82771, ← c82772] at e_1346_6
  have e_1346_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_7
  simp only [← c82773, ← c82774, k2, ← c82775] at e_1346_7
  have hr := e_1346_7
  simp only [e_1346_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12509)) := by
  have c82776 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82777 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82778 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82779 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82780 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82781 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_8
  simp only [← c82776, ← c82777, ← c82778] at e_1346_8
  have e_1346_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_9
  simp only [← c82779, ← c82780, k2, ← c82781] at e_1346_9
  have hr := e_1346_9
  simp only [e_1346_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12510)) := by
  have c82782 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82783 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82784 := (publicBatchWrapper89_copies2587 a h).1
  have c82785 := (publicBatchWrapper89_copies2587 a h).2.1
  have c82786 := (publicBatchWrapper89_copies2587 a h).2.2.1
  have c82787 := (publicBatchWrapper89_copies2587 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_10
  simp only [← c82782, ← c82783, ← c82784] at e_1346_10
  have e_1346_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_11
  simp only [← c82785, ← c82786, k2, ← c82787] at e_1346_11
  have hr := e_1346_11
  simp only [e_1346_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12511)) := by
  have c82788 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.1
  have c82789 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.1
  have c82790 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.1
  have c82791 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.1
  have c82792 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.1
  have c82793 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_12
  simp only [← c82788, ← c82789, ← c82790] at e_1346_12
  have e_1346_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_13
  simp only [← c82791, ← c82792, k2, ← c82793] at e_1346_13
  have hr := e_1346_13
  simp only [e_1346_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12512)) := by
  have c82794 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82795 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82796 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82797 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82798 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82799 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_14
  simp only [← c82794, ← c82795, ← c82796] at e_1346_14
  have e_1346_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_15
  simp only [← c82797, ← c82798, k2, ← c82799] at e_1346_15
  have hr := e_1346_15
  simp only [e_1346_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12513)) := by
  have c82800 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82801 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82802 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82803 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82804 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82805 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_16
  simp only [← c82800, ← c82801, ← c82802] at e_1346_16
  have e_1346_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_17
  simp only [← c82803, ← c82804, k2, ← c82805] at e_1346_17
  have hr := e_1346_17
  simp only [e_1346_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12514)) := by
  have c82806 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82807 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82808 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82809 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82810 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82811 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_18
  simp only [← c82806, ← c82807, ← c82808] at e_1346_18
  have e_1346_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1346 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_19
  simp only [← c82809, ← c82810, k2, ← c82811] at e_1346_19
  have hr := e_1346_19
  simp only [e_1346_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12515)) := by
  have c82812 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82813 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82814 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82815 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82816 := (publicBatchWrapper89_copies2588 a h).1
  have c82817 := (publicBatchWrapper89_copies2588 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_0
  simp only [← c82812, ← c82813, ← c82814] at e_1347_0
  have e_1347_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_1
  simp only [← c82815, ← c82816, k2, ← c82817] at e_1347_1
  have hr := e_1347_1
  simp only [e_1347_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12516)) := by
  have c82818 := (publicBatchWrapper89_copies2588 a h).2.2.1
  have c82819 := (publicBatchWrapper89_copies2588 a h).2.2.2.1
  have c82820 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.1
  have c82821 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.1
  have c82822 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.1
  have c82823 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_2
  simp only [← c82818, ← c82819, ← c82820] at e_1347_2
  have e_1347_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_3
  simp only [← c82821, ← c82822, k2, ← c82823] at e_1347_3
  have hr := e_1347_3
  simp only [e_1347_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12517)) := by
  have c82824 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.1
  have c82825 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.1
  have c82826 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82827 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82828 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82829 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_4
  simp only [← c82824, ← c82825, ← c82826] at e_1347_4
  have e_1347_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_5
  simp only [← c82827, ← c82828, k2, ← c82829] at e_1347_5
  have hr := e_1347_5
  simp only [e_1347_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12518)) := by
  have c82830 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82831 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82832 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82833 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82834 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82835 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_6
  simp only [← c82830, ← c82831, ← c82832] at e_1347_6
  have e_1347_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_7
  simp only [← c82833, ← c82834, k2, ← c82835] at e_1347_7
  have hr := e_1347_7
  simp only [e_1347_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12519)) := by
  have c82836 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82837 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82838 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82839 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82840 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82841 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_8
  simp only [← c82836, ← c82837, ← c82838] at e_1347_8
  have e_1347_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_9
  simp only [← c82839, ← c82840, k2, ← c82841] at e_1347_9
  have hr := e_1347_9
  simp only [e_1347_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12520)) := by
  have c82842 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82843 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82844 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82845 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82846 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82847 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_10
  simp only [← c82842, ← c82843, ← c82844] at e_1347_10
  have e_1347_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_11
  simp only [← c82845, ← c82846, k2, ← c82847] at e_1347_11
  have hr := e_1347_11
  simp only [e_1347_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12521)) := by
  have c82848 := (publicBatchWrapper89_copies2589 a h).1
  have c82849 := (publicBatchWrapper89_copies2589 a h).2.1
  have c82850 := (publicBatchWrapper89_copies2589 a h).2.2.1
  have c82851 := (publicBatchWrapper89_copies2589 a h).2.2.2.1
  have c82852 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.1
  have c82853 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_12
  simp only [← c82848, ← c82849, ← c82850] at e_1347_12
  have e_1347_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_13
  simp only [← c82851, ← c82852, k2, ← c82853] at e_1347_13
  have hr := e_1347_13
  simp only [e_1347_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12522)) := by
  have c82854 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.1
  have c82855 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.1
  have c82856 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.1
  have c82857 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.1
  have c82858 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82859 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_14
  simp only [← c82854, ← c82855, ← c82856] at e_1347_14
  have e_1347_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_15
  simp only [← c82857, ← c82858, k2, ← c82859] at e_1347_15
  have hr := e_1347_15
  simp only [e_1347_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12523)) := by
  have c82860 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82861 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82862 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82863 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82864 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82865 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_16
  simp only [← c82860, ← c82861, ← c82862] at e_1347_16
  have e_1347_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_17
  simp only [← c82863, ← c82864, k2, ← c82865] at e_1347_17
  have hr := e_1347_17
  simp only [e_1347_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12524)) := by
  have c82866 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82867 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82868 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82869 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82870 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82871 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_18
  simp only [← c82866, ← c82867, ← c82868] at e_1347_18
  have e_1347_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1347 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_19
  simp only [← c82869, ← c82870, k2, ← c82871] at e_1347_19
  have hr := e_1347_19
  simp only [e_1347_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12525)) := by
  have c82872 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82873 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82874 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82875 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82876 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82877 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_0
  simp only [← c82872, ← c82873, ← c82874] at e_1348_0
  have e_1348_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_1
  simp only [← c82875, ← c82876, k2, ← c82877] at e_1348_1
  have hr := e_1348_1
  simp only [e_1348_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12526)) := by
  have c82878 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82879 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82880 := (publicBatchWrapper89_copies2590 a h).1
  have c82881 := (publicBatchWrapper89_copies2590 a h).2.1
  have c82882 := (publicBatchWrapper89_copies2590 a h).2.2.1
  have c82883 := (publicBatchWrapper89_copies2590 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_2
  simp only [← c82878, ← c82879, ← c82880] at e_1348_2
  have e_1348_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_3
  simp only [← c82881, ← c82882, k2, ← c82883] at e_1348_3
  have hr := e_1348_3
  simp only [e_1348_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12527)) := by
  have c82884 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.1
  have c82885 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.1
  have c82886 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.1
  have c82887 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.1
  have c82888 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.1
  have c82889 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_4
  simp only [← c82884, ← c82885, ← c82886] at e_1348_4
  have e_1348_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_5
  simp only [← c82887, ← c82888, k2, ← c82889] at e_1348_5
  have hr := e_1348_5
  simp only [e_1348_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12528)) := by
  have c82890 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82891 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82892 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82893 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82894 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82895 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_6
  simp only [← c82890, ← c82891, ← c82892] at e_1348_6
  have e_1348_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_7
  simp only [← c82893, ← c82894, k2, ← c82895] at e_1348_7
  have hr := e_1348_7
  simp only [e_1348_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12529)) := by
  have c82896 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82897 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82898 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82899 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82900 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82901 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_8
  simp only [← c82896, ← c82897, ← c82898] at e_1348_8
  have e_1348_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_9
  simp only [← c82899, ← c82900, k2, ← c82901] at e_1348_9
  have hr := e_1348_9
  simp only [e_1348_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12530)) := by
  have c82902 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82903 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82904 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82905 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82906 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82907 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_10
  simp only [← c82902, ← c82903, ← c82904] at e_1348_10
  have e_1348_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_11
  simp only [← c82905, ← c82906, k2, ← c82907] at e_1348_11
  have hr := e_1348_11
  simp only [e_1348_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12531)) := by
  have c82908 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82909 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82910 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82911 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82912 := (publicBatchWrapper89_copies2591 a h).1
  have c82913 := (publicBatchWrapper89_copies2591 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_12
  simp only [← c82908, ← c82909, ← c82910] at e_1348_12
  have e_1348_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_13
  simp only [← c82911, ← c82912, k2, ← c82913] at e_1348_13
  have hr := e_1348_13
  simp only [e_1348_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12532)) := by
  have c82914 := (publicBatchWrapper89_copies2591 a h).2.2.1
  have c82915 := (publicBatchWrapper89_copies2591 a h).2.2.2.1
  have c82916 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.1
  have c82917 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.1
  have c82918 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.1
  have c82919 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_14
  simp only [← c82914, ← c82915, ← c82916] at e_1348_14
  have e_1348_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_15
  simp only [← c82917, ← c82918, k2, ← c82919] at e_1348_15
  have hr := e_1348_15
  simp only [e_1348_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12533)) := by
  have c82920 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.1
  have c82921 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.1
  have c82922 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82923 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82924 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82925 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_16
  simp only [← c82920, ← c82921, ← c82922] at e_1348_16
  have e_1348_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_17
  simp only [← c82923, ← c82924, k2, ← c82925] at e_1348_17
  have hr := e_1348_17
  simp only [e_1348_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12534)) := by
  have c82926 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82927 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82928 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82929 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82930 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82931 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_18
  simp only [← c82926, ← c82927, ← c82928] at e_1348_18
  have e_1348_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1348 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_19
  simp only [← c82929, ← c82930, k2, ← c82931] at e_1348_19
  have hr := e_1348_19
  simp only [e_1348_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12535)) := by
  have c82932 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82933 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82934 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82935 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82936 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82937 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_0
  simp only [← c82932, ← c82933, ← c82934] at e_1349_0
  have e_1349_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_1
  simp only [← c82935, ← c82936, k2, ← c82937] at e_1349_1
  have hr := e_1349_1
  simp only [e_1349_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12536)) := by
  have c82938 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82939 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82940 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82941 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82942 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82943 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_2
  simp only [← c82938, ← c82939, ← c82940] at e_1349_2
  have e_1349_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_3
  simp only [← c82941, ← c82942, k2, ← c82943] at e_1349_3
  have hr := e_1349_3
  simp only [e_1349_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12537)) := by
  have c82944 := (publicBatchWrapper89_copies2592 a h).1
  have c82945 := (publicBatchWrapper89_copies2592 a h).2.1
  have c82946 := (publicBatchWrapper89_copies2592 a h).2.2.1
  have c82947 := (publicBatchWrapper89_copies2592 a h).2.2.2.1
  have c82948 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.1
  have c82949 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_4
  simp only [← c82944, ← c82945, ← c82946] at e_1349_4
  have e_1349_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_5
  simp only [← c82947, ← c82948, k2, ← c82949] at e_1349_5
  have hr := e_1349_5
  simp only [e_1349_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12538)) := by
  have c82950 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.1
  have c82951 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.1
  have c82952 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.1
  have c82953 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.1
  have c82954 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82955 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_6
  simp only [← c82950, ← c82951, ← c82952] at e_1349_6
  have e_1349_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_7
  simp only [← c82953, ← c82954, k2, ← c82955] at e_1349_7
  have hr := e_1349_7
  simp only [e_1349_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12539)) := by
  have c82956 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82957 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82958 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82959 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82960 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82961 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_8
  simp only [← c82956, ← c82957, ← c82958] at e_1349_8
  have e_1349_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_9
  simp only [← c82959, ← c82960, k2, ← c82961] at e_1349_9
  have hr := e_1349_9
  simp only [e_1349_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12540)) := by
  have c82962 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82963 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82964 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82965 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82966 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82967 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_10
  simp only [← c82962, ← c82963, ← c82964] at e_1349_10
  have e_1349_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_11
  simp only [← c82965, ← c82966, k2, ← c82967] at e_1349_11
  have hr := e_1349_11
  simp only [e_1349_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12541)) := by
  have c82968 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82969 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82970 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82971 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82972 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82973 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_12
  simp only [← c82968, ← c82969, ← c82970] at e_1349_12
  have e_1349_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_13
  simp only [← c82971, ← c82972, k2, ← c82973] at e_1349_13
  have hr := e_1349_13
  simp only [e_1349_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12542)) := by
  have c82974 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82975 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82976 := (publicBatchWrapper89_copies2593 a h).1
  have c82977 := (publicBatchWrapper89_copies2593 a h).2.1
  have c82978 := (publicBatchWrapper89_copies2593 a h).2.2.1
  have c82979 := (publicBatchWrapper89_copies2593 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_14
  simp only [← c82974, ← c82975, ← c82976] at e_1349_14
  have e_1349_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_15
  simp only [← c82977, ← c82978, k2, ← c82979] at e_1349_15
  have hr := e_1349_15
  simp only [e_1349_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12543)) := by
  have c82980 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.1
  have c82981 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.1
  have c82982 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.1
  have c82983 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.1
  have c82984 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.1
  have c82985 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_16
  simp only [← c82980, ← c82981, ← c82982] at e_1349_16
  have e_1349_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_17
  simp only [← c82983, ← c82984, k2, ← c82985] at e_1349_17
  have hr := e_1349_17
  simp only [e_1349_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12544)) := by
  have c82986 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82987 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82988 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82989 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82990 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82991 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_18
  simp only [← c82986, ← c82987, ← c82988] at e_1349_18
  have e_1349_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1349 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_19
  simp only [← c82989, ← c82990, k2, ← c82991] at e_1349_19
  have hr := e_1349_19
  simp only [e_1349_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12545)) := by
  have c82992 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82993 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82994 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82995 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82996 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82997 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_0
  simp only [← c82992, ← c82993, ← c82994] at e_1350_0
  have e_1350_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_1
  simp only [← c82995, ← c82996, k2, ← c82997] at e_1350_1
  have hr := e_1350_1
  simp only [e_1350_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12546)) := by
  have c82998 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82999 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83000 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83001 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83002 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83003 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_2
  simp only [← c82998, ← c82999, ← c83000] at e_1350_2
  have e_1350_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_3
  simp only [← c83001, ← c83002, k2, ← c83003] at e_1350_3
  have hr := e_1350_3
  simp only [e_1350_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12547)) := by
  have c83004 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83005 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83006 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83007 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83008 := (publicBatchWrapper89_copies2594 a h).1
  have c83009 := (publicBatchWrapper89_copies2594 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_4
  simp only [← c83004, ← c83005, ← c83006] at e_1350_4
  have e_1350_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_5
  simp only [← c83007, ← c83008, k2, ← c83009] at e_1350_5
  have hr := e_1350_5
  simp only [e_1350_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12548)) := by
  have c83010 := (publicBatchWrapper89_copies2594 a h).2.2.1
  have c83011 := (publicBatchWrapper89_copies2594 a h).2.2.2.1
  have c83012 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.1
  have c83013 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.1
  have c83014 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.1
  have c83015 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_6
  simp only [← c83010, ← c83011, ← c83012] at e_1350_6
  have e_1350_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_7
  simp only [← c83013, ← c83014, k2, ← c83015] at e_1350_7
  have hr := e_1350_7
  simp only [e_1350_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12549)) := by
  have c83016 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.1
  have c83017 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.1
  have c83018 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83019 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83020 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83021 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_8
  simp only [← c83016, ← c83017, ← c83018] at e_1350_8
  have e_1350_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_9
  simp only [← c83019, ← c83020, k2, ← c83021] at e_1350_9
  have hr := e_1350_9
  simp only [e_1350_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12550)) := by
  have c83022 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83023 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83024 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83025 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83026 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83027 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_10
  simp only [← c83022, ← c83023, ← c83024] at e_1350_10
  have e_1350_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_11
  simp only [← c83025, ← c83026, k2, ← c83027] at e_1350_11
  have hr := e_1350_11
  simp only [e_1350_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12551)) := by
  have c83028 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83029 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83030 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83031 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83032 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83033 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_12
  simp only [← c83028, ← c83029, ← c83030] at e_1350_12
  have e_1350_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_13
  simp only [← c83031, ← c83032, k2, ← c83033] at e_1350_13
  have hr := e_1350_13
  simp only [e_1350_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12552)) := by
  have c83034 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83035 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83036 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83037 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83038 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83039 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_14
  simp only [← c83034, ← c83035, ← c83036] at e_1350_14
  have e_1350_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_15
  simp only [← c83037, ← c83038, k2, ← c83039] at e_1350_15
  have hr := e_1350_15
  simp only [e_1350_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12553)) := by
  have c83040 := (publicBatchWrapper89_copies2595 a h).1
  have c83041 := (publicBatchWrapper89_copies2595 a h).2.1
  have c83042 := (publicBatchWrapper89_copies2595 a h).2.2.1
  have c83043 := (publicBatchWrapper89_copies2595 a h).2.2.2.1
  have c83044 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.1
  have c83045 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_16
  simp only [← c83040, ← c83041, ← c83042] at e_1350_16
  have e_1350_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_17
  simp only [← c83043, ← c83044, k2, ← c83045] at e_1350_17
  have hr := e_1350_17
  simp only [e_1350_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12554)) := by
  have c83046 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.1
  have c83047 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.1
  have c83048 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.1
  have c83049 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.1
  have c83050 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83051 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_18
  simp only [← c83046, ← c83047, ← c83048] at e_1350_18
  have e_1350_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1350 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_19
  simp only [← c83049, ← c83050, k2, ← c83051] at e_1350_19
  have hr := e_1350_19
  simp only [e_1350_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12555)) := by
  have c83052 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83053 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83054 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83055 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83056 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83057 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_0
  simp only [← c83052, ← c83053, ← c83054] at e_1351_0
  have e_1351_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_1
  simp only [← c83055, ← c83056, k2, ← c83057] at e_1351_1
  have hr := e_1351_1
  simp only [e_1351_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12556)) := by
  have c83058 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83059 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83060 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83061 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83062 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83063 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_2
  simp only [← c83058, ← c83059, ← c83060] at e_1351_2
  have e_1351_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_3
  simp only [← c83061, ← c83062, k2, ← c83063] at e_1351_3
  have hr := e_1351_3
  simp only [e_1351_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12557)) := by
  have c83064 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83065 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83066 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83067 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83068 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83069 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_4
  simp only [← c83064, ← c83065, ← c83066] at e_1351_4
  have e_1351_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_5
  simp only [← c83067, ← c83068, k2, ← c83069] at e_1351_5
  have hr := e_1351_5
  simp only [e_1351_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12558)) := by
  have c83070 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83071 := (publicBatchWrapper89_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83072 := (publicBatchWrapper89_copies2596 a h).1
  have c83073 := (publicBatchWrapper89_copies2596 a h).2.1
  have c83074 := (publicBatchWrapper89_copies2596 a h).2.2.1
  have c83075 := (publicBatchWrapper89_copies2596 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_6
  simp only [← c83070, ← c83071, ← c83072] at e_1351_6
  have e_1351_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_7
  simp only [← c83073, ← c83074, k2, ← c83075] at e_1351_7
  have hr := e_1351_7
  simp only [e_1351_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12559)) := by
  have c83076 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.1
  have c83077 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.1
  have c83078 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.1
  have c83079 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.1
  have c83080 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.1
  have c83081 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_8
  simp only [← c83076, ← c83077, ← c83078] at e_1351_8
  have e_1351_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_9
  simp only [← c83079, ← c83080, k2, ← c83081] at e_1351_9
  have hr := e_1351_9
  simp only [e_1351_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12560)) := by
  have c83082 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83083 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83084 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83085 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83086 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83087 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_10
  simp only [← c83082, ← c83083, ← c83084] at e_1351_10
  have e_1351_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_11
  simp only [← c83085, ← c83086, k2, ← c83087] at e_1351_11
  have hr := e_1351_11
  simp only [e_1351_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12561)) := by
  have c83088 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83089 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83090 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83091 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83092 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83093 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_12
  simp only [← c83088, ← c83089, ← c83090] at e_1351_12
  have e_1351_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_13
  simp only [← c83091, ← c83092, k2, ← c83093] at e_1351_13
  have hr := e_1351_13
  simp only [e_1351_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12562)) := by
  have c83094 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83095 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83096 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83097 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83098 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83099 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_14
  simp only [← c83094, ← c83095, ← c83096] at e_1351_14
  have e_1351_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_15
  simp only [← c83097, ← c83098, k2, ← c83099] at e_1351_15
  have hr := e_1351_15
  simp only [e_1351_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12563)) := by
  have c83100 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83101 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83102 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83103 := (publicBatchWrapper89_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83104 := (publicBatchWrapper89_copies2597 a h).1
  have c83105 := (publicBatchWrapper89_copies2597 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_16
  simp only [← c83100, ← c83101, ← c83102] at e_1351_16
  have e_1351_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_17
  simp only [← c83103, ← c83104, k2, ← c83105] at e_1351_17
  have hr := e_1351_17
  simp only [e_1351_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1351 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12564)) := by
  have c83106 := (publicBatchWrapper89_copies2597 a h).2.2.1
  have c83107 := (publicBatchWrapper89_copies2597 a h).2.2.2.1
  have c83108 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.1
  have c83109 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.1
  have c83110 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.1
  have c83111 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1351_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_18
  simp only [← c83106, ← c83107, ← c83108] at e_1351_18
  have e_1351_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1351 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_19
  simp only [← c83109, ← c83110, k2, ← c83111] at e_1351_19
  have hr := e_1351_19
  simp only [e_1351_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12565)) := by
  have c83112 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.1
  have c83113 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.1
  have c83114 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83115 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83116 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83117 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_0
  simp only [← c83112, ← c83113, ← c83114] at e_1352_0
  have e_1352_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_1
  simp only [← c83115, ← c83116, k2, ← c83117] at e_1352_1
  have hr := e_1352_1
  simp only [e_1352_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12566)) := by
  have c83118 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83119 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83120 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83121 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83122 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83123 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_2
  simp only [← c83118, ← c83119, ← c83120] at e_1352_2
  have e_1352_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_3
  simp only [← c83121, ← c83122, k2, ← c83123] at e_1352_3
  have hr := e_1352_3
  simp only [e_1352_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12567)) := by
  have c83124 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83125 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83126 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83127 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83128 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83129 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_4
  simp only [← c83124, ← c83125, ← c83126] at e_1352_4
  have e_1352_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_5
  simp only [← c83127, ← c83128, k2, ← c83129] at e_1352_5
  have hr := e_1352_5
  simp only [e_1352_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12568)) := by
  have c83130 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83131 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83132 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83133 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83134 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83135 := (publicBatchWrapper89_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_6
  simp only [← c83130, ← c83131, ← c83132] at e_1352_6
  have e_1352_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_7
  simp only [← c83133, ← c83134, k2, ← c83135] at e_1352_7
  have hr := e_1352_7
  simp only [e_1352_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12569)) := by
  have c83136 := (publicBatchWrapper89_copies2598 a h).1
  have c83137 := (publicBatchWrapper89_copies2598 a h).2.1
  have c83138 := (publicBatchWrapper89_copies2598 a h).2.2.1
  have c83139 := (publicBatchWrapper89_copies2598 a h).2.2.2.1
  have c83140 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.1
  have c83141 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_8
  simp only [← c83136, ← c83137, ← c83138] at e_1352_8
  have e_1352_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_9
  simp only [← c83139, ← c83140, k2, ← c83141] at e_1352_9
  have hr := e_1352_9
  simp only [e_1352_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12610)) := by
  have c83142 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.1
  have c83143 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.1
  have c83144 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.1
  have c83145 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.1
  have c83146 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83147 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_10
  simp only [← c83142, ← c83143, ← c83144] at e_1352_10
  have e_1352_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_11
  simp only [← c83145, ← c83146, k2, ← c83147] at e_1352_11
  have hr := e_1352_11
  simp only [e_1352_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12611)) := by
  have c83148 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83149 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83150 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83151 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83152 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83153 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_12
  simp only [← c83148, ← c83149, ← c83150] at e_1352_12
  have e_1352_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_13
  simp only [← c83151, ← c83152, k2, ← c83153] at e_1352_13
  have hr := e_1352_13
  simp only [e_1352_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12612)) := by
  have c83154 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83155 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83156 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83157 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83158 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83159 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_14
  simp only [← c83154, ← c83155, ← c83156] at e_1352_14
  have e_1352_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_15
  simp only [← c83157, ← c83158, k2, ← c83159] at e_1352_15
  have hr := e_1352_15
  simp only [e_1352_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12613)) := by
  have c83160 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83161 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83162 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83163 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83164 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83165 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_16
  simp only [← c83160, ← c83161, ← c83162] at e_1352_16
  have e_1352_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_17
  simp only [← c83163, ← c83164, k2, ← c83165] at e_1352_17
  have hr := e_1352_17
  simp only [e_1352_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1352 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12614)) := by
  have c83166 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83167 := (publicBatchWrapper89_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83168 := (publicBatchWrapper89_copies2599 a h).1
  have c83169 := (publicBatchWrapper89_copies2599 a h).2.1
  have c83170 := (publicBatchWrapper89_copies2599 a h).2.2.1
  have c83171 := (publicBatchWrapper89_copies2599 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1352_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_18
  simp only [← c83166, ← c83167, ← c83168] at e_1352_18
  have e_1352_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1352 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_19
  simp only [← c83169, ← c83170, k2, ← c83171] at e_1352_19
  have hr := e_1352_19
  simp only [e_1352_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12615)) := by
  have c83172 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.1
  have c83173 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.1
  have c83174 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.1
  have c83175 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.1
  have c83176 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.1
  have c83177 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_0
  simp only [← c83172, ← c83173, ← c83174] at e_1353_0
  have e_1353_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_1
  simp only [← c83175, ← c83176, k2, ← c83177] at e_1353_1
  have hr := e_1353_1
  simp only [e_1353_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12616)) := by
  have c83178 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83179 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83180 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83181 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83182 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83183 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_2
  simp only [← c83178, ← c83179, ← c83180] at e_1353_2
  have e_1353_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_3
  simp only [← c83181, ← c83182, k2, ← c83183] at e_1353_3
  have hr := e_1353_3
  simp only [e_1353_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12617)) := by
  have c83184 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83185 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83186 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83187 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83188 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83189 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_4
  simp only [← c83184, ← c83185, ← c83186] at e_1353_4
  have e_1353_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_5
  simp only [← c83187, ← c83188, k2, ← c83189] at e_1353_5
  have hr := e_1353_5
  simp only [e_1353_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12618)) := by
  have c83190 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83191 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83192 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83193 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83194 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83195 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_6
  simp only [← c83190, ← c83191, ← c83192] at e_1353_6
  have e_1353_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_7
  simp only [← c83193, ← c83194, k2, ← c83195] at e_1353_7
  have hr := e_1353_7
  simp only [e_1353_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12619)) := by
  have c83196 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83197 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83198 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83199 := (publicBatchWrapper89_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83200 := (publicBatchWrapper89_copies2600 a h).1
  have c83201 := (publicBatchWrapper89_copies2600 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_8
  simp only [← c83196, ← c83197, ← c83198] at e_1353_8
  have e_1353_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_9
  simp only [← c83199, ← c83200, k2, ← c83201] at e_1353_9
  have hr := e_1353_9
  simp only [e_1353_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12620)) := by
  have c83202 := (publicBatchWrapper89_copies2600 a h).2.2.1
  have c83203 := (publicBatchWrapper89_copies2600 a h).2.2.2.1
  have c83204 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.1
  have c83205 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.1
  have c83206 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.1
  have c83207 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_10
  simp only [← c83202, ← c83203, ← c83204] at e_1353_10
  have e_1353_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_11
  simp only [← c83205, ← c83206, k2, ← c83207] at e_1353_11
  have hr := e_1353_11
  simp only [e_1353_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12621)) := by
  have c83208 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.1
  have c83209 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.1
  have c83210 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83211 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83212 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83213 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_12
  simp only [← c83208, ← c83209, ← c83210] at e_1353_12
  have e_1353_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_13
  simp only [← c83211, ← c83212, k2, ← c83213] at e_1353_13
  have hr := e_1353_13
  simp only [e_1353_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12622)) := by
  have c83214 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83215 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83216 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83217 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83218 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83219 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_14
  simp only [← c83214, ← c83215, ← c83216] at e_1353_14
  have e_1353_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_15
  simp only [← c83217, ← c83218, k2, ← c83219] at e_1353_15
  have hr := e_1353_15
  simp only [e_1353_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12623)) := by
  have c83220 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83221 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83222 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83223 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83224 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83225 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_16
  simp only [← c83220, ← c83221, ← c83222] at e_1353_16
  have e_1353_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_17
  simp only [← c83223, ← c83224, k2, ← c83225] at e_1353_17
  have hr := e_1353_17
  simp only [e_1353_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1353 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12624)) := by
  have c83226 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83227 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83228 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83229 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83230 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83231 := (publicBatchWrapper89_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1353_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_18
  simp only [← c83226, ← c83227, ← c83228] at e_1353_18
  have e_1353_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1353 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_19
  simp only [← c83229, ← c83230, k2, ← c83231] at e_1353_19
  have hr := e_1353_19
  simp only [e_1353_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12625)) := by
  have c83232 := (publicBatchWrapper89_copies2601 a h).1
  have c83233 := (publicBatchWrapper89_copies2601 a h).2.1
  have c83234 := (publicBatchWrapper89_copies2601 a h).2.2.1
  have c83235 := (publicBatchWrapper89_copies2601 a h).2.2.2.1
  have c83236 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.1
  have c83237 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_0
  simp only [← c83232, ← c83233, ← c83234] at e_1354_0
  have e_1354_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_1
  simp only [← c83235, ← c83236, k2, ← c83237] at e_1354_1
  have hr := e_1354_1
  simp only [e_1354_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12626)) := by
  have c83238 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.1
  have c83239 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.1
  have c83240 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.1
  have c83241 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.1
  have c83242 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83243 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_2
  simp only [← c83238, ← c83239, ← c83240] at e_1354_2
  have e_1354_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_3
  simp only [← c83241, ← c83242, k2, ← c83243] at e_1354_3
  have hr := e_1354_3
  simp only [e_1354_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12627)) := by
  have c83244 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83245 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83246 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83247 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83248 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83249 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_4
  simp only [← c83244, ← c83245, ← c83246] at e_1354_4
  have e_1354_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_5
  simp only [← c83247, ← c83248, k2, ← c83249] at e_1354_5
  have hr := e_1354_5
  simp only [e_1354_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12628)) := by
  have c83250 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83251 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83252 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83253 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83254 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83255 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_6
  simp only [← c83250, ← c83251, ← c83252] at e_1354_6
  have e_1354_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_7
  simp only [← c83253, ← c83254, k2, ← c83255] at e_1354_7
  have hr := e_1354_7
  simp only [e_1354_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12629)) := by
  have c83256 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83257 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83258 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83259 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83260 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83261 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_8
  simp only [← c83256, ← c83257, ← c83258] at e_1354_8
  have e_1354_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_9
  simp only [← c83259, ← c83260, k2, ← c83261] at e_1354_9
  have hr := e_1354_9
  simp only [e_1354_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12630)) := by
  have c83262 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83263 := (publicBatchWrapper89_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83264 := (publicBatchWrapper89_copies2602 a h).1
  have c83265 := (publicBatchWrapper89_copies2602 a h).2.1
  have c83266 := (publicBatchWrapper89_copies2602 a h).2.2.1
  have c83267 := (publicBatchWrapper89_copies2602 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_10
  simp only [← c83262, ← c83263, ← c83264] at e_1354_10
  have e_1354_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_11
  simp only [← c83265, ← c83266, k2, ← c83267] at e_1354_11
  have hr := e_1354_11
  simp only [e_1354_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12631)) := by
  have c83268 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.1
  have c83269 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.1
  have c83270 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.1
  have c83271 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.1
  have c83272 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.1
  have c83273 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_12
  simp only [← c83268, ← c83269, ← c83270] at e_1354_12
  have e_1354_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_13
  simp only [← c83271, ← c83272, k2, ← c83273] at e_1354_13
  have hr := e_1354_13
  simp only [e_1354_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12632)) := by
  have c83274 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83275 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83276 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83277 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83278 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83279 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_14
  simp only [← c83274, ← c83275, ← c83276] at e_1354_14
  have e_1354_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_15
  simp only [← c83277, ← c83278, k2, ← c83279] at e_1354_15
  have hr := e_1354_15
  simp only [e_1354_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12633)) := by
  have c83280 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83281 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83282 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83283 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83284 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83285 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_16
  simp only [← c83280, ← c83281, ← c83282] at e_1354_16
  have e_1354_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_17
  simp only [← c83283, ← c83284, k2, ← c83285] at e_1354_17
  have hr := e_1354_17
  simp only [e_1354_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1354 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12634)) := by
  have c83286 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83287 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83288 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83289 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83290 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83291 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1354_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_18
  simp only [← c83286, ← c83287, ← c83288] at e_1354_18
  have e_1354_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1354 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_19
  simp only [← c83289, ← c83290, k2, ← c83291] at e_1354_19
  have hr := e_1354_19
  simp only [e_1354_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12635)) := by
  have c83292 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83293 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83294 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83295 := (publicBatchWrapper89_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83296 := (publicBatchWrapper89_copies2603 a h).1
  have c83297 := (publicBatchWrapper89_copies2603 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_0
  simp only [← c83292, ← c83293, ← c83294] at e_1355_0
  have e_1355_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_1
  simp only [← c83295, ← c83296, k2, ← c83297] at e_1355_1
  have hr := e_1355_1
  simp only [e_1355_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12636)) := by
  have c83298 := (publicBatchWrapper89_copies2603 a h).2.2.1
  have c83299 := (publicBatchWrapper89_copies2603 a h).2.2.2.1
  have c83300 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.1
  have c83301 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.1
  have c83302 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.1
  have c83303 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_2
  simp only [← c83298, ← c83299, ← c83300] at e_1355_2
  have e_1355_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_3
  simp only [← c83301, ← c83302, k2, ← c83303] at e_1355_3
  have hr := e_1355_3
  simp only [e_1355_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12637)) := by
  have c83304 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.1
  have c83305 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.1
  have c83306 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83307 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83308 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83309 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_4
  simp only [← c83304, ← c83305, ← c83306] at e_1355_4
  have e_1355_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_5
  simp only [← c83307, ← c83308, k2, ← c83309] at e_1355_5
  have hr := e_1355_5
  simp only [e_1355_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12638)) := by
  have c83310 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83311 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83312 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83313 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83314 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83315 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_6
  simp only [← c83310, ← c83311, ← c83312] at e_1355_6
  have e_1355_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_7
  simp only [← c83313, ← c83314, k2, ← c83315] at e_1355_7
  have hr := e_1355_7
  simp only [e_1355_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12639)) := by
  have c83316 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83317 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83318 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83319 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83320 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83321 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_8
  simp only [← c83316, ← c83317, ← c83318] at e_1355_8
  have e_1355_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_9
  simp only [← c83319, ← c83320, k2, ← c83321] at e_1355_9
  have hr := e_1355_9
  simp only [e_1355_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12640)) := by
  have c83322 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83323 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83324 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83325 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83326 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83327 := (publicBatchWrapper89_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_10
  simp only [← c83322, ← c83323, ← c83324] at e_1355_10
  have e_1355_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_11
  simp only [← c83325, ← c83326, k2, ← c83327] at e_1355_11
  have hr := e_1355_11
  simp only [e_1355_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12641)) := by
  have c83328 := (publicBatchWrapper89_copies2604 a h).1
  have c83329 := (publicBatchWrapper89_copies2604 a h).2.1
  have c83330 := (publicBatchWrapper89_copies2604 a h).2.2.1
  have c83331 := (publicBatchWrapper89_copies2604 a h).2.2.2.1
  have c83332 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.1
  have c83333 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_12
  simp only [← c83328, ← c83329, ← c83330] at e_1355_12
  have e_1355_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_13
  simp only [← c83331, ← c83332, k2, ← c83333] at e_1355_13
  have hr := e_1355_13
  simp only [e_1355_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12642)) := by
  have c83334 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.1
  have c83335 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.1
  have c83336 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.1
  have c83337 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.1
  have c83338 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83339 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_14
  simp only [← c83334, ← c83335, ← c83336] at e_1355_14
  have e_1355_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_15
  simp only [← c83337, ← c83338, k2, ← c83339] at e_1355_15
  have hr := e_1355_15
  simp only [e_1355_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12643)) := by
  have c83340 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83341 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83342 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83343 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83344 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83345 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_16
  simp only [← c83340, ← c83341, ← c83342] at e_1355_16
  have e_1355_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_17
  simp only [← c83343, ← c83344, k2, ← c83345] at e_1355_17
  have hr := e_1355_17
  simp only [e_1355_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1355 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12644)) := by
  have c83346 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83347 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83348 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83349 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83350 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83351 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1355_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_18
  simp only [← c83346, ← c83347, ← c83348] at e_1355_18
  have e_1355_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1355 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_19
  simp only [← c83349, ← c83350, k2, ← c83351] at e_1355_19
  have hr := e_1355_19
  simp only [e_1355_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12645)) := by
  have c83352 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83353 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83354 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83355 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83356 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83357 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_0
  simp only [← c83352, ← c83353, ← c83354] at e_1356_0
  have e_1356_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_1
  simp only [← c83355, ← c83356, k2, ← c83357] at e_1356_1
  have hr := e_1356_1
  simp only [e_1356_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12646)) := by
  have c83358 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83359 := (publicBatchWrapper89_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83360 := (publicBatchWrapper89_copies2605 a h).1
  have c83361 := (publicBatchWrapper89_copies2605 a h).2.1
  have c83362 := (publicBatchWrapper89_copies2605 a h).2.2.1
  have c83363 := (publicBatchWrapper89_copies2605 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_2
  simp only [← c83358, ← c83359, ← c83360] at e_1356_2
  have e_1356_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_3
  simp only [← c83361, ← c83362, k2, ← c83363] at e_1356_3
  have hr := e_1356_3
  simp only [e_1356_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12647)) := by
  have c83364 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.1
  have c83365 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.1
  have c83366 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.1
  have c83367 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.1
  have c83368 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.1
  have c83369 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_4
  simp only [← c83364, ← c83365, ← c83366] at e_1356_4
  have e_1356_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_5
  simp only [← c83367, ← c83368, k2, ← c83369] at e_1356_5
  have hr := e_1356_5
  simp only [e_1356_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12648)) := by
  have c83370 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83371 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83372 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83373 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83374 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83375 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_6
  simp only [← c83370, ← c83371, ← c83372] at e_1356_6
  have e_1356_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_7
  simp only [← c83373, ← c83374, k2, ← c83375] at e_1356_7
  have hr := e_1356_7
  simp only [e_1356_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12649)) := by
  have c83376 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83377 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83378 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83379 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83380 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83381 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_8
  simp only [← c83376, ← c83377, ← c83378] at e_1356_8
  have e_1356_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_9
  simp only [← c83379, ← c83380, k2, ← c83381] at e_1356_9
  have hr := e_1356_9
  simp only [e_1356_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12650)) := by
  have c83382 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83383 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83384 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83385 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83386 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83387 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_10
  simp only [← c83382, ← c83383, ← c83384] at e_1356_10
  have e_1356_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_11
  simp only [← c83385, ← c83386, k2, ← c83387] at e_1356_11
  have hr := e_1356_11
  simp only [e_1356_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12651)) := by
  have c83388 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83389 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83390 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83391 := (publicBatchWrapper89_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83392 := (publicBatchWrapper89_copies2606 a h).1
  have c83393 := (publicBatchWrapper89_copies2606 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_12
  simp only [← c83388, ← c83389, ← c83390] at e_1356_12
  have e_1356_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_13
  simp only [← c83391, ← c83392, k2, ← c83393] at e_1356_13
  have hr := e_1356_13
  simp only [e_1356_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12652)) := by
  have c83394 := (publicBatchWrapper89_copies2606 a h).2.2.1
  have c83395 := (publicBatchWrapper89_copies2606 a h).2.2.2.1
  have c83396 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.1
  have c83397 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.1
  have c83398 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.1
  have c83399 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_14
  simp only [← c83394, ← c83395, ← c83396] at e_1356_14
  have e_1356_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_15
  simp only [← c83397, ← c83398, k2, ← c83399] at e_1356_15
  have hr := e_1356_15
  simp only [e_1356_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12653)) := by
  have c83400 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.1
  have c83401 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.1
  have c83402 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83403 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83404 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83405 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_16
  simp only [← c83400, ← c83401, ← c83402] at e_1356_16
  have e_1356_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_17
  simp only [← c83403, ← c83404, k2, ← c83405] at e_1356_17
  have hr := e_1356_17
  simp only [e_1356_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1356 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12654)) := by
  have c83406 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83407 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83408 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83409 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83410 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83411 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1356_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_18
  simp only [← c83406, ← c83407, ← c83408] at e_1356_18
  have e_1356_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1356 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_19
  simp only [← c83409, ← c83410, k2, ← c83411] at e_1356_19
  have hr := e_1356_19
  simp only [e_1356_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12655)) := by
  have c83412 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83413 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83414 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83415 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83416 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83417 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_0
  simp only [← c83412, ← c83413, ← c83414] at e_1357_0
  have e_1357_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_1
  simp only [← c83415, ← c83416, k2, ← c83417] at e_1357_1
  have hr := e_1357_1
  simp only [e_1357_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12656)) := by
  have c83418 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83419 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83420 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83421 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83422 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83423 := (publicBatchWrapper89_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_2
  simp only [← c83418, ← c83419, ← c83420] at e_1357_2
  have e_1357_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_3
  simp only [← c83421, ← c83422, k2, ← c83423] at e_1357_3
  have hr := e_1357_3
  simp only [e_1357_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12657)) := by
  have c83424 := (publicBatchWrapper89_copies2607 a h).1
  have c83425 := (publicBatchWrapper89_copies2607 a h).2.1
  have c83426 := (publicBatchWrapper89_copies2607 a h).2.2.1
  have c83427 := (publicBatchWrapper89_copies2607 a h).2.2.2.1
  have c83428 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.1
  have c83429 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_4
  simp only [← c83424, ← c83425, ← c83426] at e_1357_4
  have e_1357_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_5
  simp only [← c83427, ← c83428, k2, ← c83429] at e_1357_5
  have hr := e_1357_5
  simp only [e_1357_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12658)) := by
  have c83430 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.1
  have c83431 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.1
  have c83432 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.1
  have c83433 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.1
  have c83434 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83435 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_6
  simp only [← c83430, ← c83431, ← c83432] at e_1357_6
  have e_1357_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_7
  simp only [← c83433, ← c83434, k2, ← c83435] at e_1357_7
  have hr := e_1357_7
  simp only [e_1357_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12659)) := by
  have c83436 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83437 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83438 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83439 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83440 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83441 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_8
  simp only [← c83436, ← c83437, ← c83438] at e_1357_8
  have e_1357_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_9
  simp only [← c83439, ← c83440, k2, ← c83441] at e_1357_9
  have hr := e_1357_9
  simp only [e_1357_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12660)) := by
  have c83442 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83443 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83444 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83445 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83446 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83447 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_10
  simp only [← c83442, ← c83443, ← c83444] at e_1357_10
  have e_1357_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_11
  simp only [← c83445, ← c83446, k2, ← c83447] at e_1357_11
  have hr := e_1357_11
  simp only [e_1357_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12661)) := by
  have c83448 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83449 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83450 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83451 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83452 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83453 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_12
  simp only [← c83448, ← c83449, ← c83450] at e_1357_12
  have e_1357_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_13
  simp only [← c83451, ← c83452, k2, ← c83453] at e_1357_13
  have hr := e_1357_13
  simp only [e_1357_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12662)) := by
  have c83454 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83455 := (publicBatchWrapper89_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83456 := (publicBatchWrapper89_copies2608 a h).1
  have c83457 := (publicBatchWrapper89_copies2608 a h).2.1
  have c83458 := (publicBatchWrapper89_copies2608 a h).2.2.1
  have c83459 := (publicBatchWrapper89_copies2608 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_14
  simp only [← c83454, ← c83455, ← c83456] at e_1357_14
  have e_1357_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_15
  simp only [← c83457, ← c83458, k2, ← c83459] at e_1357_15
  have hr := e_1357_15
  simp only [e_1357_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12663)) := by
  have c83460 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.1
  have c83461 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.1
  have c83462 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.1
  have c83463 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.1
  have c83464 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.1
  have c83465 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_16
  simp only [← c83460, ← c83461, ← c83462] at e_1357_16
  have e_1357_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_17
  simp only [← c83463, ← c83464, k2, ← c83465] at e_1357_17
  have hr := e_1357_17
  simp only [e_1357_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1357 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12664)) := by
  have c83466 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83467 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83468 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83469 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83470 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83471 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1357_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_18
  simp only [← c83466, ← c83467, ← c83468] at e_1357_18
  have e_1357_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1357 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_19
  simp only [← c83469, ← c83470, k2, ← c83471] at e_1357_19
  have hr := e_1357_19
  simp only [e_1357_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12665)) := by
  have c83472 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83473 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83474 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83475 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83476 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83477 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_0
  simp only [← c83472, ← c83473, ← c83474] at e_1358_0
  have e_1358_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_1
  simp only [← c83475, ← c83476, k2, ← c83477] at e_1358_1
  have hr := e_1358_1
  simp only [e_1358_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12666)) := by
  have c83478 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83479 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83480 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83481 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83482 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83483 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_2
  simp only [← c83478, ← c83479, ← c83480] at e_1358_2
  have e_1358_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_3
  simp only [← c83481, ← c83482, k2, ← c83483] at e_1358_3
  have hr := e_1358_3
  simp only [e_1358_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12667)) := by
  have c83484 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83485 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83486 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83487 := (publicBatchWrapper89_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83488 := (publicBatchWrapper89_copies2609 a h).1
  have c83489 := (publicBatchWrapper89_copies2609 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_4
  simp only [← c83484, ← c83485, ← c83486] at e_1358_4
  have e_1358_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_5
  simp only [← c83487, ← c83488, k2, ← c83489] at e_1358_5
  have hr := e_1358_5
  simp only [e_1358_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12668)) := by
  have c83490 := (publicBatchWrapper89_copies2609 a h).2.2.1
  have c83491 := (publicBatchWrapper89_copies2609 a h).2.2.2.1
  have c83492 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.1
  have c83493 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.1
  have c83494 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.1
  have c83495 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_6
  simp only [← c83490, ← c83491, ← c83492] at e_1358_6
  have e_1358_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_7
  simp only [← c83493, ← c83494, k2, ← c83495] at e_1358_7
  have hr := e_1358_7
  simp only [e_1358_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12669)) := by
  have c83496 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.1
  have c83497 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.1
  have c83498 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83499 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83500 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83501 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_8
  simp only [← c83496, ← c83497, ← c83498] at e_1358_8
  have e_1358_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_9
  simp only [← c83499, ← c83500, k2, ← c83501] at e_1358_9
  have hr := e_1358_9
  simp only [e_1358_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12670)) := by
  have c83502 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83503 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83504 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83505 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83506 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83507 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_10
  simp only [← c83502, ← c83503, ← c83504] at e_1358_10
  have e_1358_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_11
  simp only [← c83505, ← c83506, k2, ← c83507] at e_1358_11
  have hr := e_1358_11
  simp only [e_1358_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12671)) := by
  have c83508 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83509 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83510 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83511 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83512 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83513 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_12
  simp only [← c83508, ← c83509, ← c83510] at e_1358_12
  have e_1358_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_13
  simp only [← c83511, ← c83512, k2, ← c83513] at e_1358_13
  have hr := e_1358_13
  simp only [e_1358_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12672)) := by
  have c83514 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83515 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83516 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83517 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83518 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83519 := (publicBatchWrapper89_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_14
  simp only [← c83514, ← c83515, ← c83516] at e_1358_14
  have e_1358_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_15
  simp only [← c83517, ← c83518, k2, ← c83519] at e_1358_15
  have hr := e_1358_15
  simp only [e_1358_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12673)) := by
  have c83520 := (publicBatchWrapper89_copies2610 a h).1
  have c83521 := (publicBatchWrapper89_copies2610 a h).2.1
  have c83522 := (publicBatchWrapper89_copies2610 a h).2.2.1
  have c83523 := (publicBatchWrapper89_copies2610 a h).2.2.2.1
  have c83524 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.1
  have c83525 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_16
  simp only [← c83520, ← c83521, ← c83522] at e_1358_16
  have e_1358_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_17
  simp only [← c83523, ← c83524, k2, ← c83525] at e_1358_17
  have hr := e_1358_17
  simp only [e_1358_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1358 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12674)) := by
  have c83526 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.1
  have c83527 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.1
  have c83528 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.1
  have c83529 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.1
  have c83530 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83531 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1358_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_18
  simp only [← c83526, ← c83527, ← c83528] at e_1358_18
  have e_1358_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1358 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_19
  simp only [← c83529, ← c83530, k2, ← c83531] at e_1358_19
  have hr := e_1358_19
  simp only [e_1358_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12675)) := by
  have c83532 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83533 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83534 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83535 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83536 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83537 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_0
  simp only [← c83532, ← c83533, ← c83534] at e_1359_0
  have e_1359_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_1
  simp only [← c83535, ← c83536, k2, ← c83537] at e_1359_1
  have hr := e_1359_1
  simp only [e_1359_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12676)) := by
  have c83538 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83539 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83540 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83541 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83542 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83543 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_2
  simp only [← c83538, ← c83539, ← c83540] at e_1359_2
  have e_1359_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_3
  simp only [← c83541, ← c83542, k2, ← c83543] at e_1359_3
  have hr := e_1359_3
  simp only [e_1359_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12677)) := by
  have c83544 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83545 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83546 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83547 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83548 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83549 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_4
  simp only [← c83544, ← c83545, ← c83546] at e_1359_4
  have e_1359_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_5
  simp only [← c83547, ← c83548, k2, ← c83549] at e_1359_5
  have hr := e_1359_5
  simp only [e_1359_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12678)) := by
  have c83550 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83551 := (publicBatchWrapper89_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83552 := (publicBatchWrapper89_copies2611 a h).1
  have c83553 := (publicBatchWrapper89_copies2611 a h).2.1
  have c83554 := (publicBatchWrapper89_copies2611 a h).2.2.1
  have c83555 := (publicBatchWrapper89_copies2611 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_6
  simp only [← c83550, ← c83551, ← c83552] at e_1359_6
  have e_1359_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_7
  simp only [← c83553, ← c83554, k2, ← c83555] at e_1359_7
  have hr := e_1359_7
  simp only [e_1359_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12679)) := by
  have c83556 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.1
  have c83557 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.1
  have c83558 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.1
  have c83559 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.1
  have c83560 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.1
  have c83561 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_8
  simp only [← c83556, ← c83557, ← c83558] at e_1359_8
  have e_1359_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_9
  simp only [← c83559, ← c83560, k2, ← c83561] at e_1359_9
  have hr := e_1359_9
  simp only [e_1359_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12680)) := by
  have c83562 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83563 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83564 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83565 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83566 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83567 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_10
  simp only [← c83562, ← c83563, ← c83564] at e_1359_10
  have e_1359_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_11
  simp only [← c83565, ← c83566, k2, ← c83567] at e_1359_11
  have hr := e_1359_11
  simp only [e_1359_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12681)) := by
  have c83568 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83569 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83570 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83571 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83572 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83573 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_12
  simp only [← c83568, ← c83569, ← c83570] at e_1359_12
  have e_1359_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_13
  simp only [← c83571, ← c83572, k2, ← c83573] at e_1359_13
  have hr := e_1359_13
  simp only [e_1359_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12682)) := by
  have c83574 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83575 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83576 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83577 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83578 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83579 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_14
  simp only [← c83574, ← c83575, ← c83576] at e_1359_14
  have e_1359_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_15
  simp only [← c83577, ← c83578, k2, ← c83579] at e_1359_15
  have hr := e_1359_15
  simp only [e_1359_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12683)) := by
  have c83580 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83581 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83582 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83583 := (publicBatchWrapper89_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83584 := (publicBatchWrapper89_copies2612 a h).1
  have c83585 := (publicBatchWrapper89_copies2612 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_16
  simp only [← c83580, ← c83581, ← c83582] at e_1359_16
  have e_1359_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_17
  simp only [← c83583, ← c83584, k2, ← c83585] at e_1359_17
  have hr := e_1359_17
  simp only [e_1359_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1359 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12684)) := by
  have c83586 := (publicBatchWrapper89_copies2612 a h).2.2.1
  have c83587 := (publicBatchWrapper89_copies2612 a h).2.2.2.1
  have c83588 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.1
  have c83589 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.1
  have c83590 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.1
  have c83591 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1359_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_18
  simp only [← c83586, ← c83587, ← c83588] at e_1359_18
  have e_1359_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1359 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_19
  simp only [← c83589, ← c83590, k2, ← c83591] at e_1359_19
  have hr := e_1359_19
  simp only [e_1359_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12685)) := by
  have c83592 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.1
  have c83593 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.1
  have c83594 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83595 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83596 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83597 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_0
  simp only [← c83592, ← c83593, ← c83594] at e_1360_0
  have e_1360_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_1
  simp only [← c83595, ← c83596, k2, ← c83597] at e_1360_1
  have hr := e_1360_1
  simp only [e_1360_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12686)) := by
  have c83598 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83599 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83600 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83601 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83602 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83603 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_2
  simp only [← c83598, ← c83599, ← c83600] at e_1360_2
  have e_1360_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_3
  simp only [← c83601, ← c83602, k2, ← c83603] at e_1360_3
  have hr := e_1360_3
  simp only [e_1360_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12687)) := by
  have c83604 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83605 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83606 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83607 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83608 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83609 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_4
  simp only [← c83604, ← c83605, ← c83606] at e_1360_4
  have e_1360_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_5
  simp only [← c83607, ← c83608, k2, ← c83609] at e_1360_5
  have hr := e_1360_5
  simp only [e_1360_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12688)) := by
  have c83610 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83611 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83612 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83613 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83614 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83615 := (publicBatchWrapper89_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_6
  simp only [← c83610, ← c83611, ← c83612] at e_1360_6
  have e_1360_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_7
  simp only [← c83613, ← c83614, k2, ← c83615] at e_1360_7
  have hr := e_1360_7
  simp only [e_1360_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12689)) := by
  have c83616 := (publicBatchWrapper89_copies2613 a h).1
  have c83617 := (publicBatchWrapper89_copies2613 a h).2.1
  have c83618 := (publicBatchWrapper89_copies2613 a h).2.2.1
  have c83619 := (publicBatchWrapper89_copies2613 a h).2.2.2.1
  have c83620 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.1
  have c83621 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_8
  simp only [← c83616, ← c83617, ← c83618] at e_1360_8
  have e_1360_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_9
  simp only [← c83619, ← c83620, k2, ← c83621] at e_1360_9
  have hr := e_1360_9
  simp only [e_1360_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12690)) := by
  have c83622 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.1
  have c83623 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.1
  have c83624 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.1
  have c83625 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.1
  have c83626 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83627 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_10
  simp only [← c83622, ← c83623, ← c83624] at e_1360_10
  have e_1360_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_11
  simp only [← c83625, ← c83626, k2, ← c83627] at e_1360_11
  have hr := e_1360_11
  simp only [e_1360_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12691)) := by
  have c83628 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83629 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83630 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83631 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83632 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83633 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_12
  simp only [← c83628, ← c83629, ← c83630] at e_1360_12
  have e_1360_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_13
  simp only [← c83631, ← c83632, k2, ← c83633] at e_1360_13
  have hr := e_1360_13
  simp only [e_1360_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12692)) := by
  have c83634 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83635 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83636 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83637 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83638 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83639 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_14
  simp only [← c83634, ← c83635, ← c83636] at e_1360_14
  have e_1360_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_15
  simp only [← c83637, ← c83638, k2, ← c83639] at e_1360_15
  have hr := e_1360_15
  simp only [e_1360_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12693)) := by
  have c83640 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83641 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83642 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83643 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83644 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83645 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_16
  simp only [← c83640, ← c83641, ← c83642] at e_1360_16
  have e_1360_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_17
  simp only [← c83643, ← c83644, k2, ← c83645] at e_1360_17
  have hr := e_1360_17
  simp only [e_1360_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1360 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12694)) := by
  have c83646 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83647 := (publicBatchWrapper89_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83648 := (publicBatchWrapper89_copies2614 a h).1
  have c83649 := (publicBatchWrapper89_copies2614 a h).2.1
  have c83650 := (publicBatchWrapper89_copies2614 a h).2.2.1
  have c83651 := (publicBatchWrapper89_copies2614 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1360_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_18
  simp only [← c83646, ← c83647, ← c83648] at e_1360_18
  have e_1360_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1360 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_19
  simp only [← c83649, ← c83650, k2, ← c83651] at e_1360_19
  have hr := e_1360_19
  simp only [e_1360_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12695)) := by
  have c83652 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.1
  have c83653 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.1
  have c83654 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.1
  have c83655 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.1
  have c83656 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.1
  have c83657 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_0
  simp only [← c83652, ← c83653, ← c83654] at e_1361_0
  have e_1361_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_1
  simp only [← c83655, ← c83656, k2, ← c83657] at e_1361_1
  have hr := e_1361_1
  simp only [e_1361_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12696)) := by
  have c83658 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83659 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83660 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83661 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83662 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83663 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_2
  simp only [← c83658, ← c83659, ← c83660] at e_1361_2
  have e_1361_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_3
  simp only [← c83661, ← c83662, k2, ← c83663] at e_1361_3
  have hr := e_1361_3
  simp only [e_1361_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12697)) := by
  have c83664 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83665 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83666 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83667 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83668 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83669 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_4
  simp only [← c83664, ← c83665, ← c83666] at e_1361_4
  have e_1361_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_5
  simp only [← c83667, ← c83668, k2, ← c83669] at e_1361_5
  have hr := e_1361_5
  simp only [e_1361_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12698)) := by
  have c83670 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83671 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83672 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83673 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83674 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83675 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_6
  simp only [← c83670, ← c83671, ← c83672] at e_1361_6
  have e_1361_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_7
  simp only [← c83673, ← c83674, k2, ← c83675] at e_1361_7
  have hr := e_1361_7
  simp only [e_1361_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12699)) := by
  have c83676 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83677 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83678 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83679 := (publicBatchWrapper89_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83680 := (publicBatchWrapper89_copies2615 a h).1
  have c83681 := (publicBatchWrapper89_copies2615 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_8
  simp only [← c83676, ← c83677, ← c83678] at e_1361_8
  have e_1361_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_9
  simp only [← c83679, ← c83680, k2, ← c83681] at e_1361_9
  have hr := e_1361_9
  simp only [e_1361_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12700)) := by
  have c83682 := (publicBatchWrapper89_copies2615 a h).2.2.1
  have c83683 := (publicBatchWrapper89_copies2615 a h).2.2.2.1
  have c83684 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.1
  have c83685 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.1
  have c83686 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.1
  have c83687 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_10
  simp only [← c83682, ← c83683, ← c83684] at e_1361_10
  have e_1361_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_11
  simp only [← c83685, ← c83686, k2, ← c83687] at e_1361_11
  have hr := e_1361_11
  simp only [e_1361_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12701)) := by
  have c83688 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.1
  have c83689 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.1
  have c83690 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83691 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83692 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83693 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_12
  simp only [← c83688, ← c83689, ← c83690] at e_1361_12
  have e_1361_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_13
  simp only [← c83691, ← c83692, k2, ← c83693] at e_1361_13
  have hr := e_1361_13
  simp only [e_1361_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12702)) := by
  have c83694 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83695 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83696 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83697 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83698 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83699 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_14
  simp only [← c83694, ← c83695, ← c83696] at e_1361_14
  have e_1361_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_15
  simp only [← c83697, ← c83698, k2, ← c83699] at e_1361_15
  have hr := e_1361_15
  simp only [e_1361_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12703)) := by
  have c83700 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83701 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83702 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83703 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83704 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83705 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_16
  simp only [← c83700, ← c83701, ← c83702] at e_1361_16
  have e_1361_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_17
  simp only [← c83703, ← c83704, k2, ← c83705] at e_1361_17
  have hr := e_1361_17
  simp only [e_1361_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1361 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12704)) := by
  have c83706 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83707 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83708 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83709 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83710 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83711 := (publicBatchWrapper89_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1361_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_18
  simp only [← c83706, ← c83707, ← c83708] at e_1361_18
  have e_1361_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1361 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_19
  simp only [← c83709, ← c83710, k2, ← c83711] at e_1361_19
  have hr := e_1361_19
  simp only [e_1361_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12705)) := by
  have c83712 := (publicBatchWrapper89_copies2616 a h).1
  have c83713 := (publicBatchWrapper89_copies2616 a h).2.1
  have c83714 := (publicBatchWrapper89_copies2616 a h).2.2.1
  have c83715 := (publicBatchWrapper89_copies2616 a h).2.2.2.1
  have c83716 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.1
  have c83717 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_0
  simp only [← c83712, ← c83713, ← c83714] at e_1362_0
  have e_1362_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_1
  simp only [← c83715, ← c83716, k2, ← c83717] at e_1362_1
  have hr := e_1362_1
  simp only [e_1362_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12706)) := by
  have c83718 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.1
  have c83719 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.1
  have c83720 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.1
  have c83721 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.1
  have c83722 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83723 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_2
  simp only [← c83718, ← c83719, ← c83720] at e_1362_2
  have e_1362_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_3
  simp only [← c83721, ← c83722, k2, ← c83723] at e_1362_3
  have hr := e_1362_3
  simp only [e_1362_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12707)) := by
  have c83724 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83725 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83726 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83727 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83728 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83729 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_4
  simp only [← c83724, ← c83725, ← c83726] at e_1362_4
  have e_1362_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_5
  simp only [← c83727, ← c83728, k2, ← c83729] at e_1362_5
  have hr := e_1362_5
  simp only [e_1362_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12708)) := by
  have c83730 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83731 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83732 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83733 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83734 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83735 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_6
  simp only [← c83730, ← c83731, ← c83732] at e_1362_6
  have e_1362_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_7
  simp only [← c83733, ← c83734, k2, ← c83735] at e_1362_7
  have hr := e_1362_7
  simp only [e_1362_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12709)) := by
  have c83736 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83737 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83738 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83739 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83740 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83741 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_8
  simp only [← c83736, ← c83737, ← c83738] at e_1362_8
  have e_1362_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_9
  simp only [← c83739, ← c83740, k2, ← c83741] at e_1362_9
  have hr := e_1362_9
  simp only [e_1362_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12710)) := by
  have c83742 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83743 := (publicBatchWrapper89_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83744 := (publicBatchWrapper89_copies2617 a h).1
  have c83745 := (publicBatchWrapper89_copies2617 a h).2.1
  have c83746 := (publicBatchWrapper89_copies2617 a h).2.2.1
  have c83747 := (publicBatchWrapper89_copies2617 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_10
  simp only [← c83742, ← c83743, ← c83744] at e_1362_10
  have e_1362_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_11
  simp only [← c83745, ← c83746, k2, ← c83747] at e_1362_11
  have hr := e_1362_11
  simp only [e_1362_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12711)) := by
  have c83748 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.1
  have c83749 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.1
  have c83750 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.1
  have c83751 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.1
  have c83752 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.1
  have c83753 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_12
  simp only [← c83748, ← c83749, ← c83750] at e_1362_12
  have e_1362_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_13
  simp only [← c83751, ← c83752, k2, ← c83753] at e_1362_13
  have hr := e_1362_13
  simp only [e_1362_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12712)) := by
  have c83754 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83755 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83756 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83757 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83758 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83759 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_14
  simp only [← c83754, ← c83755, ← c83756] at e_1362_14
  have e_1362_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_15
  simp only [← c83757, ← c83758, k2, ← c83759] at e_1362_15
  have hr := e_1362_15
  simp only [e_1362_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12713)) := by
  have c83760 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83761 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83762 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83763 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83764 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83765 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_16
  simp only [← c83760, ← c83761, ← c83762] at e_1362_16
  have e_1362_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_17
  simp only [← c83763, ← c83764, k2, ← c83765] at e_1362_17
  have hr := e_1362_17
  simp only [e_1362_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1362 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12714)) := by
  have c83766 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83767 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83768 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83769 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83770 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83771 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1362_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_18
  simp only [← c83766, ← c83767, ← c83768] at e_1362_18
  have e_1362_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1362 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_19
  simp only [← c83769, ← c83770, k2, ← c83771] at e_1362_19
  have hr := e_1362_19
  simp only [e_1362_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12715)) := by
  have c83772 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83773 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83774 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83775 := (publicBatchWrapper89_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83776 := (publicBatchWrapper89_copies2618 a h).1
  have c83777 := (publicBatchWrapper89_copies2618 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_0
  simp only [← c83772, ← c83773, ← c83774] at e_1363_0
  have e_1363_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_1
  simp only [← c83775, ← c83776, k2, ← c83777] at e_1363_1
  have hr := e_1363_1
  simp only [e_1363_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12716)) := by
  have c83778 := (publicBatchWrapper89_copies2618 a h).2.2.1
  have c83779 := (publicBatchWrapper89_copies2618 a h).2.2.2.1
  have c83780 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.1
  have c83781 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.1
  have c83782 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.1
  have c83783 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_2
  simp only [← c83778, ← c83779, ← c83780] at e_1363_2
  have e_1363_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_3
  simp only [← c83781, ← c83782, k2, ← c83783] at e_1363_3
  have hr := e_1363_3
  simp only [e_1363_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12717)) := by
  have c83784 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.1
  have c83785 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.1
  have c83786 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83787 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83788 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83789 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_4
  simp only [← c83784, ← c83785, ← c83786] at e_1363_4
  have e_1363_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_5
  simp only [← c83787, ← c83788, k2, ← c83789] at e_1363_5
  have hr := e_1363_5
  simp only [e_1363_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12718)) := by
  have c83790 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83791 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83792 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83793 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83794 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83795 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_6
  simp only [← c83790, ← c83791, ← c83792] at e_1363_6
  have e_1363_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_7
  simp only [← c83793, ← c83794, k2, ← c83795] at e_1363_7
  have hr := e_1363_7
  simp only [e_1363_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12719)) := by
  have c83796 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83797 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83798 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83799 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83800 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83801 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_8
  simp only [← c83796, ← c83797, ← c83798] at e_1363_8
  have e_1363_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_9
  simp only [← c83799, ← c83800, k2, ← c83801] at e_1363_9
  have hr := e_1363_9
  simp only [e_1363_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12720)) := by
  have c83802 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83803 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83804 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83805 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83806 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83807 := (publicBatchWrapper89_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_10
  simp only [← c83802, ← c83803, ← c83804] at e_1363_10
  have e_1363_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_11
  simp only [← c83805, ← c83806, k2, ← c83807] at e_1363_11
  have hr := e_1363_11
  simp only [e_1363_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12721)) := by
  have c83808 := (publicBatchWrapper89_copies2619 a h).1
  have c83809 := (publicBatchWrapper89_copies2619 a h).2.1
  have c83810 := (publicBatchWrapper89_copies2619 a h).2.2.1
  have c83811 := (publicBatchWrapper89_copies2619 a h).2.2.2.1
  have c83812 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.1
  have c83813 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_12
  simp only [← c83808, ← c83809, ← c83810] at e_1363_12
  have e_1363_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_13
  simp only [← c83811, ← c83812, k2, ← c83813] at e_1363_13
  have hr := e_1363_13
  simp only [e_1363_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12722)) := by
  have c83814 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.1
  have c83815 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.1
  have c83816 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.1
  have c83817 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.1
  have c83818 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83819 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_14
  simp only [← c83814, ← c83815, ← c83816] at e_1363_14
  have e_1363_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_15
  simp only [← c83817, ← c83818, k2, ← c83819] at e_1363_15
  have hr := e_1363_15
  simp only [e_1363_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12723)) := by
  have c83820 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83821 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83822 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83823 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83824 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83825 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_16
  simp only [← c83820, ← c83821, ← c83822] at e_1363_16
  have e_1363_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_17
  simp only [← c83823, ← c83824, k2, ← c83825] at e_1363_17
  have hr := e_1363_17
  simp only [e_1363_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1363 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12724)) := by
  have c83826 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83827 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83828 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83829 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83830 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83831 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1363_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_18
  simp only [← c83826, ← c83827, ← c83828] at e_1363_18
  have e_1363_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1363 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_19
  simp only [← c83829, ← c83830, k2, ← c83831] at e_1363_19
  have hr := e_1363_19
  simp only [e_1363_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12725)) := by
  have c83832 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83833 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83834 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83835 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83836 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83837 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_0
  simp only [← c83832, ← c83833, ← c83834] at e_1364_0
  have e_1364_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_1
  simp only [← c83835, ← c83836, k2, ← c83837] at e_1364_1
  have hr := e_1364_1
  simp only [e_1364_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12726)) := by
  have c83838 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83839 := (publicBatchWrapper89_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83840 := (publicBatchWrapper89_copies2620 a h).1
  have c83841 := (publicBatchWrapper89_copies2620 a h).2.1
  have c83842 := (publicBatchWrapper89_copies2620 a h).2.2.1
  have c83843 := (publicBatchWrapper89_copies2620 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_2
  simp only [← c83838, ← c83839, ← c83840] at e_1364_2
  have e_1364_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_3
  simp only [← c83841, ← c83842, k2, ← c83843] at e_1364_3
  have hr := e_1364_3
  simp only [e_1364_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12727)) := by
  have c83844 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.1
  have c83845 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.1
  have c83846 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.1
  have c83847 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.1
  have c83848 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.1
  have c83849 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_4
  simp only [← c83844, ← c83845, ← c83846] at e_1364_4
  have e_1364_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_5
  simp only [← c83847, ← c83848, k2, ← c83849] at e_1364_5
  have hr := e_1364_5
  simp only [e_1364_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12728)) := by
  have c83850 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83851 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83852 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83853 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83854 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83855 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_6
  simp only [← c83850, ← c83851, ← c83852] at e_1364_6
  have e_1364_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_7
  simp only [← c83853, ← c83854, k2, ← c83855] at e_1364_7
  have hr := e_1364_7
  simp only [e_1364_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12729)) := by
  have c83856 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83857 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83858 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83859 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83860 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83861 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_8
  simp only [← c83856, ← c83857, ← c83858] at e_1364_8
  have e_1364_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_9
  simp only [← c83859, ← c83860, k2, ← c83861] at e_1364_9
  have hr := e_1364_9
  simp only [e_1364_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12730)) := by
  have c83862 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83863 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83864 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83865 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83866 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83867 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_10
  simp only [← c83862, ← c83863, ← c83864] at e_1364_10
  have e_1364_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_11
  simp only [← c83865, ← c83866, k2, ← c83867] at e_1364_11
  have hr := e_1364_11
  simp only [e_1364_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12731)) := by
  have c83868 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83869 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83870 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83871 := (publicBatchWrapper89_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83872 := (publicBatchWrapper89_copies2621 a h).1
  have c83873 := (publicBatchWrapper89_copies2621 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_12
  simp only [← c83868, ← c83869, ← c83870] at e_1364_12
  have e_1364_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_13
  simp only [← c83871, ← c83872, k2, ← c83873] at e_1364_13
  have hr := e_1364_13
  simp only [e_1364_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12732)) := by
  have c83874 := (publicBatchWrapper89_copies2621 a h).2.2.1
  have c83875 := (publicBatchWrapper89_copies2621 a h).2.2.2.1
  have c83876 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.1
  have c83877 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.1
  have c83878 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.1
  have c83879 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_14
  simp only [← c83874, ← c83875, ← c83876] at e_1364_14
  have e_1364_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_15
  simp only [← c83877, ← c83878, k2, ← c83879] at e_1364_15
  have hr := e_1364_15
  simp only [e_1364_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12733)) := by
  have c83880 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.1
  have c83881 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.1
  have c83882 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83883 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83884 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83885 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_16
  simp only [← c83880, ← c83881, ← c83882] at e_1364_16
  have e_1364_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_17
  simp only [← c83883, ← c83884, k2, ← c83885] at e_1364_17
  have hr := e_1364_17
  simp only [e_1364_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1364 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12734)) := by
  have c83886 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83887 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83888 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83889 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83890 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83891 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1364_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_18
  simp only [← c83886, ← c83887, ← c83888] at e_1364_18
  have e_1364_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1364 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_19
  simp only [← c83889, ← c83890, k2, ← c83891] at e_1364_19
  have hr := e_1364_19
  simp only [e_1364_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12735)) := by
  have c83892 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83893 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83894 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83895 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83896 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83897 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_0
  simp only [← c83892, ← c83893, ← c83894] at e_1365_0
  have e_1365_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_1
  simp only [← c83895, ← c83896, k2, ← c83897] at e_1365_1
  have hr := e_1365_1
  simp only [e_1365_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12736)) := by
  have c83898 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83899 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83900 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83901 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83902 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83903 := (publicBatchWrapper89_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_2
  simp only [← c83898, ← c83899, ← c83900] at e_1365_2
  have e_1365_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_3
  simp only [← c83901, ← c83902, k2, ← c83903] at e_1365_3
  have hr := e_1365_3
  simp only [e_1365_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12737)) := by
  have c83904 := (publicBatchWrapper89_copies2622 a h).1
  have c83905 := (publicBatchWrapper89_copies2622 a h).2.1
  have c83906 := (publicBatchWrapper89_copies2622 a h).2.2.1
  have c83907 := (publicBatchWrapper89_copies2622 a h).2.2.2.1
  have c83908 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.1
  have c83909 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_4
  simp only [← c83904, ← c83905, ← c83906] at e_1365_4
  have e_1365_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_5
  simp only [← c83907, ← c83908, k2, ← c83909] at e_1365_5
  have hr := e_1365_5
  simp only [e_1365_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12738)) := by
  have c83910 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.1
  have c83911 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.1
  have c83912 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.1
  have c83913 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.1
  have c83914 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83915 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_6
  simp only [← c83910, ← c83911, ← c83912] at e_1365_6
  have e_1365_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_7
  simp only [← c83913, ← c83914, k2, ← c83915] at e_1365_7
  have hr := e_1365_7
  simp only [e_1365_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12739)) := by
  have c83916 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83917 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83918 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83919 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83920 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83921 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_8
  simp only [← c83916, ← c83917, ← c83918] at e_1365_8
  have e_1365_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_9
  simp only [← c83919, ← c83920, k2, ← c83921] at e_1365_9
  have hr := e_1365_9
  simp only [e_1365_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12740)) := by
  have c83922 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83923 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83924 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83925 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83926 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83927 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_10
  simp only [← c83922, ← c83923, ← c83924] at e_1365_10
  have e_1365_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_11
  simp only [← c83925, ← c83926, k2, ← c83927] at e_1365_11
  have hr := e_1365_11
  simp only [e_1365_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12741)) := by
  have c83928 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83929 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83930 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83931 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83932 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83933 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_12
  simp only [← c83928, ← c83929, ← c83930] at e_1365_12
  have e_1365_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_13
  simp only [← c83931, ← c83932, k2, ← c83933] at e_1365_13
  have hr := e_1365_13
  simp only [e_1365_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12742)) := by
  have c83934 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83935 := (publicBatchWrapper89_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83936 := (publicBatchWrapper89_copies2623 a h).1
  have c83937 := (publicBatchWrapper89_copies2623 a h).2.1
  have c83938 := (publicBatchWrapper89_copies2623 a h).2.2.1
  have c83939 := (publicBatchWrapper89_copies2623 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_14
  simp only [← c83934, ← c83935, ← c83936] at e_1365_14
  have e_1365_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_15
  simp only [← c83937, ← c83938, k2, ← c83939] at e_1365_15
  have hr := e_1365_15
  simp only [e_1365_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12743)) := by
  have c83940 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.1
  have c83941 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.1
  have c83942 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.1
  have c83943 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.1
  have c83944 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.1
  have c83945 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_16
  simp only [← c83940, ← c83941, ← c83942] at e_1365_16
  have e_1365_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_17
  simp only [← c83943, ← c83944, k2, ← c83945] at e_1365_17
  have hr := e_1365_17
  simp only [e_1365_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1365 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12744)) := by
  have c83946 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83947 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83948 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83949 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83950 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83951 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1365_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_18
  simp only [← c83946, ← c83947, ← c83948] at e_1365_18
  have e_1365_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1365 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_19
  simp only [← c83949, ← c83950, k2, ← c83951] at e_1365_19
  have hr := e_1365_19
  simp only [e_1365_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12745)) := by
  have c83952 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83953 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83954 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83955 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83956 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83957 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_0
  simp only [← c83952, ← c83953, ← c83954] at e_1366_0
  have e_1366_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_1
  simp only [← c83955, ← c83956, k2, ← c83957] at e_1366_1
  have hr := e_1366_1
  simp only [e_1366_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12746)) := by
  have c83958 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83959 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83960 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83961 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83962 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83963 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_2
  simp only [← c83958, ← c83959, ← c83960] at e_1366_2
  have e_1366_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_3
  simp only [← c83961, ← c83962, k2, ← c83963] at e_1366_3
  have hr := e_1366_3
  simp only [e_1366_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12747)) := by
  have c83964 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83965 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83966 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83967 := (publicBatchWrapper89_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83968 := (publicBatchWrapper89_copies2624 a h).1
  have c83969 := (publicBatchWrapper89_copies2624 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_4
  simp only [← c83964, ← c83965, ← c83966] at e_1366_4
  have e_1366_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_5
  simp only [← c83967, ← c83968, k2, ← c83969] at e_1366_5
  have hr := e_1366_5
  simp only [e_1366_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12748)) := by
  have c83970 := (publicBatchWrapper89_copies2624 a h).2.2.1
  have c83971 := (publicBatchWrapper89_copies2624 a h).2.2.2.1
  have c83972 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.1
  have c83973 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.1
  have c83974 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.1
  have c83975 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_6
  simp only [← c83970, ← c83971, ← c83972] at e_1366_6
  have e_1366_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_7
  simp only [← c83973, ← c83974, k2, ← c83975] at e_1366_7
  have hr := e_1366_7
  simp only [e_1366_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12749)) := by
  have c83976 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.1
  have c83977 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.1
  have c83978 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83979 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83980 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83981 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_8
  simp only [← c83976, ← c83977, ← c83978] at e_1366_8
  have e_1366_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_9
  simp only [← c83979, ← c83980, k2, ← c83981] at e_1366_9
  have hr := e_1366_9
  simp only [e_1366_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12750)) := by
  have c83982 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83983 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83984 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83985 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83986 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83987 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_10
  simp only [← c83982, ← c83983, ← c83984] at e_1366_10
  have e_1366_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_11
  simp only [← c83985, ← c83986, k2, ← c83987] at e_1366_11
  have hr := e_1366_11
  simp only [e_1366_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12751)) := by
  have c83988 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83989 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83990 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83991 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83992 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83993 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_12
  simp only [← c83988, ← c83989, ← c83990] at e_1366_12
  have e_1366_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_13
  simp only [← c83991, ← c83992, k2, ← c83993] at e_1366_13
  have hr := e_1366_13
  simp only [e_1366_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12752)) := by
  have c83994 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83995 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83996 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83997 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83998 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83999 := (publicBatchWrapper89_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_14
  simp only [← c83994, ← c83995, ← c83996] at e_1366_14
  have e_1366_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_15
  simp only [← c83997, ← c83998, k2, ← c83999] at e_1366_15
  have hr := e_1366_15
  simp only [e_1366_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12753)) := by
  have c84000 := (publicBatchWrapper89_copies2625 a h).1
  have c84001 := (publicBatchWrapper89_copies2625 a h).2.1
  have c84002 := (publicBatchWrapper89_copies2625 a h).2.2.1
  have c84003 := (publicBatchWrapper89_copies2625 a h).2.2.2.1
  have c84004 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.1
  have c84005 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_16
  simp only [← c84000, ← c84001, ← c84002] at e_1366_16
  have e_1366_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_17
  simp only [← c84003, ← c84004, k2, ← c84005] at e_1366_17
  have hr := e_1366_17
  simp only [e_1366_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1366 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12754)) := by
  have c84006 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.1
  have c84007 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.1
  have c84008 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.1
  have c84009 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.1
  have c84010 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84011 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1366_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_18
  simp only [← c84006, ← c84007, ← c84008] at e_1366_18
  have e_1366_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1366 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_19
  simp only [← c84009, ← c84010, k2, ← c84011] at e_1366_19
  have hr := e_1366_19
  simp only [e_1366_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12755)) := by
  have c84012 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84013 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84014 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84015 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84016 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84017 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_0
  simp only [← c84012, ← c84013, ← c84014] at e_1367_0
  have e_1367_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_1
  simp only [← c84015, ← c84016, k2, ← c84017] at e_1367_1
  have hr := e_1367_1
  simp only [e_1367_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12756)) := by
  have c84018 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84019 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84020 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84021 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84022 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84023 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_2
  simp only [← c84018, ← c84019, ← c84020] at e_1367_2
  have e_1367_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_3
  simp only [← c84021, ← c84022, k2, ← c84023] at e_1367_3
  have hr := e_1367_3
  simp only [e_1367_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12757)) := by
  have c84024 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84025 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84026 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84027 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84028 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84029 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_4
  simp only [← c84024, ← c84025, ← c84026] at e_1367_4
  have e_1367_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_5
  simp only [← c84027, ← c84028, k2, ← c84029] at e_1367_5
  have hr := e_1367_5
  simp only [e_1367_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12758)) := by
  have c84030 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84031 := (publicBatchWrapper89_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84032 := (publicBatchWrapper89_copies2626 a h).1
  have c84033 := (publicBatchWrapper89_copies2626 a h).2.1
  have c84034 := (publicBatchWrapper89_copies2626 a h).2.2.1
  have c84035 := (publicBatchWrapper89_copies2626 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_6
  simp only [← c84030, ← c84031, ← c84032] at e_1367_6
  have e_1367_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_7
  simp only [← c84033, ← c84034, k2, ← c84035] at e_1367_7
  have hr := e_1367_7
  simp only [e_1367_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12759)) := by
  have c84036 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.1
  have c84037 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.1
  have c84038 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.1
  have c84039 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.1
  have c84040 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.1
  have c84041 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_8
  simp only [← c84036, ← c84037, ← c84038] at e_1367_8
  have e_1367_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_9
  simp only [← c84039, ← c84040, k2, ← c84041] at e_1367_9
  have hr := e_1367_9
  simp only [e_1367_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12760)) := by
  have c84042 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84043 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84044 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84045 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84046 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84047 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_10
  simp only [← c84042, ← c84043, ← c84044] at e_1367_10
  have e_1367_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_11
  simp only [← c84045, ← c84046, k2, ← c84047] at e_1367_11
  have hr := e_1367_11
  simp only [e_1367_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12761)) := by
  have c84048 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84049 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84050 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84051 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84052 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84053 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_12
  simp only [← c84048, ← c84049, ← c84050] at e_1367_12
  have e_1367_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_13
  simp only [← c84051, ← c84052, k2, ← c84053] at e_1367_13
  have hr := e_1367_13
  simp only [e_1367_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12762)) := by
  have c84054 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84055 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84056 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84057 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84058 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84059 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_14
  simp only [← c84054, ← c84055, ← c84056] at e_1367_14
  have e_1367_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_15
  simp only [← c84057, ← c84058, k2, ← c84059] at e_1367_15
  have hr := e_1367_15
  simp only [e_1367_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12763)) := by
  have c84060 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84061 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84062 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84063 := (publicBatchWrapper89_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84064 := (publicBatchWrapper89_copies2627 a h).1
  have c84065 := (publicBatchWrapper89_copies2627 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_16
  simp only [← c84060, ← c84061, ← c84062] at e_1367_16
  have e_1367_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_17
  simp only [← c84063, ← c84064, k2, ← c84065] at e_1367_17
  have hr := e_1367_17
  simp only [e_1367_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1367 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12764)) := by
  have c84066 := (publicBatchWrapper89_copies2627 a h).2.2.1
  have c84067 := (publicBatchWrapper89_copies2627 a h).2.2.2.1
  have c84068 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.1
  have c84069 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.1
  have c84070 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.1
  have c84071 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1367_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_18
  simp only [← c84066, ← c84067, ← c84068] at e_1367_18
  have e_1367_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1367 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_19
  simp only [← c84069, ← c84070, k2, ← c84071] at e_1367_19
  have hr := e_1367_19
  simp only [e_1367_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12765)) := by
  have c84072 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.1
  have c84073 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.1
  have c84074 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84075 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84076 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84077 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_0
  simp only [← c84072, ← c84073, ← c84074] at e_1368_0
  have e_1368_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_1
  simp only [← c84075, ← c84076, k2, ← c84077] at e_1368_1
  have hr := e_1368_1
  simp only [e_1368_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12766)) := by
  have c84078 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84079 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84080 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84081 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84082 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84083 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_2
  simp only [← c84078, ← c84079, ← c84080] at e_1368_2
  have e_1368_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_3
  simp only [← c84081, ← c84082, k2, ← c84083] at e_1368_3
  have hr := e_1368_3
  simp only [e_1368_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12767)) := by
  have c84084 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84085 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84086 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84087 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84088 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84089 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_4
  simp only [← c84084, ← c84085, ← c84086] at e_1368_4
  have e_1368_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_5
  simp only [← c84087, ← c84088, k2, ← c84089] at e_1368_5
  have hr := e_1368_5
  simp only [e_1368_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12768)) := by
  have c84090 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84091 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84092 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84093 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84094 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84095 := (publicBatchWrapper89_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_6
  simp only [← c84090, ← c84091, ← c84092] at e_1368_6
  have e_1368_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_7
  simp only [← c84093, ← c84094, k2, ← c84095] at e_1368_7
  have hr := e_1368_7
  simp only [e_1368_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12769)) := by
  have c84096 := (publicBatchWrapper89_copies2628 a h).1
  have c84097 := (publicBatchWrapper89_copies2628 a h).2.1
  have c84098 := (publicBatchWrapper89_copies2628 a h).2.2.1
  have c84099 := (publicBatchWrapper89_copies2628 a h).2.2.2.1
  have c84100 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.1
  have c84101 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_8
  simp only [← c84096, ← c84097, ← c84098] at e_1368_8
  have e_1368_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_9
  simp only [← c84099, ← c84100, k2, ← c84101] at e_1368_9
  have hr := e_1368_9
  simp only [e_1368_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12810)) := by
  have c84102 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.1
  have c84103 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.1
  have c84104 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.1
  have c84105 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.1
  have c84106 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84107 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_10
  simp only [← c84102, ← c84103, ← c84104] at e_1368_10
  have e_1368_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_11
  simp only [← c84105, ← c84106, k2, ← c84107] at e_1368_11
  have hr := e_1368_11
  simp only [e_1368_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12811)) := by
  have c84108 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84109 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84110 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84111 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84112 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84113 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_12
  simp only [← c84108, ← c84109, ← c84110] at e_1368_12
  have e_1368_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_13
  simp only [← c84111, ← c84112, k2, ← c84113] at e_1368_13
  have hr := e_1368_13
  simp only [e_1368_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12812)) := by
  have c84114 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84115 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84116 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84117 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84118 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84119 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_14
  simp only [← c84114, ← c84115, ← c84116] at e_1368_14
  have e_1368_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_15
  simp only [← c84117, ← c84118, k2, ← c84119] at e_1368_15
  have hr := e_1368_15
  simp only [e_1368_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12813)) := by
  have c84120 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84121 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84122 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84123 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84124 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84125 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_16
  simp only [← c84120, ← c84121, ← c84122] at e_1368_16
  have e_1368_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_17
  simp only [← c84123, ← c84124, k2, ← c84125] at e_1368_17
  have hr := e_1368_17
  simp only [e_1368_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1368 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12814)) := by
  have c84126 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84127 := (publicBatchWrapper89_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84128 := (publicBatchWrapper89_copies2629 a h).1
  have c84129 := (publicBatchWrapper89_copies2629 a h).2.1
  have c84130 := (publicBatchWrapper89_copies2629 a h).2.2.1
  have c84131 := (publicBatchWrapper89_copies2629 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1368_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_18
  simp only [← c84126, ← c84127, ← c84128] at e_1368_18
  have e_1368_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1368 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_19
  simp only [← c84129, ← c84130, k2, ← c84131] at e_1368_19
  have hr := e_1368_19
  simp only [e_1368_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12815)) := by
  have c84132 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.1
  have c84133 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.1
  have c84134 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.1
  have c84135 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.1
  have c84136 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.1
  have c84137 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_0
  simp only [← c84132, ← c84133, ← c84134] at e_1369_0
  have e_1369_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_1
  simp only [← c84135, ← c84136, k2, ← c84137] at e_1369_1
  have hr := e_1369_1
  simp only [e_1369_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12816)) := by
  have c84138 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84139 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84140 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84141 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84142 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84143 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_2
  simp only [← c84138, ← c84139, ← c84140] at e_1369_2
  have e_1369_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_3
  simp only [← c84141, ← c84142, k2, ← c84143] at e_1369_3
  have hr := e_1369_3
  simp only [e_1369_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12817)) := by
  have c84144 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84145 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84146 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84147 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84148 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84149 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_4
  simp only [← c84144, ← c84145, ← c84146] at e_1369_4
  have e_1369_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_5
  simp only [← c84147, ← c84148, k2, ← c84149] at e_1369_5
  have hr := e_1369_5
  simp only [e_1369_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12818)) := by
  have c84150 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84151 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84152 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84153 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84154 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84155 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_6
  simp only [← c84150, ← c84151, ← c84152] at e_1369_6
  have e_1369_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_7
  simp only [← c84153, ← c84154, k2, ← c84155] at e_1369_7
  have hr := e_1369_7
  simp only [e_1369_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12819)) := by
  have c84156 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84157 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84158 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84159 := (publicBatchWrapper89_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84160 := (publicBatchWrapper89_copies2630 a h).1
  have c84161 := (publicBatchWrapper89_copies2630 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_8
  simp only [← c84156, ← c84157, ← c84158] at e_1369_8
  have e_1369_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_9
  simp only [← c84159, ← c84160, k2, ← c84161] at e_1369_9
  have hr := e_1369_9
  simp only [e_1369_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12820)) := by
  have c84162 := (publicBatchWrapper89_copies2630 a h).2.2.1
  have c84163 := (publicBatchWrapper89_copies2630 a h).2.2.2.1
  have c84164 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.1
  have c84165 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.1
  have c84166 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.1
  have c84167 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_10
  simp only [← c84162, ← c84163, ← c84164] at e_1369_10
  have e_1369_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_11
  simp only [← c84165, ← c84166, k2, ← c84167] at e_1369_11
  have hr := e_1369_11
  simp only [e_1369_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12821)) := by
  have c84168 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.1
  have c84169 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.1
  have c84170 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84171 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84172 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84173 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_12
  simp only [← c84168, ← c84169, ← c84170] at e_1369_12
  have e_1369_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_13
  simp only [← c84171, ← c84172, k2, ← c84173] at e_1369_13
  have hr := e_1369_13
  simp only [e_1369_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12822)) := by
  have c84174 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84175 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84176 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84177 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84178 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84179 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_14
  simp only [← c84174, ← c84175, ← c84176] at e_1369_14
  have e_1369_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_15
  simp only [← c84177, ← c84178, k2, ← c84179] at e_1369_15
  have hr := e_1369_15
  simp only [e_1369_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12823)) := by
  have c84180 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84181 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84182 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84183 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84184 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84185 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_16
  simp only [← c84180, ← c84181, ← c84182] at e_1369_16
  have e_1369_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_17
  simp only [← c84183, ← c84184, k2, ← c84185] at e_1369_17
  have hr := e_1369_17
  simp only [e_1369_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1369 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12824)) := by
  have c84186 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84187 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84188 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84189 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84190 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84191 := (publicBatchWrapper89_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1369_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_18
  simp only [← c84186, ← c84187, ← c84188] at e_1369_18
  have e_1369_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1369 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_19
  simp only [← c84189, ← c84190, k2, ← c84191] at e_1369_19
  have hr := e_1369_19
  simp only [e_1369_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12825)) := by
  have c84192 := (publicBatchWrapper89_copies2631 a h).1
  have c84193 := (publicBatchWrapper89_copies2631 a h).2.1
  have c84194 := (publicBatchWrapper89_copies2631 a h).2.2.1
  have c84195 := (publicBatchWrapper89_copies2631 a h).2.2.2.1
  have c84196 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.1
  have c84197 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_0
  simp only [← c84192, ← c84193, ← c84194] at e_1370_0
  have e_1370_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_1
  simp only [← c84195, ← c84196, k2, ← c84197] at e_1370_1
  have hr := e_1370_1
  simp only [e_1370_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12826)) := by
  have c84198 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.1
  have c84199 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.1
  have c84200 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.1
  have c84201 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.1
  have c84202 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84203 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_2
  simp only [← c84198, ← c84199, ← c84200] at e_1370_2
  have e_1370_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_3
  simp only [← c84201, ← c84202, k2, ← c84203] at e_1370_3
  have hr := e_1370_3
  simp only [e_1370_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12827)) := by
  have c84204 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84205 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84206 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84207 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84208 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84209 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_4
  simp only [← c84204, ← c84205, ← c84206] at e_1370_4
  have e_1370_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_5
  simp only [← c84207, ← c84208, k2, ← c84209] at e_1370_5
  have hr := e_1370_5
  simp only [e_1370_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12828)) := by
  have c84210 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84211 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84212 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84213 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84214 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84215 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_6
  simp only [← c84210, ← c84211, ← c84212] at e_1370_6
  have e_1370_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_7
  simp only [← c84213, ← c84214, k2, ← c84215] at e_1370_7
  have hr := e_1370_7
  simp only [e_1370_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12829)) := by
  have c84216 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84217 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84218 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84219 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84220 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84221 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_8
  simp only [← c84216, ← c84217, ← c84218] at e_1370_8
  have e_1370_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_9
  simp only [← c84219, ← c84220, k2, ← c84221] at e_1370_9
  have hr := e_1370_9
  simp only [e_1370_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12830)) := by
  have c84222 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84223 := (publicBatchWrapper89_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84224 := (publicBatchWrapper89_copies2632 a h).1
  have c84225 := (publicBatchWrapper89_copies2632 a h).2.1
  have c84226 := (publicBatchWrapper89_copies2632 a h).2.2.1
  have c84227 := (publicBatchWrapper89_copies2632 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_10
  simp only [← c84222, ← c84223, ← c84224] at e_1370_10
  have e_1370_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_11
  simp only [← c84225, ← c84226, k2, ← c84227] at e_1370_11
  have hr := e_1370_11
  simp only [e_1370_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12831)) := by
  have c84228 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.1
  have c84229 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.1
  have c84230 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.1
  have c84231 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.1
  have c84232 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.1
  have c84233 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_12
  simp only [← c84228, ← c84229, ← c84230] at e_1370_12
  have e_1370_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_13
  simp only [← c84231, ← c84232, k2, ← c84233] at e_1370_13
  have hr := e_1370_13
  simp only [e_1370_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12832)) := by
  have c84234 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84235 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84236 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84237 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84238 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84239 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_14
  simp only [← c84234, ← c84235, ← c84236] at e_1370_14
  have e_1370_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_15
  simp only [← c84237, ← c84238, k2, ← c84239] at e_1370_15
  have hr := e_1370_15
  simp only [e_1370_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12833)) := by
  have c84240 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84241 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84242 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84243 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84244 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84245 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_16
  simp only [← c84240, ← c84241, ← c84242] at e_1370_16
  have e_1370_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_17
  simp only [← c84243, ← c84244, k2, ← c84245] at e_1370_17
  have hr := e_1370_17
  simp only [e_1370_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1370 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12834)) := by
  have c84246 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84247 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84248 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84249 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84250 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84251 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1370_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_18
  simp only [← c84246, ← c84247, ← c84248] at e_1370_18
  have e_1370_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1370 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_19
  simp only [← c84249, ← c84250, k2, ← c84251] at e_1370_19
  have hr := e_1370_19
  simp only [e_1370_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12835)) := by
  have c84252 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84253 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84254 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84255 := (publicBatchWrapper89_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84256 := (publicBatchWrapper89_copies2633 a h).1
  have c84257 := (publicBatchWrapper89_copies2633 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_0
  simp only [← c84252, ← c84253, ← c84254] at e_1371_0
  have e_1371_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_1
  simp only [← c84255, ← c84256, k2, ← c84257] at e_1371_1
  have hr := e_1371_1
  simp only [e_1371_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12836)) := by
  have c84258 := (publicBatchWrapper89_copies2633 a h).2.2.1
  have c84259 := (publicBatchWrapper89_copies2633 a h).2.2.2.1
  have c84260 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.1
  have c84261 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.1
  have c84262 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.1
  have c84263 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_2
  simp only [← c84258, ← c84259, ← c84260] at e_1371_2
  have e_1371_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_3
  simp only [← c84261, ← c84262, k2, ← c84263] at e_1371_3
  have hr := e_1371_3
  simp only [e_1371_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12837)) := by
  have c84264 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.1
  have c84265 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.1
  have c84266 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84267 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84268 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84269 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_4
  simp only [← c84264, ← c84265, ← c84266] at e_1371_4
  have e_1371_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_5
  simp only [← c84267, ← c84268, k2, ← c84269] at e_1371_5
  have hr := e_1371_5
  simp only [e_1371_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12838)) := by
  have c84270 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84271 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84272 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84273 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84274 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84275 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_6
  simp only [← c84270, ← c84271, ← c84272] at e_1371_6
  have e_1371_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_7
  simp only [← c84273, ← c84274, k2, ← c84275] at e_1371_7
  have hr := e_1371_7
  simp only [e_1371_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12839)) := by
  have c84276 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84277 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84278 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84279 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84280 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84281 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_8
  simp only [← c84276, ← c84277, ← c84278] at e_1371_8
  have e_1371_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_9
  simp only [← c84279, ← c84280, k2, ← c84281] at e_1371_9
  have hr := e_1371_9
  simp only [e_1371_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12840)) := by
  have c84282 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84283 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84284 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84285 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84286 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84287 := (publicBatchWrapper89_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_10
  simp only [← c84282, ← c84283, ← c84284] at e_1371_10
  have e_1371_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_11
  simp only [← c84285, ← c84286, k2, ← c84287] at e_1371_11
  have hr := e_1371_11
  simp only [e_1371_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12841)) := by
  have c84288 := (publicBatchWrapper89_copies2634 a h).1
  have c84289 := (publicBatchWrapper89_copies2634 a h).2.1
  have c84290 := (publicBatchWrapper89_copies2634 a h).2.2.1
  have c84291 := (publicBatchWrapper89_copies2634 a h).2.2.2.1
  have c84292 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.1
  have c84293 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_12
  simp only [← c84288, ← c84289, ← c84290] at e_1371_12
  have e_1371_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_13
  simp only [← c84291, ← c84292, k2, ← c84293] at e_1371_13
  have hr := e_1371_13
  simp only [e_1371_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12842)) := by
  have c84294 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.1
  have c84295 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.1
  have c84296 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.1
  have c84297 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.1
  have c84298 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84299 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_14
  simp only [← c84294, ← c84295, ← c84296] at e_1371_14
  have e_1371_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_15
  simp only [← c84297, ← c84298, k2, ← c84299] at e_1371_15
  have hr := e_1371_15
  simp only [e_1371_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12843)) := by
  have c84300 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84301 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84302 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84303 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84304 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84305 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_16
  simp only [← c84300, ← c84301, ← c84302] at e_1371_16
  have e_1371_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_17
  simp only [← c84303, ← c84304, k2, ← c84305] at e_1371_17
  have hr := e_1371_17
  simp only [e_1371_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1371 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12844)) := by
  have c84306 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84307 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84308 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84309 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84310 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84311 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1371_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_18
  simp only [← c84306, ← c84307, ← c84308] at e_1371_18
  have e_1371_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1371 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_19
  simp only [← c84309, ← c84310, k2, ← c84311] at e_1371_19
  have hr := e_1371_19
  simp only [e_1371_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12845)) := by
  have c84312 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84313 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84314 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84315 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84316 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84317 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_0
  simp only [← c84312, ← c84313, ← c84314] at e_1372_0
  have e_1372_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_1
  simp only [← c84315, ← c84316, k2, ← c84317] at e_1372_1
  have hr := e_1372_1
  simp only [e_1372_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12846)) := by
  have c84318 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84319 := (publicBatchWrapper89_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84320 := (publicBatchWrapper89_copies2635 a h).1
  have c84321 := (publicBatchWrapper89_copies2635 a h).2.1
  have c84322 := (publicBatchWrapper89_copies2635 a h).2.2.1
  have c84323 := (publicBatchWrapper89_copies2635 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_2
  simp only [← c84318, ← c84319, ← c84320] at e_1372_2
  have e_1372_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_3
  simp only [← c84321, ← c84322, k2, ← c84323] at e_1372_3
  have hr := e_1372_3
  simp only [e_1372_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12847)) := by
  have c84324 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.1
  have c84325 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.1
  have c84326 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.1
  have c84327 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.1
  have c84328 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.1
  have c84329 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_4
  simp only [← c84324, ← c84325, ← c84326] at e_1372_4
  have e_1372_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_5
  simp only [← c84327, ← c84328, k2, ← c84329] at e_1372_5
  have hr := e_1372_5
  simp only [e_1372_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12848)) := by
  have c84330 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84331 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84332 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84333 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84334 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84335 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_6
  simp only [← c84330, ← c84331, ← c84332] at e_1372_6
  have e_1372_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_7
  simp only [← c84333, ← c84334, k2, ← c84335] at e_1372_7
  have hr := e_1372_7
  simp only [e_1372_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12849)) := by
  have c84336 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84337 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84338 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84339 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84340 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84341 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_8
  simp only [← c84336, ← c84337, ← c84338] at e_1372_8
  have e_1372_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_9
  simp only [← c84339, ← c84340, k2, ← c84341] at e_1372_9
  have hr := e_1372_9
  simp only [e_1372_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12850)) := by
  have c84342 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84343 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84344 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84345 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84346 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84347 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_10
  simp only [← c84342, ← c84343, ← c84344] at e_1372_10
  have e_1372_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_11
  simp only [← c84345, ← c84346, k2, ← c84347] at e_1372_11
  have hr := e_1372_11
  simp only [e_1372_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12851)) := by
  have c84348 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84349 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84350 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84351 := (publicBatchWrapper89_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84352 := (publicBatchWrapper89_copies2636 a h).1
  have c84353 := (publicBatchWrapper89_copies2636 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_12
  simp only [← c84348, ← c84349, ← c84350] at e_1372_12
  have e_1372_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_13
  simp only [← c84351, ← c84352, k2, ← c84353] at e_1372_13
  have hr := e_1372_13
  simp only [e_1372_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12852)) := by
  have c84354 := (publicBatchWrapper89_copies2636 a h).2.2.1
  have c84355 := (publicBatchWrapper89_copies2636 a h).2.2.2.1
  have c84356 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.1
  have c84357 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.1
  have c84358 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.1
  have c84359 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_14
  simp only [← c84354, ← c84355, ← c84356] at e_1372_14
  have e_1372_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_15
  simp only [← c84357, ← c84358, k2, ← c84359] at e_1372_15
  have hr := e_1372_15
  simp only [e_1372_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12853)) := by
  have c84360 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.1
  have c84361 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.1
  have c84362 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84363 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84364 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84365 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_16
  simp only [← c84360, ← c84361, ← c84362] at e_1372_16
  have e_1372_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_17
  simp only [← c84363, ← c84364, k2, ← c84365] at e_1372_17
  have hr := e_1372_17
  simp only [e_1372_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1372 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12854)) := by
  have c84366 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84367 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84368 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84369 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84370 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84371 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1372_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_18
  simp only [← c84366, ← c84367, ← c84368] at e_1372_18
  have e_1372_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1372 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_19
  simp only [← c84369, ← c84370, k2, ← c84371] at e_1372_19
  have hr := e_1372_19
  simp only [e_1372_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12855)) := by
  have c84372 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84373 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84374 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84375 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84376 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84377 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_0
  simp only [← c84372, ← c84373, ← c84374] at e_1373_0
  have e_1373_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_1
  simp only [← c84375, ← c84376, k2, ← c84377] at e_1373_1
  have hr := e_1373_1
  simp only [e_1373_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12856)) := by
  have c84378 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84379 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84380 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84381 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84382 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84383 := (publicBatchWrapper89_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_2
  simp only [← c84378, ← c84379, ← c84380] at e_1373_2
  have e_1373_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_3
  simp only [← c84381, ← c84382, k2, ← c84383] at e_1373_3
  have hr := e_1373_3
  simp only [e_1373_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12857)) := by
  have c84384 := (publicBatchWrapper89_copies2637 a h).1
  have c84385 := (publicBatchWrapper89_copies2637 a h).2.1
  have c84386 := (publicBatchWrapper89_copies2637 a h).2.2.1
  have c84387 := (publicBatchWrapper89_copies2637 a h).2.2.2.1
  have c84388 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.1
  have c84389 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_4
  simp only [← c84384, ← c84385, ← c84386] at e_1373_4
  have e_1373_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_5
  simp only [← c84387, ← c84388, k2, ← c84389] at e_1373_5
  have hr := e_1373_5
  simp only [e_1373_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12858)) := by
  have c84390 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.1
  have c84391 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.1
  have c84392 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.1
  have c84393 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.1
  have c84394 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84395 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_6
  simp only [← c84390, ← c84391, ← c84392] at e_1373_6
  have e_1373_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_7
  simp only [← c84393, ← c84394, k2, ← c84395] at e_1373_7
  have hr := e_1373_7
  simp only [e_1373_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12859)) := by
  have c84396 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84397 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84398 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84399 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84400 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84401 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_8
  simp only [← c84396, ← c84397, ← c84398] at e_1373_8
  have e_1373_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_9
  simp only [← c84399, ← c84400, k2, ← c84401] at e_1373_9
  have hr := e_1373_9
  simp only [e_1373_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12860)) := by
  have c84402 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84403 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84404 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84405 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84406 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84407 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_10
  simp only [← c84402, ← c84403, ← c84404] at e_1373_10
  have e_1373_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_11
  simp only [← c84405, ← c84406, k2, ← c84407] at e_1373_11
  have hr := e_1373_11
  simp only [e_1373_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12861)) := by
  have c84408 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84409 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84410 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84411 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84412 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84413 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_12
  simp only [← c84408, ← c84409, ← c84410] at e_1373_12
  have e_1373_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_13
  simp only [← c84411, ← c84412, k2, ← c84413] at e_1373_13
  have hr := e_1373_13
  simp only [e_1373_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12862)) := by
  have c84414 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84415 := (publicBatchWrapper89_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84416 := (publicBatchWrapper89_copies2638 a h).1
  have c84417 := (publicBatchWrapper89_copies2638 a h).2.1
  have c84418 := (publicBatchWrapper89_copies2638 a h).2.2.1
  have c84419 := (publicBatchWrapper89_copies2638 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_14
  simp only [← c84414, ← c84415, ← c84416] at e_1373_14
  have e_1373_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_15
  simp only [← c84417, ← c84418, k2, ← c84419] at e_1373_15
  have hr := e_1373_15
  simp only [e_1373_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12863)) := by
  have c84420 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.1
  have c84421 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.1
  have c84422 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.1
  have c84423 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.1
  have c84424 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.1
  have c84425 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_16
  simp only [← c84420, ← c84421, ← c84422] at e_1373_16
  have e_1373_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_17
  simp only [← c84423, ← c84424, k2, ← c84425] at e_1373_17
  have hr := e_1373_17
  simp only [e_1373_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1373 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12864)) := by
  have c84426 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84427 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84428 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84429 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84430 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84431 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1373_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_18
  simp only [← c84426, ← c84427, ← c84428] at e_1373_18
  have e_1373_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1373 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_19
  simp only [← c84429, ← c84430, k2, ← c84431] at e_1373_19
  have hr := e_1373_19
  simp only [e_1373_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12865)) := by
  have c84432 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84433 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84434 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84435 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84436 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84437 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_0
  simp only [← c84432, ← c84433, ← c84434] at e_1374_0
  have e_1374_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_1
  simp only [← c84435, ← c84436, k2, ← c84437] at e_1374_1
  have hr := e_1374_1
  simp only [e_1374_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12866)) := by
  have c84438 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84439 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84440 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84441 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84442 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84443 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_2
  simp only [← c84438, ← c84439, ← c84440] at e_1374_2
  have e_1374_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_3
  simp only [← c84441, ← c84442, k2, ← c84443] at e_1374_3
  have hr := e_1374_3
  simp only [e_1374_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12867)) := by
  have c84444 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84445 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84446 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84447 := (publicBatchWrapper89_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84448 := (publicBatchWrapper89_copies2639 a h).1
  have c84449 := (publicBatchWrapper89_copies2639 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_4
  simp only [← c84444, ← c84445, ← c84446] at e_1374_4
  have e_1374_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_5
  simp only [← c84447, ← c84448, k2, ← c84449] at e_1374_5
  have hr := e_1374_5
  simp only [e_1374_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12868)) := by
  have c84450 := (publicBatchWrapper89_copies2639 a h).2.2.1
  have c84451 := (publicBatchWrapper89_copies2639 a h).2.2.2.1
  have c84452 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.1
  have c84453 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.1
  have c84454 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.1
  have c84455 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_6
  simp only [← c84450, ← c84451, ← c84452] at e_1374_6
  have e_1374_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_7
  simp only [← c84453, ← c84454, k2, ← c84455] at e_1374_7
  have hr := e_1374_7
  simp only [e_1374_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12869)) := by
  have c84456 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.1
  have c84457 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.1
  have c84458 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84459 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84460 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84461 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_8
  simp only [← c84456, ← c84457, ← c84458] at e_1374_8
  have e_1374_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_9
  simp only [← c84459, ← c84460, k2, ← c84461] at e_1374_9
  have hr := e_1374_9
  simp only [e_1374_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12870)) := by
  have c84462 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84463 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84464 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84465 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84466 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84467 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_10
  simp only [← c84462, ← c84463, ← c84464] at e_1374_10
  have e_1374_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_11
  simp only [← c84465, ← c84466, k2, ← c84467] at e_1374_11
  have hr := e_1374_11
  simp only [e_1374_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12871)) := by
  have c84468 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84469 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84470 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84471 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84472 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84473 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_12
  simp only [← c84468, ← c84469, ← c84470] at e_1374_12
  have e_1374_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_13
  simp only [← c84471, ← c84472, k2, ← c84473] at e_1374_13
  have hr := e_1374_13
  simp only [e_1374_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12872)) := by
  have c84474 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84475 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84476 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84477 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84478 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84479 := (publicBatchWrapper89_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_14
  simp only [← c84474, ← c84475, ← c84476] at e_1374_14
  have e_1374_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_15
  simp only [← c84477, ← c84478, k2, ← c84479] at e_1374_15
  have hr := e_1374_15
  simp only [e_1374_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12873)) := by
  have c84480 := (publicBatchWrapper89_copies2640 a h).1
  have c84481 := (publicBatchWrapper89_copies2640 a h).2.1
  have c84482 := (publicBatchWrapper89_copies2640 a h).2.2.1
  have c84483 := (publicBatchWrapper89_copies2640 a h).2.2.2.1
  have c84484 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.1
  have c84485 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_16
  simp only [← c84480, ← c84481, ← c84482] at e_1374_16
  have e_1374_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_17
  simp only [← c84483, ← c84484, k2, ← c84485] at e_1374_17
  have hr := e_1374_17
  simp only [e_1374_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1374 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12874)) := by
  have c84486 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.1
  have c84487 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.1
  have c84488 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.1
  have c84489 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.1
  have c84490 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84491 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1374_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_18
  simp only [← c84486, ← c84487, ← c84488] at e_1374_18
  have e_1374_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1374 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_19
  simp only [← c84489, ← c84490, k2, ← c84491] at e_1374_19
  have hr := e_1374_19
  simp only [e_1374_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12875)) := by
  have c84492 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84493 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84494 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84495 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84496 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84497 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_0
  simp only [← c84492, ← c84493, ← c84494] at e_1375_0
  have e_1375_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_1
  simp only [← c84495, ← c84496, k2, ← c84497] at e_1375_1
  have hr := e_1375_1
  simp only [e_1375_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12876)) := by
  have c84498 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84499 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84500 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84501 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84502 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84503 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_2
  simp only [← c84498, ← c84499, ← c84500] at e_1375_2
  have e_1375_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_3
  simp only [← c84501, ← c84502, k2, ← c84503] at e_1375_3
  have hr := e_1375_3
  simp only [e_1375_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12877)) := by
  have c84504 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84505 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84506 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84507 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84508 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84509 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_4
  simp only [← c84504, ← c84505, ← c84506] at e_1375_4
  have e_1375_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_5
  simp only [← c84507, ← c84508, k2, ← c84509] at e_1375_5
  have hr := e_1375_5
  simp only [e_1375_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12878)) := by
  have c84510 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84511 := (publicBatchWrapper89_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84512 := (publicBatchWrapper89_copies2641 a h).1
  have c84513 := (publicBatchWrapper89_copies2641 a h).2.1
  have c84514 := (publicBatchWrapper89_copies2641 a h).2.2.1
  have c84515 := (publicBatchWrapper89_copies2641 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_6
  simp only [← c84510, ← c84511, ← c84512] at e_1375_6
  have e_1375_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_7
  simp only [← c84513, ← c84514, k2, ← c84515] at e_1375_7
  have hr := e_1375_7
  simp only [e_1375_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12879)) := by
  have c84516 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.1
  have c84517 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.1
  have c84518 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.1
  have c84519 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.1
  have c84520 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.1
  have c84521 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_8
  simp only [← c84516, ← c84517, ← c84518] at e_1375_8
  have e_1375_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_9
  simp only [← c84519, ← c84520, k2, ← c84521] at e_1375_9
  have hr := e_1375_9
  simp only [e_1375_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12880)) := by
  have c84522 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84523 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84524 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84525 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84526 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84527 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_10
  simp only [← c84522, ← c84523, ← c84524] at e_1375_10
  have e_1375_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_11
  simp only [← c84525, ← c84526, k2, ← c84527] at e_1375_11
  have hr := e_1375_11
  simp only [e_1375_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12881)) := by
  have c84528 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84529 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84530 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84531 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84532 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84533 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_12
  simp only [← c84528, ← c84529, ← c84530] at e_1375_12
  have e_1375_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_13
  simp only [← c84531, ← c84532, k2, ← c84533] at e_1375_13
  have hr := e_1375_13
  simp only [e_1375_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12882)) := by
  have c84534 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84535 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84536 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84537 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84538 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84539 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_14
  simp only [← c84534, ← c84535, ← c84536] at e_1375_14
  have e_1375_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_15
  simp only [← c84537, ← c84538, k2, ← c84539] at e_1375_15
  have hr := e_1375_15
  simp only [e_1375_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12883)) := by
  have c84540 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84541 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84542 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84543 := (publicBatchWrapper89_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84544 := (publicBatchWrapper89_copies2642 a h).1
  have c84545 := (publicBatchWrapper89_copies2642 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_16
  simp only [← c84540, ← c84541, ← c84542] at e_1375_16
  have e_1375_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_17
  simp only [← c84543, ← c84544, k2, ← c84545] at e_1375_17
  have hr := e_1375_17
  simp only [e_1375_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1375 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12884)) := by
  have c84546 := (publicBatchWrapper89_copies2642 a h).2.2.1
  have c84547 := (publicBatchWrapper89_copies2642 a h).2.2.2.1
  have c84548 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.1
  have c84549 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.1
  have c84550 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.1
  have c84551 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1375_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_18
  simp only [← c84546, ← c84547, ← c84548] at e_1375_18
  have e_1375_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1375 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_19
  simp only [← c84549, ← c84550, k2, ← c84551] at e_1375_19
  have hr := e_1375_19
  simp only [e_1375_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12885)) := by
  have c84552 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.1
  have c84553 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.1
  have c84554 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84555 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84556 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84557 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_0
  simp only [← c84552, ← c84553, ← c84554] at e_1376_0
  have e_1376_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_1
  simp only [← c84555, ← c84556, k2, ← c84557] at e_1376_1
  have hr := e_1376_1
  simp only [e_1376_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12886)) := by
  have c84558 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84559 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84560 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84561 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84562 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84563 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_2
  simp only [← c84558, ← c84559, ← c84560] at e_1376_2
  have e_1376_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_3
  simp only [← c84561, ← c84562, k2, ← c84563] at e_1376_3
  have hr := e_1376_3
  simp only [e_1376_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12887)) := by
  have c84564 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84565 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84566 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84567 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84568 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84569 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_4
  simp only [← c84564, ← c84565, ← c84566] at e_1376_4
  have e_1376_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_5
  simp only [← c84567, ← c84568, k2, ← c84569] at e_1376_5
  have hr := e_1376_5
  simp only [e_1376_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12888)) := by
  have c84570 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84571 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84572 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84573 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84574 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84575 := (publicBatchWrapper89_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_6
  simp only [← c84570, ← c84571, ← c84572] at e_1376_6
  have e_1376_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_7
  simp only [← c84573, ← c84574, k2, ← c84575] at e_1376_7
  have hr := e_1376_7
  simp only [e_1376_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12889)) := by
  have c84576 := (publicBatchWrapper89_copies2643 a h).1
  have c84577 := (publicBatchWrapper89_copies2643 a h).2.1
  have c84578 := (publicBatchWrapper89_copies2643 a h).2.2.1
  have c84579 := (publicBatchWrapper89_copies2643 a h).2.2.2.1
  have c84580 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.1
  have c84581 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_8
  simp only [← c84576, ← c84577, ← c84578] at e_1376_8
  have e_1376_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_9
  simp only [← c84579, ← c84580, k2, ← c84581] at e_1376_9
  have hr := e_1376_9
  simp only [e_1376_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12890)) := by
  have c84582 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.1
  have c84583 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.1
  have c84584 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.1
  have c84585 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.1
  have c84586 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84587 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_10
  simp only [← c84582, ← c84583, ← c84584] at e_1376_10
  have e_1376_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_11
  simp only [← c84585, ← c84586, k2, ← c84587] at e_1376_11
  have hr := e_1376_11
  simp only [e_1376_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12891)) := by
  have c84588 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84589 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84590 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84591 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84592 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84593 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_12
  simp only [← c84588, ← c84589, ← c84590] at e_1376_12
  have e_1376_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_13
  simp only [← c84591, ← c84592, k2, ← c84593] at e_1376_13
  have hr := e_1376_13
  simp only [e_1376_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12892)) := by
  have c84594 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84595 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84596 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84597 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84598 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84599 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_14
  simp only [← c84594, ← c84595, ← c84596] at e_1376_14
  have e_1376_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_15
  simp only [← c84597, ← c84598, k2, ← c84599] at e_1376_15
  have hr := e_1376_15
  simp only [e_1376_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12893)) := by
  have c84600 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84601 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84602 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84603 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84604 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84605 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_16
  simp only [← c84600, ← c84601, ← c84602] at e_1376_16
  have e_1376_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_17
  simp only [← c84603, ← c84604, k2, ← c84605] at e_1376_17
  have hr := e_1376_17
  simp only [e_1376_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1376 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12894)) := by
  have c84606 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84607 := (publicBatchWrapper89_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84608 := (publicBatchWrapper89_copies2644 a h).1
  have c84609 := (publicBatchWrapper89_copies2644 a h).2.1
  have c84610 := (publicBatchWrapper89_copies2644 a h).2.2.1
  have c84611 := (publicBatchWrapper89_copies2644 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1376_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_18
  simp only [← c84606, ← c84607, ← c84608] at e_1376_18
  have e_1376_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1376 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_19
  simp only [← c84609, ← c84610, k2, ← c84611] at e_1376_19
  have hr := e_1376_19
  simp only [e_1376_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12895)) := by
  have c84612 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.1
  have c84613 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.1
  have c84614 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.1
  have c84615 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.1
  have c84616 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.1
  have c84617 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_0
  simp only [← c84612, ← c84613, ← c84614] at e_1377_0
  have e_1377_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_1
  simp only [← c84615, ← c84616, k2, ← c84617] at e_1377_1
  have hr := e_1377_1
  simp only [e_1377_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12896)) := by
  have c84618 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84619 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84620 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84621 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84622 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84623 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_2
  simp only [← c84618, ← c84619, ← c84620] at e_1377_2
  have e_1377_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_3
  simp only [← c84621, ← c84622, k2, ← c84623] at e_1377_3
  have hr := e_1377_3
  simp only [e_1377_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12897)) := by
  have c84624 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84625 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84626 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84627 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84628 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84629 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_4
  simp only [← c84624, ← c84625, ← c84626] at e_1377_4
  have e_1377_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_5
  simp only [← c84627, ← c84628, k2, ← c84629] at e_1377_5
  have hr := e_1377_5
  simp only [e_1377_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12898)) := by
  have c84630 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84631 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84632 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84633 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84634 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84635 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_6
  simp only [← c84630, ← c84631, ← c84632] at e_1377_6
  have e_1377_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_7
  simp only [← c84633, ← c84634, k2, ← c84635] at e_1377_7
  have hr := e_1377_7
  simp only [e_1377_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12899)) := by
  have c84636 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84637 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84638 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84639 := (publicBatchWrapper89_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84640 := (publicBatchWrapper89_copies2645 a h).1
  have c84641 := (publicBatchWrapper89_copies2645 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_8
  simp only [← c84636, ← c84637, ← c84638] at e_1377_8
  have e_1377_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_9
  simp only [← c84639, ← c84640, k2, ← c84641] at e_1377_9
  have hr := e_1377_9
  simp only [e_1377_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12900)) := by
  have c84642 := (publicBatchWrapper89_copies2645 a h).2.2.1
  have c84643 := (publicBatchWrapper89_copies2645 a h).2.2.2.1
  have c84644 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.1
  have c84645 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.1
  have c84646 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.1
  have c84647 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_10
  simp only [← c84642, ← c84643, ← c84644] at e_1377_10
  have e_1377_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_11
  simp only [← c84645, ← c84646, k2, ← c84647] at e_1377_11
  have hr := e_1377_11
  simp only [e_1377_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12901)) := by
  have c84648 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.1
  have c84649 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.1
  have c84650 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84651 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84652 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84653 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_12
  simp only [← c84648, ← c84649, ← c84650] at e_1377_12
  have e_1377_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_13
  simp only [← c84651, ← c84652, k2, ← c84653] at e_1377_13
  have hr := e_1377_13
  simp only [e_1377_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12902)) := by
  have c84654 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84655 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84656 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84657 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84658 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84659 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_14
  simp only [← c84654, ← c84655, ← c84656] at e_1377_14
  have e_1377_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_15
  simp only [← c84657, ← c84658, k2, ← c84659] at e_1377_15
  have hr := e_1377_15
  simp only [e_1377_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12903)) := by
  have c84660 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84661 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84662 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84663 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84664 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84665 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_16
  simp only [← c84660, ← c84661, ← c84662] at e_1377_16
  have e_1377_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_17
  simp only [← c84663, ← c84664, k2, ← c84665] at e_1377_17
  have hr := e_1377_17
  simp only [e_1377_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1377 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12904)) := by
  have c84666 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84667 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84668 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84669 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84670 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84671 := (publicBatchWrapper89_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1377_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_18
  simp only [← c84666, ← c84667, ← c84668] at e_1377_18
  have e_1377_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1377 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_19
  simp only [← c84669, ← c84670, k2, ← c84671] at e_1377_19
  have hr := e_1377_19
  simp only [e_1377_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12905)) := by
  have c84672 := (publicBatchWrapper89_copies2646 a h).1
  have c84673 := (publicBatchWrapper89_copies2646 a h).2.1
  have c84674 := (publicBatchWrapper89_copies2646 a h).2.2.1
  have c84675 := (publicBatchWrapper89_copies2646 a h).2.2.2.1
  have c84676 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.1
  have c84677 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_0
  simp only [← c84672, ← c84673, ← c84674] at e_1378_0
  have e_1378_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_1
  simp only [← c84675, ← c84676, k2, ← c84677] at e_1378_1
  have hr := e_1378_1
  simp only [e_1378_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12906)) := by
  have c84678 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.1
  have c84679 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.1
  have c84680 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.1
  have c84681 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.1
  have c84682 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84683 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_2
  simp only [← c84678, ← c84679, ← c84680] at e_1378_2
  have e_1378_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_3
  simp only [← c84681, ← c84682, k2, ← c84683] at e_1378_3
  have hr := e_1378_3
  simp only [e_1378_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12907)) := by
  have c84684 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84685 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84686 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84687 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84688 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84689 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_4
  simp only [← c84684, ← c84685, ← c84686] at e_1378_4
  have e_1378_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_5
  simp only [← c84687, ← c84688, k2, ← c84689] at e_1378_5
  have hr := e_1378_5
  simp only [e_1378_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12908)) := by
  have c84690 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84691 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84692 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84693 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84694 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84695 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_6
  simp only [← c84690, ← c84691, ← c84692] at e_1378_6
  have e_1378_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_7
  simp only [← c84693, ← c84694, k2, ← c84695] at e_1378_7
  have hr := e_1378_7
  simp only [e_1378_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12909)) := by
  have c84696 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84697 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84698 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84699 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84700 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84701 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_8
  simp only [← c84696, ← c84697, ← c84698] at e_1378_8
  have e_1378_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_9
  simp only [← c84699, ← c84700, k2, ← c84701] at e_1378_9
  have hr := e_1378_9
  simp only [e_1378_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12910)) := by
  have c84702 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84703 := (publicBatchWrapper89_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84704 := (publicBatchWrapper89_copies2647 a h).1
  have c84705 := (publicBatchWrapper89_copies2647 a h).2.1
  have c84706 := (publicBatchWrapper89_copies2647 a h).2.2.1
  have c84707 := (publicBatchWrapper89_copies2647 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_10
  simp only [← c84702, ← c84703, ← c84704] at e_1378_10
  have e_1378_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_11
  simp only [← c84705, ← c84706, k2, ← c84707] at e_1378_11
  have hr := e_1378_11
  simp only [e_1378_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12911)) := by
  have c84708 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.1
  have c84709 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.1
  have c84710 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.1
  have c84711 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.1
  have c84712 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.1
  have c84713 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_12
  simp only [← c84708, ← c84709, ← c84710] at e_1378_12
  have e_1378_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_13
  simp only [← c84711, ← c84712, k2, ← c84713] at e_1378_13
  have hr := e_1378_13
  simp only [e_1378_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12912)) := by
  have c84714 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84715 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84716 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84717 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84718 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84719 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_14
  simp only [← c84714, ← c84715, ← c84716] at e_1378_14
  have e_1378_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_15
  simp only [← c84717, ← c84718, k2, ← c84719] at e_1378_15
  have hr := e_1378_15
  simp only [e_1378_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12913)) := by
  have c84720 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84721 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84722 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84723 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84724 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84725 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_16
  simp only [← c84720, ← c84721, ← c84722] at e_1378_16
  have e_1378_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_17
  simp only [← c84723, ← c84724, k2, ← c84725] at e_1378_17
  have hr := e_1378_17
  simp only [e_1378_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1378 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12914)) := by
  have c84726 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84727 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84728 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84729 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84730 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84731 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1378_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_18
  simp only [← c84726, ← c84727, ← c84728] at e_1378_18
  have e_1378_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1378 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_19
  simp only [← c84729, ← c84730, k2, ← c84731] at e_1378_19
  have hr := e_1378_19
  simp only [e_1378_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12915)) := by
  have c84732 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84733 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84734 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84735 := (publicBatchWrapper89_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84736 := (publicBatchWrapper89_copies2648 a h).1
  have c84737 := (publicBatchWrapper89_copies2648 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_0
  simp only [← c84732, ← c84733, ← c84734] at e_1379_0
  have e_1379_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_1
  simp only [← c84735, ← c84736, k2, ← c84737] at e_1379_1
  have hr := e_1379_1
  simp only [e_1379_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12916)) := by
  have c84738 := (publicBatchWrapper89_copies2648 a h).2.2.1
  have c84739 := (publicBatchWrapper89_copies2648 a h).2.2.2.1
  have c84740 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.1
  have c84741 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.1
  have c84742 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.1
  have c84743 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_2
  simp only [← c84738, ← c84739, ← c84740] at e_1379_2
  have e_1379_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_3
  simp only [← c84741, ← c84742, k2, ← c84743] at e_1379_3
  have hr := e_1379_3
  simp only [e_1379_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12917)) := by
  have c84744 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.1
  have c84745 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.1
  have c84746 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84747 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84748 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84749 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_4
  simp only [← c84744, ← c84745, ← c84746] at e_1379_4
  have e_1379_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_5
  simp only [← c84747, ← c84748, k2, ← c84749] at e_1379_5
  have hr := e_1379_5
  simp only [e_1379_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12918)) := by
  have c84750 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84751 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84752 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84753 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84754 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84755 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_6
  simp only [← c84750, ← c84751, ← c84752] at e_1379_6
  have e_1379_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_7
  simp only [← c84753, ← c84754, k2, ← c84755] at e_1379_7
  have hr := e_1379_7
  simp only [e_1379_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12919)) := by
  have c84756 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84757 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84758 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84759 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84760 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84761 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_8
  simp only [← c84756, ← c84757, ← c84758] at e_1379_8
  have e_1379_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_9
  simp only [← c84759, ← c84760, k2, ← c84761] at e_1379_9
  have hr := e_1379_9
  simp only [e_1379_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12920)) := by
  have c84762 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84763 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84764 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84765 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84766 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84767 := (publicBatchWrapper89_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_10
  simp only [← c84762, ← c84763, ← c84764] at e_1379_10
  have e_1379_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_11
  simp only [← c84765, ← c84766, k2, ← c84767] at e_1379_11
  have hr := e_1379_11
  simp only [e_1379_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12921)) := by
  have c84768 := (publicBatchWrapper89_copies2649 a h).1
  have c84769 := (publicBatchWrapper89_copies2649 a h).2.1
  have c84770 := (publicBatchWrapper89_copies2649 a h).2.2.1
  have c84771 := (publicBatchWrapper89_copies2649 a h).2.2.2.1
  have c84772 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.1
  have c84773 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_12
  simp only [← c84768, ← c84769, ← c84770] at e_1379_12
  have e_1379_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_13
  simp only [← c84771, ← c84772, k2, ← c84773] at e_1379_13
  have hr := e_1379_13
  simp only [e_1379_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12922)) := by
  have c84774 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.1
  have c84775 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.1
  have c84776 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.1
  have c84777 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.1
  have c84778 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84779 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_14
  simp only [← c84774, ← c84775, ← c84776] at e_1379_14
  have e_1379_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_15
  simp only [← c84777, ← c84778, k2, ← c84779] at e_1379_15
  have hr := e_1379_15
  simp only [e_1379_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12923)) := by
  have c84780 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84781 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84782 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84783 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84784 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84785 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_16
  simp only [← c84780, ← c84781, ← c84782] at e_1379_16
  have e_1379_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_17
  simp only [← c84783, ← c84784, k2, ← c84785] at e_1379_17
  have hr := e_1379_17
  simp only [e_1379_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1379 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12924)) := by
  have c84786 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84787 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84788 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84789 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84790 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84791 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1379_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_18
  simp only [← c84786, ← c84787, ← c84788] at e_1379_18
  have e_1379_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1379 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_19
  simp only [← c84789, ← c84790, k2, ← c84791] at e_1379_19
  have hr := e_1379_19
  simp only [e_1379_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12925)) := by
  have c84792 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84793 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84794 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84795 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84796 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84797 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_0
  simp only [← c84792, ← c84793, ← c84794] at e_1380_0
  have e_1380_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_1
  simp only [← c84795, ← c84796, k2, ← c84797] at e_1380_1
  have hr := e_1380_1
  simp only [e_1380_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12926)) := by
  have c84798 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84799 := (publicBatchWrapper89_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84800 := (publicBatchWrapper89_copies2650 a h).1
  have c84801 := (publicBatchWrapper89_copies2650 a h).2.1
  have c84802 := (publicBatchWrapper89_copies2650 a h).2.2.1
  have c84803 := (publicBatchWrapper89_copies2650 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_2
  simp only [← c84798, ← c84799, ← c84800] at e_1380_2
  have e_1380_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_3
  simp only [← c84801, ← c84802, k2, ← c84803] at e_1380_3
  have hr := e_1380_3
  simp only [e_1380_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12927)) := by
  have c84804 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.1
  have c84805 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.1
  have c84806 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.1
  have c84807 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.1
  have c84808 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.1
  have c84809 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_4
  simp only [← c84804, ← c84805, ← c84806] at e_1380_4
  have e_1380_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_5
  simp only [← c84807, ← c84808, k2, ← c84809] at e_1380_5
  have hr := e_1380_5
  simp only [e_1380_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12928)) := by
  have c84810 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84811 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84812 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84813 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84814 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84815 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_6
  simp only [← c84810, ← c84811, ← c84812] at e_1380_6
  have e_1380_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_7
  simp only [← c84813, ← c84814, k2, ← c84815] at e_1380_7
  have hr := e_1380_7
  simp only [e_1380_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12929)) := by
  have c84816 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84817 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84818 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84819 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84820 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84821 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_8
  simp only [← c84816, ← c84817, ← c84818] at e_1380_8
  have e_1380_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_9
  simp only [← c84819, ← c84820, k2, ← c84821] at e_1380_9
  have hr := e_1380_9
  simp only [e_1380_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12930)) := by
  have c84822 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84823 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84824 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84825 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84826 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84827 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_10
  simp only [← c84822, ← c84823, ← c84824] at e_1380_10
  have e_1380_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_11
  simp only [← c84825, ← c84826, k2, ← c84827] at e_1380_11
  have hr := e_1380_11
  simp only [e_1380_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12931)) := by
  have c84828 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84829 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84830 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84831 := (publicBatchWrapper89_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84832 := (publicBatchWrapper89_copies2651 a h).1
  have c84833 := (publicBatchWrapper89_copies2651 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_12
  simp only [← c84828, ← c84829, ← c84830] at e_1380_12
  have e_1380_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_13
  simp only [← c84831, ← c84832, k2, ← c84833] at e_1380_13
  have hr := e_1380_13
  simp only [e_1380_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12932)) := by
  have c84834 := (publicBatchWrapper89_copies2651 a h).2.2.1
  have c84835 := (publicBatchWrapper89_copies2651 a h).2.2.2.1
  have c84836 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.1
  have c84837 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.1
  have c84838 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.1
  have c84839 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_14
  simp only [← c84834, ← c84835, ← c84836] at e_1380_14
  have e_1380_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_15
  simp only [← c84837, ← c84838, k2, ← c84839] at e_1380_15
  have hr := e_1380_15
  simp only [e_1380_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12933)) := by
  have c84840 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.1
  have c84841 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.1
  have c84842 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84843 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84844 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84845 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_16
  simp only [← c84840, ← c84841, ← c84842] at e_1380_16
  have e_1380_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_17
  simp only [← c84843, ← c84844, k2, ← c84845] at e_1380_17
  have hr := e_1380_17
  simp only [e_1380_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1380 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12934)) := by
  have c84846 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84847 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84848 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84849 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84850 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84851 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1380_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_18
  simp only [← c84846, ← c84847, ← c84848] at e_1380_18
  have e_1380_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1380 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_19
  simp only [← c84849, ← c84850, k2, ← c84851] at e_1380_19
  have hr := e_1380_19
  simp only [e_1380_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12935)) := by
  have c84852 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84853 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84854 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84855 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84856 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84857 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_0
  simp only [← c84852, ← c84853, ← c84854] at e_1381_0
  have e_1381_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_1
  simp only [← c84855, ← c84856, k2, ← c84857] at e_1381_1
  have hr := e_1381_1
  simp only [e_1381_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12936)) := by
  have c84858 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84859 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84860 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84861 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84862 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84863 := (publicBatchWrapper89_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_2
  simp only [← c84858, ← c84859, ← c84860] at e_1381_2
  have e_1381_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_3
  simp only [← c84861, ← c84862, k2, ← c84863] at e_1381_3
  have hr := e_1381_3
  simp only [e_1381_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12937)) := by
  have c84864 := (publicBatchWrapper89_copies2652 a h).1
  have c84865 := (publicBatchWrapper89_copies2652 a h).2.1
  have c84866 := (publicBatchWrapper89_copies2652 a h).2.2.1
  have c84867 := (publicBatchWrapper89_copies2652 a h).2.2.2.1
  have c84868 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.1
  have c84869 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_4
  simp only [← c84864, ← c84865, ← c84866] at e_1381_4
  have e_1381_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_5
  simp only [← c84867, ← c84868, k2, ← c84869] at e_1381_5
  have hr := e_1381_5
  simp only [e_1381_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12938)) := by
  have c84870 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.1
  have c84871 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.1
  have c84872 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.1
  have c84873 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.1
  have c84874 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84875 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_6
  simp only [← c84870, ← c84871, ← c84872] at e_1381_6
  have e_1381_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_7
  simp only [← c84873, ← c84874, k2, ← c84875] at e_1381_7
  have hr := e_1381_7
  simp only [e_1381_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12939)) := by
  have c84876 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84877 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84878 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84879 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84880 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84881 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_8
  simp only [← c84876, ← c84877, ← c84878] at e_1381_8
  have e_1381_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_9
  simp only [← c84879, ← c84880, k2, ← c84881] at e_1381_9
  have hr := e_1381_9
  simp only [e_1381_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12940)) := by
  have c84882 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84883 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84884 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84885 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84886 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84887 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_10
  simp only [← c84882, ← c84883, ← c84884] at e_1381_10
  have e_1381_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_11
  simp only [← c84885, ← c84886, k2, ← c84887] at e_1381_11
  have hr := e_1381_11
  simp only [e_1381_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12941)) := by
  have c84888 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84889 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84890 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84891 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84892 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84893 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_12
  simp only [← c84888, ← c84889, ← c84890] at e_1381_12
  have e_1381_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_13
  simp only [← c84891, ← c84892, k2, ← c84893] at e_1381_13
  have hr := e_1381_13
  simp only [e_1381_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12942)) := by
  have c84894 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84895 := (publicBatchWrapper89_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84896 := (publicBatchWrapper89_copies2653 a h).1
  have c84897 := (publicBatchWrapper89_copies2653 a h).2.1
  have c84898 := (publicBatchWrapper89_copies2653 a h).2.2.1
  have c84899 := (publicBatchWrapper89_copies2653 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_14
  simp only [← c84894, ← c84895, ← c84896] at e_1381_14
  have e_1381_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_15
  simp only [← c84897, ← c84898, k2, ← c84899] at e_1381_15
  have hr := e_1381_15
  simp only [e_1381_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12943)) := by
  have c84900 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.1
  have c84901 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.1
  have c84902 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.1
  have c84903 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.1
  have c84904 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.1
  have c84905 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_16
  simp only [← c84900, ← c84901, ← c84902] at e_1381_16
  have e_1381_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_17
  simp only [← c84903, ← c84904, k2, ← c84905] at e_1381_17
  have hr := e_1381_17
  simp only [e_1381_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1381 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12944)) := by
  have c84906 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84907 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84908 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84909 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84910 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84911 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1381_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_18
  simp only [← c84906, ← c84907, ← c84908] at e_1381_18
  have e_1381_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1381 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_19
  simp only [← c84909, ← c84910, k2, ← c84911] at e_1381_19
  have hr := e_1381_19
  simp only [e_1381_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12945)) := by
  have c84912 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84913 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84914 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84915 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84916 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84917 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_0
  simp only [← c84912, ← c84913, ← c84914] at e_1382_0
  have e_1382_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_1
  simp only [← c84915, ← c84916, k2, ← c84917] at e_1382_1
  have hr := e_1382_1
  simp only [e_1382_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12946)) := by
  have c84918 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84919 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84920 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84921 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84922 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84923 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_2
  simp only [← c84918, ← c84919, ← c84920] at e_1382_2
  have e_1382_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_3
  simp only [← c84921, ← c84922, k2, ← c84923] at e_1382_3
  have hr := e_1382_3
  simp only [e_1382_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12947)) := by
  have c84924 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84925 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84926 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84927 := (publicBatchWrapper89_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84928 := (publicBatchWrapper89_copies2654 a h).1
  have c84929 := (publicBatchWrapper89_copies2654 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_4
  simp only [← c84924, ← c84925, ← c84926] at e_1382_4
  have e_1382_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_5
  simp only [← c84927, ← c84928, k2, ← c84929] at e_1382_5
  have hr := e_1382_5
  simp only [e_1382_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12948)) := by
  have c84930 := (publicBatchWrapper89_copies2654 a h).2.2.1
  have c84931 := (publicBatchWrapper89_copies2654 a h).2.2.2.1
  have c84932 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.1
  have c84933 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.1
  have c84934 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.1
  have c84935 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_6
  simp only [← c84930, ← c84931, ← c84932] at e_1382_6
  have e_1382_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_7
  simp only [← c84933, ← c84934, k2, ← c84935] at e_1382_7
  have hr := e_1382_7
  simp only [e_1382_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12949)) := by
  have c84936 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.1
  have c84937 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.1
  have c84938 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84939 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84940 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84941 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_8
  simp only [← c84936, ← c84937, ← c84938] at e_1382_8
  have e_1382_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_9
  simp only [← c84939, ← c84940, k2, ← c84941] at e_1382_9
  have hr := e_1382_9
  simp only [e_1382_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12950)) := by
  have c84942 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84943 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84944 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84945 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84946 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84947 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_10
  simp only [← c84942, ← c84943, ← c84944] at e_1382_10
  have e_1382_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_11
  simp only [← c84945, ← c84946, k2, ← c84947] at e_1382_11
  have hr := e_1382_11
  simp only [e_1382_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12951)) := by
  have c84948 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84949 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84950 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84951 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84952 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84953 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_12
  simp only [← c84948, ← c84949, ← c84950] at e_1382_12
  have e_1382_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_13
  simp only [← c84951, ← c84952, k2, ← c84953] at e_1382_13
  have hr := e_1382_13
  simp only [e_1382_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12952)) := by
  have c84954 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84955 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84956 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84957 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84958 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84959 := (publicBatchWrapper89_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_14
  simp only [← c84954, ← c84955, ← c84956] at e_1382_14
  have e_1382_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_15
  simp only [← c84957, ← c84958, k2, ← c84959] at e_1382_15
  have hr := e_1382_15
  simp only [e_1382_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12953)) := by
  have c84960 := (publicBatchWrapper89_copies2655 a h).1
  have c84961 := (publicBatchWrapper89_copies2655 a h).2.1
  have c84962 := (publicBatchWrapper89_copies2655 a h).2.2.1
  have c84963 := (publicBatchWrapper89_copies2655 a h).2.2.2.1
  have c84964 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.1
  have c84965 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_16
  simp only [← c84960, ← c84961, ← c84962] at e_1382_16
  have e_1382_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_17
  simp only [← c84963, ← c84964, k2, ← c84965] at e_1382_17
  have hr := e_1382_17
  simp only [e_1382_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1382 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12954)) := by
  have c84966 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.1
  have c84967 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.1
  have c84968 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.1
  have c84969 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.1
  have c84970 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84971 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1382_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_18
  simp only [← c84966, ← c84967, ← c84968] at e_1382_18
  have e_1382_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1382 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_19
  simp only [← c84969, ← c84970, k2, ← c84971] at e_1382_19
  have hr := e_1382_19
  simp only [e_1382_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12955)) := by
  have c84972 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84973 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84974 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84975 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84976 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84977 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_0
  simp only [← c84972, ← c84973, ← c84974] at e_1383_0
  have e_1383_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_1
  simp only [← c84975, ← c84976, k2, ← c84977] at e_1383_1
  have hr := e_1383_1
  simp only [e_1383_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12956)) := by
  have c84978 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84979 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84980 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84981 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84982 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84983 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_2
  simp only [← c84978, ← c84979, ← c84980] at e_1383_2
  have e_1383_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_3
  simp only [← c84981, ← c84982, k2, ← c84983] at e_1383_3
  have hr := e_1383_3
  simp only [e_1383_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12957)) := by
  have c84984 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84985 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84986 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84987 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84988 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84989 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_4
  simp only [← c84984, ← c84985, ← c84986] at e_1383_4
  have e_1383_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_5
  simp only [← c84987, ← c84988, k2, ← c84989] at e_1383_5
  have hr := e_1383_5
  simp only [e_1383_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12958)) := by
  have c84990 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84991 := (publicBatchWrapper89_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84992 := (publicBatchWrapper89_copies2656 a h).1
  have c84993 := (publicBatchWrapper89_copies2656 a h).2.1
  have c84994 := (publicBatchWrapper89_copies2656 a h).2.2.1
  have c84995 := (publicBatchWrapper89_copies2656 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_6
  simp only [← c84990, ← c84991, ← c84992] at e_1383_6
  have e_1383_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_7
  simp only [← c84993, ← c84994, k2, ← c84995] at e_1383_7
  have hr := e_1383_7
  simp only [e_1383_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12959)) := by
  have c84996 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.1
  have c84997 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.1
  have c84998 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.1
  have c84999 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.1
  have c85000 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.1
  have c85001 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_8
  simp only [← c84996, ← c84997, ← c84998] at e_1383_8
  have e_1383_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_9
  simp only [← c84999, ← c85000, k2, ← c85001] at e_1383_9
  have hr := e_1383_9
  simp only [e_1383_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12960)) := by
  have c85002 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85003 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85004 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85005 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85006 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85007 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_10
  simp only [← c85002, ← c85003, ← c85004] at e_1383_10
  have e_1383_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_11
  simp only [← c85005, ← c85006, k2, ← c85007] at e_1383_11
  have hr := e_1383_11
  simp only [e_1383_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12961)) := by
  have c85008 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85009 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85010 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85011 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85012 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85013 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_12
  simp only [← c85008, ← c85009, ← c85010] at e_1383_12
  have e_1383_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_13
  simp only [← c85011, ← c85012, k2, ← c85013] at e_1383_13
  have hr := e_1383_13
  simp only [e_1383_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12962)) := by
  have c85014 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85015 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85016 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85017 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85018 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85019 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_14
  simp only [← c85014, ← c85015, ← c85016] at e_1383_14
  have e_1383_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_15
  simp only [← c85017, ← c85018, k2, ← c85019] at e_1383_15
  have hr := e_1383_15
  simp only [e_1383_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12963)) := by
  have c85020 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85021 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85022 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85023 := (publicBatchWrapper89_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85024 := (publicBatchWrapper89_copies2657 a h).1
  have c85025 := (publicBatchWrapper89_copies2657 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_16
  simp only [← c85020, ← c85021, ← c85022] at e_1383_16
  have e_1383_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_17
  simp only [← c85023, ← c85024, k2, ← c85025] at e_1383_17
  have hr := e_1383_17
  simp only [e_1383_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1383 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12964)) := by
  have c85026 := (publicBatchWrapper89_copies2657 a h).2.2.1
  have c85027 := (publicBatchWrapper89_copies2657 a h).2.2.2.1
  have c85028 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.1
  have c85029 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.1
  have c85030 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.1
  have c85031 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1383_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_18
  simp only [← c85026, ← c85027, ← c85028] at e_1383_18
  have e_1383_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1383 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_19
  simp only [← c85029, ← c85030, k2, ← c85031] at e_1383_19
  have hr := e_1383_19
  simp only [e_1383_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12965)) := by
  have c85032 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.1
  have c85033 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.1
  have c85034 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85035 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85036 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85037 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_0
  simp only [← c85032, ← c85033, ← c85034] at e_1384_0
  have e_1384_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_1
  simp only [← c85035, ← c85036, k2, ← c85037] at e_1384_1
  have hr := e_1384_1
  simp only [e_1384_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12966)) := by
  have c85038 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85039 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85040 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85041 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85042 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85043 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_2
  simp only [← c85038, ← c85039, ← c85040] at e_1384_2
  have e_1384_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_3
  simp only [← c85041, ← c85042, k2, ← c85043] at e_1384_3
  have hr := e_1384_3
  simp only [e_1384_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12967)) := by
  have c85044 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85045 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85046 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85047 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85048 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85049 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_4
  simp only [← c85044, ← c85045, ← c85046] at e_1384_4
  have e_1384_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_5
  simp only [← c85047, ← c85048, k2, ← c85049] at e_1384_5
  have hr := e_1384_5
  simp only [e_1384_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12968)) := by
  have c85050 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85051 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85052 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85053 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85054 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85055 := (publicBatchWrapper89_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_6
  simp only [← c85050, ← c85051, ← c85052] at e_1384_6
  have e_1384_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_7
  simp only [← c85053, ← c85054, k2, ← c85055] at e_1384_7
  have hr := e_1384_7
  simp only [e_1384_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12969)) := by
  have c85056 := (publicBatchWrapper89_copies2658 a h).1
  have c85057 := (publicBatchWrapper89_copies2658 a h).2.1
  have c85058 := (publicBatchWrapper89_copies2658 a h).2.2.1
  have c85059 := (publicBatchWrapper89_copies2658 a h).2.2.2.1
  have c85060 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.1
  have c85061 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_8
  simp only [← c85056, ← c85057, ← c85058] at e_1384_8
  have e_1384_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_9
  simp only [← c85059, ← c85060, k2, ← c85061] at e_1384_9
  have hr := e_1384_9
  simp only [e_1384_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13010)) := by
  have c85062 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.1
  have c85063 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.1
  have c85064 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.1
  have c85065 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.1
  have c85066 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85067 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_10
  simp only [← c85062, ← c85063, ← c85064] at e_1384_10
  have e_1384_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_11
  simp only [← c85065, ← c85066, k2, ← c85067] at e_1384_11
  have hr := e_1384_11
  simp only [e_1384_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13011)) := by
  have c85068 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85069 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85070 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85071 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85072 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85073 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_12
  simp only [← c85068, ← c85069, ← c85070] at e_1384_12
  have e_1384_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_13
  simp only [← c85071, ← c85072, k2, ← c85073] at e_1384_13
  have hr := e_1384_13
  simp only [e_1384_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13012)) := by
  have c85074 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85075 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85076 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85077 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85078 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85079 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_14
  simp only [← c85074, ← c85075, ← c85076] at e_1384_14
  have e_1384_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_15
  simp only [← c85077, ← c85078, k2, ← c85079] at e_1384_15
  have hr := e_1384_15
  simp only [e_1384_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13013)) := by
  have c85080 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85081 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85082 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85083 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85084 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85085 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_16
  simp only [← c85080, ← c85081, ← c85082] at e_1384_16
  have e_1384_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_17
  simp only [← c85083, ← c85084, k2, ← c85085] at e_1384_17
  have hr := e_1384_17
  simp only [e_1384_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1384 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13014)) := by
  have c85086 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85087 := (publicBatchWrapper89_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85088 := (publicBatchWrapper89_copies2659 a h).1
  have c85089 := (publicBatchWrapper89_copies2659 a h).2.1
  have c85090 := (publicBatchWrapper89_copies2659 a h).2.2.1
  have c85091 := (publicBatchWrapper89_copies2659 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1384_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_18
  simp only [← c85086, ← c85087, ← c85088] at e_1384_18
  have e_1384_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1384 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_19
  simp only [← c85089, ← c85090, k2, ← c85091] at e_1384_19
  have hr := e_1384_19
  simp only [e_1384_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13015)) := by
  have c85092 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.1
  have c85093 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.1
  have c85094 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.1
  have c85095 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.1
  have c85096 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.1
  have c85097 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_0
  simp only [← c85092, ← c85093, ← c85094] at e_1385_0
  have e_1385_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_1
  simp only [← c85095, ← c85096, k2, ← c85097] at e_1385_1
  have hr := e_1385_1
  simp only [e_1385_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13016)) := by
  have c85098 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85099 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85100 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85101 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85102 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85103 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_2
  simp only [← c85098, ← c85099, ← c85100] at e_1385_2
  have e_1385_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_3
  simp only [← c85101, ← c85102, k2, ← c85103] at e_1385_3
  have hr := e_1385_3
  simp only [e_1385_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13017)) := by
  have c85104 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85105 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85106 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85107 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85108 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85109 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_4
  simp only [← c85104, ← c85105, ← c85106] at e_1385_4
  have e_1385_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_5
  simp only [← c85107, ← c85108, k2, ← c85109] at e_1385_5
  have hr := e_1385_5
  simp only [e_1385_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13018)) := by
  have c85110 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85111 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85112 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85113 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85114 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85115 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_6
  simp only [← c85110, ← c85111, ← c85112] at e_1385_6
  have e_1385_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_7
  simp only [← c85113, ← c85114, k2, ← c85115] at e_1385_7
  have hr := e_1385_7
  simp only [e_1385_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13019)) := by
  have c85116 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85117 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85118 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85119 := (publicBatchWrapper89_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85120 := (publicBatchWrapper89_copies2660 a h).1
  have c85121 := (publicBatchWrapper89_copies2660 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_8
  simp only [← c85116, ← c85117, ← c85118] at e_1385_8
  have e_1385_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_9
  simp only [← c85119, ← c85120, k2, ← c85121] at e_1385_9
  have hr := e_1385_9
  simp only [e_1385_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13020)) := by
  have c85122 := (publicBatchWrapper89_copies2660 a h).2.2.1
  have c85123 := (publicBatchWrapper89_copies2660 a h).2.2.2.1
  have c85124 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.1
  have c85125 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.1
  have c85126 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.1
  have c85127 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_10
  simp only [← c85122, ← c85123, ← c85124] at e_1385_10
  have e_1385_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_11
  simp only [← c85125, ← c85126, k2, ← c85127] at e_1385_11
  have hr := e_1385_11
  simp only [e_1385_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13021)) := by
  have c85128 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.1
  have c85129 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.1
  have c85130 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85131 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85132 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85133 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_12
  simp only [← c85128, ← c85129, ← c85130] at e_1385_12
  have e_1385_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_13
  simp only [← c85131, ← c85132, k2, ← c85133] at e_1385_13
  have hr := e_1385_13
  simp only [e_1385_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13022)) := by
  have c85134 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85135 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85136 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85137 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85138 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85139 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_14
  simp only [← c85134, ← c85135, ← c85136] at e_1385_14
  have e_1385_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_15
  simp only [← c85137, ← c85138, k2, ← c85139] at e_1385_15
  have hr := e_1385_15
  simp only [e_1385_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13023)) := by
  have c85140 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85141 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85142 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85143 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85144 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85145 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_16
  simp only [← c85140, ← c85141, ← c85142] at e_1385_16
  have e_1385_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_17
  simp only [← c85143, ← c85144, k2, ← c85145] at e_1385_17
  have hr := e_1385_17
  simp only [e_1385_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1385 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13024)) := by
  have c85146 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85147 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85148 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85149 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85150 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85151 := (publicBatchWrapper89_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1385_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_18
  simp only [← c85146, ← c85147, ← c85148] at e_1385_18
  have e_1385_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1385 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_19
  simp only [← c85149, ← c85150, k2, ← c85151] at e_1385_19
  have hr := e_1385_19
  simp only [e_1385_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13025)) := by
  have c85152 := (publicBatchWrapper89_copies2661 a h).1
  have c85153 := (publicBatchWrapper89_copies2661 a h).2.1
  have c85154 := (publicBatchWrapper89_copies2661 a h).2.2.1
  have c85155 := (publicBatchWrapper89_copies2661 a h).2.2.2.1
  have c85156 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.1
  have c85157 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_0
  simp only [← c85152, ← c85153, ← c85154] at e_1386_0
  have e_1386_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_1
  simp only [← c85155, ← c85156, k2, ← c85157] at e_1386_1
  have hr := e_1386_1
  simp only [e_1386_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13026)) := by
  have c85158 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.1
  have c85159 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.1
  have c85160 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.1
  have c85161 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.1
  have c85162 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85163 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_2
  simp only [← c85158, ← c85159, ← c85160] at e_1386_2
  have e_1386_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_3
  simp only [← c85161, ← c85162, k2, ← c85163] at e_1386_3
  have hr := e_1386_3
  simp only [e_1386_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13027)) := by
  have c85164 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85165 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85166 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85167 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85168 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85169 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_4
  simp only [← c85164, ← c85165, ← c85166] at e_1386_4
  have e_1386_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_5
  simp only [← c85167, ← c85168, k2, ← c85169] at e_1386_5
  have hr := e_1386_5
  simp only [e_1386_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13028)) := by
  have c85170 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85171 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85172 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85173 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85174 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85175 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_6
  simp only [← c85170, ← c85171, ← c85172] at e_1386_6
  have e_1386_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_7
  simp only [← c85173, ← c85174, k2, ← c85175] at e_1386_7
  have hr := e_1386_7
  simp only [e_1386_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13029)) := by
  have c85176 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85177 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85178 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85179 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85180 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85181 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_8
  simp only [← c85176, ← c85177, ← c85178] at e_1386_8
  have e_1386_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_9
  simp only [← c85179, ← c85180, k2, ← c85181] at e_1386_9
  have hr := e_1386_9
  simp only [e_1386_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13030)) := by
  have c85182 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85183 := (publicBatchWrapper89_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85184 := (publicBatchWrapper89_copies2662 a h).1
  have c85185 := (publicBatchWrapper89_copies2662 a h).2.1
  have c85186 := (publicBatchWrapper89_copies2662 a h).2.2.1
  have c85187 := (publicBatchWrapper89_copies2662 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_10
  simp only [← c85182, ← c85183, ← c85184] at e_1386_10
  have e_1386_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_11
  simp only [← c85185, ← c85186, k2, ← c85187] at e_1386_11
  have hr := e_1386_11
  simp only [e_1386_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13031)) := by
  have c85188 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.1
  have c85189 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.1
  have c85190 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.1
  have c85191 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.1
  have c85192 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.1
  have c85193 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_12
  simp only [← c85188, ← c85189, ← c85190] at e_1386_12
  have e_1386_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_13
  simp only [← c85191, ← c85192, k2, ← c85193] at e_1386_13
  have hr := e_1386_13
  simp only [e_1386_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13032)) := by
  have c85194 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85195 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85196 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85197 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85198 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85199 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_14
  simp only [← c85194, ← c85195, ← c85196] at e_1386_14
  have e_1386_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_15
  simp only [← c85197, ← c85198, k2, ← c85199] at e_1386_15
  have hr := e_1386_15
  simp only [e_1386_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13033)) := by
  have c85200 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85201 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85202 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85203 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85204 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85205 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_16
  simp only [← c85200, ← c85201, ← c85202] at e_1386_16
  have e_1386_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_17
  simp only [← c85203, ← c85204, k2, ← c85205] at e_1386_17
  have hr := e_1386_17
  simp only [e_1386_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1386 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13034)) := by
  have c85206 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85207 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85208 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85209 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85210 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85211 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1386_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_18
  simp only [← c85206, ← c85207, ← c85208] at e_1386_18
  have e_1386_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1386 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_19
  simp only [← c85209, ← c85210, k2, ← c85211] at e_1386_19
  have hr := e_1386_19
  simp only [e_1386_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13035)) := by
  have c85212 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85213 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85214 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85215 := (publicBatchWrapper89_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85216 := (publicBatchWrapper89_copies2663 a h).1
  have c85217 := (publicBatchWrapper89_copies2663 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_0
  simp only [← c85212, ← c85213, ← c85214] at e_1387_0
  have e_1387_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_1
  simp only [← c85215, ← c85216, k2, ← c85217] at e_1387_1
  have hr := e_1387_1
  simp only [e_1387_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13036)) := by
  have c85218 := (publicBatchWrapper89_copies2663 a h).2.2.1
  have c85219 := (publicBatchWrapper89_copies2663 a h).2.2.2.1
  have c85220 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.1
  have c85221 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.1
  have c85222 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.1
  have c85223 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_2
  simp only [← c85218, ← c85219, ← c85220] at e_1387_2
  have e_1387_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_3
  simp only [← c85221, ← c85222, k2, ← c85223] at e_1387_3
  have hr := e_1387_3
  simp only [e_1387_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13037)) := by
  have c85224 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.1
  have c85225 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.1
  have c85226 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85227 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85228 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85229 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_4
  simp only [← c85224, ← c85225, ← c85226] at e_1387_4
  have e_1387_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_5
  simp only [← c85227, ← c85228, k2, ← c85229] at e_1387_5
  have hr := e_1387_5
  simp only [e_1387_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13038)) := by
  have c85230 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85231 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85232 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85233 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85234 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85235 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_6
  simp only [← c85230, ← c85231, ← c85232] at e_1387_6
  have e_1387_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_7
  simp only [← c85233, ← c85234, k2, ← c85235] at e_1387_7
  have hr := e_1387_7
  simp only [e_1387_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13039)) := by
  have c85236 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85237 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85238 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85239 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85240 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85241 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_8
  simp only [← c85236, ← c85237, ← c85238] at e_1387_8
  have e_1387_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_9
  simp only [← c85239, ← c85240, k2, ← c85241] at e_1387_9
  have hr := e_1387_9
  simp only [e_1387_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13040)) := by
  have c85242 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85243 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85244 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85245 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85246 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85247 := (publicBatchWrapper89_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_10
  simp only [← c85242, ← c85243, ← c85244] at e_1387_10
  have e_1387_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_11
  simp only [← c85245, ← c85246, k2, ← c85247] at e_1387_11
  have hr := e_1387_11
  simp only [e_1387_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13041)) := by
  have c85248 := (publicBatchWrapper89_copies2664 a h).1
  have c85249 := (publicBatchWrapper89_copies2664 a h).2.1
  have c85250 := (publicBatchWrapper89_copies2664 a h).2.2.1
  have c85251 := (publicBatchWrapper89_copies2664 a h).2.2.2.1
  have c85252 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.1
  have c85253 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_12
  simp only [← c85248, ← c85249, ← c85250] at e_1387_12
  have e_1387_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_13
  simp only [← c85251, ← c85252, k2, ← c85253] at e_1387_13
  have hr := e_1387_13
  simp only [e_1387_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13042)) := by
  have c85254 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.1
  have c85255 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.1
  have c85256 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.1
  have c85257 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.1
  have c85258 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85259 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_14
  simp only [← c85254, ← c85255, ← c85256] at e_1387_14
  have e_1387_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_15
  simp only [← c85257, ← c85258, k2, ← c85259] at e_1387_15
  have hr := e_1387_15
  simp only [e_1387_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13043)) := by
  have c85260 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85261 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85262 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85263 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85264 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85265 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_16
  simp only [← c85260, ← c85261, ← c85262] at e_1387_16
  have e_1387_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_17
  simp only [← c85263, ← c85264, k2, ← c85265] at e_1387_17
  have hr := e_1387_17
  simp only [e_1387_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1387 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13044)) := by
  have c85266 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85267 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85268 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85269 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85270 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85271 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1387_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_18
  simp only [← c85266, ← c85267, ← c85268] at e_1387_18
  have e_1387_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1387 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_19
  simp only [← c85269, ← c85270, k2, ← c85271] at e_1387_19
  have hr := e_1387_19
  simp only [e_1387_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13045)) := by
  have c85272 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85273 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85274 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85275 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85276 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85277 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_0
  simp only [← c85272, ← c85273, ← c85274] at e_1388_0
  have e_1388_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_1
  simp only [← c85275, ← c85276, k2, ← c85277] at e_1388_1
  have hr := e_1388_1
  simp only [e_1388_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13046)) := by
  have c85278 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85279 := (publicBatchWrapper89_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85280 := (publicBatchWrapper89_copies2665 a h).1
  have c85281 := (publicBatchWrapper89_copies2665 a h).2.1
  have c85282 := (publicBatchWrapper89_copies2665 a h).2.2.1
  have c85283 := (publicBatchWrapper89_copies2665 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_2
  simp only [← c85278, ← c85279, ← c85280] at e_1388_2
  have e_1388_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_3
  simp only [← c85281, ← c85282, k2, ← c85283] at e_1388_3
  have hr := e_1388_3
  simp only [e_1388_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13047)) := by
  have c85284 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.1
  have c85285 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.1
  have c85286 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.1
  have c85287 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.1
  have c85288 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.1
  have c85289 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_4
  simp only [← c85284, ← c85285, ← c85286] at e_1388_4
  have e_1388_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_5
  simp only [← c85287, ← c85288, k2, ← c85289] at e_1388_5
  have hr := e_1388_5
  simp only [e_1388_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13048)) := by
  have c85290 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85291 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85292 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85293 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85294 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85295 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_6
  simp only [← c85290, ← c85291, ← c85292] at e_1388_6
  have e_1388_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_7
  simp only [← c85293, ← c85294, k2, ← c85295] at e_1388_7
  have hr := e_1388_7
  simp only [e_1388_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13049)) := by
  have c85296 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85297 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85298 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85299 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85300 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85301 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_8
  simp only [← c85296, ← c85297, ← c85298] at e_1388_8
  have e_1388_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_9
  simp only [← c85299, ← c85300, k2, ← c85301] at e_1388_9
  have hr := e_1388_9
  simp only [e_1388_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13050)) := by
  have c85302 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85303 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85304 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85305 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85306 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85307 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_10
  simp only [← c85302, ← c85303, ← c85304] at e_1388_10
  have e_1388_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_11
  simp only [← c85305, ← c85306, k2, ← c85307] at e_1388_11
  have hr := e_1388_11
  simp only [e_1388_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13051)) := by
  have c85308 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85309 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85310 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85311 := (publicBatchWrapper89_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85312 := (publicBatchWrapper89_copies2666 a h).1
  have c85313 := (publicBatchWrapper89_copies2666 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_12
  simp only [← c85308, ← c85309, ← c85310] at e_1388_12
  have e_1388_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_13
  simp only [← c85311, ← c85312, k2, ← c85313] at e_1388_13
  have hr := e_1388_13
  simp only [e_1388_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13052)) := by
  have c85314 := (publicBatchWrapper89_copies2666 a h).2.2.1
  have c85315 := (publicBatchWrapper89_copies2666 a h).2.2.2.1
  have c85316 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.1
  have c85317 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.1
  have c85318 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.1
  have c85319 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_14
  simp only [← c85314, ← c85315, ← c85316] at e_1388_14
  have e_1388_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_15
  simp only [← c85317, ← c85318, k2, ← c85319] at e_1388_15
  have hr := e_1388_15
  simp only [e_1388_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13053)) := by
  have c85320 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.1
  have c85321 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.1
  have c85322 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85323 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85324 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85325 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_16
  simp only [← c85320, ← c85321, ← c85322] at e_1388_16
  have e_1388_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_17
  simp only [← c85323, ← c85324, k2, ← c85325] at e_1388_17
  have hr := e_1388_17
  simp only [e_1388_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1388 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13054)) := by
  have c85326 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85327 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85328 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85329 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85330 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85331 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1388_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_18
  simp only [← c85326, ← c85327, ← c85328] at e_1388_18
  have e_1388_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1388 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_19
  simp only [← c85329, ← c85330, k2, ← c85331] at e_1388_19
  have hr := e_1388_19
  simp only [e_1388_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13055)) := by
  have c85332 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85333 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85334 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85335 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85336 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85337 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_0
  simp only [← c85332, ← c85333, ← c85334] at e_1389_0
  have e_1389_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_1
  simp only [← c85335, ← c85336, k2, ← c85337] at e_1389_1
  have hr := e_1389_1
  simp only [e_1389_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13056)) := by
  have c85338 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85339 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85340 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85341 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85342 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85343 := (publicBatchWrapper89_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_2
  simp only [← c85338, ← c85339, ← c85340] at e_1389_2
  have e_1389_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_3
  simp only [← c85341, ← c85342, k2, ← c85343] at e_1389_3
  have hr := e_1389_3
  simp only [e_1389_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13057)) := by
  have c85344 := (publicBatchWrapper89_copies2667 a h).1
  have c85345 := (publicBatchWrapper89_copies2667 a h).2.1
  have c85346 := (publicBatchWrapper89_copies2667 a h).2.2.1
  have c85347 := (publicBatchWrapper89_copies2667 a h).2.2.2.1
  have c85348 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.1
  have c85349 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_4
  simp only [← c85344, ← c85345, ← c85346] at e_1389_4
  have e_1389_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_5
  simp only [← c85347, ← c85348, k2, ← c85349] at e_1389_5
  have hr := e_1389_5
  simp only [e_1389_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13058)) := by
  have c85350 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.1
  have c85351 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.1
  have c85352 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.1
  have c85353 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.1
  have c85354 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85355 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_6
  simp only [← c85350, ← c85351, ← c85352] at e_1389_6
  have e_1389_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_7
  simp only [← c85353, ← c85354, k2, ← c85355] at e_1389_7
  have hr := e_1389_7
  simp only [e_1389_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13059)) := by
  have c85356 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85357 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85358 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85359 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85360 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85361 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_8
  simp only [← c85356, ← c85357, ← c85358] at e_1389_8
  have e_1389_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_9
  simp only [← c85359, ← c85360, k2, ← c85361] at e_1389_9
  have hr := e_1389_9
  simp only [e_1389_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13060)) := by
  have c85362 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85363 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85364 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85365 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85366 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85367 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_10
  simp only [← c85362, ← c85363, ← c85364] at e_1389_10
  have e_1389_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_11
  simp only [← c85365, ← c85366, k2, ← c85367] at e_1389_11
  have hr := e_1389_11
  simp only [e_1389_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13061)) := by
  have c85368 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85369 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85370 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85371 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85372 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85373 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_12
  simp only [← c85368, ← c85369, ← c85370] at e_1389_12
  have e_1389_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_13
  simp only [← c85371, ← c85372, k2, ← c85373] at e_1389_13
  have hr := e_1389_13
  simp only [e_1389_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13062)) := by
  have c85374 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85375 := (publicBatchWrapper89_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85376 := (publicBatchWrapper89_copies2668 a h).1
  have c85377 := (publicBatchWrapper89_copies2668 a h).2.1
  have c85378 := (publicBatchWrapper89_copies2668 a h).2.2.1
  have c85379 := (publicBatchWrapper89_copies2668 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_14
  simp only [← c85374, ← c85375, ← c85376] at e_1389_14
  have e_1389_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_15
  simp only [← c85377, ← c85378, k2, ← c85379] at e_1389_15
  have hr := e_1389_15
  simp only [e_1389_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13063)) := by
  have c85380 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.1
  have c85381 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.1
  have c85382 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.1
  have c85383 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.1
  have c85384 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.1
  have c85385 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_16
  simp only [← c85380, ← c85381, ← c85382] at e_1389_16
  have e_1389_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_17
  simp only [← c85383, ← c85384, k2, ← c85385] at e_1389_17
  have hr := e_1389_17
  simp only [e_1389_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1389 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13064)) := by
  have c85386 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85387 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85388 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85389 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85390 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85391 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1389_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_18
  simp only [← c85386, ← c85387, ← c85388] at e_1389_18
  have e_1389_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1389 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_19
  simp only [← c85389, ← c85390, k2, ← c85391] at e_1389_19
  have hr := e_1389_19
  simp only [e_1389_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13065)) := by
  have c85392 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85393 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85394 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85395 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85396 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85397 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_0
  simp only [← c85392, ← c85393, ← c85394] at e_1390_0
  have e_1390_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_1
  simp only [← c85395, ← c85396, k2, ← c85397] at e_1390_1
  have hr := e_1390_1
  simp only [e_1390_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13066)) := by
  have c85398 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85399 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85400 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85401 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85402 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85403 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_2
  simp only [← c85398, ← c85399, ← c85400] at e_1390_2
  have e_1390_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_3
  simp only [← c85401, ← c85402, k2, ← c85403] at e_1390_3
  have hr := e_1390_3
  simp only [e_1390_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13067)) := by
  have c85404 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85405 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85406 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85407 := (publicBatchWrapper89_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85408 := (publicBatchWrapper89_copies2669 a h).1
  have c85409 := (publicBatchWrapper89_copies2669 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_4
  simp only [← c85404, ← c85405, ← c85406] at e_1390_4
  have e_1390_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_5
  simp only [← c85407, ← c85408, k2, ← c85409] at e_1390_5
  have hr := e_1390_5
  simp only [e_1390_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13068)) := by
  have c85410 := (publicBatchWrapper89_copies2669 a h).2.2.1
  have c85411 := (publicBatchWrapper89_copies2669 a h).2.2.2.1
  have c85412 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.1
  have c85413 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.1
  have c85414 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.1
  have c85415 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_6
  simp only [← c85410, ← c85411, ← c85412] at e_1390_6
  have e_1390_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_7
  simp only [← c85413, ← c85414, k2, ← c85415] at e_1390_7
  have hr := e_1390_7
  simp only [e_1390_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13069)) := by
  have c85416 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.1
  have c85417 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.1
  have c85418 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85419 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85420 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85421 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_8
  simp only [← c85416, ← c85417, ← c85418] at e_1390_8
  have e_1390_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_9
  simp only [← c85419, ← c85420, k2, ← c85421] at e_1390_9
  have hr := e_1390_9
  simp only [e_1390_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13070)) := by
  have c85422 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85423 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85424 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85425 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85426 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85427 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_10
  simp only [← c85422, ← c85423, ← c85424] at e_1390_10
  have e_1390_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_11
  simp only [← c85425, ← c85426, k2, ← c85427] at e_1390_11
  have hr := e_1390_11
  simp only [e_1390_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13071)) := by
  have c85428 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85429 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85430 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85431 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85432 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85433 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_12
  simp only [← c85428, ← c85429, ← c85430] at e_1390_12
  have e_1390_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_13
  simp only [← c85431, ← c85432, k2, ← c85433] at e_1390_13
  have hr := e_1390_13
  simp only [e_1390_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13072)) := by
  have c85434 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85435 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85436 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85437 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85438 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85439 := (publicBatchWrapper89_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_14
  simp only [← c85434, ← c85435, ← c85436] at e_1390_14
  have e_1390_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_15
  simp only [← c85437, ← c85438, k2, ← c85439] at e_1390_15
  have hr := e_1390_15
  simp only [e_1390_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13073)) := by
  have c85440 := (publicBatchWrapper89_copies2670 a h).1
  have c85441 := (publicBatchWrapper89_copies2670 a h).2.1
  have c85442 := (publicBatchWrapper89_copies2670 a h).2.2.1
  have c85443 := (publicBatchWrapper89_copies2670 a h).2.2.2.1
  have c85444 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.1
  have c85445 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_16
  simp only [← c85440, ← c85441, ← c85442] at e_1390_16
  have e_1390_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_17
  simp only [← c85443, ← c85444, k2, ← c85445] at e_1390_17
  have hr := e_1390_17
  simp only [e_1390_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1390 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13074)) := by
  have c85446 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.1
  have c85447 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.1
  have c85448 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.1
  have c85449 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.1
  have c85450 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85451 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1390_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_18
  simp only [← c85446, ← c85447, ← c85448] at e_1390_18
  have e_1390_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1390 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_19
  simp only [← c85449, ← c85450, k2, ← c85451] at e_1390_19
  have hr := e_1390_19
  simp only [e_1390_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13075)) := by
  have c85452 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85453 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85454 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85455 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85456 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85457 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_0
  simp only [← c85452, ← c85453, ← c85454] at e_1391_0
  have e_1391_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_1
  simp only [← c85455, ← c85456, k2, ← c85457] at e_1391_1
  have hr := e_1391_1
  simp only [e_1391_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13076)) := by
  have c85458 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85459 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85460 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85461 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85462 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85463 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_2
  simp only [← c85458, ← c85459, ← c85460] at e_1391_2
  have e_1391_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_3
  simp only [← c85461, ← c85462, k2, ← c85463] at e_1391_3
  have hr := e_1391_3
  simp only [e_1391_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13077)) := by
  have c85464 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85465 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85466 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85467 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85468 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85469 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_4
  simp only [← c85464, ← c85465, ← c85466] at e_1391_4
  have e_1391_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_5
  simp only [← c85467, ← c85468, k2, ← c85469] at e_1391_5
  have hr := e_1391_5
  simp only [e_1391_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13078)) := by
  have c85470 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85471 := (publicBatchWrapper89_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85472 := (publicBatchWrapper89_copies2671 a h).1
  have c85473 := (publicBatchWrapper89_copies2671 a h).2.1
  have c85474 := (publicBatchWrapper89_copies2671 a h).2.2.1
  have c85475 := (publicBatchWrapper89_copies2671 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_6
  simp only [← c85470, ← c85471, ← c85472] at e_1391_6
  have e_1391_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_7
  simp only [← c85473, ← c85474, k2, ← c85475] at e_1391_7
  have hr := e_1391_7
  simp only [e_1391_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13079)) := by
  have c85476 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.1
  have c85477 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.1
  have c85478 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.1
  have c85479 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.1
  have c85480 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.1
  have c85481 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_8
  simp only [← c85476, ← c85477, ← c85478] at e_1391_8
  have e_1391_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_9
  simp only [← c85479, ← c85480, k2, ← c85481] at e_1391_9
  have hr := e_1391_9
  simp only [e_1391_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13080)) := by
  have c85482 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85483 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85484 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85485 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85486 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85487 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_10
  simp only [← c85482, ← c85483, ← c85484] at e_1391_10
  have e_1391_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_11
  simp only [← c85485, ← c85486, k2, ← c85487] at e_1391_11
  have hr := e_1391_11
  simp only [e_1391_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13081)) := by
  have c85488 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85489 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85490 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85491 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85492 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85493 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_12
  simp only [← c85488, ← c85489, ← c85490] at e_1391_12
  have e_1391_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_13
  simp only [← c85491, ← c85492, k2, ← c85493] at e_1391_13
  have hr := e_1391_13
  simp only [e_1391_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13082)) := by
  have c85494 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85495 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85496 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85497 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85498 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85499 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_14
  simp only [← c85494, ← c85495, ← c85496] at e_1391_14
  have e_1391_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_15
  simp only [← c85497, ← c85498, k2, ← c85499] at e_1391_15
  have hr := e_1391_15
  simp only [e_1391_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13083)) := by
  have c85500 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85501 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85502 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85503 := (publicBatchWrapper89_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85504 := (publicBatchWrapper89_copies2672 a h).1
  have c85505 := (publicBatchWrapper89_copies2672 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_16
  simp only [← c85500, ← c85501, ← c85502] at e_1391_16
  have e_1391_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_17
  simp only [← c85503, ← c85504, k2, ← c85505] at e_1391_17
  have hr := e_1391_17
  simp only [e_1391_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1391 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13084)) := by
  have c85506 := (publicBatchWrapper89_copies2672 a h).2.2.1
  have c85507 := (publicBatchWrapper89_copies2672 a h).2.2.2.1
  have c85508 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.1
  have c85509 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.1
  have c85510 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.1
  have c85511 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1391_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_18
  simp only [← c85506, ← c85507, ← c85508] at e_1391_18
  have e_1391_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1391 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_19
  simp only [← c85509, ← c85510, k2, ← c85511] at e_1391_19
  have hr := e_1391_19
  simp only [e_1391_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13085)) := by
  have c85512 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.1
  have c85513 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.1
  have c85514 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85515 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85516 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85517 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_0
  simp only [← c85512, ← c85513, ← c85514] at e_1392_0
  have e_1392_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_1
  simp only [← c85515, ← c85516, k2, ← c85517] at e_1392_1
  have hr := e_1392_1
  simp only [e_1392_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13086)) := by
  have c85518 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85519 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85520 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85521 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85522 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85523 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_2
  simp only [← c85518, ← c85519, ← c85520] at e_1392_2
  have e_1392_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_3
  simp only [← c85521, ← c85522, k2, ← c85523] at e_1392_3
  have hr := e_1392_3
  simp only [e_1392_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13087)) := by
  have c85524 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85525 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85526 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85527 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85528 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85529 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_4
  simp only [← c85524, ← c85525, ← c85526] at e_1392_4
  have e_1392_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_5
  simp only [← c85527, ← c85528, k2, ← c85529] at e_1392_5
  have hr := e_1392_5
  simp only [e_1392_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13088)) := by
  have c85530 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85531 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85532 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85533 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85534 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85535 := (publicBatchWrapper89_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_6
  simp only [← c85530, ← c85531, ← c85532] at e_1392_6
  have e_1392_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_7
  simp only [← c85533, ← c85534, k2, ← c85535] at e_1392_7
  have hr := e_1392_7
  simp only [e_1392_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13089)) := by
  have c85536 := (publicBatchWrapper89_copies2673 a h).1
  have c85537 := (publicBatchWrapper89_copies2673 a h).2.1
  have c85538 := (publicBatchWrapper89_copies2673 a h).2.2.1
  have c85539 := (publicBatchWrapper89_copies2673 a h).2.2.2.1
  have c85540 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.1
  have c85541 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_8
  simp only [← c85536, ← c85537, ← c85538] at e_1392_8
  have e_1392_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_9
  simp only [← c85539, ← c85540, k2, ← c85541] at e_1392_9
  have hr := e_1392_9
  simp only [e_1392_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13090)) := by
  have c85542 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.1
  have c85543 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.1
  have c85544 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.1
  have c85545 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.1
  have c85546 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85547 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_10
  simp only [← c85542, ← c85543, ← c85544] at e_1392_10
  have e_1392_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_11
  simp only [← c85545, ← c85546, k2, ← c85547] at e_1392_11
  have hr := e_1392_11
  simp only [e_1392_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13091)) := by
  have c85548 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85549 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85550 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85551 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85552 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85553 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_12
  simp only [← c85548, ← c85549, ← c85550] at e_1392_12
  have e_1392_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_13
  simp only [← c85551, ← c85552, k2, ← c85553] at e_1392_13
  have hr := e_1392_13
  simp only [e_1392_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13092)) := by
  have c85554 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85555 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85556 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85557 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85558 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85559 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_14
  simp only [← c85554, ← c85555, ← c85556] at e_1392_14
  have e_1392_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_15
  simp only [← c85557, ← c85558, k2, ← c85559] at e_1392_15
  have hr := e_1392_15
  simp only [e_1392_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13093)) := by
  have c85560 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85561 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85562 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85563 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85564 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85565 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_16
  simp only [← c85560, ← c85561, ← c85562] at e_1392_16
  have e_1392_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_17
  simp only [← c85563, ← c85564, k2, ← c85565] at e_1392_17
  have hr := e_1392_17
  simp only [e_1392_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1392 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13094)) := by
  have c85566 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85567 := (publicBatchWrapper89_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85568 := (publicBatchWrapper89_copies2674 a h).1
  have c85569 := (publicBatchWrapper89_copies2674 a h).2.1
  have c85570 := (publicBatchWrapper89_copies2674 a h).2.2.1
  have c85571 := (publicBatchWrapper89_copies2674 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1392_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_18
  simp only [← c85566, ← c85567, ← c85568] at e_1392_18
  have e_1392_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1392 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_19
  simp only [← c85569, ← c85570, k2, ← c85571] at e_1392_19
  have hr := e_1392_19
  simp only [e_1392_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13095)) := by
  have c85572 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.1
  have c85573 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.1
  have c85574 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.1
  have c85575 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.1
  have c85576 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.1
  have c85577 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_0
  simp only [← c85572, ← c85573, ← c85574] at e_1393_0
  have e_1393_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_1
  simp only [← c85575, ← c85576, k2, ← c85577] at e_1393_1
  have hr := e_1393_1
  simp only [e_1393_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13096)) := by
  have c85578 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85579 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85580 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85581 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85582 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85583 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_2
  simp only [← c85578, ← c85579, ← c85580] at e_1393_2
  have e_1393_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_3
  simp only [← c85581, ← c85582, k2, ← c85583] at e_1393_3
  have hr := e_1393_3
  simp only [e_1393_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13097)) := by
  have c85584 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85585 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85586 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85587 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85588 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85589 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_4
  simp only [← c85584, ← c85585, ← c85586] at e_1393_4
  have e_1393_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_5
  simp only [← c85587, ← c85588, k2, ← c85589] at e_1393_5
  have hr := e_1393_5
  simp only [e_1393_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13098)) := by
  have c85590 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85591 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85592 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85593 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85594 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85595 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_6
  simp only [← c85590, ← c85591, ← c85592] at e_1393_6
  have e_1393_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_7
  simp only [← c85593, ← c85594, k2, ← c85595] at e_1393_7
  have hr := e_1393_7
  simp only [e_1393_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13099)) := by
  have c85596 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85597 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85598 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85599 := (publicBatchWrapper89_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85600 := (publicBatchWrapper89_copies2675 a h).1
  have c85601 := (publicBatchWrapper89_copies2675 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_8
  simp only [← c85596, ← c85597, ← c85598] at e_1393_8
  have e_1393_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_9
  simp only [← c85599, ← c85600, k2, ← c85601] at e_1393_9
  have hr := e_1393_9
  simp only [e_1393_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13100)) := by
  have c85602 := (publicBatchWrapper89_copies2675 a h).2.2.1
  have c85603 := (publicBatchWrapper89_copies2675 a h).2.2.2.1
  have c85604 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.1
  have c85605 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.1
  have c85606 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.1
  have c85607 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_10
  simp only [← c85602, ← c85603, ← c85604] at e_1393_10
  have e_1393_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_11
  simp only [← c85605, ← c85606, k2, ← c85607] at e_1393_11
  have hr := e_1393_11
  simp only [e_1393_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13101)) := by
  have c85608 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.1
  have c85609 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.1
  have c85610 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85611 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85612 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85613 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_12
  simp only [← c85608, ← c85609, ← c85610] at e_1393_12
  have e_1393_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_13
  simp only [← c85611, ← c85612, k2, ← c85613] at e_1393_13
  have hr := e_1393_13
  simp only [e_1393_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13102)) := by
  have c85614 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85615 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85616 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85617 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85618 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85619 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_14
  simp only [← c85614, ← c85615, ← c85616] at e_1393_14
  have e_1393_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_15
  simp only [← c85617, ← c85618, k2, ← c85619] at e_1393_15
  have hr := e_1393_15
  simp only [e_1393_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13103)) := by
  have c85620 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85621 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85622 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85623 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85624 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85625 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_16
  simp only [← c85620, ← c85621, ← c85622] at e_1393_16
  have e_1393_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_17
  simp only [← c85623, ← c85624, k2, ← c85625] at e_1393_17
  have hr := e_1393_17
  simp only [e_1393_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1393 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13104)) := by
  have c85626 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85627 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85628 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85629 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85630 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85631 := (publicBatchWrapper89_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1393_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_18
  simp only [← c85626, ← c85627, ← c85628] at e_1393_18
  have e_1393_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1393 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_19
  simp only [← c85629, ← c85630, k2, ← c85631] at e_1393_19
  have hr := e_1393_19
  simp only [e_1393_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13105)) := by
  have c85632 := (publicBatchWrapper89_copies2676 a h).1
  have c85633 := (publicBatchWrapper89_copies2676 a h).2.1
  have c85634 := (publicBatchWrapper89_copies2676 a h).2.2.1
  have c85635 := (publicBatchWrapper89_copies2676 a h).2.2.2.1
  have c85636 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.1
  have c85637 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_0
  simp only [← c85632, ← c85633, ← c85634] at e_1394_0
  have e_1394_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_1
  simp only [← c85635, ← c85636, k2, ← c85637] at e_1394_1
  have hr := e_1394_1
  simp only [e_1394_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13106)) := by
  have c85638 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.1
  have c85639 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.1
  have c85640 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.1
  have c85641 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.1
  have c85642 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85643 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_2
  simp only [← c85638, ← c85639, ← c85640] at e_1394_2
  have e_1394_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_3
  simp only [← c85641, ← c85642, k2, ← c85643] at e_1394_3
  have hr := e_1394_3
  simp only [e_1394_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13107)) := by
  have c85644 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85645 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85646 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85647 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85648 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85649 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_4
  simp only [← c85644, ← c85645, ← c85646] at e_1394_4
  have e_1394_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_5
  simp only [← c85647, ← c85648, k2, ← c85649] at e_1394_5
  have hr := e_1394_5
  simp only [e_1394_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13108)) := by
  have c85650 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85651 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85652 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85653 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85654 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85655 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_6
  simp only [← c85650, ← c85651, ← c85652] at e_1394_6
  have e_1394_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_7
  simp only [← c85653, ← c85654, k2, ← c85655] at e_1394_7
  have hr := e_1394_7
  simp only [e_1394_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13109)) := by
  have c85656 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85657 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85658 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85659 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85660 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85661 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_8
  simp only [← c85656, ← c85657, ← c85658] at e_1394_8
  have e_1394_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_9
  simp only [← c85659, ← c85660, k2, ← c85661] at e_1394_9
  have hr := e_1394_9
  simp only [e_1394_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13110)) := by
  have c85662 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85663 := (publicBatchWrapper89_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85664 := (publicBatchWrapper89_copies2677 a h).1
  have c85665 := (publicBatchWrapper89_copies2677 a h).2.1
  have c85666 := (publicBatchWrapper89_copies2677 a h).2.2.1
  have c85667 := (publicBatchWrapper89_copies2677 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_10
  simp only [← c85662, ← c85663, ← c85664] at e_1394_10
  have e_1394_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_11
  simp only [← c85665, ← c85666, k2, ← c85667] at e_1394_11
  have hr := e_1394_11
  simp only [e_1394_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13111)) := by
  have c85668 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.1
  have c85669 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.1
  have c85670 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.1
  have c85671 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.1
  have c85672 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.1
  have c85673 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_12
  simp only [← c85668, ← c85669, ← c85670] at e_1394_12
  have e_1394_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_13
  simp only [← c85671, ← c85672, k2, ← c85673] at e_1394_13
  have hr := e_1394_13
  simp only [e_1394_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13112)) := by
  have c85674 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85675 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85676 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85677 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85678 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85679 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_14
  simp only [← c85674, ← c85675, ← c85676] at e_1394_14
  have e_1394_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_15
  simp only [← c85677, ← c85678, k2, ← c85679] at e_1394_15
  have hr := e_1394_15
  simp only [e_1394_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13113)) := by
  have c85680 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85681 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85682 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85683 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85684 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85685 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_16
  simp only [← c85680, ← c85681, ← c85682] at e_1394_16
  have e_1394_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_17
  simp only [← c85683, ← c85684, k2, ← c85685] at e_1394_17
  have hr := e_1394_17
  simp only [e_1394_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1394 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13114)) := by
  have c85686 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85687 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85688 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85689 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85690 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85691 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1394_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_18
  simp only [← c85686, ← c85687, ← c85688] at e_1394_18
  have e_1394_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1394 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_19
  simp only [← c85689, ← c85690, k2, ← c85691] at e_1394_19
  have hr := e_1394_19
  simp only [e_1394_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13115)) := by
  have c85692 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85693 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85694 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85695 := (publicBatchWrapper89_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85696 := (publicBatchWrapper89_copies2678 a h).1
  have c85697 := (publicBatchWrapper89_copies2678 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_0
  simp only [← c85692, ← c85693, ← c85694] at e_1395_0
  have e_1395_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_1
  simp only [← c85695, ← c85696, k2, ← c85697] at e_1395_1
  have hr := e_1395_1
  simp only [e_1395_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13116)) := by
  have c85698 := (publicBatchWrapper89_copies2678 a h).2.2.1
  have c85699 := (publicBatchWrapper89_copies2678 a h).2.2.2.1
  have c85700 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.1
  have c85701 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.1
  have c85702 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.1
  have c85703 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_2
  simp only [← c85698, ← c85699, ← c85700] at e_1395_2
  have e_1395_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_3
  simp only [← c85701, ← c85702, k2, ← c85703] at e_1395_3
  have hr := e_1395_3
  simp only [e_1395_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13117)) := by
  have c85704 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.1
  have c85705 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.1
  have c85706 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85707 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85708 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85709 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_4
  simp only [← c85704, ← c85705, ← c85706] at e_1395_4
  have e_1395_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_5
  simp only [← c85707, ← c85708, k2, ← c85709] at e_1395_5
  have hr := e_1395_5
  simp only [e_1395_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13118)) := by
  have c85710 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85711 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85712 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85713 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85714 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85715 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_6
  simp only [← c85710, ← c85711, ← c85712] at e_1395_6
  have e_1395_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_7
  simp only [← c85713, ← c85714, k2, ← c85715] at e_1395_7
  have hr := e_1395_7
  simp only [e_1395_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13119)) := by
  have c85716 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85717 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85718 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85719 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85720 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85721 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_8
  simp only [← c85716, ← c85717, ← c85718] at e_1395_8
  have e_1395_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_9
  simp only [← c85719, ← c85720, k2, ← c85721] at e_1395_9
  have hr := e_1395_9
  simp only [e_1395_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13120)) := by
  have c85722 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85723 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85724 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85725 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85726 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85727 := (publicBatchWrapper89_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_10
  simp only [← c85722, ← c85723, ← c85724] at e_1395_10
  have e_1395_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_11
  simp only [← c85725, ← c85726, k2, ← c85727] at e_1395_11
  have hr := e_1395_11
  simp only [e_1395_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13121)) := by
  have c85728 := (publicBatchWrapper89_copies2679 a h).1
  have c85729 := (publicBatchWrapper89_copies2679 a h).2.1
  have c85730 := (publicBatchWrapper89_copies2679 a h).2.2.1
  have c85731 := (publicBatchWrapper89_copies2679 a h).2.2.2.1
  have c85732 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.1
  have c85733 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_12
  simp only [← c85728, ← c85729, ← c85730] at e_1395_12
  have e_1395_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_13
  simp only [← c85731, ← c85732, k2, ← c85733] at e_1395_13
  have hr := e_1395_13
  simp only [e_1395_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13122)) := by
  have c85734 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.1
  have c85735 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.1
  have c85736 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.1
  have c85737 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.1
  have c85738 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85739 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_14
  simp only [← c85734, ← c85735, ← c85736] at e_1395_14
  have e_1395_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_15
  simp only [← c85737, ← c85738, k2, ← c85739] at e_1395_15
  have hr := e_1395_15
  simp only [e_1395_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13123)) := by
  have c85740 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85741 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85742 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85743 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85744 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85745 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_16
  simp only [← c85740, ← c85741, ← c85742] at e_1395_16
  have e_1395_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_17
  simp only [← c85743, ← c85744, k2, ← c85745] at e_1395_17
  have hr := e_1395_17
  simp only [e_1395_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1395 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13124)) := by
  have c85746 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85747 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85748 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85749 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85750 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85751 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1395_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_18
  simp only [← c85746, ← c85747, ← c85748] at e_1395_18
  have e_1395_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1395 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_19
  simp only [← c85749, ← c85750, k2, ← c85751] at e_1395_19
  have hr := e_1395_19
  simp only [e_1395_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13125)) := by
  have c85752 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85753 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85754 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85755 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85756 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85757 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_0
  simp only [← c85752, ← c85753, ← c85754] at e_1396_0
  have e_1396_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_1
  simp only [← c85755, ← c85756, k2, ← c85757] at e_1396_1
  have hr := e_1396_1
  simp only [e_1396_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13126)) := by
  have c85758 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85759 := (publicBatchWrapper89_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85760 := (publicBatchWrapper89_copies2680 a h).1
  have c85761 := (publicBatchWrapper89_copies2680 a h).2.1
  have c85762 := (publicBatchWrapper89_copies2680 a h).2.2.1
  have c85763 := (publicBatchWrapper89_copies2680 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_2
  simp only [← c85758, ← c85759, ← c85760] at e_1396_2
  have e_1396_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_3
  simp only [← c85761, ← c85762, k2, ← c85763] at e_1396_3
  have hr := e_1396_3
  simp only [e_1396_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13127)) := by
  have c85764 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.1
  have c85765 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.1
  have c85766 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.1
  have c85767 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.1
  have c85768 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.1
  have c85769 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_4
  simp only [← c85764, ← c85765, ← c85766] at e_1396_4
  have e_1396_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_5
  simp only [← c85767, ← c85768, k2, ← c85769] at e_1396_5
  have hr := e_1396_5
  simp only [e_1396_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13128)) := by
  have c85770 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85771 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85772 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85773 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85774 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85775 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_6
  simp only [← c85770, ← c85771, ← c85772] at e_1396_6
  have e_1396_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_7
  simp only [← c85773, ← c85774, k2, ← c85775] at e_1396_7
  have hr := e_1396_7
  simp only [e_1396_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13129)) := by
  have c85776 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85777 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85778 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85779 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85780 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85781 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_8
  simp only [← c85776, ← c85777, ← c85778] at e_1396_8
  have e_1396_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_9
  simp only [← c85779, ← c85780, k2, ← c85781] at e_1396_9
  have hr := e_1396_9
  simp only [e_1396_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13130)) := by
  have c85782 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85783 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85784 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85785 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85786 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85787 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_10
  simp only [← c85782, ← c85783, ← c85784] at e_1396_10
  have e_1396_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_11
  simp only [← c85785, ← c85786, k2, ← c85787] at e_1396_11
  have hr := e_1396_11
  simp only [e_1396_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13131)) := by
  have c85788 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85789 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85790 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85791 := (publicBatchWrapper89_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85792 := (publicBatchWrapper89_copies2681 a h).1
  have c85793 := (publicBatchWrapper89_copies2681 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_12
  simp only [← c85788, ← c85789, ← c85790] at e_1396_12
  have e_1396_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_13
  simp only [← c85791, ← c85792, k2, ← c85793] at e_1396_13
  have hr := e_1396_13
  simp only [e_1396_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13132)) := by
  have c85794 := (publicBatchWrapper89_copies2681 a h).2.2.1
  have c85795 := (publicBatchWrapper89_copies2681 a h).2.2.2.1
  have c85796 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.1
  have c85797 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.1
  have c85798 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.1
  have c85799 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_14
  simp only [← c85794, ← c85795, ← c85796] at e_1396_14
  have e_1396_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_15
  simp only [← c85797, ← c85798, k2, ← c85799] at e_1396_15
  have hr := e_1396_15
  simp only [e_1396_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13133)) := by
  have c85800 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.1
  have c85801 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.1
  have c85802 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85803 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85804 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85805 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_16
  simp only [← c85800, ← c85801, ← c85802] at e_1396_16
  have e_1396_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_17
  simp only [← c85803, ← c85804, k2, ← c85805] at e_1396_17
  have hr := e_1396_17
  simp only [e_1396_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1396 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13134)) := by
  have c85806 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85807 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85808 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85809 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85810 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85811 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1396_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_18
  simp only [← c85806, ← c85807, ← c85808] at e_1396_18
  have e_1396_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1396 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_19
  simp only [← c85809, ← c85810, k2, ← c85811] at e_1396_19
  have hr := e_1396_19
  simp only [e_1396_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13135)) := by
  have c85812 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85813 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85814 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85815 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85816 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85817 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_0
  simp only [← c85812, ← c85813, ← c85814] at e_1397_0
  have e_1397_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_1
  simp only [← c85815, ← c85816, k2, ← c85817] at e_1397_1
  have hr := e_1397_1
  simp only [e_1397_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13136)) := by
  have c85818 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85819 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85820 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85821 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85822 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85823 := (publicBatchWrapper89_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_2
  simp only [← c85818, ← c85819, ← c85820] at e_1397_2
  have e_1397_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_3
  simp only [← c85821, ← c85822, k2, ← c85823] at e_1397_3
  have hr := e_1397_3
  simp only [e_1397_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13137)) := by
  have c85824 := (publicBatchWrapper89_copies2682 a h).1
  have c85825 := (publicBatchWrapper89_copies2682 a h).2.1
  have c85826 := (publicBatchWrapper89_copies2682 a h).2.2.1
  have c85827 := (publicBatchWrapper89_copies2682 a h).2.2.2.1
  have c85828 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.1
  have c85829 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_4
  simp only [← c85824, ← c85825, ← c85826] at e_1397_4
  have e_1397_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_5
  simp only [← c85827, ← c85828, k2, ← c85829] at e_1397_5
  have hr := e_1397_5
  simp only [e_1397_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13138)) := by
  have c85830 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.1
  have c85831 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.1
  have c85832 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.1
  have c85833 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.1
  have c85834 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85835 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_6
  simp only [← c85830, ← c85831, ← c85832] at e_1397_6
  have e_1397_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_7
  simp only [← c85833, ← c85834, k2, ← c85835] at e_1397_7
  have hr := e_1397_7
  simp only [e_1397_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13139)) := by
  have c85836 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85837 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85838 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85839 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85840 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85841 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_8
  simp only [← c85836, ← c85837, ← c85838] at e_1397_8
  have e_1397_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_9
  simp only [← c85839, ← c85840, k2, ← c85841] at e_1397_9
  have hr := e_1397_9
  simp only [e_1397_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13140)) := by
  have c85842 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85843 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85844 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85845 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85846 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85847 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_10
  simp only [← c85842, ← c85843, ← c85844] at e_1397_10
  have e_1397_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_11
  simp only [← c85845, ← c85846, k2, ← c85847] at e_1397_11
  have hr := e_1397_11
  simp only [e_1397_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13141)) := by
  have c85848 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85849 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85850 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85851 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85852 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85853 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_12
  simp only [← c85848, ← c85849, ← c85850] at e_1397_12
  have e_1397_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_13
  simp only [← c85851, ← c85852, k2, ← c85853] at e_1397_13
  have hr := e_1397_13
  simp only [e_1397_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13142)) := by
  have c85854 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85855 := (publicBatchWrapper89_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85856 := (publicBatchWrapper89_copies2683 a h).1
  have c85857 := (publicBatchWrapper89_copies2683 a h).2.1
  have c85858 := (publicBatchWrapper89_copies2683 a h).2.2.1
  have c85859 := (publicBatchWrapper89_copies2683 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_14
  simp only [← c85854, ← c85855, ← c85856] at e_1397_14
  have e_1397_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_15
  simp only [← c85857, ← c85858, k2, ← c85859] at e_1397_15
  have hr := e_1397_15
  simp only [e_1397_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13143)) := by
  have c85860 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.1
  have c85861 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.1
  have c85862 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.1
  have c85863 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.1
  have c85864 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.1
  have c85865 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_16
  simp only [← c85860, ← c85861, ← c85862] at e_1397_16
  have e_1397_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_17
  simp only [← c85863, ← c85864, k2, ← c85865] at e_1397_17
  have hr := e_1397_17
  simp only [e_1397_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1397 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13144)) := by
  have c85866 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85867 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85868 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85869 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85870 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85871 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1397_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_18
  simp only [← c85866, ← c85867, ← c85868] at e_1397_18
  have e_1397_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1397 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_19
  simp only [← c85869, ← c85870, k2, ← c85871] at e_1397_19
  have hr := e_1397_19
  simp only [e_1397_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13145)) := by
  have c85872 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85873 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85874 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85875 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85876 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85877 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_0
  simp only [← c85872, ← c85873, ← c85874] at e_1398_0
  have e_1398_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_1
  simp only [← c85875, ← c85876, k2, ← c85877] at e_1398_1
  have hr := e_1398_1
  simp only [e_1398_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13146)) := by
  have c85878 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85879 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85880 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85881 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85882 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85883 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_2
  simp only [← c85878, ← c85879, ← c85880] at e_1398_2
  have e_1398_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_3
  simp only [← c85881, ← c85882, k2, ← c85883] at e_1398_3
  have hr := e_1398_3
  simp only [e_1398_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13147)) := by
  have c85884 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85885 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85886 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85887 := (publicBatchWrapper89_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85888 := (publicBatchWrapper89_copies2684 a h).1
  have c85889 := (publicBatchWrapper89_copies2684 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_4
  simp only [← c85884, ← c85885, ← c85886] at e_1398_4
  have e_1398_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_5
  simp only [← c85887, ← c85888, k2, ← c85889] at e_1398_5
  have hr := e_1398_5
  simp only [e_1398_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13148)) := by
  have c85890 := (publicBatchWrapper89_copies2684 a h).2.2.1
  have c85891 := (publicBatchWrapper89_copies2684 a h).2.2.2.1
  have c85892 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.1
  have c85893 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.1
  have c85894 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.1
  have c85895 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_6
  simp only [← c85890, ← c85891, ← c85892] at e_1398_6
  have e_1398_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_7
  simp only [← c85893, ← c85894, k2, ← c85895] at e_1398_7
  have hr := e_1398_7
  simp only [e_1398_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13149)) := by
  have c85896 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.1
  have c85897 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.1
  have c85898 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85899 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85900 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85901 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_8
  simp only [← c85896, ← c85897, ← c85898] at e_1398_8
  have e_1398_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_9
  simp only [← c85899, ← c85900, k2, ← c85901] at e_1398_9
  have hr := e_1398_9
  simp only [e_1398_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13150)) := by
  have c85902 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85903 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85904 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85905 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85906 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85907 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_10
  simp only [← c85902, ← c85903, ← c85904] at e_1398_10
  have e_1398_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_11
  simp only [← c85905, ← c85906, k2, ← c85907] at e_1398_11
  have hr := e_1398_11
  simp only [e_1398_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13151)) := by
  have c85908 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85909 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85910 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85911 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85912 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85913 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_12
  simp only [← c85908, ← c85909, ← c85910] at e_1398_12
  have e_1398_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_13
  simp only [← c85911, ← c85912, k2, ← c85913] at e_1398_13
  have hr := e_1398_13
  simp only [e_1398_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13152)) := by
  have c85914 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85915 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85916 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85917 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85918 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85919 := (publicBatchWrapper89_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_14
  simp only [← c85914, ← c85915, ← c85916] at e_1398_14
  have e_1398_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_15
  simp only [← c85917, ← c85918, k2, ← c85919] at e_1398_15
  have hr := e_1398_15
  simp only [e_1398_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13153)) := by
  have c85920 := (publicBatchWrapper89_copies2685 a h).1
  have c85921 := (publicBatchWrapper89_copies2685 a h).2.1
  have c85922 := (publicBatchWrapper89_copies2685 a h).2.2.1
  have c85923 := (publicBatchWrapper89_copies2685 a h).2.2.2.1
  have c85924 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.1
  have c85925 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_16
  simp only [← c85920, ← c85921, ← c85922] at e_1398_16
  have e_1398_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_17
  simp only [← c85923, ← c85924, k2, ← c85925] at e_1398_17
  have hr := e_1398_17
  simp only [e_1398_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1398 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13154)) := by
  have c85926 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.1
  have c85927 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.1
  have c85928 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.1
  have c85929 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.1
  have c85930 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85931 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1398_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_18
  simp only [← c85926, ← c85927, ← c85928] at e_1398_18
  have e_1398_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_19
  simp only [← c85929, ← c85930, k2, ← c85931] at e_1398_19
  have hr := e_1398_19
  simp only [e_1398_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13155)) := by
  have c85932 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85933 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85934 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85935 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85936 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85937 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_0
  simp only [← c85932, ← c85933, ← c85934] at e_1399_0
  have e_1399_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_1
  simp only [← c85935, ← c85936, k2, ← c85937] at e_1399_1
  have hr := e_1399_1
  simp only [e_1399_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13156)) := by
  have c85938 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85939 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85940 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85941 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85942 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85943 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_2
  simp only [← c85938, ← c85939, ← c85940] at e_1399_2
  have e_1399_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_3
  simp only [← c85941, ← c85942, k2, ← c85943] at e_1399_3
  have hr := e_1399_3
  simp only [e_1399_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13157)) := by
  have c85944 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85945 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85946 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85947 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85948 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85949 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_4
  simp only [← c85944, ← c85945, ← c85946] at e_1399_4
  have e_1399_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_5
  simp only [← c85947, ← c85948, k2, ← c85949] at e_1399_5
  have hr := e_1399_5
  simp only [e_1399_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13158)) := by
  have c85950 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85951 := (publicBatchWrapper89_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85952 := (publicBatchWrapper89_copies2686 a h).1
  have c85953 := (publicBatchWrapper89_copies2686 a h).2.1
  have c85954 := (publicBatchWrapper89_copies2686 a h).2.2.1
  have c85955 := (publicBatchWrapper89_copies2686 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_6
  simp only [← c85950, ← c85951, ← c85952] at e_1399_6
  have e_1399_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_7
  simp only [← c85953, ← c85954, k2, ← c85955] at e_1399_7
  have hr := e_1399_7
  simp only [e_1399_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13159)) := by
  have c85956 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.1
  have c85957 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.1
  have c85958 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.1
  have c85959 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.1
  have c85960 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.1
  have c85961 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_8
  simp only [← c85956, ← c85957, ← c85958] at e_1399_8
  have e_1399_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_9
  simp only [← c85959, ← c85960, k2, ← c85961] at e_1399_9
  have hr := e_1399_9
  simp only [e_1399_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13160)) := by
  have c85962 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85963 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85964 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85965 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85966 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85967 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_10
  simp only [← c85962, ← c85963, ← c85964] at e_1399_10
  have e_1399_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_11
  simp only [← c85965, ← c85966, k2, ← c85967] at e_1399_11
  have hr := e_1399_11
  simp only [e_1399_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13161)) := by
  have c85968 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85969 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85970 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85971 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85972 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85973 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_12
  simp only [← c85968, ← c85969, ← c85970] at e_1399_12
  have e_1399_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_13
  simp only [← c85971, ← c85972, k2, ← c85973] at e_1399_13
  have hr := e_1399_13
  simp only [e_1399_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13162)) := by
  have c85974 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85975 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85976 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85977 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85978 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85979 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_14
  simp only [← c85974, ← c85975, ← c85976] at e_1399_14
  have e_1399_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_15
  simp only [← c85977, ← c85978, k2, ← c85979] at e_1399_15
  have hr := e_1399_15
  simp only [e_1399_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13163)) := by
  have c85980 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85981 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85982 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85983 := (publicBatchWrapper89_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85984 := (publicBatchWrapper89_copies2687 a h).1
  have c85985 := (publicBatchWrapper89_copies2687 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_16
  simp only [← c85980, ← c85981, ← c85982] at e_1399_16
  have e_1399_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_17
  simp only [← c85983, ← c85984, k2, ← c85985] at e_1399_17
  have hr := e_1399_17
  simp only [e_1399_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1399 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13164)) := by
  have c85986 := (publicBatchWrapper89_copies2687 a h).2.2.1
  have c85987 := (publicBatchWrapper89_copies2687 a h).2.2.2.1
  have c85988 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.1
  have c85989 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.1
  have c85990 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.1
  have c85991 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1399_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_18
  simp only [← c85986, ← c85987, ← c85988] at e_1399_18
  have e_1399_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_19
  simp only [← c85989, ← c85990, k2, ← c85991] at e_1399_19
  have hr := e_1399_19
  simp only [e_1399_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13165)) := by
  have c85992 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.1
  have c85993 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.1
  have c85994 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85995 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85996 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85997 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_0
  simp only [← c85992, ← c85993, ← c85994] at e_1400_0
  have e_1400_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_1
  simp only [← c85995, ← c85996, k2, ← c85997] at e_1400_1
  have hr := e_1400_1
  simp only [e_1400_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13166)) := by
  have c85998 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85999 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86000 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86001 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86002 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86003 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_2
  simp only [← c85998, ← c85999, ← c86000] at e_1400_2
  have e_1400_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_3
  simp only [← c86001, ← c86002, k2, ← c86003] at e_1400_3
  have hr := e_1400_3
  simp only [e_1400_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13167)) := by
  have c86004 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86005 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86006 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86007 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86008 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86009 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_4
  simp only [← c86004, ← c86005, ← c86006] at e_1400_4
  have e_1400_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_5
  simp only [← c86007, ← c86008, k2, ← c86009] at e_1400_5
  have hr := e_1400_5
  simp only [e_1400_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13168)) := by
  have c86010 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86011 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86012 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86013 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86014 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86015 := (publicBatchWrapper89_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_6
  simp only [← c86010, ← c86011, ← c86012] at e_1400_6
  have e_1400_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_7
  simp only [← c86013, ← c86014, k2, ← c86015] at e_1400_7
  have hr := e_1400_7
  simp only [e_1400_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13169)) := by
  have c86016 := (publicBatchWrapper89_copies2688 a h).1
  have c86017 := (publicBatchWrapper89_copies2688 a h).2.1
  have c86018 := (publicBatchWrapper89_copies2688 a h).2.2.1
  have c86019 := (publicBatchWrapper89_copies2688 a h).2.2.2.1
  have c86020 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.1
  have c86021 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_8
  simp only [← c86016, ← c86017, ← c86018] at e_1400_8
  have e_1400_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_9
  simp only [← c86019, ← c86020, k2, ← c86021] at e_1400_9
  have hr := e_1400_9
  simp only [e_1400_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13210)) := by
  have c86022 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.1
  have c86023 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.1
  have c86024 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.1
  have c86025 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.1
  have c86026 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86027 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_10
  simp only [← c86022, ← c86023, ← c86024] at e_1400_10
  have e_1400_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_11
  simp only [← c86025, ← c86026, k2, ← c86027] at e_1400_11
  have hr := e_1400_11
  simp only [e_1400_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13211)) := by
  have c86028 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86029 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86030 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86031 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86032 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86033 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_12
  simp only [← c86028, ← c86029, ← c86030] at e_1400_12
  have e_1400_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_13
  simp only [← c86031, ← c86032, k2, ← c86033] at e_1400_13
  have hr := e_1400_13
  simp only [e_1400_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13212)) := by
  have c86034 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86035 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86036 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86037 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86038 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86039 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_14
  simp only [← c86034, ← c86035, ← c86036] at e_1400_14
  have e_1400_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_15
  simp only [← c86037, ← c86038, k2, ← c86039] at e_1400_15
  have hr := e_1400_15
  simp only [e_1400_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13213)) := by
  have c86040 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86041 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86042 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86043 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86044 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86045 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_16
  simp only [← c86040, ← c86041, ← c86042] at e_1400_16
  have e_1400_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_17
  simp only [← c86043, ← c86044, k2, ← c86045] at e_1400_17
  have hr := e_1400_17
  simp only [e_1400_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1400 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13214)) := by
  have c86046 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86047 := (publicBatchWrapper89_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86048 := (publicBatchWrapper89_copies2689 a h).1
  have c86049 := (publicBatchWrapper89_copies2689 a h).2.1
  have c86050 := (publicBatchWrapper89_copies2689 a h).2.2.1
  have c86051 := (publicBatchWrapper89_copies2689 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1400_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_18
  simp only [← c86046, ← c86047, ← c86048] at e_1400_18
  have e_1400_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_19
  simp only [← c86049, ← c86050, k2, ← c86051] at e_1400_19
  have hr := e_1400_19
  simp only [e_1400_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13215)) := by
  have c86052 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.1
  have c86053 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.1
  have c86054 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.1
  have c86055 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.1
  have c86056 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.1
  have c86057 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_0
  simp only [← c86052, ← c86053, ← c86054] at e_1401_0
  have e_1401_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_1
  simp only [← c86055, ← c86056, k2, ← c86057] at e_1401_1
  have hr := e_1401_1
  simp only [e_1401_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13216)) := by
  have c86058 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86059 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86060 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86061 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86062 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86063 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_2
  simp only [← c86058, ← c86059, ← c86060] at e_1401_2
  have e_1401_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_3
  simp only [← c86061, ← c86062, k2, ← c86063] at e_1401_3
  have hr := e_1401_3
  simp only [e_1401_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13217)) := by
  have c86064 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86065 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86066 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86067 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86068 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86069 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_4
  simp only [← c86064, ← c86065, ← c86066] at e_1401_4
  have e_1401_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_5
  simp only [← c86067, ← c86068, k2, ← c86069] at e_1401_5
  have hr := e_1401_5
  simp only [e_1401_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13218)) := by
  have c86070 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86071 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86072 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86073 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86074 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86075 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_6
  simp only [← c86070, ← c86071, ← c86072] at e_1401_6
  have e_1401_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_7
  simp only [← c86073, ← c86074, k2, ← c86075] at e_1401_7
  have hr := e_1401_7
  simp only [e_1401_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13219)) := by
  have c86076 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86077 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86078 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86079 := (publicBatchWrapper89_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86080 := (publicBatchWrapper89_copies2690 a h).1
  have c86081 := (publicBatchWrapper89_copies2690 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_8
  simp only [← c86076, ← c86077, ← c86078] at e_1401_8
  have e_1401_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_9
  simp only [← c86079, ← c86080, k2, ← c86081] at e_1401_9
  have hr := e_1401_9
  simp only [e_1401_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13220)) := by
  have c86082 := (publicBatchWrapper89_copies2690 a h).2.2.1
  have c86083 := (publicBatchWrapper89_copies2690 a h).2.2.2.1
  have c86084 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.1
  have c86085 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.1
  have c86086 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.1
  have c86087 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_10
  simp only [← c86082, ← c86083, ← c86084] at e_1401_10
  have e_1401_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_11
  simp only [← c86085, ← c86086, k2, ← c86087] at e_1401_11
  have hr := e_1401_11
  simp only [e_1401_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13221)) := by
  have c86088 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.1
  have c86089 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.1
  have c86090 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86091 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86092 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86093 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_12
  simp only [← c86088, ← c86089, ← c86090] at e_1401_12
  have e_1401_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_13
  simp only [← c86091, ← c86092, k2, ← c86093] at e_1401_13
  have hr := e_1401_13
  simp only [e_1401_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13222)) := by
  have c86094 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86095 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86096 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86097 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86098 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86099 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_14
  simp only [← c86094, ← c86095, ← c86096] at e_1401_14
  have e_1401_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_15
  simp only [← c86097, ← c86098, k2, ← c86099] at e_1401_15
  have hr := e_1401_15
  simp only [e_1401_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13223)) := by
  have c86100 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86101 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86102 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86103 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86104 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86105 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_16
  simp only [← c86100, ← c86101, ← c86102] at e_1401_16
  have e_1401_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_17
  simp only [← c86103, ← c86104, k2, ← c86105] at e_1401_17
  have hr := e_1401_17
  simp only [e_1401_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1401 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13224)) := by
  have c86106 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86107 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86108 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86109 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86110 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86111 := (publicBatchWrapper89_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1401_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_18
  simp only [← c86106, ← c86107, ← c86108] at e_1401_18
  have e_1401_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_19
  simp only [← c86109, ← c86110, k2, ← c86111] at e_1401_19
  have hr := e_1401_19
  simp only [e_1401_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13225)) := by
  have c86112 := (publicBatchWrapper89_copies2691 a h).1
  have c86113 := (publicBatchWrapper89_copies2691 a h).2.1
  have c86114 := (publicBatchWrapper89_copies2691 a h).2.2.1
  have c86115 := (publicBatchWrapper89_copies2691 a h).2.2.2.1
  have c86116 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.1
  have c86117 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_0
  simp only [← c86112, ← c86113, ← c86114] at e_1402_0
  have e_1402_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_1
  simp only [← c86115, ← c86116, k2, ← c86117] at e_1402_1
  have hr := e_1402_1
  simp only [e_1402_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13226)) := by
  have c86118 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.1
  have c86119 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.1
  have c86120 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.1
  have c86121 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.1
  have c86122 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86123 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_2
  simp only [← c86118, ← c86119, ← c86120] at e_1402_2
  have e_1402_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_3
  simp only [← c86121, ← c86122, k2, ← c86123] at e_1402_3
  have hr := e_1402_3
  simp only [e_1402_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13227)) := by
  have c86124 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86125 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86126 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86127 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86128 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86129 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_4
  simp only [← c86124, ← c86125, ← c86126] at e_1402_4
  have e_1402_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_5
  simp only [← c86127, ← c86128, k2, ← c86129] at e_1402_5
  have hr := e_1402_5
  simp only [e_1402_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13228)) := by
  have c86130 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86131 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86132 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86133 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86134 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86135 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_6
  simp only [← c86130, ← c86131, ← c86132] at e_1402_6
  have e_1402_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_7
  simp only [← c86133, ← c86134, k2, ← c86135] at e_1402_7
  have hr := e_1402_7
  simp only [e_1402_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13229)) := by
  have c86136 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86137 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86138 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86139 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86140 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86141 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_8
  simp only [← c86136, ← c86137, ← c86138] at e_1402_8
  have e_1402_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_9
  simp only [← c86139, ← c86140, k2, ← c86141] at e_1402_9
  have hr := e_1402_9
  simp only [e_1402_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13230)) := by
  have c86142 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86143 := (publicBatchWrapper89_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86144 := (publicBatchWrapper89_copies2692 a h).1
  have c86145 := (publicBatchWrapper89_copies2692 a h).2.1
  have c86146 := (publicBatchWrapper89_copies2692 a h).2.2.1
  have c86147 := (publicBatchWrapper89_copies2692 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_10
  simp only [← c86142, ← c86143, ← c86144] at e_1402_10
  have e_1402_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_11
  simp only [← c86145, ← c86146, k2, ← c86147] at e_1402_11
  have hr := e_1402_11
  simp only [e_1402_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13231)) := by
  have c86148 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.1
  have c86149 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.1
  have c86150 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.1
  have c86151 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.1
  have c86152 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.1
  have c86153 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_12
  simp only [← c86148, ← c86149, ← c86150] at e_1402_12
  have e_1402_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_13
  simp only [← c86151, ← c86152, k2, ← c86153] at e_1402_13
  have hr := e_1402_13
  simp only [e_1402_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13232)) := by
  have c86154 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86155 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86156 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86157 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86158 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86159 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_14
  simp only [← c86154, ← c86155, ← c86156] at e_1402_14
  have e_1402_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_15
  simp only [← c86157, ← c86158, k2, ← c86159] at e_1402_15
  have hr := e_1402_15
  simp only [e_1402_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13233)) := by
  have c86160 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86161 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86162 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86163 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86164 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86165 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_16
  simp only [← c86160, ← c86161, ← c86162] at e_1402_16
  have e_1402_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_17
  simp only [← c86163, ← c86164, k2, ← c86165] at e_1402_17
  have hr := e_1402_17
  simp only [e_1402_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1402 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13234)) := by
  have c86166 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86167 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86168 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86169 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86170 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86171 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1402_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_18
  simp only [← c86166, ← c86167, ← c86168] at e_1402_18
  have e_1402_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_19
  simp only [← c86169, ← c86170, k2, ← c86171] at e_1402_19
  have hr := e_1402_19
  simp only [e_1402_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13235)) := by
  have c86172 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86173 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86174 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86175 := (publicBatchWrapper89_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86176 := (publicBatchWrapper89_copies2693 a h).1
  have c86177 := (publicBatchWrapper89_copies2693 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_0
  simp only [← c86172, ← c86173, ← c86174] at e_1403_0
  have e_1403_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_1
  simp only [← c86175, ← c86176, k2, ← c86177] at e_1403_1
  have hr := e_1403_1
  simp only [e_1403_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13236)) := by
  have c86178 := (publicBatchWrapper89_copies2693 a h).2.2.1
  have c86179 := (publicBatchWrapper89_copies2693 a h).2.2.2.1
  have c86180 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.1
  have c86181 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.1
  have c86182 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.1
  have c86183 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_2
  simp only [← c86178, ← c86179, ← c86180] at e_1403_2
  have e_1403_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_3
  simp only [← c86181, ← c86182, k2, ← c86183] at e_1403_3
  have hr := e_1403_3
  simp only [e_1403_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13237)) := by
  have c86184 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.1
  have c86185 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.1
  have c86186 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86187 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86188 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86189 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_4
  simp only [← c86184, ← c86185, ← c86186] at e_1403_4
  have e_1403_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_5
  simp only [← c86187, ← c86188, k2, ← c86189] at e_1403_5
  have hr := e_1403_5
  simp only [e_1403_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13238)) := by
  have c86190 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86191 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86192 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86193 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86194 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86195 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_6
  simp only [← c86190, ← c86191, ← c86192] at e_1403_6
  have e_1403_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_7
  simp only [← c86193, ← c86194, k2, ← c86195] at e_1403_7
  have hr := e_1403_7
  simp only [e_1403_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13239)) := by
  have c86196 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86197 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86198 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86199 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86200 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86201 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_8
  simp only [← c86196, ← c86197, ← c86198] at e_1403_8
  have e_1403_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_9
  simp only [← c86199, ← c86200, k2, ← c86201] at e_1403_9
  have hr := e_1403_9
  simp only [e_1403_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13240)) := by
  have c86202 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86203 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86204 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86205 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86206 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86207 := (publicBatchWrapper89_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_10
  simp only [← c86202, ← c86203, ← c86204] at e_1403_10
  have e_1403_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_11
  simp only [← c86205, ← c86206, k2, ← c86207] at e_1403_11
  have hr := e_1403_11
  simp only [e_1403_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13241)) := by
  have c86208 := (publicBatchWrapper89_copies2694 a h).1
  have c86209 := (publicBatchWrapper89_copies2694 a h).2.1
  have c86210 := (publicBatchWrapper89_copies2694 a h).2.2.1
  have c86211 := (publicBatchWrapper89_copies2694 a h).2.2.2.1
  have c86212 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.1
  have c86213 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_12
  simp only [← c86208, ← c86209, ← c86210] at e_1403_12
  have e_1403_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_13
  simp only [← c86211, ← c86212, k2, ← c86213] at e_1403_13
  have hr := e_1403_13
  simp only [e_1403_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13242)) := by
  have c86214 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.1
  have c86215 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.1
  have c86216 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.1
  have c86217 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.1
  have c86218 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86219 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_14
  simp only [← c86214, ← c86215, ← c86216] at e_1403_14
  have e_1403_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_15
  simp only [← c86217, ← c86218, k2, ← c86219] at e_1403_15
  have hr := e_1403_15
  simp only [e_1403_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13243)) := by
  have c86220 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86221 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86222 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86223 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86224 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86225 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_16
  simp only [← c86220, ← c86221, ← c86222] at e_1403_16
  have e_1403_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_17
  simp only [← c86223, ← c86224, k2, ← c86225] at e_1403_17
  have hr := e_1403_17
  simp only [e_1403_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1403 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13244)) := by
  have c86226 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86227 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86228 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86229 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86230 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86231 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1403_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_18
  simp only [← c86226, ← c86227, ← c86228] at e_1403_18
  have e_1403_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_19
  simp only [← c86229, ← c86230, k2, ← c86231] at e_1403_19
  have hr := e_1403_19
  simp only [e_1403_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13245)) := by
  have c86232 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86233 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86234 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86235 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86236 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86237 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_0
  simp only [← c86232, ← c86233, ← c86234] at e_1404_0
  have e_1404_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_1
  simp only [← c86235, ← c86236, k2, ← c86237] at e_1404_1
  have hr := e_1404_1
  simp only [e_1404_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13246)) := by
  have c86238 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86239 := (publicBatchWrapper89_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86240 := (publicBatchWrapper89_copies2695 a h).1
  have c86241 := (publicBatchWrapper89_copies2695 a h).2.1
  have c86242 := (publicBatchWrapper89_copies2695 a h).2.2.1
  have c86243 := (publicBatchWrapper89_copies2695 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_2
  simp only [← c86238, ← c86239, ← c86240] at e_1404_2
  have e_1404_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_3
  simp only [← c86241, ← c86242, k2, ← c86243] at e_1404_3
  have hr := e_1404_3
  simp only [e_1404_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13247)) := by
  have c86244 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.1
  have c86245 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.1
  have c86246 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.1
  have c86247 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.1
  have c86248 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.1
  have c86249 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_4
  simp only [← c86244, ← c86245, ← c86246] at e_1404_4
  have e_1404_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_5
  simp only [← c86247, ← c86248, k2, ← c86249] at e_1404_5
  have hr := e_1404_5
  simp only [e_1404_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13248)) := by
  have c86250 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86251 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86252 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86253 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86254 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86255 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_6
  simp only [← c86250, ← c86251, ← c86252] at e_1404_6
  have e_1404_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_7
  simp only [← c86253, ← c86254, k2, ← c86255] at e_1404_7
  have hr := e_1404_7
  simp only [e_1404_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13249)) := by
  have c86256 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86257 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86258 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86259 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86260 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86261 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_8
  simp only [← c86256, ← c86257, ← c86258] at e_1404_8
  have e_1404_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_9
  simp only [← c86259, ← c86260, k2, ← c86261] at e_1404_9
  have hr := e_1404_9
  simp only [e_1404_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13250)) := by
  have c86262 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86263 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86264 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86265 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86266 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86267 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_10
  simp only [← c86262, ← c86263, ← c86264] at e_1404_10
  have e_1404_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_11
  simp only [← c86265, ← c86266, k2, ← c86267] at e_1404_11
  have hr := e_1404_11
  simp only [e_1404_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13251)) := by
  have c86268 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86269 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86270 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86271 := (publicBatchWrapper89_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86272 := (publicBatchWrapper89_copies2696 a h).1
  have c86273 := (publicBatchWrapper89_copies2696 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_12
  simp only [← c86268, ← c86269, ← c86270] at e_1404_12
  have e_1404_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_13
  simp only [← c86271, ← c86272, k2, ← c86273] at e_1404_13
  have hr := e_1404_13
  simp only [e_1404_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13252)) := by
  have c86274 := (publicBatchWrapper89_copies2696 a h).2.2.1
  have c86275 := (publicBatchWrapper89_copies2696 a h).2.2.2.1
  have c86276 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.1
  have c86277 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.1
  have c86278 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.1
  have c86279 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_14
  simp only [← c86274, ← c86275, ← c86276] at e_1404_14
  have e_1404_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_15
  simp only [← c86277, ← c86278, k2, ← c86279] at e_1404_15
  have hr := e_1404_15
  simp only [e_1404_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13253)) := by
  have c86280 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.1
  have c86281 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.1
  have c86282 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86283 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86284 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86285 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_16
  simp only [← c86280, ← c86281, ← c86282] at e_1404_16
  have e_1404_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_17
  simp only [← c86283, ← c86284, k2, ← c86285] at e_1404_17
  have hr := e_1404_17
  simp only [e_1404_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1404 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13254)) := by
  have c86286 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86287 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86288 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86289 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86290 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86291 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1404_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_18
  simp only [← c86286, ← c86287, ← c86288] at e_1404_18
  have e_1404_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_19
  simp only [← c86289, ← c86290, k2, ← c86291] at e_1404_19
  have hr := e_1404_19
  simp only [e_1404_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13255)) := by
  have c86292 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86293 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86294 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86295 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86296 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86297 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_0
  simp only [← c86292, ← c86293, ← c86294] at e_1405_0
  have e_1405_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_1
  simp only [← c86295, ← c86296, k2, ← c86297] at e_1405_1
  have hr := e_1405_1
  simp only [e_1405_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13256)) := by
  have c86298 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86299 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86300 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86301 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86302 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86303 := (publicBatchWrapper89_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_2
  simp only [← c86298, ← c86299, ← c86300] at e_1405_2
  have e_1405_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_3
  simp only [← c86301, ← c86302, k2, ← c86303] at e_1405_3
  have hr := e_1405_3
  simp only [e_1405_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13257)) := by
  have c86304 := (publicBatchWrapper89_copies2697 a h).1
  have c86305 := (publicBatchWrapper89_copies2697 a h).2.1
  have c86306 := (publicBatchWrapper89_copies2697 a h).2.2.1
  have c86307 := (publicBatchWrapper89_copies2697 a h).2.2.2.1
  have c86308 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.1
  have c86309 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_4
  simp only [← c86304, ← c86305, ← c86306] at e_1405_4
  have e_1405_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_5
  simp only [← c86307, ← c86308, k2, ← c86309] at e_1405_5
  have hr := e_1405_5
  simp only [e_1405_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13258)) := by
  have c86310 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.1
  have c86311 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.1
  have c86312 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.1
  have c86313 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.1
  have c86314 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86315 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_6
  simp only [← c86310, ← c86311, ← c86312] at e_1405_6
  have e_1405_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_7
  simp only [← c86313, ← c86314, k2, ← c86315] at e_1405_7
  have hr := e_1405_7
  simp only [e_1405_6] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
