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

theorem publicBatchWrapper86_f2214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 240 39) = band (a (.virt 18355)) (a (.virt 18357)) := by
  have c15677 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15678 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15679 := (publicBatchWrapper86_copies489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_240_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row240 (by decide)
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
  have e_240_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row240 (by decide)
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
  have e_240_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row240 (by decide)
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
  have e_236_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_9
  simp only [← c15689, ← c15690, k3, ← c15691] at e_236_9
  have e_241_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row241 (by decide)
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
  have e_239_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_18
  simp only [← c15693, k3, ← c15694, k3, ← c15695] at e_239_18
  have e_239_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_19
  simp only [← c15696, ← c15697, k3, ← c15698] at e_239_19
  have e_240_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_12
  simp only [← c15699, ← c15700, ← c15701] at e_240_12
  have e_240_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_13
  simp only [← c15702, ← c15703, ← c15704] at e_240_13
  have e_242_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row242 (by decide)
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
  have e_236_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_10
  simp only [← c15713, ← c15714, k3, ← c15715] at e_236_10
  have e_241_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row241 (by decide)
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
  have e_240_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_14
  simp only [← c15723, ← c15724, ← c15725] at e_240_14
  have e_240_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_15
  simp only [← c15726, ← c15727, ← c15728] at e_240_15
  have e_242_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_1
  simp only [← c15717, k3, ← c15718, k3, ← c15719] at e_242_1
  have e_242_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_2
  simp only [← c15720, ← c15721, k3, ← c15722] at e_242_2
  have e_242_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row242 (by decide)
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
  have e_236_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_11
  simp only [← c15737, ← c15738, k3, ← c15739] at e_236_11
  have e_241_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row241 (by decide)
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
  have e_240_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_16
  simp only [← c15747, ← c15748, ← c15749] at e_240_16
  have e_240_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_17
  simp only [← c15750, ← c15751, ← c15752] at e_240_17
  have e_242_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_4
  simp only [← c15741, k3, ← c15742, k3, ← c15743] at e_242_4
  have e_242_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_5
  simp only [← c15744, ← c15745, k3, ← c15746] at e_242_5
  have e_242_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row242 (by decide)
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
  have e_240_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_18
  simp only [← c15764, ← c15765, ← c15766] at e_240_18
  have e_240_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_19
  simp only [← c15767, ← c15768, ← c15769] at e_240_19
  have e_242_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_7
  simp only [← c15758, k3, ← c15759, k3, ← c15760] at e_242_7
  have e_242_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_8
  simp only [← c15761, ← c15762, k3, ← c15763] at e_242_8
  have e_242_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row242 (by decide)
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
  have e_242_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_10
  simp only [← c15775, k3, ← c15776, k3, ← c15777] at e_242_10
  have e_242_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_11
  simp only [← c15778, ← c15779, k3, ← c15780] at e_242_11
  have e_242_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_12
  simp only [← c15787, ← c15788, k3, ← c15789] at e_242_12
  have e_243_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_0
  simp only [← c15781, ← c15782, ← c15783] at e_243_0
  have e_243_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row243 (by decide)
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
  have e_242_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_13
  simp only [← c15792, k3, ← c15793, k3, ← c15794] at e_242_13
  have e_242_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_14
  simp only [← c15795, ← c15796, k3, ← c15797] at e_242_14
  have e_242_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_15
  simp only [← c15804, ← c15805, k3, ← c15806] at e_242_15
  have e_243_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_2
  simp only [← c15798, ← c15799, ← c15800] at e_243_2
  have e_243_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row243 (by decide)
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
  have e_243_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row243 (by decide)
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
  have e_243_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row243 (by decide)
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
  have e_243_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row243 (by decide)
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
  have e_236_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_12
  simp only [← c15821, ← c15822, k3, ← c15823] at e_236_12
  have e_241_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row241 (by decide)
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
  have e_242_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_16
  simp only [← c15825, k3, ← c15826, k3, ← c15827] at e_242_16
  have e_242_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_17
  simp only [← c15828, ← c15829, k3, ← c15830] at e_242_17
  have e_242_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_18
  simp only [← c15837, ← c15838, k3, ← c15839] at e_242_18
  have e_243_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_7
  simp only [← c15831, ← c15832, ← c15833] at e_243_7
  have e_243_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row243 (by decide)
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
  have e_236_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_13
  simp only [← c15845, ← c15846, k3, ← c15847] at e_236_13
  have e_241_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row241 (by decide)
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
  have e_242_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_19
  simp only [← c15849, k3, ← c15850, k3, ← c15851] at e_242_19
  have e_243_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_9
  simp only [← c15855, ← c15856, ← c15857] at e_243_9
  have e_243_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_10
  simp only [← c15858, ← c15859, ← c15860] at e_243_10
  have e_244_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_0
  simp only [← c15852, ← c15853, k3, ← c15854] at e_244_0
  have e_244_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row244 (by decide)
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
  have e_236_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_14
  simp only [← c15869, ← c15870, k3, ← c15871] at e_236_14
  have e_241_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row241 (by decide)
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
  have e_243_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_11
  simp only [← c15879, ← c15880, ← c15881] at e_243_11
  have e_243_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_12
  simp only [← c15882, ← c15883, ← c15884] at e_243_12
  have e_244_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_2
  simp only [← c15873, k3, ← c15874, k3, ← c15875] at e_244_2
  have e_244_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_3
  simp only [← c15876, ← c15877, k3, ← c15878] at e_244_3
  have e_244_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row244 (by decide)
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
  have e_243_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_13
  simp only [← c15896, ← c15897, ← c15898] at e_243_13
  have e_243_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_14
  simp only [← c15899, ← c15900, ← c15901] at e_243_14
  have e_244_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_5
  simp only [← c15890, k3, ← c15891, k3, ← c15892] at e_244_5
  have e_244_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_6
  simp only [← c15893, ← c15894, k3, ← c15895] at e_244_6
  have e_244_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row244 (by decide)
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
  have e_243_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_15
  simp only [← c15913, ← c15914, ← c15915] at e_243_15
  have e_243_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_16
  simp only [← c15916, ← c15917, ← c15918] at e_243_16
  have e_244_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_8
  simp only [← c15907, k3, ← c15908, k3, ← c15909] at e_244_8
  have e_244_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_9
  simp only [← c15910, ← c15911, k3, ← c15912] at e_244_9
  have e_244_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row244 (by decide)
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
  have e_243_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_17
  simp only [← c15930, ← c15931, ← c15932] at e_243_17
  have e_243_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_18
  simp only [← c15933, ← c15934, ← c15935] at e_243_18
  have e_244_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_11
  simp only [← c15924, k3, ← c15925, k3, ← c15926] at e_244_11
  have e_244_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_12
  simp only [← c15927, ← c15928, k3, ← c15929] at e_244_12
  have e_244_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row244 (by decide)
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
  have e_243_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row243 (by decide)
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
  have e_245_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row245 (by decide)
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
  have e_245_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row245 (by decide)
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
  have e_236_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_15
  simp only [← c15953, ← c15954, k3, ← c15955] at e_236_15
  have e_241_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row241 (by decide)
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

theorem publicBatchWrapper86_f2249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8203)) (a (.wire 165 55)) (a (.virt 18387)) (a (.virt 18388)) := by
  have c15957 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15958 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15959 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15960 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15961 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15962 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15963 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15964 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15965 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15966 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15967 := (publicBatchWrapper86_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15968 := (publicBatchWrapper86_copies499 a h).1
  have c15969 := (publicBatchWrapper86_copies499 a h).2.1
  have c15970 := (publicBatchWrapper86_copies499 a h).2.2.1
  have c15971 := (publicBatchWrapper86_copies499 a h).2.2.2.1
  have c15972 := (publicBatchWrapper86_copies499 a h).2.2.2.2.1
  have c15973 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_244_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_14
  simp only [← c15957, k3, ← c15958, k3, ← c15959] at e_244_14
  have e_244_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_15
  simp only [← c15960, ← c15961, k3, ← c15962] at e_244_15
  have e_244_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_16
  simp only [← c15969, ← c15970, k3, ← c15971] at e_244_16
  have e_245_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_2
  simp only [← c15963, ← c15964, ← c15965] at e_245_2
  have e_245_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_3
  simp only [← c15966, ← c15967, ← c15968] at e_245_3
  refine ⟨?_, ?_⟩
  · have hc := e_245_2
    simp only [e_244_15] at hc
    linear_combination c15972.trans k2 - hc
  · have hc := e_244_16
    simp only [e_244_14, e_245_3, e_244_15] at hc
    linear_combination c15973.trans k2 - hc

theorem publicBatchWrapper86_f2250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 67) = bor (a (.wire 41 71)) (a (.virt 18387)) := by
  have c15974 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.1
  have c15975 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.1
  have c15976 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.1
  have c15977 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.1
  have c15978 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15979 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_16
  simp only [← c15977, ← c15978, k3, ← c15979] at e_236_16
  have e_241_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_8
  simp only [← c15974, ← c15975, ← c15976] at e_241_8
  have hr := e_236_16
  simp only [e_241_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 67) = a (.virt 201) := by
  have c15980 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15980

theorem publicBatchWrapper86_f2252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8204)) (a (.wire 165 63)) (a (.virt 18389)) (a (.virt 18390)) := by
  have c15981 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15982 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15983 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15984 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15985 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15986 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15987 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15988 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15989 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15990 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15991 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15992 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15993 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15994 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15995 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15996 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15997 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_244_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_17
  simp only [← c15981, k3, ← c15982, k3, ← c15983] at e_244_17
  have e_244_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_18
  simp only [← c15984, ← c15985, k3, ← c15986] at e_244_18
  have e_244_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_19
  simp only [← c15993, ← c15994, k3, ← c15995] at e_244_19
  have e_245_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_4
  simp only [← c15987, ← c15988, ← c15989] at e_245_4
  have e_245_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_5
  simp only [← c15990, ← c15991, ← c15992] at e_245_5
  refine ⟨?_, ?_⟩
  · have hc := e_245_4
    simp only [e_244_18] at hc
    linear_combination c15996.trans k2 - hc
  · have hc := e_244_19
    simp only [e_244_17, e_245_5, e_244_18] at hc
    linear_combination c15997.trans k2 - hc

theorem publicBatchWrapper86_f2253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 71) = bor (a (.wire 41 71)) (a (.virt 18389)) := by
  have c15998 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15999 := (publicBatchWrapper86_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16000 := (publicBatchWrapper86_copies500 a h).1
  have c16001 := (publicBatchWrapper86_copies500 a h).2.1
  have c16002 := (publicBatchWrapper86_copies500 a h).2.2.1
  have c16003 := (publicBatchWrapper86_copies500 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_17
  simp only [← c16001, ← c16002, k3, ← c16003] at e_236_17
  have e_241_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_9
  simp only [← c15998, ← c15999, ← c16000] at e_241_9
  have hr := e_236_17
  simp only [e_241_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 71) = a (.virt 201) := by
  have c16004 := (publicBatchWrapper86_copies500 a h).2.2.2.2.1
  exact c16004

theorem publicBatchWrapper86_f2255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8205)) (a (.wire 165 15)) (a (.virt 18391)) (a (.virt 18392)) := by
  have c16005 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.1
  have c16006 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.1
  have c16007 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.1
  have c16008 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.1
  have c16009 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.1
  have c16010 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16011 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16012 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16013 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16014 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16015 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16016 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16017 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16018 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16019 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16020 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16021 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_245_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_6
  simp only [← c16011, ← c16012, ← c16013] at e_245_6
  have e_245_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_7
  simp only [← c16014, ← c16015, ← c16016] at e_245_7
  have e_246_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_0
  simp only [← c16005, k3, ← c16006, k3, ← c16007] at e_246_0
  have e_246_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_1
  simp only [← c16008, ← c16009, k3, ← c16010] at e_246_1
  have e_246_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_2
  simp only [← c16017, ← c16018, k3, ← c16019] at e_246_2
  refine ⟨?_, ?_⟩
  · have hc := e_245_6
    simp only [e_246_1] at hc
    linear_combination c16020.trans k2 - hc
  · have hc := e_246_2
    simp only [e_246_0, e_245_7, e_246_1] at hc
    linear_combination c16021.trans k2 - hc

theorem publicBatchWrapper86_f2256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8206)) (a (.wire 165 23)) (a (.virt 18393)) (a (.virt 18394)) := by
  have c16022 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16023 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16024 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16025 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16026 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16027 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16028 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16029 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16030 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16031 := (publicBatchWrapper86_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16032 := (publicBatchWrapper86_copies501 a h).1
  have c16033 := (publicBatchWrapper86_copies501 a h).2.1
  have c16034 := (publicBatchWrapper86_copies501 a h).2.2.1
  have c16035 := (publicBatchWrapper86_copies501 a h).2.2.2.1
  have c16036 := (publicBatchWrapper86_copies501 a h).2.2.2.2.1
  have c16037 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.1
  have c16038 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_245_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_8
  simp only [← c16028, ← c16029, ← c16030] at e_245_8
  have e_245_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_9
  simp only [← c16031, ← c16032, ← c16033] at e_245_9
  have e_246_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_3
  simp only [← c16022, k3, ← c16023, k3, ← c16024] at e_246_3
  have e_246_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_4
  simp only [← c16025, ← c16026, k3, ← c16027] at e_246_4
  have e_246_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_5
  simp only [← c16034, ← c16035, k3, ← c16036] at e_246_5
  refine ⟨?_, ?_⟩
  · have hc := e_245_8
    simp only [e_246_4] at hc
    linear_combination c16037.trans k2 - hc
  · have hc := e_246_5
    simp only [e_246_3, e_245_9, e_246_4] at hc
    linear_combination c16038.trans k2 - hc

theorem publicBatchWrapper86_f2257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8207)) (a (.wire 165 31)) (a (.virt 18395)) (a (.virt 18396)) := by
  have c16039 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.1
  have c16040 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.1
  have c16041 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.1
  have c16042 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16043 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16044 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16045 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16046 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16047 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16048 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16049 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16050 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16051 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16052 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16053 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16054 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16055 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_245_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_10
  simp only [← c16045, ← c16046, ← c16047] at e_245_10
  have e_245_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_11
  simp only [← c16048, ← c16049, ← c16050] at e_245_11
  have e_246_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_6
  simp only [← c16039, k3, ← c16040, k3, ← c16041] at e_246_6
  have e_246_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_7
  simp only [← c16042, ← c16043, k3, ← c16044] at e_246_7
  have e_246_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_8
  simp only [← c16051, ← c16052, k3, ← c16053] at e_246_8
  refine ⟨?_, ?_⟩
  · have hc := e_245_10
    simp only [e_246_7] at hc
    linear_combination c16054.trans k2 - hc
  · have hc := e_246_8
    simp only [e_246_6, e_245_11, e_246_7] at hc
    linear_combination c16055.trans k2 - hc

theorem publicBatchWrapper86_f2258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8208)) (a (.wire 165 39)) (a (.virt 18397)) (a (.virt 18398)) := by
  have c16056 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16057 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16058 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16059 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16060 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16061 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16062 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16063 := (publicBatchWrapper86_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16064 := (publicBatchWrapper86_copies502 a h).1
  have c16065 := (publicBatchWrapper86_copies502 a h).2.1
  have c16066 := (publicBatchWrapper86_copies502 a h).2.2.1
  have c16067 := (publicBatchWrapper86_copies502 a h).2.2.2.1
  have c16068 := (publicBatchWrapper86_copies502 a h).2.2.2.2.1
  have c16069 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.1
  have c16070 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.1
  have c16071 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.1
  have c16072 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_245_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_12
  simp only [← c16062, ← c16063, ← c16064] at e_245_12
  have e_245_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_13
  simp only [← c16065, ← c16066, ← c16067] at e_245_13
  have e_246_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_9
  simp only [← c16056, k3, ← c16057, k3, ← c16058] at e_246_9
  have e_246_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_10
  simp only [← c16059, ← c16060, k3, ← c16061] at e_246_10
  have e_246_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_11
  simp only [← c16068, ← c16069, k3, ← c16070] at e_246_11
  refine ⟨?_, ?_⟩
  · have hc := e_245_12
    simp only [e_246_10] at hc
    linear_combination c16071.trans k2 - hc
  · have hc := e_246_11
    simp only [e_246_9, e_245_13, e_246_10] at hc
    linear_combination c16072.trans k2 - hc

theorem publicBatchWrapper86_f2259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 245 59) = band (a (.virt 18391)) (a (.virt 18393)) := by
  have c16073 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.1
  have c16074 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16075 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_245_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_14
  simp only [← c16073, ← c16074, ← c16075] at e_245_14
  have hr := e_245_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 245 63) = band (a (.virt 18395)) (a (.virt 18397)) := by
  have c16076 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16077 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16078 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_245_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_15
  simp only [← c16076, ← c16077, ← c16078] at e_245_15
  have hr := e_245_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 245 67) = band (a (.wire 245 59)) (a (.wire 245 63)) := by
  have c16079 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16080 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16081 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_245_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_16
  simp only [← c16079, ← c16080, ← c16081] at e_245_16
  have hr := e_245_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 75) = bor (a (.wire 41 71)) (a (.wire 245 67)) := by
  have c16082 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16083 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16084 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16085 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16086 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16087 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_18
  simp only [← c16085, ← c16086, k3, ← c16087] at e_236_18
  have e_241_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_10
  simp only [← c16082, ← c16083, ← c16084] at e_241_10
  have hr := e_236_18
  simp only [e_241_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 75) = a (.virt 201) := by
  have c16088 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16088

theorem publicBatchWrapper86_f2264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8403)) (a (.wire 165 55)) (a (.virt 18399)) (a (.virt 18400)) := by
  have c16089 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16090 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16091 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16092 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16093 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16094 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16095 := (publicBatchWrapper86_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16096 := (publicBatchWrapper86_copies503 a h).1
  have c16097 := (publicBatchWrapper86_copies503 a h).2.1
  have c16098 := (publicBatchWrapper86_copies503 a h).2.2.1
  have c16099 := (publicBatchWrapper86_copies503 a h).2.2.2.1
  have c16100 := (publicBatchWrapper86_copies503 a h).2.2.2.2.1
  have c16101 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.1
  have c16102 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.1
  have c16103 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.1
  have c16104 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.1
  have c16105 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_245_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_17
  simp only [← c16095, ← c16096, ← c16097] at e_245_17
  have e_245_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_18
  simp only [← c16098, ← c16099, ← c16100] at e_245_18
  have e_246_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_12
  simp only [← c16089, k3, ← c16090, k3, ← c16091] at e_246_12
  have e_246_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_13
  simp only [← c16092, ← c16093, k3, ← c16094] at e_246_13
  have e_246_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_14
  simp only [← c16101, ← c16102, k3, ← c16103] at e_246_14
  refine ⟨?_, ?_⟩
  · have hc := e_245_17
    simp only [e_246_13] at hc
    linear_combination c16104.trans k2 - hc
  · have hc := e_246_14
    simp only [e_246_12, e_245_18, e_246_13] at hc
    linear_combination c16105.trans k2 - hc

theorem publicBatchWrapper86_f2265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 79) = bor (a (.wire 42 35)) (a (.virt 18399)) := by
  have c16106 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16107 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16108 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16109 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16110 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16111 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_236_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_19
  simp only [← c16109, ← c16110, k3, ← c16111] at e_236_19
  have e_241_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_11
  simp only [← c16106, ← c16107, ← c16108] at e_241_11
  have hr := e_236_19
  simp only [e_241_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 236 79) = a (.virt 201) := by
  have c16112 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16112

theorem publicBatchWrapper86_f2267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8404)) (a (.wire 165 63)) (a (.virt 18401)) (a (.virt 18402)) := by
  have c16113 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16114 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16115 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16116 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16117 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16118 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16119 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16120 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16121 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16122 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16123 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16124 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16125 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16126 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16127 := (publicBatchWrapper86_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16128 := (publicBatchWrapper86_copies504 a h).1
  have c16129 := (publicBatchWrapper86_copies504 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_245_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_19
  simp only [← c16119, ← c16120, ← c16121] at e_245_19
  have e_246_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_15
  simp only [← c16113, k3, ← c16114, k3, ← c16115] at e_246_15
  have e_246_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_16
  simp only [← c16116, ← c16117, k3, ← c16118] at e_246_16
  have e_246_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_17
  simp only [← c16125, ← c16126, k3, ← c16127] at e_246_17
  have e_247_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_0
  simp only [← c16122, ← c16123, ← c16124] at e_247_0
  refine ⟨?_, ?_⟩
  · have hc := e_245_19
    simp only [e_246_16] at hc
    linear_combination c16128.trans k2 - hc
  · have hc := e_246_17
    simp only [e_246_15, e_247_0, e_246_16] at hc
    linear_combination c16129.trans k2 - hc

theorem publicBatchWrapper86_f2268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 3) = bor (a (.wire 42 35)) (a (.virt 18401)) := by
  have c16130 := (publicBatchWrapper86_copies504 a h).2.2.1
  have c16131 := (publicBatchWrapper86_copies504 a h).2.2.2.1
  have c16132 := (publicBatchWrapper86_copies504 a h).2.2.2.2.1
  have c16133 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.1
  have c16134 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.1
  have c16135 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_241_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_12
  simp only [← c16130, ← c16131, ← c16132] at e_241_12
  have e_248_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_0
  simp only [← c16133, ← c16134, k3, ← c16135] at e_248_0
  have hr := e_248_0
  simp only [e_241_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 3) = a (.virt 201) := by
  have c16136 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.1
  exact c16136

theorem publicBatchWrapper86_f2270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8405)) (a (.wire 165 15)) (a (.virt 18403)) (a (.virt 18404)) := by
  have c16137 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.1
  have c16138 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16139 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16140 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16141 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16142 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16143 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16144 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16145 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16146 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16147 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16148 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16149 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16150 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16151 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16152 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16153 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_246_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_18
  simp only [← c16137, k3, ← c16138, k3, ← c16139] at e_246_18
  have e_246_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_19
  simp only [← c16140, ← c16141, k3, ← c16142] at e_246_19
  have e_247_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_1
  simp only [← c16143, ← c16144, ← c16145] at e_247_1
  have e_247_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_2
  simp only [← c16146, ← c16147, ← c16148] at e_247_2
  have e_249_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_0
  simp only [← c16149, ← c16150, k3, ← c16151] at e_249_0
  refine ⟨?_, ?_⟩
  · have hc := e_247_1
    simp only [e_246_19] at hc
    linear_combination c16152.trans k2 - hc
  · have hc := e_249_0
    simp only [e_246_18, e_247_2, e_246_19] at hc
    linear_combination c16153.trans k2 - hc

theorem publicBatchWrapper86_f2271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8406)) (a (.wire 165 23)) (a (.virt 18405)) (a (.virt 18406)) := by
  have c16154 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16155 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16156 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16157 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16158 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16159 := (publicBatchWrapper86_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16160 := (publicBatchWrapper86_copies505 a h).1
  have c16161 := (publicBatchWrapper86_copies505 a h).2.1
  have c16162 := (publicBatchWrapper86_copies505 a h).2.2.1
  have c16163 := (publicBatchWrapper86_copies505 a h).2.2.2.1
  have c16164 := (publicBatchWrapper86_copies505 a h).2.2.2.2.1
  have c16165 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.1
  have c16166 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.1
  have c16167 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.1
  have c16168 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.1
  have c16169 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.1
  have c16170 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_247_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_3
  simp only [← c16160, ← c16161, ← c16162] at e_247_3
  have e_247_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_4
  simp only [← c16163, ← c16164, ← c16165] at e_247_4
  have e_249_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_1
  simp only [← c16154, k3, ← c16155, k3, ← c16156] at e_249_1
  have e_249_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_2
  simp only [← c16157, ← c16158, k3, ← c16159] at e_249_2
  have e_249_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_3
  simp only [← c16166, ← c16167, k3, ← c16168] at e_249_3
  refine ⟨?_, ?_⟩
  · have hc := e_247_3
    simp only [e_249_2] at hc
    linear_combination c16169.trans k2 - hc
  · have hc := e_249_3
    simp only [e_249_1, e_247_4, e_249_2] at hc
    linear_combination c16170.trans k2 - hc

theorem publicBatchWrapper86_f2272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8407)) (a (.wire 165 31)) (a (.virt 18407)) (a (.virt 18408)) := by
  have c16171 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16172 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16173 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16174 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16175 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16176 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16177 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16178 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16179 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16180 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16181 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16182 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16183 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16184 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16185 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16186 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16187 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_247_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_5
  simp only [← c16177, ← c16178, ← c16179] at e_247_5
  have e_247_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_6
  simp only [← c16180, ← c16181, ← c16182] at e_247_6
  have e_249_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_4
  simp only [← c16171, k3, ← c16172, k3, ← c16173] at e_249_4
  have e_249_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_5
  simp only [← c16174, ← c16175, k3, ← c16176] at e_249_5
  have e_249_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_6
  simp only [← c16183, ← c16184, k3, ← c16185] at e_249_6
  refine ⟨?_, ?_⟩
  · have hc := e_247_5
    simp only [e_249_5] at hc
    linear_combination c16186.trans k2 - hc
  · have hc := e_249_6
    simp only [e_249_4, e_247_6, e_249_5] at hc
    linear_combination c16187.trans k2 - hc

theorem publicBatchWrapper86_f2273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8408)) (a (.wire 165 39)) (a (.virt 18409)) (a (.virt 18410)) := by
  have c16188 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16189 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16190 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16191 := (publicBatchWrapper86_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16192 := (publicBatchWrapper86_copies506 a h).1
  have c16193 := (publicBatchWrapper86_copies506 a h).2.1
  have c16194 := (publicBatchWrapper86_copies506 a h).2.2.1
  have c16195 := (publicBatchWrapper86_copies506 a h).2.2.2.1
  have c16196 := (publicBatchWrapper86_copies506 a h).2.2.2.2.1
  have c16197 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.1
  have c16198 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.1
  have c16199 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.1
  have c16200 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.1
  have c16201 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.1
  have c16202 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16203 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16204 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_247_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_7
  simp only [← c16194, ← c16195, ← c16196] at e_247_7
  have e_247_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_8
  simp only [← c16197, ← c16198, ← c16199] at e_247_8
  have e_249_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_7
  simp only [← c16188, k3, ← c16189, k3, ← c16190] at e_249_7
  have e_249_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_8
  simp only [← c16191, ← c16192, k3, ← c16193] at e_249_8
  have e_249_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_9
  simp only [← c16200, ← c16201, k3, ← c16202] at e_249_9
  refine ⟨?_, ?_⟩
  · have hc := e_247_7
    simp only [e_249_8] at hc
    linear_combination c16203.trans k2 - hc
  · have hc := e_249_9
    simp only [e_249_7, e_247_8, e_249_8] at hc
    linear_combination c16204.trans k2 - hc

theorem publicBatchWrapper86_f2274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 247 39) = band (a (.virt 18403)) (a (.virt 18405)) := by
  have c16205 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16206 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16207 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_247_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_9
  simp only [← c16205, ← c16206, ← c16207] at e_247_9
  have hr := e_247_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 247 43) = band (a (.virt 18407)) (a (.virt 18409)) := by
  have c16208 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16209 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16210 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_247_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_10
  simp only [← c16208, ← c16209, ← c16210] at e_247_10
  have hr := e_247_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 247 47) = band (a (.wire 247 39)) (a (.wire 247 43)) := by
  have c16211 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16212 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16213 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_247_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_11
  simp only [← c16211, ← c16212, ← c16213] at e_247_11
  have hr := e_247_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 7) = bor (a (.wire 42 35)) (a (.wire 247 47)) := by
  have c16214 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16215 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16216 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16217 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16218 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16219 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_241_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_13
  simp only [← c16214, ← c16215, ← c16216] at e_241_13
  have e_248_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_1
  simp only [← c16217, ← c16218, k3, ← c16219] at e_248_1
  have hr := e_248_1
  simp only [e_241_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 7) = a (.virt 201) := by
  have c16220 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16220

theorem publicBatchWrapper86_f2279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8603)) (a (.wire 165 55)) (a (.virt 18411)) (a (.virt 18412)) := by
  have c16221 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16222 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16223 := (publicBatchWrapper86_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16224 := (publicBatchWrapper86_copies507 a h).1
  have c16225 := (publicBatchWrapper86_copies507 a h).2.1
  have c16226 := (publicBatchWrapper86_copies507 a h).2.2.1
  have c16227 := (publicBatchWrapper86_copies507 a h).2.2.2.1
  have c16228 := (publicBatchWrapper86_copies507 a h).2.2.2.2.1
  have c16229 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.1
  have c16230 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.1
  have c16231 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.1
  have c16232 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.1
  have c16233 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.1
  have c16234 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16235 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16236 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16237 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_247_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_12
  simp only [← c16227, ← c16228, ← c16229] at e_247_12
  have e_247_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_13
  simp only [← c16230, ← c16231, ← c16232] at e_247_13
  have e_249_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_10
  simp only [← c16221, k3, ← c16222, k3, ← c16223] at e_249_10
  have e_249_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_11
  simp only [← c16224, ← c16225, k3, ← c16226] at e_249_11
  have e_249_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_12
  simp only [← c16233, ← c16234, k3, ← c16235] at e_249_12
  refine ⟨?_, ?_⟩
  · have hc := e_247_12
    simp only [e_249_11] at hc
    linear_combination c16236.trans k2 - hc
  · have hc := e_249_12
    simp only [e_249_10, e_247_13, e_249_11] at hc
    linear_combination c16237.trans k2 - hc

theorem publicBatchWrapper86_f2280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 11) = bor (a (.wire 42 79)) (a (.virt 18411)) := by
  have c16238 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16239 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16240 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16241 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16242 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16243 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_241_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_14
  simp only [← c16238, ← c16239, ← c16240] at e_241_14
  have e_248_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_2
  simp only [← c16241, ← c16242, k3, ← c16243] at e_248_2
  have hr := e_248_2
  simp only [e_241_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 11) = a (.virt 201) := by
  have c16244 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16244

theorem publicBatchWrapper86_f2282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8604)) (a (.wire 165 63)) (a (.virt 18413)) (a (.virt 18414)) := by
  have c16245 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16246 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16247 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16248 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16249 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16250 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16251 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16252 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16253 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16254 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16255 := (publicBatchWrapper86_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16256 := (publicBatchWrapper86_copies508 a h).1
  have c16257 := (publicBatchWrapper86_copies508 a h).2.1
  have c16258 := (publicBatchWrapper86_copies508 a h).2.2.1
  have c16259 := (publicBatchWrapper86_copies508 a h).2.2.2.1
  have c16260 := (publicBatchWrapper86_copies508 a h).2.2.2.2.1
  have c16261 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_247_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_14
  simp only [← c16251, ← c16252, ← c16253] at e_247_14
  have e_247_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_15
  simp only [← c16254, ← c16255, ← c16256] at e_247_15
  have e_249_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_13
  simp only [← c16245, k3, ← c16246, k3, ← c16247] at e_249_13
  have e_249_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_14
  simp only [← c16248, ← c16249, k3, ← c16250] at e_249_14
  have e_249_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_15
  simp only [← c16257, ← c16258, k3, ← c16259] at e_249_15
  refine ⟨?_, ?_⟩
  · have hc := e_247_14
    simp only [e_249_14] at hc
    linear_combination c16260.trans k2 - hc
  · have hc := e_249_15
    simp only [e_249_13, e_247_15, e_249_14] at hc
    linear_combination c16261.trans k2 - hc

theorem publicBatchWrapper86_f2283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 15) = bor (a (.wire 42 79)) (a (.virt 18413)) := by
  have c16262 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.1
  have c16263 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.1
  have c16264 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.1
  have c16265 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.1
  have c16266 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16267 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_241_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_15
  simp only [← c16262, ← c16263, ← c16264] at e_241_15
  have e_248_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_3
  simp only [← c16265, ← c16266, k3, ← c16267] at e_248_3
  have hr := e_248_3
  simp only [e_241_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 15) = a (.virt 201) := by
  have c16268 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16268

theorem publicBatchWrapper86_f2285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8605)) (a (.wire 165 15)) (a (.virt 18415)) (a (.virt 18416)) := by
  have c16269 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16270 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16271 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16272 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16273 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16274 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16275 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16276 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16277 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16278 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16279 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16280 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16281 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16282 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16283 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16284 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16285 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_247_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_16
  simp only [← c16275, ← c16276, ← c16277] at e_247_16
  have e_247_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_17
  simp only [← c16278, ← c16279, ← c16280] at e_247_17
  have e_249_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_16
  simp only [← c16269, k3, ← c16270, k3, ← c16271] at e_249_16
  have e_249_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_17
  simp only [← c16272, ← c16273, k3, ← c16274] at e_249_17
  have e_249_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_18
  simp only [← c16281, ← c16282, k3, ← c16283] at e_249_18
  refine ⟨?_, ?_⟩
  · have hc := e_247_16
    simp only [e_249_17] at hc
    linear_combination c16284.trans k2 - hc
  · have hc := e_249_18
    simp only [e_249_16, e_247_17, e_249_17] at hc
    linear_combination c16285.trans k2 - hc

theorem publicBatchWrapper86_f2286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8606)) (a (.wire 165 23)) (a (.virt 18417)) (a (.virt 18418)) := by
  have c16286 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16287 := (publicBatchWrapper86_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16288 := (publicBatchWrapper86_copies509 a h).1
  have c16289 := (publicBatchWrapper86_copies509 a h).2.1
  have c16290 := (publicBatchWrapper86_copies509 a h).2.2.1
  have c16291 := (publicBatchWrapper86_copies509 a h).2.2.2.1
  have c16292 := (publicBatchWrapper86_copies509 a h).2.2.2.2.1
  have c16293 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.1
  have c16294 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.1
  have c16295 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.1
  have c16296 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.1
  have c16297 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.1
  have c16298 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16299 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16300 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16301 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16302 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_247_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_18
  simp only [← c16292, ← c16293, ← c16294] at e_247_18
  have e_247_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_19
  simp only [← c16295, ← c16296, ← c16297] at e_247_19
  have e_249_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_19
  simp only [← c16286, k3, ← c16287, k3, ← c16288] at e_249_19
  have e_250_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_0
  simp only [← c16289, ← c16290, k3, ← c16291] at e_250_0
  have e_250_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_1
  simp only [← c16298, ← c16299, k3, ← c16300] at e_250_1
  refine ⟨?_, ?_⟩
  · have hc := e_247_18
    simp only [e_250_0] at hc
    linear_combination c16301.trans k2 - hc
  · have hc := e_250_1
    simp only [e_249_19, e_247_19, e_250_0] at hc
    linear_combination c16302.trans k2 - hc

theorem publicBatchWrapper86_f2287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8607)) (a (.wire 165 31)) (a (.virt 18419)) (a (.virt 18420)) := by
  have c16303 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16304 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16305 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16306 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16307 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16308 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16309 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16310 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16311 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16312 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16313 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16314 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16315 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16316 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16317 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16318 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16319 := (publicBatchWrapper86_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_250_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_2
  simp only [← c16303, k3, ← c16304, k3, ← c16305] at e_250_2
  have e_250_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_3
  simp only [← c16306, ← c16307, k3, ← c16308] at e_250_3
  have e_250_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_4
  simp only [← c16315, ← c16316, k3, ← c16317] at e_250_4
  have e_251_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_0
  simp only [← c16309, ← c16310, ← c16311] at e_251_0
  have e_251_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_1
  simp only [← c16312, ← c16313, ← c16314] at e_251_1
  refine ⟨?_, ?_⟩
  · have hc := e_251_0
    simp only [e_250_3] at hc
    linear_combination c16318.trans k2 - hc
  · have hc := e_250_4
    simp only [e_250_2, e_251_1, e_250_3] at hc
    linear_combination c16319.trans k2 - hc

theorem publicBatchWrapper86_f2288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8608)) (a (.wire 165 39)) (a (.virt 18421)) (a (.virt 18422)) := by
  have c16320 := (publicBatchWrapper86_copies510 a h).1
  have c16321 := (publicBatchWrapper86_copies510 a h).2.1
  have c16322 := (publicBatchWrapper86_copies510 a h).2.2.1
  have c16323 := (publicBatchWrapper86_copies510 a h).2.2.2.1
  have c16324 := (publicBatchWrapper86_copies510 a h).2.2.2.2.1
  have c16325 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.1
  have c16326 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.1
  have c16327 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.1
  have c16328 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.1
  have c16329 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.1
  have c16330 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16331 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16332 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16333 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16334 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16335 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16336 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_250_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_5
  simp only [← c16320, k3, ← c16321, k3, ← c16322] at e_250_5
  have e_250_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_6
  simp only [← c16323, ← c16324, k3, ← c16325] at e_250_6
  have e_250_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_7
  simp only [← c16332, ← c16333, k3, ← c16334] at e_250_7
  have e_251_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_2
  simp only [← c16326, ← c16327, ← c16328] at e_251_2
  have e_251_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_3
  simp only [← c16329, ← c16330, ← c16331] at e_251_3
  refine ⟨?_, ?_⟩
  · have hc := e_251_2
    simp only [e_250_6] at hc
    linear_combination c16335.trans k2 - hc
  · have hc := e_250_7
    simp only [e_250_5, e_251_3, e_250_6] at hc
    linear_combination c16336.trans k2 - hc

theorem publicBatchWrapper86_f2289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 251 19) = band (a (.virt 18415)) (a (.virt 18417)) := by
  have c16337 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16338 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16339 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_251_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_4
  simp only [← c16337, ← c16338, ← c16339] at e_251_4
  have hr := e_251_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 251 23) = band (a (.virt 18419)) (a (.virt 18421)) := by
  have c16340 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16341 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16342 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_251_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_5
  simp only [← c16340, ← c16341, ← c16342] at e_251_5
  have hr := e_251_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 251 27) = band (a (.wire 251 19)) (a (.wire 251 23)) := by
  have c16343 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16344 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16345 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_251_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_6
  simp only [← c16343, ← c16344, ← c16345] at e_251_6
  have hr := e_251_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 19) = bor (a (.wire 42 79)) (a (.wire 251 27)) := by
  have c16346 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16347 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16348 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16349 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16350 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16351 := (publicBatchWrapper86_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_241_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_16
  simp only [← c16346, ← c16347, ← c16348] at e_241_16
  have e_248_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_4
  simp only [← c16349, ← c16350, k3, ← c16351] at e_248_4
  have hr := e_248_4
  simp only [e_241_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 19) = a (.virt 201) := by
  have c16352 := (publicBatchWrapper86_copies511 a h).1
  exact c16352

theorem publicBatchWrapper86_f2294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8803)) (a (.wire 165 55)) (a (.virt 18423)) (a (.virt 18424)) := by
  have c16353 := (publicBatchWrapper86_copies511 a h).2.1
  have c16354 := (publicBatchWrapper86_copies511 a h).2.2.1
  have c16355 := (publicBatchWrapper86_copies511 a h).2.2.2.1
  have c16356 := (publicBatchWrapper86_copies511 a h).2.2.2.2.1
  have c16357 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.1
  have c16358 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.1
  have c16359 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.1
  have c16360 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.1
  have c16361 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.1
  have c16362 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16363 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16364 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16365 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16366 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16367 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16368 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16369 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_250_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_8
  simp only [← c16353, k3, ← c16354, k3, ← c16355] at e_250_8
  have e_250_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_9
  simp only [← c16356, ← c16357, k3, ← c16358] at e_250_9
  have e_250_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_10
  simp only [← c16365, ← c16366, k3, ← c16367] at e_250_10
  have e_251_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_7
  simp only [← c16359, ← c16360, ← c16361] at e_251_7
  have e_251_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_8
  simp only [← c16362, ← c16363, ← c16364] at e_251_8
  refine ⟨?_, ?_⟩
  · have hc := e_251_7
    simp only [e_250_9] at hc
    linear_combination c16368.trans k2 - hc
  · have hc := e_250_10
    simp only [e_250_8, e_251_8, e_250_9] at hc
    linear_combination c16369.trans k2 - hc

theorem publicBatchWrapper86_f2295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 23) = bor (a (.wire 44 43)) (a (.virt 18423)) := by
  have c16370 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16371 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16372 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16373 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16374 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16375 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_241_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_17
  simp only [← c16370, ← c16371, ← c16372] at e_241_17
  have e_248_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_5
  simp only [← c16373, ← c16374, k3, ← c16375] at e_248_5
  have hr := e_248_5
  simp only [e_241_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 23) = a (.virt 201) := by
  have c16376 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16376

theorem publicBatchWrapper86_f2297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8804)) (a (.wire 165 63)) (a (.virt 18425)) (a (.virt 18426)) := by
  have c16377 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16378 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16379 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16380 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16381 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16382 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16383 := (publicBatchWrapper86_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16384 := (publicBatchWrapper86_copies512 a h).1
  have c16385 := (publicBatchWrapper86_copies512 a h).2.1
  have c16386 := (publicBatchWrapper86_copies512 a h).2.2.1
  have c16387 := (publicBatchWrapper86_copies512 a h).2.2.2.1
  have c16388 := (publicBatchWrapper86_copies512 a h).2.2.2.2.1
  have c16389 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.1
  have c16390 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.1
  have c16391 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.1
  have c16392 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.1
  have c16393 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_250_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_11
  simp only [← c16377, k3, ← c16378, k3, ← c16379] at e_250_11
  have e_250_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_12
  simp only [← c16380, ← c16381, k3, ← c16382] at e_250_12
  have e_250_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_13
  simp only [← c16389, ← c16390, k3, ← c16391] at e_250_13
  have e_251_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_9
  simp only [← c16383, ← c16384, ← c16385] at e_251_9
  have e_251_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_10
  simp only [← c16386, ← c16387, ← c16388] at e_251_10
  refine ⟨?_, ?_⟩
  · have hc := e_251_9
    simp only [e_250_12] at hc
    linear_combination c16392.trans k2 - hc
  · have hc := e_250_13
    simp only [e_250_11, e_251_10, e_250_12] at hc
    linear_combination c16393.trans k2 - hc

theorem publicBatchWrapper86_f2298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 27) = bor (a (.wire 44 43)) (a (.virt 18425)) := by
  have c16394 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16395 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16396 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16397 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16398 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16399 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_241_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_18
  simp only [← c16394, ← c16395, ← c16396] at e_241_18
  have e_248_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_6
  simp only [← c16397, ← c16398, k3, ← c16399] at e_248_6
  have hr := e_248_6
  simp only [e_241_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 27) = a (.virt 201) := by
  have c16400 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16400

theorem publicBatchWrapper86_f2300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8805)) (a (.wire 165 15)) (a (.virt 18427)) (a (.virt 18428)) := by
  have c16401 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16402 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16403 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16404 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16405 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16406 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16407 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16408 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16409 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16410 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16411 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16412 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16413 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16414 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16415 := (publicBatchWrapper86_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16416 := (publicBatchWrapper86_copies513 a h).1
  have c16417 := (publicBatchWrapper86_copies513 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_250_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_14
  simp only [← c16401, k3, ← c16402, k3, ← c16403] at e_250_14
  have e_250_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_15
  simp only [← c16404, ← c16405, k3, ← c16406] at e_250_15
  have e_250_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_16
  simp only [← c16413, ← c16414, k3, ← c16415] at e_250_16
  have e_251_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_11
  simp only [← c16407, ← c16408, ← c16409] at e_251_11
  have e_251_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_12
  simp only [← c16410, ← c16411, ← c16412] at e_251_12
  refine ⟨?_, ?_⟩
  · have hc := e_251_11
    simp only [e_250_15] at hc
    linear_combination c16416.trans k2 - hc
  · have hc := e_250_16
    simp only [e_250_14, e_251_12, e_250_15] at hc
    linear_combination c16417.trans k2 - hc

theorem publicBatchWrapper86_f2301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8806)) (a (.wire 165 23)) (a (.virt 18429)) (a (.virt 18430)) := by
  have c16418 := (publicBatchWrapper86_copies513 a h).2.2.1
  have c16419 := (publicBatchWrapper86_copies513 a h).2.2.2.1
  have c16420 := (publicBatchWrapper86_copies513 a h).2.2.2.2.1
  have c16421 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.1
  have c16422 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.1
  have c16423 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.1
  have c16424 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.1
  have c16425 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.1
  have c16426 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16427 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16428 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16429 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16430 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16431 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16432 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16433 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16434 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_250_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_17
  simp only [← c16418, k3, ← c16419, k3, ← c16420] at e_250_17
  have e_250_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_18
  simp only [← c16421, ← c16422, k3, ← c16423] at e_250_18
  have e_250_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_19
  simp only [← c16430, ← c16431, k3, ← c16432] at e_250_19
  have e_251_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_13
  simp only [← c16424, ← c16425, ← c16426] at e_251_13
  have e_251_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_14
  simp only [← c16427, ← c16428, ← c16429] at e_251_14
  refine ⟨?_, ?_⟩
  · have hc := e_251_13
    simp only [e_250_18] at hc
    linear_combination c16433.trans k2 - hc
  · have hc := e_250_19
    simp only [e_250_17, e_251_14, e_250_18] at hc
    linear_combination c16434.trans k2 - hc

theorem publicBatchWrapper86_f2302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8807)) (a (.wire 165 31)) (a (.virt 18431)) (a (.virt 18432)) := by
  have c16435 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16436 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16437 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16438 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16439 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16440 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16441 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16442 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16443 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16444 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16445 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16446 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16447 := (publicBatchWrapper86_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16448 := (publicBatchWrapper86_copies514 a h).1
  have c16449 := (publicBatchWrapper86_copies514 a h).2.1
  have c16450 := (publicBatchWrapper86_copies514 a h).2.2.1
  have c16451 := (publicBatchWrapper86_copies514 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_251_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_15
  simp only [← c16441, ← c16442, ← c16443] at e_251_15
  have e_251_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_16
  simp only [← c16444, ← c16445, ← c16446] at e_251_16
  have e_252_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_0
  simp only [← c16435, k3, ← c16436, k3, ← c16437] at e_252_0
  have e_252_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_1
  simp only [← c16438, ← c16439, k3, ← c16440] at e_252_1
  have e_252_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_2
  simp only [← c16447, ← c16448, k3, ← c16449] at e_252_2
  refine ⟨?_, ?_⟩
  · have hc := e_251_15
    simp only [e_252_1] at hc
    linear_combination c16450.trans k2 - hc
  · have hc := e_252_2
    simp only [e_252_0, e_251_16, e_252_1] at hc
    linear_combination c16451.trans k2 - hc

theorem publicBatchWrapper86_f2303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8808)) (a (.wire 165 39)) (a (.virt 18433)) (a (.virt 18434)) := by
  have c16452 := (publicBatchWrapper86_copies514 a h).2.2.2.2.1
  have c16453 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.1
  have c16454 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.1
  have c16455 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.1
  have c16456 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.1
  have c16457 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.1
  have c16458 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16459 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16460 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16461 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16462 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16463 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16464 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16465 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16466 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16467 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16468 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_251_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_17
  simp only [← c16458, ← c16459, ← c16460] at e_251_17
  have e_251_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_18
  simp only [← c16461, ← c16462, ← c16463] at e_251_18
  have e_252_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_3
  simp only [← c16452, k3, ← c16453, k3, ← c16454] at e_252_3
  have e_252_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_4
  simp only [← c16455, ← c16456, k3, ← c16457] at e_252_4
  have e_252_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_5
  simp only [← c16464, ← c16465, k3, ← c16466] at e_252_5
  refine ⟨?_, ?_⟩
  · have hc := e_251_17
    simp only [e_252_4] at hc
    linear_combination c16467.trans k2 - hc
  · have hc := e_252_5
    simp only [e_252_3, e_251_18, e_252_4] at hc
    linear_combination c16468.trans k2 - hc

theorem publicBatchWrapper86_f2304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 251 79) = band (a (.virt 18427)) (a (.virt 18429)) := by
  have c16469 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16470 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16471 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_251_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_19
  simp only [← c16469, ← c16470, ← c16471] at e_251_19
  have hr := e_251_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 253 3) = band (a (.virt 18431)) (a (.virt 18433)) := by
  have c16472 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16473 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16474 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_253_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_0
  simp only [← c16472, ← c16473, ← c16474] at e_253_0
  have hr := e_253_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 253 7) = band (a (.wire 251 79)) (a (.wire 253 3)) := by
  have c16475 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16476 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16477 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_253_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_1
  simp only [← c16475, ← c16476, ← c16477] at e_253_1
  have hr := e_253_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 31) = bor (a (.wire 44 43)) (a (.wire 253 7)) := by
  have c16478 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16479 := (publicBatchWrapper86_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16480 := (publicBatchWrapper86_copies515 a h).1
  have c16481 := (publicBatchWrapper86_copies515 a h).2.1
  have c16482 := (publicBatchWrapper86_copies515 a h).2.2.1
  have c16483 := (publicBatchWrapper86_copies515 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_241_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_19
  simp only [← c16478, ← c16479, ← c16480] at e_241_19
  have e_248_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_7
  simp only [← c16481, ← c16482, k3, ← c16483] at e_248_7
  have hr := e_248_7
  simp only [e_241_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 31) = a (.virt 201) := by
  have c16484 := (publicBatchWrapper86_copies515 a h).2.2.2.2.1
  exact c16484

theorem publicBatchWrapper86_f2309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9003)) (a (.wire 165 55)) (a (.virt 18435)) (a (.virt 18436)) := by
  have c16485 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.1
  have c16486 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.1
  have c16487 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.1
  have c16488 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.1
  have c16489 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.1
  have c16490 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16491 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16492 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16493 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16494 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16495 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16496 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16497 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16498 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16499 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16500 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16501 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_252_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_6
  simp only [← c16485, k3, ← c16486, k3, ← c16487] at e_252_6
  have e_252_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_7
  simp only [← c16488, ← c16489, k3, ← c16490] at e_252_7
  have e_252_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_8
  simp only [← c16497, ← c16498, k3, ← c16499] at e_252_8
  have e_253_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_2
  simp only [← c16491, ← c16492, ← c16493] at e_253_2
  have e_253_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_3
  simp only [← c16494, ← c16495, ← c16496] at e_253_3
  refine ⟨?_, ?_⟩
  · have hc := e_253_2
    simp only [e_252_7] at hc
    linear_combination c16500.trans k2 - hc
  · have hc := e_252_8
    simp only [e_252_6, e_253_3, e_252_7] at hc
    linear_combination c16501.trans k2 - hc

theorem publicBatchWrapper86_f2310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 35) = bor (a (.wire 46 7)) (a (.virt 18435)) := by
  have c16502 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16503 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16504 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16505 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16506 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16507 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_8
  simp only [← c16505, ← c16506, k3, ← c16507] at e_248_8
  have e_254_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_0
  simp only [← c16502, ← c16503, ← c16504] at e_254_0
  have hr := e_248_8
  simp only [e_254_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 35) = a (.virt 201) := by
  have c16508 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16508

theorem publicBatchWrapper86_f2312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9004)) (a (.wire 165 63)) (a (.virt 18437)) (a (.virt 18438)) := by
  have c16509 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16510 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16511 := (publicBatchWrapper86_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16512 := (publicBatchWrapper86_copies516 a h).1
  have c16513 := (publicBatchWrapper86_copies516 a h).2.1
  have c16514 := (publicBatchWrapper86_copies516 a h).2.2.1
  have c16515 := (publicBatchWrapper86_copies516 a h).2.2.2.1
  have c16516 := (publicBatchWrapper86_copies516 a h).2.2.2.2.1
  have c16517 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.1
  have c16518 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.1
  have c16519 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.1
  have c16520 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.1
  have c16521 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.1
  have c16522 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16523 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16524 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16525 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_252_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_9
  simp only [← c16509, k3, ← c16510, k3, ← c16511] at e_252_9
  have e_252_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_10
  simp only [← c16512, ← c16513, k3, ← c16514] at e_252_10
  have e_252_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_11
  simp only [← c16521, ← c16522, k3, ← c16523] at e_252_11
  have e_253_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_4
  simp only [← c16515, ← c16516, ← c16517] at e_253_4
  have e_253_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_5
  simp only [← c16518, ← c16519, ← c16520] at e_253_5
  refine ⟨?_, ?_⟩
  · have hc := e_253_4
    simp only [e_252_10] at hc
    linear_combination c16524.trans k2 - hc
  · have hc := e_252_11
    simp only [e_252_9, e_253_5, e_252_10] at hc
    linear_combination c16525.trans k2 - hc

theorem publicBatchWrapper86_f2313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 39) = bor (a (.wire 46 7)) (a (.virt 18437)) := by
  have c16526 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16527 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16528 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16529 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16530 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16531 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_9
  simp only [← c16529, ← c16530, k3, ← c16531] at e_248_9
  have e_254_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_1
  simp only [← c16526, ← c16527, ← c16528] at e_254_1
  have hr := e_248_9
  simp only [e_254_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 39) = a (.virt 201) := by
  have c16532 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16532

theorem publicBatchWrapper86_f2315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9005)) (a (.wire 165 15)) (a (.virt 18439)) (a (.virt 18440)) := by
  have c16533 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16534 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16535 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16536 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16537 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16538 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16539 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16540 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16541 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16542 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16543 := (publicBatchWrapper86_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16544 := (publicBatchWrapper86_copies517 a h).1
  have c16545 := (publicBatchWrapper86_copies517 a h).2.1
  have c16546 := (publicBatchWrapper86_copies517 a h).2.2.1
  have c16547 := (publicBatchWrapper86_copies517 a h).2.2.2.1
  have c16548 := (publicBatchWrapper86_copies517 a h).2.2.2.2.1
  have c16549 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_252_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_12
  simp only [← c16533, k3, ← c16534, k3, ← c16535] at e_252_12
  have e_252_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_13
  simp only [← c16536, ← c16537, k3, ← c16538] at e_252_13
  have e_252_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_14
  simp only [← c16545, ← c16546, k3, ← c16547] at e_252_14
  have e_253_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_6
  simp only [← c16539, ← c16540, ← c16541] at e_253_6
  have e_253_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_7
  simp only [← c16542, ← c16543, ← c16544] at e_253_7
  refine ⟨?_, ?_⟩
  · have hc := e_253_6
    simp only [e_252_13] at hc
    linear_combination c16548.trans k2 - hc
  · have hc := e_252_14
    simp only [e_252_12, e_253_7, e_252_13] at hc
    linear_combination c16549.trans k2 - hc

theorem publicBatchWrapper86_f2316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9006)) (a (.wire 165 23)) (a (.virt 18441)) (a (.virt 18442)) := by
  have c16550 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.1
  have c16551 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.1
  have c16552 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.1
  have c16553 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.1
  have c16554 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16555 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16556 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16557 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16558 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16559 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16560 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16561 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16562 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16563 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16564 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16565 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16566 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_252_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_15
  simp only [← c16550, k3, ← c16551, k3, ← c16552] at e_252_15
  have e_252_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_16
  simp only [← c16553, ← c16554, k3, ← c16555] at e_252_16
  have e_252_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_17
  simp only [← c16562, ← c16563, k3, ← c16564] at e_252_17
  have e_253_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_8
  simp only [← c16556, ← c16557, ← c16558] at e_253_8
  have e_253_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_9
  simp only [← c16559, ← c16560, ← c16561] at e_253_9
  refine ⟨?_, ?_⟩
  · have hc := e_253_8
    simp only [e_252_16] at hc
    linear_combination c16565.trans k2 - hc
  · have hc := e_252_17
    simp only [e_252_15, e_253_9, e_252_16] at hc
    linear_combination c16566.trans k2 - hc

theorem publicBatchWrapper86_f2317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9007)) (a (.wire 165 31)) (a (.virt 18443)) (a (.virt 18444)) := by
  have c16567 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16568 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16569 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16570 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16571 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16572 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16573 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16574 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16575 := (publicBatchWrapper86_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16576 := (publicBatchWrapper86_copies518 a h).1
  have c16577 := (publicBatchWrapper86_copies518 a h).2.1
  have c16578 := (publicBatchWrapper86_copies518 a h).2.2.1
  have c16579 := (publicBatchWrapper86_copies518 a h).2.2.2.1
  have c16580 := (publicBatchWrapper86_copies518 a h).2.2.2.2.1
  have c16581 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.1
  have c16582 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.1
  have c16583 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_252_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_18
  simp only [← c16567, k3, ← c16568, k3, ← c16569] at e_252_18
  have e_252_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_19
  simp only [← c16570, ← c16571, k3, ← c16572] at e_252_19
  have e_253_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_10
  simp only [← c16573, ← c16574, ← c16575] at e_253_10
  have e_253_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_11
  simp only [← c16576, ← c16577, ← c16578] at e_253_11
  have e_255_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_0
  simp only [← c16579, ← c16580, k3, ← c16581] at e_255_0
  refine ⟨?_, ?_⟩
  · have hc := e_253_10
    simp only [e_252_19] at hc
    linear_combination c16582.trans k2 - hc
  · have hc := e_255_0
    simp only [e_252_18, e_253_11, e_252_19] at hc
    linear_combination c16583.trans k2 - hc

theorem publicBatchWrapper86_f2318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9008)) (a (.wire 165 39)) (a (.virt 18445)) (a (.virt 18446)) := by
  have c16584 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.1
  have c16585 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.1
  have c16586 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16587 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16588 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16589 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16590 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16591 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16592 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16593 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16594 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16595 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16596 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16597 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16598 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16599 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16600 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_253_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_12
  simp only [← c16590, ← c16591, ← c16592] at e_253_12
  have e_253_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_13
  simp only [← c16593, ← c16594, ← c16595] at e_253_13
  have e_255_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_1
  simp only [← c16584, k3, ← c16585, k3, ← c16586] at e_255_1
  have e_255_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_2
  simp only [← c16587, ← c16588, k3, ← c16589] at e_255_2
  have e_255_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_3
  simp only [← c16596, ← c16597, k3, ← c16598] at e_255_3
  refine ⟨?_, ?_⟩
  · have hc := e_253_12
    simp only [e_255_2] at hc
    linear_combination c16599.trans k2 - hc
  · have hc := e_255_3
    simp only [e_255_1, e_253_13, e_255_2] at hc
    linear_combination c16600.trans k2 - hc

theorem publicBatchWrapper86_f2319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 253 59) = band (a (.virt 18439)) (a (.virt 18441)) := by
  have c16601 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16602 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16603 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_253_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_14
  simp only [← c16601, ← c16602, ← c16603] at e_253_14
  have hr := e_253_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 253 63) = band (a (.virt 18443)) (a (.virt 18445)) := by
  have c16604 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16605 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16606 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_253_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_15
  simp only [← c16604, ← c16605, ← c16606] at e_253_15
  have hr := e_253_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 253 67) = band (a (.wire 253 59)) (a (.wire 253 63)) := by
  have c16607 := (publicBatchWrapper86_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16608 := (publicBatchWrapper86_copies519 a h).1
  have c16609 := (publicBatchWrapper86_copies519 a h).2.1
  have e_253_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_16
  simp only [← c16607, ← c16608, ← c16609] at e_253_16
  have hr := e_253_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 43) = bor (a (.wire 46 7)) (a (.wire 253 67)) := by
  have c16610 := (publicBatchWrapper86_copies519 a h).2.2.1
  have c16611 := (publicBatchWrapper86_copies519 a h).2.2.2.1
  have c16612 := (publicBatchWrapper86_copies519 a h).2.2.2.2.1
  have c16613 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.1
  have c16614 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.1
  have c16615 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_10
  simp only [← c16613, ← c16614, k3, ← c16615] at e_248_10
  have e_254_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_2
  simp only [← c16610, ← c16611, ← c16612] at e_254_2
  have hr := e_248_10
  simp only [e_254_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 43) = a (.virt 201) := by
  have c16616 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.1
  exact c16616

theorem publicBatchWrapper86_f2324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9203)) (a (.wire 165 55)) (a (.virt 18447)) (a (.virt 18448)) := by
  have c16617 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.1
  have c16618 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16619 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16620 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16621 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16622 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16623 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16624 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16625 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16626 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16627 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16628 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16629 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16630 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16631 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16632 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16633 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_253_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_17
  simp only [← c16623, ← c16624, ← c16625] at e_253_17
  have e_253_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_18
  simp only [← c16626, ← c16627, ← c16628] at e_253_18
  have e_255_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_4
  simp only [← c16617, k3, ← c16618, k3, ← c16619] at e_255_4
  have e_255_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_5
  simp only [← c16620, ← c16621, k3, ← c16622] at e_255_5
  have e_255_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_6
  simp only [← c16629, ← c16630, k3, ← c16631] at e_255_6
  refine ⟨?_, ?_⟩
  · have hc := e_253_17
    simp only [e_255_5] at hc
    linear_combination c16632.trans k2 - hc
  · have hc := e_255_6
    simp only [e_255_4, e_253_18, e_255_5] at hc
    linear_combination c16633.trans k2 - hc

theorem publicBatchWrapper86_f2325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 47) = bor (a (.wire 46 51)) (a (.virt 18447)) := by
  have c16634 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16635 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16636 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16637 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16638 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16639 := (publicBatchWrapper86_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_11
  simp only [← c16637, ← c16638, k3, ← c16639] at e_248_11
  have e_254_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_3
  simp only [← c16634, ← c16635, ← c16636] at e_254_3
  have hr := e_248_11
  simp only [e_254_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 47) = a (.virt 201) := by
  have c16640 := (publicBatchWrapper86_copies520 a h).1
  exact c16640

theorem publicBatchWrapper86_f2327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9204)) (a (.wire 165 63)) (a (.virt 18449)) (a (.virt 18450)) := by
  have c16641 := (publicBatchWrapper86_copies520 a h).2.1
  have c16642 := (publicBatchWrapper86_copies520 a h).2.2.1
  have c16643 := (publicBatchWrapper86_copies520 a h).2.2.2.1
  have c16644 := (publicBatchWrapper86_copies520 a h).2.2.2.2.1
  have c16645 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.1
  have c16646 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.1
  have c16647 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.1
  have c16648 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.1
  have c16649 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.1
  have c16650 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16651 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16652 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16653 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16654 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16655 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16656 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16657 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_253_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_19
  simp only [← c16647, ← c16648, ← c16649] at e_253_19
  have e_255_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_7
  simp only [← c16641, k3, ← c16642, k3, ← c16643] at e_255_7
  have e_255_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_8
  simp only [← c16644, ← c16645, k3, ← c16646] at e_255_8
  have e_255_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_9
  simp only [← c16653, ← c16654, k3, ← c16655] at e_255_9
  have e_256_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_0
  simp only [← c16650, ← c16651, ← c16652] at e_256_0
  refine ⟨?_, ?_⟩
  · have hc := e_253_19
    simp only [e_255_8] at hc
    linear_combination c16656.trans k2 - hc
  · have hc := e_255_9
    simp only [e_255_7, e_256_0, e_255_8] at hc
    linear_combination c16657.trans k2 - hc

theorem publicBatchWrapper86_f2328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 51) = bor (a (.wire 46 51)) (a (.virt 18449)) := by
  have c16658 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16659 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16660 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16661 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16662 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16663 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_12
  simp only [← c16661, ← c16662, k3, ← c16663] at e_248_12
  have e_254_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_4
  simp only [← c16658, ← c16659, ← c16660] at e_254_4
  have hr := e_248_12
  simp only [e_254_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 51) = a (.virt 201) := by
  have c16664 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16664

theorem publicBatchWrapper86_f2330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9205)) (a (.wire 165 15)) (a (.virt 18451)) (a (.virt 18452)) := by
  have c16665 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16666 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16667 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16668 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16669 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16670 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16671 := (publicBatchWrapper86_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16672 := (publicBatchWrapper86_copies521 a h).1
  have c16673 := (publicBatchWrapper86_copies521 a h).2.1
  have c16674 := (publicBatchWrapper86_copies521 a h).2.2.1
  have c16675 := (publicBatchWrapper86_copies521 a h).2.2.2.1
  have c16676 := (publicBatchWrapper86_copies521 a h).2.2.2.2.1
  have c16677 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.1
  have c16678 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.1
  have c16679 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.1
  have c16680 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.1
  have c16681 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_255_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_10
  simp only [← c16665, k3, ← c16666, k3, ← c16667] at e_255_10
  have e_255_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_11
  simp only [← c16668, ← c16669, k3, ← c16670] at e_255_11
  have e_255_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_12
  simp only [← c16677, ← c16678, k3, ← c16679] at e_255_12
  have e_256_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_1
  simp only [← c16671, ← c16672, ← c16673] at e_256_1
  have e_256_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_2
  simp only [← c16674, ← c16675, ← c16676] at e_256_2
  refine ⟨?_, ?_⟩
  · have hc := e_256_1
    simp only [e_255_11] at hc
    linear_combination c16680.trans k2 - hc
  · have hc := e_255_12
    simp only [e_255_10, e_256_2, e_255_11] at hc
    linear_combination c16681.trans k2 - hc

theorem publicBatchWrapper86_f2331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9206)) (a (.wire 165 23)) (a (.virt 18453)) (a (.virt 18454)) := by
  have c16682 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16683 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16684 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16685 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16686 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16687 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16688 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16689 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16690 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16691 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16692 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16693 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16694 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16695 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16696 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16697 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16698 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_255_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_13
  simp only [← c16682, k3, ← c16683, k3, ← c16684] at e_255_13
  have e_255_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_14
  simp only [← c16685, ← c16686, k3, ← c16687] at e_255_14
  have e_255_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_15
  simp only [← c16694, ← c16695, k3, ← c16696] at e_255_15
  have e_256_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_3
  simp only [← c16688, ← c16689, ← c16690] at e_256_3
  have e_256_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_4
  simp only [← c16691, ← c16692, ← c16693] at e_256_4
  refine ⟨?_, ?_⟩
  · have hc := e_256_3
    simp only [e_255_14] at hc
    linear_combination c16697.trans k2 - hc
  · have hc := e_255_15
    simp only [e_255_13, e_256_4, e_255_14] at hc
    linear_combination c16698.trans k2 - hc

theorem publicBatchWrapper86_f2332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9207)) (a (.wire 165 31)) (a (.virt 18455)) (a (.virt 18456)) := by
  have c16699 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16700 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16701 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16702 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16703 := (publicBatchWrapper86_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16704 := (publicBatchWrapper86_copies522 a h).1
  have c16705 := (publicBatchWrapper86_copies522 a h).2.1
  have c16706 := (publicBatchWrapper86_copies522 a h).2.2.1
  have c16707 := (publicBatchWrapper86_copies522 a h).2.2.2.1
  have c16708 := (publicBatchWrapper86_copies522 a h).2.2.2.2.1
  have c16709 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.1
  have c16710 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.1
  have c16711 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.1
  have c16712 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.1
  have c16713 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.1
  have c16714 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16715 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_255_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_16
  simp only [← c16699, k3, ← c16700, k3, ← c16701] at e_255_16
  have e_255_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_17
  simp only [← c16702, ← c16703, k3, ← c16704] at e_255_17
  have e_255_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_18
  simp only [← c16711, ← c16712, k3, ← c16713] at e_255_18
  have e_256_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_5
  simp only [← c16705, ← c16706, ← c16707] at e_256_5
  have e_256_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_6
  simp only [← c16708, ← c16709, ← c16710] at e_256_6
  refine ⟨?_, ?_⟩
  · have hc := e_256_5
    simp only [e_255_17] at hc
    linear_combination c16714.trans k2 - hc
  · have hc := e_255_18
    simp only [e_255_16, e_256_6, e_255_17] at hc
    linear_combination c16715.trans k2 - hc

theorem publicBatchWrapper86_f2333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9208)) (a (.wire 165 39)) (a (.virt 18457)) (a (.virt 18458)) := by
  have c16716 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16717 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16718 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16719 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16720 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16721 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16722 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16723 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16724 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16725 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16726 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16727 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16728 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16729 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16730 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16731 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16732 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_255_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_19
  simp only [← c16716, k3, ← c16717, k3, ← c16718] at e_255_19
  have e_256_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_7
  simp only [← c16722, ← c16723, ← c16724] at e_256_7
  have e_256_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_8
  simp only [← c16725, ← c16726, ← c16727] at e_256_8
  have e_257_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_0
  simp only [← c16719, ← c16720, k3, ← c16721] at e_257_0
  have e_257_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_1
  simp only [← c16728, ← c16729, k3, ← c16730] at e_257_1
  refine ⟨?_, ?_⟩
  · have hc := e_256_7
    simp only [e_257_0] at hc
    linear_combination c16731.trans k2 - hc
  · have hc := e_257_1
    simp only [e_255_19, e_256_8, e_257_0] at hc
    linear_combination c16732.trans k2 - hc

theorem publicBatchWrapper86_f2334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 256 39) = band (a (.virt 18451)) (a (.virt 18453)) := by
  have c16733 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16734 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16735 := (publicBatchWrapper86_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_256_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_9
  simp only [← c16733, ← c16734, ← c16735] at e_256_9
  have hr := e_256_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 256 43) = band (a (.virt 18455)) (a (.virt 18457)) := by
  have c16736 := (publicBatchWrapper86_copies523 a h).1
  have c16737 := (publicBatchWrapper86_copies523 a h).2.1
  have c16738 := (publicBatchWrapper86_copies523 a h).2.2.1
  have e_256_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_10
  simp only [← c16736, ← c16737, ← c16738] at e_256_10
  have hr := e_256_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 256 47) = band (a (.wire 256 39)) (a (.wire 256 43)) := by
  have c16739 := (publicBatchWrapper86_copies523 a h).2.2.2.1
  have c16740 := (publicBatchWrapper86_copies523 a h).2.2.2.2.1
  have c16741 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.1
  have e_256_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_11
  simp only [← c16739, ← c16740, ← c16741] at e_256_11
  have hr := e_256_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 55) = bor (a (.wire 46 51)) (a (.wire 256 47)) := by
  have c16742 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.1
  have c16743 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.1
  have c16744 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.1
  have c16745 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.1
  have c16746 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16747 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_13
  simp only [← c16745, ← c16746, k3, ← c16747] at e_248_13
  have e_254_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_5
  simp only [← c16742, ← c16743, ← c16744] at e_254_5
  have hr := e_248_13
  simp only [e_254_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 55) = a (.virt 201) := by
  have c16748 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16748

theorem publicBatchWrapper86_f2339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9403)) (a (.wire 165 55)) (a (.virt 18459)) (a (.virt 18460)) := by
  have c16749 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16750 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16751 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16752 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16753 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16754 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16755 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16756 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16757 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16758 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16759 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16760 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16761 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16762 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16763 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16764 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16765 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_256_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_12
  simp only [← c16755, ← c16756, ← c16757] at e_256_12
  have e_256_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_13
  simp only [← c16758, ← c16759, ← c16760] at e_256_13
  have e_257_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_2
  simp only [← c16749, k3, ← c16750, k3, ← c16751] at e_257_2
  have e_257_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_3
  simp only [← c16752, ← c16753, k3, ← c16754] at e_257_3
  have e_257_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_4
  simp only [← c16761, ← c16762, k3, ← c16763] at e_257_4
  refine ⟨?_, ?_⟩
  · have hc := e_256_12
    simp only [e_257_3] at hc
    linear_combination c16764.trans k2 - hc
  · have hc := e_257_4
    simp only [e_257_2, e_256_13, e_257_3] at hc
    linear_combination c16765.trans k2 - hc

theorem publicBatchWrapper86_f2340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 59) = bor (a (.wire 48 15)) (a (.virt 18459)) := by
  have c16766 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16767 := (publicBatchWrapper86_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16768 := (publicBatchWrapper86_copies524 a h).1
  have c16769 := (publicBatchWrapper86_copies524 a h).2.1
  have c16770 := (publicBatchWrapper86_copies524 a h).2.2.1
  have c16771 := (publicBatchWrapper86_copies524 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_14
  simp only [← c16769, ← c16770, k3, ← c16771] at e_248_14
  have e_254_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_6
  simp only [← c16766, ← c16767, ← c16768] at e_254_6
  have hr := e_248_14
  simp only [e_254_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 59) = a (.virt 201) := by
  have c16772 := (publicBatchWrapper86_copies524 a h).2.2.2.2.1
  exact c16772

theorem publicBatchWrapper86_f2342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9404)) (a (.wire 165 63)) (a (.virt 18461)) (a (.virt 18462)) := by
  have c16773 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.1
  have c16774 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.1
  have c16775 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.1
  have c16776 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.1
  have c16777 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.1
  have c16778 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16779 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16780 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16781 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16782 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16783 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16784 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16785 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16786 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16787 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16788 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16789 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_256_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_14
  simp only [← c16779, ← c16780, ← c16781] at e_256_14
  have e_256_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_15
  simp only [← c16782, ← c16783, ← c16784] at e_256_15
  have e_257_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_5
  simp only [← c16773, k3, ← c16774, k3, ← c16775] at e_257_5
  have e_257_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_6
  simp only [← c16776, ← c16777, k3, ← c16778] at e_257_6
  have e_257_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_7
  simp only [← c16785, ← c16786, k3, ← c16787] at e_257_7
  refine ⟨?_, ?_⟩
  · have hc := e_256_14
    simp only [e_257_6] at hc
    linear_combination c16788.trans k2 - hc
  · have hc := e_257_7
    simp only [e_257_5, e_256_15, e_257_6] at hc
    linear_combination c16789.trans k2 - hc

theorem publicBatchWrapper86_f2343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 63) = bor (a (.wire 48 15)) (a (.virt 18461)) := by
  have c16790 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16791 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16792 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16793 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16794 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16795 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_15
  simp only [← c16793, ← c16794, k3, ← c16795] at e_248_15
  have e_254_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_7
  simp only [← c16790, ← c16791, ← c16792] at e_254_7
  have hr := e_248_15
  simp only [e_254_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 63) = a (.virt 201) := by
  have c16796 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16796

theorem publicBatchWrapper86_f2345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9405)) (a (.wire 165 15)) (a (.virt 18463)) (a (.virt 18464)) := by
  have c16797 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16798 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16799 := (publicBatchWrapper86_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16800 := (publicBatchWrapper86_copies525 a h).1
  have c16801 := (publicBatchWrapper86_copies525 a h).2.1
  have c16802 := (publicBatchWrapper86_copies525 a h).2.2.1
  have c16803 := (publicBatchWrapper86_copies525 a h).2.2.2.1
  have c16804 := (publicBatchWrapper86_copies525 a h).2.2.2.2.1
  have c16805 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.1
  have c16806 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.1
  have c16807 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.1
  have c16808 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.1
  have c16809 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.1
  have c16810 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16811 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16812 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16813 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_256_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_16
  simp only [← c16803, ← c16804, ← c16805] at e_256_16
  have e_256_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_17
  simp only [← c16806, ← c16807, ← c16808] at e_256_17
  have e_257_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_8
  simp only [← c16797, k3, ← c16798, k3, ← c16799] at e_257_8
  have e_257_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_9
  simp only [← c16800, ← c16801, k3, ← c16802] at e_257_9
  have e_257_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_10
  simp only [← c16809, ← c16810, k3, ← c16811] at e_257_10
  refine ⟨?_, ?_⟩
  · have hc := e_256_16
    simp only [e_257_9] at hc
    linear_combination c16812.trans k2 - hc
  · have hc := e_257_10
    simp only [e_257_8, e_256_17, e_257_9] at hc
    linear_combination c16813.trans k2 - hc

theorem publicBatchWrapper86_f2346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9406)) (a (.wire 165 23)) (a (.virt 18465)) (a (.virt 18466)) := by
  have c16814 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16815 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16816 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16817 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16818 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16819 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16820 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16821 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16822 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16823 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16824 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16825 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16826 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16827 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16828 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16829 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16830 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_256_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_18
  simp only [← c16820, ← c16821, ← c16822] at e_256_18
  have e_256_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_19
  simp only [← c16823, ← c16824, ← c16825] at e_256_19
  have e_257_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_11
  simp only [← c16814, k3, ← c16815, k3, ← c16816] at e_257_11
  have e_257_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_12
  simp only [← c16817, ← c16818, k3, ← c16819] at e_257_12
  have e_257_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_13
  simp only [← c16826, ← c16827, k3, ← c16828] at e_257_13
  refine ⟨?_, ?_⟩
  · have hc := e_256_18
    simp only [e_257_12] at hc
    linear_combination c16829.trans k2 - hc
  · have hc := e_257_13
    simp only [e_257_11, e_256_19, e_257_12] at hc
    linear_combination c16830.trans k2 - hc

theorem publicBatchWrapper86_f2347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9407)) (a (.wire 165 31)) (a (.virt 18467)) (a (.virt 18468)) := by
  have c16831 := (publicBatchWrapper86_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16832 := (publicBatchWrapper86_copies526 a h).1
  have c16833 := (publicBatchWrapper86_copies526 a h).2.1
  have c16834 := (publicBatchWrapper86_copies526 a h).2.2.1
  have c16835 := (publicBatchWrapper86_copies526 a h).2.2.2.1
  have c16836 := (publicBatchWrapper86_copies526 a h).2.2.2.2.1
  have c16837 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.1
  have c16838 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.1
  have c16839 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.1
  have c16840 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.1
  have c16841 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.1
  have c16842 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16843 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16844 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16845 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16846 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16847 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_257_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_14
  simp only [← c16831, k3, ← c16832, k3, ← c16833] at e_257_14
  have e_257_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_15
  simp only [← c16834, ← c16835, k3, ← c16836] at e_257_15
  have e_257_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_16
  simp only [← c16843, ← c16844, k3, ← c16845] at e_257_16
  have e_258_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_0
  simp only [← c16837, ← c16838, ← c16839] at e_258_0
  have e_258_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_1
  simp only [← c16840, ← c16841, ← c16842] at e_258_1
  refine ⟨?_, ?_⟩
  · have hc := e_258_0
    simp only [e_257_15] at hc
    linear_combination c16846.trans k2 - hc
  · have hc := e_257_16
    simp only [e_257_14, e_258_1, e_257_15] at hc
    linear_combination c16847.trans k2 - hc

theorem publicBatchWrapper86_f2348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9408)) (a (.wire 165 39)) (a (.virt 18469)) (a (.virt 18470)) := by
  have c16848 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16849 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16850 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16851 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16852 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16853 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16854 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16855 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16856 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16857 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16858 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16859 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16860 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16861 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16862 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16863 := (publicBatchWrapper86_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16864 := (publicBatchWrapper86_copies527 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_257_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_17
  simp only [← c16848, k3, ← c16849, k3, ← c16850] at e_257_17
  have e_257_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_18
  simp only [← c16851, ← c16852, k3, ← c16853] at e_257_18
  have e_257_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_19
  simp only [← c16860, ← c16861, k3, ← c16862] at e_257_19
  have e_258_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_2
  simp only [← c16854, ← c16855, ← c16856] at e_258_2
  have e_258_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_3
  simp only [← c16857, ← c16858, ← c16859] at e_258_3
  refine ⟨?_, ?_⟩
  · have hc := e_258_2
    simp only [e_257_18] at hc
    linear_combination c16863.trans k2 - hc
  · have hc := e_257_19
    simp only [e_257_17, e_258_3, e_257_18] at hc
    linear_combination c16864.trans k2 - hc

theorem publicBatchWrapper86_f2349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 258 19) = band (a (.virt 18463)) (a (.virt 18465)) := by
  have c16865 := (publicBatchWrapper86_copies527 a h).2.1
  have c16866 := (publicBatchWrapper86_copies527 a h).2.2.1
  have c16867 := (publicBatchWrapper86_copies527 a h).2.2.2.1
  have e_258_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_4
  simp only [← c16865, ← c16866, ← c16867] at e_258_4
  have hr := e_258_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 258 23) = band (a (.virt 18467)) (a (.virt 18469)) := by
  have c16868 := (publicBatchWrapper86_copies527 a h).2.2.2.2.1
  have c16869 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.1
  have c16870 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.1
  have e_258_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_5
  simp only [← c16868, ← c16869, ← c16870] at e_258_5
  have hr := e_258_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 258 27) = band (a (.wire 258 19)) (a (.wire 258 23)) := by
  have c16871 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.1
  have c16872 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.1
  have c16873 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.1
  have e_258_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_6
  simp only [← c16871, ← c16872, ← c16873] at e_258_6
  have hr := e_258_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 67) = bor (a (.wire 48 15)) (a (.wire 258 27)) := by
  have c16874 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16875 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16876 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16877 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16878 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16879 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_16
  simp only [← c16877, ← c16878, k3, ← c16879] at e_248_16
  have e_254_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_8
  simp only [← c16874, ← c16875, ← c16876] at e_254_8
  have hr := e_248_16
  simp only [e_254_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 67) = a (.virt 201) := by
  have c16880 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16880

theorem publicBatchWrapper86_f2354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9603)) (a (.wire 165 55)) (a (.virt 18471)) (a (.virt 18472)) := by
  have c16881 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16882 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16883 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16884 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16885 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16886 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16887 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16888 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16889 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16890 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16891 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16892 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16893 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16894 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16895 := (publicBatchWrapper86_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16896 := (publicBatchWrapper86_copies528 a h).1
  have c16897 := (publicBatchWrapper86_copies528 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_258_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_7
  simp only [← c16887, ← c16888, ← c16889] at e_258_7
  have e_258_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_8
  simp only [← c16890, ← c16891, ← c16892] at e_258_8
  have e_259_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_0
  simp only [← c16881, k3, ← c16882, k3, ← c16883] at e_259_0
  have e_259_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_1
  simp only [← c16884, ← c16885, k3, ← c16886] at e_259_1
  have e_259_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_2
  simp only [← c16893, ← c16894, k3, ← c16895] at e_259_2
  refine ⟨?_, ?_⟩
  · have hc := e_258_7
    simp only [e_259_1] at hc
    linear_combination c16896.trans k2 - hc
  · have hc := e_259_2
    simp only [e_259_0, e_258_8, e_259_1] at hc
    linear_combination c16897.trans k2 - hc

theorem publicBatchWrapper86_f2355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 71) = bor (a (.wire 48 59)) (a (.virt 18471)) := by
  have c16898 := (publicBatchWrapper86_copies528 a h).2.2.1
  have c16899 := (publicBatchWrapper86_copies528 a h).2.2.2.1
  have c16900 := (publicBatchWrapper86_copies528 a h).2.2.2.2.1
  have c16901 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.1
  have c16902 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.1
  have c16903 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_17
  simp only [← c16901, ← c16902, k3, ← c16903] at e_248_17
  have e_254_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_9
  simp only [← c16898, ← c16899, ← c16900] at e_254_9
  have hr := e_248_17
  simp only [e_254_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 71) = a (.virt 201) := by
  have c16904 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.1
  exact c16904

theorem publicBatchWrapper86_f2357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9604)) (a (.wire 165 63)) (a (.virt 18473)) (a (.virt 18474)) := by
  have c16905 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.1
  have c16906 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16907 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16908 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16909 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16910 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16911 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16912 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16913 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16914 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16915 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16916 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16917 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16918 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16919 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16920 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16921 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_258_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_9
  simp only [← c16911, ← c16912, ← c16913] at e_258_9
  have e_258_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_10
  simp only [← c16914, ← c16915, ← c16916] at e_258_10
  have e_259_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_3
  simp only [← c16905, k3, ← c16906, k3, ← c16907] at e_259_3
  have e_259_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_4
  simp only [← c16908, ← c16909, k3, ← c16910] at e_259_4
  have e_259_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_5
  simp only [← c16917, ← c16918, k3, ← c16919] at e_259_5
  refine ⟨?_, ?_⟩
  · have hc := e_258_9
    simp only [e_259_4] at hc
    linear_combination c16920.trans k2 - hc
  · have hc := e_259_5
    simp only [e_259_3, e_258_10, e_259_4] at hc
    linear_combination c16921.trans k2 - hc

theorem publicBatchWrapper86_f2358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 75) = bor (a (.wire 48 59)) (a (.virt 18473)) := by
  have c16922 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16923 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16924 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16925 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16926 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16927 := (publicBatchWrapper86_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_18
  simp only [← c16925, ← c16926, k3, ← c16927] at e_248_18
  have e_254_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_10
  simp only [← c16922, ← c16923, ← c16924] at e_254_10
  have hr := e_248_18
  simp only [e_254_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 75) = a (.virt 201) := by
  have c16928 := (publicBatchWrapper86_copies529 a h).1
  exact c16928

theorem publicBatchWrapper86_f2360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9605)) (a (.wire 165 15)) (a (.virt 18475)) (a (.virt 18476)) := by
  have c16929 := (publicBatchWrapper86_copies529 a h).2.1
  have c16930 := (publicBatchWrapper86_copies529 a h).2.2.1
  have c16931 := (publicBatchWrapper86_copies529 a h).2.2.2.1
  have c16932 := (publicBatchWrapper86_copies529 a h).2.2.2.2.1
  have c16933 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.1
  have c16934 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.1
  have c16935 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.1
  have c16936 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.1
  have c16937 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.1
  have c16938 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16939 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16940 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16941 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16942 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16943 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16944 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16945 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_258_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_11
  simp only [← c16935, ← c16936, ← c16937] at e_258_11
  have e_258_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_12
  simp only [← c16938, ← c16939, ← c16940] at e_258_12
  have e_259_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_6
  simp only [← c16929, k3, ← c16930, k3, ← c16931] at e_259_6
  have e_259_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_7
  simp only [← c16932, ← c16933, k3, ← c16934] at e_259_7
  have e_259_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_8
  simp only [← c16941, ← c16942, k3, ← c16943] at e_259_8
  refine ⟨?_, ?_⟩
  · have hc := e_258_11
    simp only [e_259_7] at hc
    linear_combination c16944.trans k2 - hc
  · have hc := e_259_8
    simp only [e_259_6, e_258_12, e_259_7] at hc
    linear_combination c16945.trans k2 - hc

theorem publicBatchWrapper86_f2361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9606)) (a (.wire 165 23)) (a (.virt 18477)) (a (.virt 18478)) := by
  have c16946 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16947 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16948 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16949 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16950 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16951 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16952 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16953 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16954 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16955 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16956 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16957 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16958 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16959 := (publicBatchWrapper86_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16960 := (publicBatchWrapper86_copies530 a h).1
  have c16961 := (publicBatchWrapper86_copies530 a h).2.1
  have c16962 := (publicBatchWrapper86_copies530 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_258_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_13
  simp only [← c16952, ← c16953, ← c16954] at e_258_13
  have e_258_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_14
  simp only [← c16955, ← c16956, ← c16957] at e_258_14
  have e_259_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_9
  simp only [← c16946, k3, ← c16947, k3, ← c16948] at e_259_9
  have e_259_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_10
  simp only [← c16949, ← c16950, k3, ← c16951] at e_259_10
  have e_259_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_11
  simp only [← c16958, ← c16959, k3, ← c16960] at e_259_11
  refine ⟨?_, ?_⟩
  · have hc := e_258_13
    simp only [e_259_10] at hc
    linear_combination c16961.trans k2 - hc
  · have hc := e_259_11
    simp only [e_259_9, e_258_14, e_259_10] at hc
    linear_combination c16962.trans k2 - hc

theorem publicBatchWrapper86_f2362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9607)) (a (.wire 165 31)) (a (.virt 18479)) (a (.virt 18480)) := by
  have c16963 := (publicBatchWrapper86_copies530 a h).2.2.2.1
  have c16964 := (publicBatchWrapper86_copies530 a h).2.2.2.2.1
  have c16965 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.1
  have c16966 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.1
  have c16967 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.1
  have c16968 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.1
  have c16969 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.1
  have c16970 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16971 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16972 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16973 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16974 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16975 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16976 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16977 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16978 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16979 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_258_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_15
  simp only [← c16969, ← c16970, ← c16971] at e_258_15
  have e_258_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_16
  simp only [← c16972, ← c16973, ← c16974] at e_258_16
  have e_259_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_12
  simp only [← c16963, k3, ← c16964, k3, ← c16965] at e_259_12
  have e_259_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_13
  simp only [← c16966, ← c16967, k3, ← c16968] at e_259_13
  have e_259_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_14
  simp only [← c16975, ← c16976, k3, ← c16977] at e_259_14
  refine ⟨?_, ?_⟩
  · have hc := e_258_15
    simp only [e_259_13] at hc
    linear_combination c16978.trans k2 - hc
  · have hc := e_259_14
    simp only [e_259_12, e_258_16, e_259_13] at hc
    linear_combination c16979.trans k2 - hc

theorem publicBatchWrapper86_f2363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9608)) (a (.wire 165 39)) (a (.virt 18481)) (a (.virt 18482)) := by
  have c16980 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16981 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16982 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16983 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16984 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16985 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16986 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16987 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16988 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16989 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16990 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16991 := (publicBatchWrapper86_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16992 := (publicBatchWrapper86_copies531 a h).1
  have c16993 := (publicBatchWrapper86_copies531 a h).2.1
  have c16994 := (publicBatchWrapper86_copies531 a h).2.2.1
  have c16995 := (publicBatchWrapper86_copies531 a h).2.2.2.1
  have c16996 := (publicBatchWrapper86_copies531 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_258_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_17
  simp only [← c16986, ← c16987, ← c16988] at e_258_17
  have e_258_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_18
  simp only [← c16989, ← c16990, ← c16991] at e_258_18
  have e_259_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_15
  simp only [← c16980, k3, ← c16981, k3, ← c16982] at e_259_15
  have e_259_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_16
  simp only [← c16983, ← c16984, k3, ← c16985] at e_259_16
  have e_259_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_17
  simp only [← c16992, ← c16993, k3, ← c16994] at e_259_17
  refine ⟨?_, ?_⟩
  · have hc := e_258_17
    simp only [e_259_16] at hc
    linear_combination c16995.trans k2 - hc
  · have hc := e_259_17
    simp only [e_259_15, e_258_18, e_259_16] at hc
    linear_combination c16996.trans k2 - hc

theorem publicBatchWrapper86_f2364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 258 79) = band (a (.virt 18475)) (a (.virt 18477)) := by
  have c16997 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.1
  have c16998 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.1
  have c16999 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.1
  have e_258_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_19
  simp only [← c16997, ← c16998, ← c16999] at e_258_19
  have hr := e_258_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 260 3) = band (a (.virt 18479)) (a (.virt 18481)) := by
  have c17000 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.1
  have c17001 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.1
  have c17002 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_260_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_0
  simp only [← c17000, ← c17001, ← c17002] at e_260_0
  have hr := e_260_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 260 7) = band (a (.wire 258 79)) (a (.wire 260 3)) := by
  have c17003 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17004 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17005 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_260_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_1
  simp only [← c17003, ← c17004, ← c17005] at e_260_1
  have hr := e_260_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 79) = bor (a (.wire 48 59)) (a (.wire 260 7)) := by
  have c17006 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17007 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17008 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17009 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17010 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17011 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_248_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_19
  simp only [← c17009, ← c17010, k3, ← c17011] at e_248_19
  have e_254_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_11
  simp only [← c17006, ← c17007, ← c17008] at e_254_11
  have hr := e_248_19
  simp only [e_254_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 248 79) = a (.virt 201) := by
  have c17012 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17012

theorem publicBatchWrapper86_f2369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9803)) (a (.wire 165 55)) (a (.virt 18483)) (a (.virt 18484)) := by
  have c17013 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17014 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17015 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17016 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17017 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17018 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17019 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17020 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17021 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17022 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17023 := (publicBatchWrapper86_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17024 := (publicBatchWrapper86_copies532 a h).1
  have c17025 := (publicBatchWrapper86_copies532 a h).2.1
  have c17026 := (publicBatchWrapper86_copies532 a h).2.2.1
  have c17027 := (publicBatchWrapper86_copies532 a h).2.2.2.1
  have c17028 := (publicBatchWrapper86_copies532 a h).2.2.2.2.1
  have c17029 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_259_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_18
  simp only [← c17013, k3, ← c17014, k3, ← c17015] at e_259_18
  have e_259_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_19
  simp only [← c17016, ← c17017, k3, ← c17018] at e_259_19
  have e_260_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_2
  simp only [← c17019, ← c17020, ← c17021] at e_260_2
  have e_260_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_3
  simp only [← c17022, ← c17023, ← c17024] at e_260_3
  have e_261_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_0
  simp only [← c17025, ← c17026, k3, ← c17027] at e_261_0
  refine ⟨?_, ?_⟩
  · have hc := e_260_2
    simp only [e_259_19] at hc
    linear_combination c17028.trans k2 - hc
  · have hc := e_261_0
    simp only [e_259_18, e_260_3, e_259_19] at hc
    linear_combination c17029.trans k2 - hc

theorem publicBatchWrapper86_f2370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 3) = bor (a (.wire 49 23)) (a (.virt 18483)) := by
  have c17030 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.1
  have c17031 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.1
  have c17032 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.1
  have c17033 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.1
  have c17034 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17035 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_254_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_12
  simp only [← c17030, ← c17031, ← c17032] at e_254_12
  have e_262_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_0
  simp only [← c17033, ← c17034, k3, ← c17035] at e_262_0
  have hr := e_262_0
  simp only [e_254_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 3) = a (.virt 201) := by
  have c17036 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17036

theorem publicBatchWrapper86_f2372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9804)) (a (.wire 165 63)) (a (.virt 18485)) (a (.virt 18486)) := by
  have c17037 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17038 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17039 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17040 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17041 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17042 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17043 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17044 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17045 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17046 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17047 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17048 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17049 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17050 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17051 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17052 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17053 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_260_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_4
  simp only [← c17043, ← c17044, ← c17045] at e_260_4
  have e_260_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_5
  simp only [← c17046, ← c17047, ← c17048] at e_260_5
  have e_261_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_1
  simp only [← c17037, k3, ← c17038, k3, ← c17039] at e_261_1
  have e_261_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_2
  simp only [← c17040, ← c17041, k3, ← c17042] at e_261_2
  have e_261_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_3
  simp only [← c17049, ← c17050, k3, ← c17051] at e_261_3
  refine ⟨?_, ?_⟩
  · have hc := e_260_4
    simp only [e_261_2] at hc
    linear_combination c17052.trans k2 - hc
  · have hc := e_261_3
    simp only [e_261_1, e_260_5, e_261_2] at hc
    linear_combination c17053.trans k2 - hc

theorem publicBatchWrapper86_f2373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 7) = bor (a (.wire 49 23)) (a (.virt 18485)) := by
  have c17054 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17055 := (publicBatchWrapper86_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17056 := (publicBatchWrapper86_copies533 a h).1
  have c17057 := (publicBatchWrapper86_copies533 a h).2.1
  have c17058 := (publicBatchWrapper86_copies533 a h).2.2.1
  have c17059 := (publicBatchWrapper86_copies533 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_254_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_13
  simp only [← c17054, ← c17055, ← c17056] at e_254_13
  have e_262_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_1
  simp only [← c17057, ← c17058, k3, ← c17059] at e_262_1
  have hr := e_262_1
  simp only [e_254_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 7) = a (.virt 201) := by
  have c17060 := (publicBatchWrapper86_copies533 a h).2.2.2.2.1
  exact c17060

theorem publicBatchWrapper86_f2375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9805)) (a (.wire 165 15)) (a (.virt 18487)) (a (.virt 18488)) := by
  have c17061 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.1
  have c17062 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.1
  have c17063 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.1
  have c17064 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.1
  have c17065 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.1
  have c17066 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17067 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17068 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17069 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17070 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17071 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17072 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17073 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17074 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17075 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17076 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17077 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_260_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_6
  simp only [← c17067, ← c17068, ← c17069] at e_260_6
  have e_260_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_7
  simp only [← c17070, ← c17071, ← c17072] at e_260_7
  have e_261_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_4
  simp only [← c17061, k3, ← c17062, k3, ← c17063] at e_261_4
  have e_261_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_5
  simp only [← c17064, ← c17065, k3, ← c17066] at e_261_5
  have e_261_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_6
  simp only [← c17073, ← c17074, k3, ← c17075] at e_261_6
  refine ⟨?_, ?_⟩
  · have hc := e_260_6
    simp only [e_261_5] at hc
    linear_combination c17076.trans k2 - hc
  · have hc := e_261_6
    simp only [e_261_4, e_260_7, e_261_5] at hc
    linear_combination c17077.trans k2 - hc

theorem publicBatchWrapper86_f2376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9806)) (a (.wire 165 23)) (a (.virt 18489)) (a (.virt 18490)) := by
  have c17078 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17079 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17080 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17081 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17082 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17083 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17084 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17085 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17086 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17087 := (publicBatchWrapper86_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17088 := (publicBatchWrapper86_copies534 a h).1
  have c17089 := (publicBatchWrapper86_copies534 a h).2.1
  have c17090 := (publicBatchWrapper86_copies534 a h).2.2.1
  have c17091 := (publicBatchWrapper86_copies534 a h).2.2.2.1
  have c17092 := (publicBatchWrapper86_copies534 a h).2.2.2.2.1
  have c17093 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.1
  have c17094 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_260_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_8
  simp only [← c17084, ← c17085, ← c17086] at e_260_8
  have e_260_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_9
  simp only [← c17087, ← c17088, ← c17089] at e_260_9
  have e_261_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_7
  simp only [← c17078, k3, ← c17079, k3, ← c17080] at e_261_7
  have e_261_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_8
  simp only [← c17081, ← c17082, k3, ← c17083] at e_261_8
  have e_261_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_9
  simp only [← c17090, ← c17091, k3, ← c17092] at e_261_9
  refine ⟨?_, ?_⟩
  · have hc := e_260_8
    simp only [e_261_8] at hc
    linear_combination c17093.trans k2 - hc
  · have hc := e_261_9
    simp only [e_261_7, e_260_9, e_261_8] at hc
    linear_combination c17094.trans k2 - hc

theorem publicBatchWrapper86_f2377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9807)) (a (.wire 165 31)) (a (.virt 18491)) (a (.virt 18492)) := by
  have c17095 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.1
  have c17096 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.1
  have c17097 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.1
  have c17098 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17099 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17100 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17101 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17102 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17103 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17104 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17105 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17106 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17107 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17108 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17109 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17110 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17111 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_260_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_10
  simp only [← c17101, ← c17102, ← c17103] at e_260_10
  have e_260_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_11
  simp only [← c17104, ← c17105, ← c17106] at e_260_11
  have e_261_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_10
  simp only [← c17095, k3, ← c17096, k3, ← c17097] at e_261_10
  have e_261_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_11
  simp only [← c17098, ← c17099, k3, ← c17100] at e_261_11
  have e_261_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_12
  simp only [← c17107, ← c17108, k3, ← c17109] at e_261_12
  refine ⟨?_, ?_⟩
  · have hc := e_260_10
    simp only [e_261_11] at hc
    linear_combination c17110.trans k2 - hc
  · have hc := e_261_12
    simp only [e_261_10, e_260_11, e_261_11] at hc
    linear_combination c17111.trans k2 - hc

theorem publicBatchWrapper86_f2378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9808)) (a (.wire 165 39)) (a (.virt 18493)) (a (.virt 18494)) := by
  have c17112 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17113 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17114 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17115 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17116 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17117 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17118 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17119 := (publicBatchWrapper86_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17120 := (publicBatchWrapper86_copies535 a h).1
  have c17121 := (publicBatchWrapper86_copies535 a h).2.1
  have c17122 := (publicBatchWrapper86_copies535 a h).2.2.1
  have c17123 := (publicBatchWrapper86_copies535 a h).2.2.2.1
  have c17124 := (publicBatchWrapper86_copies535 a h).2.2.2.2.1
  have c17125 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.1
  have c17126 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.1
  have c17127 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.1
  have c17128 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_260_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_12
  simp only [← c17118, ← c17119, ← c17120] at e_260_12
  have e_260_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_13
  simp only [← c17121, ← c17122, ← c17123] at e_260_13
  have e_261_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_13
  simp only [← c17112, k3, ← c17113, k3, ← c17114] at e_261_13
  have e_261_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_14
  simp only [← c17115, ← c17116, k3, ← c17117] at e_261_14
  have e_261_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_15
  simp only [← c17124, ← c17125, k3, ← c17126] at e_261_15
  refine ⟨?_, ?_⟩
  · have hc := e_260_12
    simp only [e_261_14] at hc
    linear_combination c17127.trans k2 - hc
  · have hc := e_261_15
    simp only [e_261_13, e_260_13, e_261_14] at hc
    linear_combination c17128.trans k2 - hc

theorem publicBatchWrapper86_f2379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 260 59) = band (a (.virt 18487)) (a (.virt 18489)) := by
  have c17129 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.1
  have c17130 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17131 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_260_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_14
  simp only [← c17129, ← c17130, ← c17131] at e_260_14
  have hr := e_260_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 260 63) = band (a (.virt 18491)) (a (.virt 18493)) := by
  have c17132 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17133 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17134 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_260_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_15
  simp only [← c17132, ← c17133, ← c17134] at e_260_15
  have hr := e_260_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 260 67) = band (a (.wire 260 59)) (a (.wire 260 63)) := by
  have c17135 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17136 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17137 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_260_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_16
  simp only [← c17135, ← c17136, ← c17137] at e_260_16
  have hr := e_260_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 11) = bor (a (.wire 49 23)) (a (.wire 260 67)) := by
  have c17138 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17139 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17140 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17141 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17142 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17143 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_254_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_14
  simp only [← c17138, ← c17139, ← c17140] at e_254_14
  have e_262_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_2
  simp only [← c17141, ← c17142, k3, ← c17143] at e_262_2
  have hr := e_262_2
  simp only [e_254_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 11) = a (.virt 201) := by
  have c17144 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17144

theorem publicBatchWrapper86_f2384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10003)) (a (.wire 165 55)) (a (.virt 18495)) (a (.virt 18496)) := by
  have c17145 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17146 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17147 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17148 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17149 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17150 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17151 := (publicBatchWrapper86_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17152 := (publicBatchWrapper86_copies536 a h).1
  have c17153 := (publicBatchWrapper86_copies536 a h).2.1
  have c17154 := (publicBatchWrapper86_copies536 a h).2.2.1
  have c17155 := (publicBatchWrapper86_copies536 a h).2.2.2.1
  have c17156 := (publicBatchWrapper86_copies536 a h).2.2.2.2.1
  have c17157 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.1
  have c17158 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.1
  have c17159 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.1
  have c17160 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.1
  have c17161 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_260_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_17
  simp only [← c17151, ← c17152, ← c17153] at e_260_17
  have e_260_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_18
  simp only [← c17154, ← c17155, ← c17156] at e_260_18
  have e_261_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_16
  simp only [← c17145, k3, ← c17146, k3, ← c17147] at e_261_16
  have e_261_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_17
  simp only [← c17148, ← c17149, k3, ← c17150] at e_261_17
  have e_261_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_18
  simp only [← c17157, ← c17158, k3, ← c17159] at e_261_18
  refine ⟨?_, ?_⟩
  · have hc := e_260_17
    simp only [e_261_17] at hc
    linear_combination c17160.trans k2 - hc
  · have hc := e_261_18
    simp only [e_261_16, e_260_18, e_261_17] at hc
    linear_combination c17161.trans k2 - hc

theorem publicBatchWrapper86_f2385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 15) = bor (a (.wire 49 67)) (a (.virt 18495)) := by
  have c17162 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17163 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17164 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17165 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17166 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17167 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_254_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_15
  simp only [← c17162, ← c17163, ← c17164] at e_254_15
  have e_262_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_3
  simp only [← c17165, ← c17166, k3, ← c17167] at e_262_3
  have hr := e_262_3
  simp only [e_254_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 15) = a (.virt 201) := by
  have c17168 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17168

theorem publicBatchWrapper86_f2387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10004)) (a (.wire 165 63)) (a (.virt 18497)) (a (.virt 18498)) := by
  have c17169 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17170 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17171 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17172 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17173 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17174 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17175 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17176 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17177 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17178 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17179 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17180 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17181 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17182 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17183 := (publicBatchWrapper86_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17184 := (publicBatchWrapper86_copies537 a h).1
  have c17185 := (publicBatchWrapper86_copies537 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_260_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_19
  simp only [← c17175, ← c17176, ← c17177] at e_260_19
  have e_261_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_19
  simp only [← c17169, k3, ← c17170, k3, ← c17171] at e_261_19
  have e_263_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_0
  simp only [← c17172, ← c17173, k3, ← c17174] at e_263_0
  have e_263_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_1
  simp only [← c17181, ← c17182, k3, ← c17183] at e_263_1
  have e_264_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_0
  simp only [← c17178, ← c17179, ← c17180] at e_264_0
  refine ⟨?_, ?_⟩
  · have hc := e_260_19
    simp only [e_263_0] at hc
    linear_combination c17184.trans k2 - hc
  · have hc := e_263_1
    simp only [e_261_19, e_264_0, e_263_0] at hc
    linear_combination c17185.trans k2 - hc

theorem publicBatchWrapper86_f2388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 19) = bor (a (.wire 49 67)) (a (.virt 18497)) := by
  have c17186 := (publicBatchWrapper86_copies537 a h).2.2.1
  have c17187 := (publicBatchWrapper86_copies537 a h).2.2.2.1
  have c17188 := (publicBatchWrapper86_copies537 a h).2.2.2.2.1
  have c17189 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.1
  have c17190 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.1
  have c17191 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_254_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_16
  simp only [← c17186, ← c17187, ← c17188] at e_254_16
  have e_262_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_4
  simp only [← c17189, ← c17190, k3, ← c17191] at e_262_4
  have hr := e_262_4
  simp only [e_254_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 19) = a (.virt 201) := by
  have c17192 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.1
  exact c17192

theorem publicBatchWrapper86_f2390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10005)) (a (.wire 165 15)) (a (.virt 18499)) (a (.virt 18500)) := by
  have c17193 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.1
  have c17194 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17195 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17196 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17197 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17198 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17199 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17200 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17201 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17202 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17203 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17204 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17205 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17206 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17207 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17208 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17209 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_263_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_2
  simp only [← c17193, k3, ← c17194, k3, ← c17195] at e_263_2
  have e_263_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_3
  simp only [← c17196, ← c17197, k3, ← c17198] at e_263_3
  have e_263_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_4
  simp only [← c17205, ← c17206, k3, ← c17207] at e_263_4
  have e_264_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_1
  simp only [← c17199, ← c17200, ← c17201] at e_264_1
  have e_264_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_2
  simp only [← c17202, ← c17203, ← c17204] at e_264_2
  refine ⟨?_, ?_⟩
  · have hc := e_264_1
    simp only [e_263_3] at hc
    linear_combination c17208.trans k2 - hc
  · have hc := e_263_4
    simp only [e_263_2, e_264_2, e_263_3] at hc
    linear_combination c17209.trans k2 - hc

theorem publicBatchWrapper86_f2391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10006)) (a (.wire 165 23)) (a (.virt 18501)) (a (.virt 18502)) := by
  have c17210 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17211 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17212 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17213 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17214 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17215 := (publicBatchWrapper86_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17216 := (publicBatchWrapper86_copies538 a h).1
  have c17217 := (publicBatchWrapper86_copies538 a h).2.1
  have c17218 := (publicBatchWrapper86_copies538 a h).2.2.1
  have c17219 := (publicBatchWrapper86_copies538 a h).2.2.2.1
  have c17220 := (publicBatchWrapper86_copies538 a h).2.2.2.2.1
  have c17221 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.1
  have c17222 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.1
  have c17223 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.1
  have c17224 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.1
  have c17225 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.1
  have c17226 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_263_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_5
  simp only [← c17210, k3, ← c17211, k3, ← c17212] at e_263_5
  have e_263_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_6
  simp only [← c17213, ← c17214, k3, ← c17215] at e_263_6
  have e_263_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_7
  simp only [← c17222, ← c17223, k3, ← c17224] at e_263_7
  have e_264_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_3
  simp only [← c17216, ← c17217, ← c17218] at e_264_3
  have e_264_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_4
  simp only [← c17219, ← c17220, ← c17221] at e_264_4
  refine ⟨?_, ?_⟩
  · have hc := e_264_3
    simp only [e_263_6] at hc
    linear_combination c17225.trans k2 - hc
  · have hc := e_263_7
    simp only [e_263_5, e_264_4, e_263_6] at hc
    linear_combination c17226.trans k2 - hc

theorem publicBatchWrapper86_f2392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10007)) (a (.wire 165 31)) (a (.virt 18503)) (a (.virt 18504)) := by
  have c17227 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17228 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17229 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17230 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17231 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17232 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17233 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17234 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17235 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17236 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17237 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17238 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17239 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17240 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17241 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17242 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17243 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_263_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_8
  simp only [← c17227, k3, ← c17228, k3, ← c17229] at e_263_8
  have e_263_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_9
  simp only [← c17230, ← c17231, k3, ← c17232] at e_263_9
  have e_263_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_10
  simp only [← c17239, ← c17240, k3, ← c17241] at e_263_10
  have e_264_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_5
  simp only [← c17233, ← c17234, ← c17235] at e_264_5
  have e_264_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_6
  simp only [← c17236, ← c17237, ← c17238] at e_264_6
  refine ⟨?_, ?_⟩
  · have hc := e_264_5
    simp only [e_263_9] at hc
    linear_combination c17242.trans k2 - hc
  · have hc := e_263_10
    simp only [e_263_8, e_264_6, e_263_9] at hc
    linear_combination c17243.trans k2 - hc

theorem publicBatchWrapper86_f2393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10008)) (a (.wire 165 39)) (a (.virt 18505)) (a (.virt 18506)) := by
  have c17244 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17245 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17246 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17247 := (publicBatchWrapper86_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17248 := (publicBatchWrapper86_copies539 a h).1
  have c17249 := (publicBatchWrapper86_copies539 a h).2.1
  have c17250 := (publicBatchWrapper86_copies539 a h).2.2.1
  have c17251 := (publicBatchWrapper86_copies539 a h).2.2.2.1
  have c17252 := (publicBatchWrapper86_copies539 a h).2.2.2.2.1
  have c17253 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.1
  have c17254 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.1
  have c17255 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.1
  have c17256 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.1
  have c17257 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.1
  have c17258 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17259 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17260 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_263_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_11
  simp only [← c17244, k3, ← c17245, k3, ← c17246] at e_263_11
  have e_263_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_12
  simp only [← c17247, ← c17248, k3, ← c17249] at e_263_12
  have e_263_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_13
  simp only [← c17256, ← c17257, k3, ← c17258] at e_263_13
  have e_264_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_7
  simp only [← c17250, ← c17251, ← c17252] at e_264_7
  have e_264_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_8
  simp only [← c17253, ← c17254, ← c17255] at e_264_8
  refine ⟨?_, ?_⟩
  · have hc := e_264_7
    simp only [e_263_12] at hc
    linear_combination c17259.trans k2 - hc
  · have hc := e_263_13
    simp only [e_263_11, e_264_8, e_263_12] at hc
    linear_combination c17260.trans k2 - hc

theorem publicBatchWrapper86_f2394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 264 39) = band (a (.virt 18499)) (a (.virt 18501)) := by
  have c17261 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17262 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17263 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_264_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_9
  simp only [← c17261, ← c17262, ← c17263] at e_264_9
  have hr := e_264_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 264 43) = band (a (.virt 18503)) (a (.virt 18505)) := by
  have c17264 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17265 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17266 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_264_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_10
  simp only [← c17264, ← c17265, ← c17266] at e_264_10
  have hr := e_264_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 264 47) = band (a (.wire 264 39)) (a (.wire 264 43)) := by
  have c17267 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17268 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17269 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_264_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_11
  simp only [← c17267, ← c17268, ← c17269] at e_264_11
  have hr := e_264_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 23) = bor (a (.wire 49 67)) (a (.wire 264 47)) := by
  have c17270 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17271 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17272 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17273 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17274 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17275 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_254_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_17
  simp only [← c17270, ← c17271, ← c17272] at e_254_17
  have e_262_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_5
  simp only [← c17273, ← c17274, k3, ← c17275] at e_262_5
  have hr := e_262_5
  simp only [e_254_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 23) = a (.virt 201) := by
  have c17276 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17276

theorem publicBatchWrapper86_f2399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10203)) (a (.wire 165 55)) (a (.virt 18507)) (a (.virt 18508)) := by
  have c17277 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17278 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17279 := (publicBatchWrapper86_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17280 := (publicBatchWrapper86_copies540 a h).1
  have c17281 := (publicBatchWrapper86_copies540 a h).2.1
  have c17282 := (publicBatchWrapper86_copies540 a h).2.2.1
  have c17283 := (publicBatchWrapper86_copies540 a h).2.2.2.1
  have c17284 := (publicBatchWrapper86_copies540 a h).2.2.2.2.1
  have c17285 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.1
  have c17286 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.1
  have c17287 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.1
  have c17288 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.1
  have c17289 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.1
  have c17290 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17291 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17292 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17293 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_263_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_14
  simp only [← c17277, k3, ← c17278, k3, ← c17279] at e_263_14
  have e_263_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_15
  simp only [← c17280, ← c17281, k3, ← c17282] at e_263_15
  have e_263_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_16
  simp only [← c17289, ← c17290, k3, ← c17291] at e_263_16
  have e_264_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_12
  simp only [← c17283, ← c17284, ← c17285] at e_264_12
  have e_264_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_13
  simp only [← c17286, ← c17287, ← c17288] at e_264_13
  refine ⟨?_, ?_⟩
  · have hc := e_264_12
    simp only [e_263_15] at hc
    linear_combination c17292.trans k2 - hc
  · have hc := e_263_16
    simp only [e_263_14, e_264_13, e_263_15] at hc
    linear_combination c17293.trans k2 - hc

theorem publicBatchWrapper86_f2400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 27) = bor (a (.wire 51 31)) (a (.virt 18507)) := by
  have c17294 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17295 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17296 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17297 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17298 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17299 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_254_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_18
  simp only [← c17294, ← c17295, ← c17296] at e_254_18
  have e_262_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_6
  simp only [← c17297, ← c17298, k3, ← c17299] at e_262_6
  have hr := e_262_6
  simp only [e_254_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 27) = a (.virt 201) := by
  have c17300 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17300

theorem publicBatchWrapper86_f2402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10204)) (a (.wire 165 63)) (a (.virt 18509)) (a (.virt 18510)) := by
  have c17301 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17302 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17303 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17304 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17305 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17306 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17307 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17308 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17309 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17310 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17311 := (publicBatchWrapper86_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17312 := (publicBatchWrapper86_copies541 a h).1
  have c17313 := (publicBatchWrapper86_copies541 a h).2.1
  have c17314 := (publicBatchWrapper86_copies541 a h).2.2.1
  have c17315 := (publicBatchWrapper86_copies541 a h).2.2.2.1
  have c17316 := (publicBatchWrapper86_copies541 a h).2.2.2.2.1
  have c17317 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_263_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_17
  simp only [← c17301, k3, ← c17302, k3, ← c17303] at e_263_17
  have e_263_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_18
  simp only [← c17304, ← c17305, k3, ← c17306] at e_263_18
  have e_263_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_19
  simp only [← c17313, ← c17314, k3, ← c17315] at e_263_19
  have e_264_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_14
  simp only [← c17307, ← c17308, ← c17309] at e_264_14
  have e_264_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_15
  simp only [← c17310, ← c17311, ← c17312] at e_264_15
  refine ⟨?_, ?_⟩
  · have hc := e_264_14
    simp only [e_263_18] at hc
    linear_combination c17316.trans k2 - hc
  · have hc := e_263_19
    simp only [e_263_17, e_264_15, e_263_18] at hc
    linear_combination c17317.trans k2 - hc

theorem publicBatchWrapper86_f2403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 31) = bor (a (.wire 51 31)) (a (.virt 18509)) := by
  have c17318 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.1
  have c17319 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.1
  have c17320 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.1
  have c17321 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.1
  have c17322 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17323 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_254_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_19
  simp only [← c17318, ← c17319, ← c17320] at e_254_19
  have e_262_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_7
  simp only [← c17321, ← c17322, k3, ← c17323] at e_262_7
  have hr := e_262_7
  simp only [e_254_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 31) = a (.virt 201) := by
  have c17324 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17324

theorem publicBatchWrapper86_f2405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10205)) (a (.wire 165 15)) (a (.virt 18511)) (a (.virt 18512)) := by
  have c17325 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17326 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17327 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17328 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17329 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17330 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17331 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17332 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17333 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17334 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17335 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17336 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17337 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17338 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17339 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17340 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17341 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_264_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_16
  simp only [← c17331, ← c17332, ← c17333] at e_264_16
  have e_264_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_17
  simp only [← c17334, ← c17335, ← c17336] at e_264_17
  have e_265_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_0
  simp only [← c17325, k3, ← c17326, k3, ← c17327] at e_265_0
  have e_265_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_1
  simp only [← c17328, ← c17329, k3, ← c17330] at e_265_1
  have e_265_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_2
  simp only [← c17337, ← c17338, k3, ← c17339] at e_265_2
  refine ⟨?_, ?_⟩
  · have hc := e_264_16
    simp only [e_265_1] at hc
    linear_combination c17340.trans k2 - hc
  · have hc := e_265_2
    simp only [e_265_0, e_264_17, e_265_1] at hc
    linear_combination c17341.trans k2 - hc

theorem publicBatchWrapper86_f2406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10206)) (a (.wire 165 23)) (a (.virt 18513)) (a (.virt 18514)) := by
  have c17342 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17343 := (publicBatchWrapper86_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17344 := (publicBatchWrapper86_copies542 a h).1
  have c17345 := (publicBatchWrapper86_copies542 a h).2.1
  have c17346 := (publicBatchWrapper86_copies542 a h).2.2.1
  have c17347 := (publicBatchWrapper86_copies542 a h).2.2.2.1
  have c17348 := (publicBatchWrapper86_copies542 a h).2.2.2.2.1
  have c17349 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.1
  have c17350 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.1
  have c17351 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.1
  have c17352 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.1
  have c17353 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.1
  have c17354 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17355 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17356 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17357 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17358 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_264_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_18
  simp only [← c17348, ← c17349, ← c17350] at e_264_18
  have e_264_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_19
  simp only [← c17351, ← c17352, ← c17353] at e_264_19
  have e_265_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_3
  simp only [← c17342, k3, ← c17343, k3, ← c17344] at e_265_3
  have e_265_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_4
  simp only [← c17345, ← c17346, k3, ← c17347] at e_265_4
  have e_265_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_5
  simp only [← c17354, ← c17355, k3, ← c17356] at e_265_5
  refine ⟨?_, ?_⟩
  · have hc := e_264_18
    simp only [e_265_4] at hc
    linear_combination c17357.trans k2 - hc
  · have hc := e_265_5
    simp only [e_265_3, e_264_19, e_265_4] at hc
    linear_combination c17358.trans k2 - hc

theorem publicBatchWrapper86_f2407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10207)) (a (.wire 165 31)) (a (.virt 18515)) (a (.virt 18516)) := by
  have c17359 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17360 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17361 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17362 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17363 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17364 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17365 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17366 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17367 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17368 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17369 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17370 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17371 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17372 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17373 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17374 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17375 := (publicBatchWrapper86_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_265_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_6
  simp only [← c17359, k3, ← c17360, k3, ← c17361] at e_265_6
  have e_265_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_7
  simp only [← c17362, ← c17363, k3, ← c17364] at e_265_7
  have e_265_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_8
  simp only [← c17371, ← c17372, k3, ← c17373] at e_265_8
  have e_266_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_0
  simp only [← c17365, ← c17366, ← c17367] at e_266_0
  have e_266_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_1
  simp only [← c17368, ← c17369, ← c17370] at e_266_1
  refine ⟨?_, ?_⟩
  · have hc := e_266_0
    simp only [e_265_7] at hc
    linear_combination c17374.trans k2 - hc
  · have hc := e_265_8
    simp only [e_265_6, e_266_1, e_265_7] at hc
    linear_combination c17375.trans k2 - hc

theorem publicBatchWrapper86_f2408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10208)) (a (.wire 165 39)) (a (.virt 18517)) (a (.virt 18518)) := by
  have c17376 := (publicBatchWrapper86_copies543 a h).1
  have c17377 := (publicBatchWrapper86_copies543 a h).2.1
  have c17378 := (publicBatchWrapper86_copies543 a h).2.2.1
  have c17379 := (publicBatchWrapper86_copies543 a h).2.2.2.1
  have c17380 := (publicBatchWrapper86_copies543 a h).2.2.2.2.1
  have c17381 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.1
  have c17382 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.1
  have c17383 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.1
  have c17384 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.1
  have c17385 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.1
  have c17386 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17387 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17388 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17389 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17390 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17391 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17392 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_265_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_9
  simp only [← c17376, k3, ← c17377, k3, ← c17378] at e_265_9
  have e_265_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_10
  simp only [← c17379, ← c17380, k3, ← c17381] at e_265_10
  have e_265_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_11
  simp only [← c17388, ← c17389, k3, ← c17390] at e_265_11
  have e_266_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_2
  simp only [← c17382, ← c17383, ← c17384] at e_266_2
  have e_266_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_3
  simp only [← c17385, ← c17386, ← c17387] at e_266_3
  refine ⟨?_, ?_⟩
  · have hc := e_266_2
    simp only [e_265_10] at hc
    linear_combination c17391.trans k2 - hc
  · have hc := e_265_11
    simp only [e_265_9, e_266_3, e_265_10] at hc
    linear_combination c17392.trans k2 - hc

theorem publicBatchWrapper86_f2409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 266 19) = band (a (.virt 18511)) (a (.virt 18513)) := by
  have c17393 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17394 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17395 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_266_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_4
  simp only [← c17393, ← c17394, ← c17395] at e_266_4
  have hr := e_266_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 266 23) = band (a (.virt 18515)) (a (.virt 18517)) := by
  have c17396 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17397 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17398 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_266_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_5
  simp only [← c17396, ← c17397, ← c17398] at e_266_5
  have hr := e_266_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 266 27) = band (a (.wire 266 19)) (a (.wire 266 23)) := by
  have c17399 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17400 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17401 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_266_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_6
  simp only [← c17399, ← c17400, ← c17401] at e_266_6
  have hr := e_266_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 35) = bor (a (.wire 51 31)) (a (.wire 266 27)) := by
  have c17402 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17403 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17404 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17405 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17406 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17407 := (publicBatchWrapper86_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_8
  simp only [← c17405, ← c17406, k3, ← c17407] at e_262_8
  have e_267_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_0
  simp only [← c17402, ← c17403, ← c17404] at e_267_0
  have hr := e_262_8
  simp only [e_267_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 35) = a (.virt 201) := by
  have c17408 := (publicBatchWrapper86_copies544 a h).1
  exact c17408

theorem publicBatchWrapper86_f2414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10403)) (a (.wire 165 55)) (a (.virt 18519)) (a (.virt 18520)) := by
  have c17409 := (publicBatchWrapper86_copies544 a h).2.1
  have c17410 := (publicBatchWrapper86_copies544 a h).2.2.1
  have c17411 := (publicBatchWrapper86_copies544 a h).2.2.2.1
  have c17412 := (publicBatchWrapper86_copies544 a h).2.2.2.2.1
  have c17413 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.1
  have c17414 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.1
  have c17415 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.1
  have c17416 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.1
  have c17417 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.1
  have c17418 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17419 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17420 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17421 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17422 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17423 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17424 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17425 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_265_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_12
  simp only [← c17409, k3, ← c17410, k3, ← c17411] at e_265_12
  have e_265_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_13
  simp only [← c17412, ← c17413, k3, ← c17414] at e_265_13
  have e_265_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_14
  simp only [← c17421, ← c17422, k3, ← c17423] at e_265_14
  have e_266_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_7
  simp only [← c17415, ← c17416, ← c17417] at e_266_7
  have e_266_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_8
  simp only [← c17418, ← c17419, ← c17420] at e_266_8
  refine ⟨?_, ?_⟩
  · have hc := e_266_7
    simp only [e_265_13] at hc
    linear_combination c17424.trans k2 - hc
  · have hc := e_265_14
    simp only [e_265_12, e_266_8, e_265_13] at hc
    linear_combination c17425.trans k2 - hc

theorem publicBatchWrapper86_f2415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 39) = bor (a (.wire 51 75)) (a (.virt 18519)) := by
  have c17426 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17427 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17428 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17429 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17430 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17431 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_9
  simp only [← c17429, ← c17430, k3, ← c17431] at e_262_9
  have e_267_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_1
  simp only [← c17426, ← c17427, ← c17428] at e_267_1
  have hr := e_262_9
  simp only [e_267_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 39) = a (.virt 201) := by
  have c17432 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17432

theorem publicBatchWrapper86_f2417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10404)) (a (.wire 165 63)) (a (.virt 18521)) (a (.virt 18522)) := by
  have c17433 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17434 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17435 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17436 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17437 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17438 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17439 := (publicBatchWrapper86_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17440 := (publicBatchWrapper86_copies545 a h).1
  have c17441 := (publicBatchWrapper86_copies545 a h).2.1
  have c17442 := (publicBatchWrapper86_copies545 a h).2.2.1
  have c17443 := (publicBatchWrapper86_copies545 a h).2.2.2.1
  have c17444 := (publicBatchWrapper86_copies545 a h).2.2.2.2.1
  have c17445 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.1
  have c17446 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.1
  have c17447 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.1
  have c17448 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.1
  have c17449 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_265_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_15
  simp only [← c17433, k3, ← c17434, k3, ← c17435] at e_265_15
  have e_265_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_16
  simp only [← c17436, ← c17437, k3, ← c17438] at e_265_16
  have e_265_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_17
  simp only [← c17445, ← c17446, k3, ← c17447] at e_265_17
  have e_266_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_9
  simp only [← c17439, ← c17440, ← c17441] at e_266_9
  have e_266_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_10
  simp only [← c17442, ← c17443, ← c17444] at e_266_10
  refine ⟨?_, ?_⟩
  · have hc := e_266_9
    simp only [e_265_16] at hc
    linear_combination c17448.trans k2 - hc
  · have hc := e_265_17
    simp only [e_265_15, e_266_10, e_265_16] at hc
    linear_combination c17449.trans k2 - hc

theorem publicBatchWrapper86_f2418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 43) = bor (a (.wire 51 75)) (a (.virt 18521)) := by
  have c17450 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17451 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17452 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17453 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17454 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17455 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_10
  simp only [← c17453, ← c17454, k3, ← c17455] at e_262_10
  have e_267_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_2
  simp only [← c17450, ← c17451, ← c17452] at e_267_2
  have hr := e_262_10
  simp only [e_267_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 43) = a (.virt 201) := by
  have c17456 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17456

theorem publicBatchWrapper86_f2420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10405)) (a (.wire 165 15)) (a (.virt 18523)) (a (.virt 18524)) := by
  have c17457 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17458 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17459 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17460 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17461 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17462 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17463 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17464 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17465 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17466 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17467 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17468 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17469 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17470 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17471 := (publicBatchWrapper86_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17472 := (publicBatchWrapper86_copies546 a h).1
  have c17473 := (publicBatchWrapper86_copies546 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_265_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_18
  simp only [← c17457, k3, ← c17458, k3, ← c17459] at e_265_18
  have e_265_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_19
  simp only [← c17460, ← c17461, k3, ← c17462] at e_265_19
  have e_266_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_11
  simp only [← c17463, ← c17464, ← c17465] at e_266_11
  have e_266_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_12
  simp only [← c17466, ← c17467, ← c17468] at e_266_12
  have e_268_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_0
  simp only [← c17469, ← c17470, k3, ← c17471] at e_268_0
  refine ⟨?_, ?_⟩
  · have hc := e_266_11
    simp only [e_265_19] at hc
    linear_combination c17472.trans k2 - hc
  · have hc := e_268_0
    simp only [e_265_18, e_266_12, e_265_19] at hc
    linear_combination c17473.trans k2 - hc

theorem publicBatchWrapper86_f2421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10406)) (a (.wire 165 23)) (a (.virt 18525)) (a (.virt 18526)) := by
  have c17474 := (publicBatchWrapper86_copies546 a h).2.2.1
  have c17475 := (publicBatchWrapper86_copies546 a h).2.2.2.1
  have c17476 := (publicBatchWrapper86_copies546 a h).2.2.2.2.1
  have c17477 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.1
  have c17478 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.1
  have c17479 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.1
  have c17480 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.1
  have c17481 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.1
  have c17482 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17483 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17484 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17485 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17486 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17487 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17488 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17489 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17490 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_266_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_13
  simp only [← c17480, ← c17481, ← c17482] at e_266_13
  have e_266_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_14
  simp only [← c17483, ← c17484, ← c17485] at e_266_14
  have e_268_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_1
  simp only [← c17474, k3, ← c17475, k3, ← c17476] at e_268_1
  have e_268_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_2
  simp only [← c17477, ← c17478, k3, ← c17479] at e_268_2
  have e_268_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_3
  simp only [← c17486, ← c17487, k3, ← c17488] at e_268_3
  refine ⟨?_, ?_⟩
  · have hc := e_266_13
    simp only [e_268_2] at hc
    linear_combination c17489.trans k2 - hc
  · have hc := e_268_3
    simp only [e_268_1, e_266_14, e_268_2] at hc
    linear_combination c17490.trans k2 - hc

theorem publicBatchWrapper86_f2422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10407)) (a (.wire 165 31)) (a (.virt 18527)) (a (.virt 18528)) := by
  have c17491 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17492 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17493 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17494 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17495 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17496 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17497 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17498 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17499 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17500 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17501 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17502 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17503 := (publicBatchWrapper86_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17504 := (publicBatchWrapper86_copies547 a h).1
  have c17505 := (publicBatchWrapper86_copies547 a h).2.1
  have c17506 := (publicBatchWrapper86_copies547 a h).2.2.1
  have c17507 := (publicBatchWrapper86_copies547 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_266_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_15
  simp only [← c17497, ← c17498, ← c17499] at e_266_15
  have e_266_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_16
  simp only [← c17500, ← c17501, ← c17502] at e_266_16
  have e_268_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_4
  simp only [← c17491, k3, ← c17492, k3, ← c17493] at e_268_4
  have e_268_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_5
  simp only [← c17494, ← c17495, k3, ← c17496] at e_268_5
  have e_268_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_6
  simp only [← c17503, ← c17504, k3, ← c17505] at e_268_6
  refine ⟨?_, ?_⟩
  · have hc := e_266_15
    simp only [e_268_5] at hc
    linear_combination c17506.trans k2 - hc
  · have hc := e_268_6
    simp only [e_268_4, e_266_16, e_268_5] at hc
    linear_combination c17507.trans k2 - hc

theorem publicBatchWrapper86_f2423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10408)) (a (.wire 165 39)) (a (.virt 18529)) (a (.virt 18530)) := by
  have c17508 := (publicBatchWrapper86_copies547 a h).2.2.2.2.1
  have c17509 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.1
  have c17510 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.1
  have c17511 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.1
  have c17512 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.1
  have c17513 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.1
  have c17514 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17515 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17516 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17517 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17518 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17519 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17520 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17521 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17522 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17523 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17524 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_266_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_17
  simp only [← c17514, ← c17515, ← c17516] at e_266_17
  have e_266_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_18
  simp only [← c17517, ← c17518, ← c17519] at e_266_18
  have e_268_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_7
  simp only [← c17508, k3, ← c17509, k3, ← c17510] at e_268_7
  have e_268_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_8
  simp only [← c17511, ← c17512, k3, ← c17513] at e_268_8
  have e_268_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_9
  simp only [← c17520, ← c17521, k3, ← c17522] at e_268_9
  refine ⟨?_, ?_⟩
  · have hc := e_266_17
    simp only [e_268_8] at hc
    linear_combination c17523.trans k2 - hc
  · have hc := e_268_9
    simp only [e_268_7, e_266_18, e_268_8] at hc
    linear_combination c17524.trans k2 - hc

theorem publicBatchWrapper86_f2424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 266 79) = band (a (.virt 18523)) (a (.virt 18525)) := by
  have c17525 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17526 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17527 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_266_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_19
  simp only [← c17525, ← c17526, ← c17527] at e_266_19
  have hr := e_266_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 269 3) = band (a (.virt 18527)) (a (.virt 18529)) := by
  have c17528 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17529 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17530 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_269_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_0
  simp only [← c17528, ← c17529, ← c17530] at e_269_0
  have hr := e_269_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 269 7) = band (a (.wire 266 79)) (a (.wire 269 3)) := by
  have c17531 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17532 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17533 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_269_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_1
  simp only [← c17531, ← c17532, ← c17533] at e_269_1
  have hr := e_269_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 47) = bor (a (.wire 51 75)) (a (.wire 269 7)) := by
  have c17534 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17535 := (publicBatchWrapper86_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17536 := (publicBatchWrapper86_copies548 a h).1
  have c17537 := (publicBatchWrapper86_copies548 a h).2.1
  have c17538 := (publicBatchWrapper86_copies548 a h).2.2.1
  have c17539 := (publicBatchWrapper86_copies548 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_11
  simp only [← c17537, ← c17538, k3, ← c17539] at e_262_11
  have e_267_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_3
  simp only [← c17534, ← c17535, ← c17536] at e_267_3
  have hr := e_262_11
  simp only [e_267_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 47) = a (.virt 201) := by
  have c17540 := (publicBatchWrapper86_copies548 a h).2.2.2.2.1
  exact c17540

theorem publicBatchWrapper86_f2429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10603)) (a (.wire 165 55)) (a (.virt 18531)) (a (.virt 18532)) := by
  have c17541 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.1
  have c17542 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.1
  have c17543 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.1
  have c17544 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.1
  have c17545 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.1
  have c17546 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17547 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17548 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17549 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17550 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17551 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17552 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17553 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17554 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17555 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17556 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17557 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_268_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_10
  simp only [← c17541, k3, ← c17542, k3, ← c17543] at e_268_10
  have e_268_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_11
  simp only [← c17544, ← c17545, k3, ← c17546] at e_268_11
  have e_268_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_12
  simp only [← c17553, ← c17554, k3, ← c17555] at e_268_12
  have e_269_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_2
  simp only [← c17547, ← c17548, ← c17549] at e_269_2
  have e_269_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_3
  simp only [← c17550, ← c17551, ← c17552] at e_269_3
  refine ⟨?_, ?_⟩
  · have hc := e_269_2
    simp only [e_268_11] at hc
    linear_combination c17556.trans k2 - hc
  · have hc := e_268_12
    simp only [e_268_10, e_269_3, e_268_11] at hc
    linear_combination c17557.trans k2 - hc

theorem publicBatchWrapper86_f2430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 51) = bor (a (.wire 53 39)) (a (.virt 18531)) := by
  have c17558 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17559 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17560 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17561 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17562 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17563 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_12
  simp only [← c17561, ← c17562, k3, ← c17563] at e_262_12
  have e_267_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_4
  simp only [← c17558, ← c17559, ← c17560] at e_267_4
  have hr := e_262_12
  simp only [e_267_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 51) = a (.virt 201) := by
  have c17564 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17564

theorem publicBatchWrapper86_f2432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10604)) (a (.wire 165 63)) (a (.virt 18533)) (a (.virt 18534)) := by
  have c17565 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17566 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17567 := (publicBatchWrapper86_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17568 := (publicBatchWrapper86_copies549 a h).1
  have c17569 := (publicBatchWrapper86_copies549 a h).2.1
  have c17570 := (publicBatchWrapper86_copies549 a h).2.2.1
  have c17571 := (publicBatchWrapper86_copies549 a h).2.2.2.1
  have c17572 := (publicBatchWrapper86_copies549 a h).2.2.2.2.1
  have c17573 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.1
  have c17574 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.1
  have c17575 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.1
  have c17576 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.1
  have c17577 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.1
  have c17578 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17579 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17580 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17581 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_268_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_13
  simp only [← c17565, k3, ← c17566, k3, ← c17567] at e_268_13
  have e_268_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_14
  simp only [← c17568, ← c17569, k3, ← c17570] at e_268_14
  have e_268_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_15
  simp only [← c17577, ← c17578, k3, ← c17579] at e_268_15
  have e_269_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_4
  simp only [← c17571, ← c17572, ← c17573] at e_269_4
  have e_269_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_5
  simp only [← c17574, ← c17575, ← c17576] at e_269_5
  refine ⟨?_, ?_⟩
  · have hc := e_269_4
    simp only [e_268_14] at hc
    linear_combination c17580.trans k2 - hc
  · have hc := e_268_15
    simp only [e_268_13, e_269_5, e_268_14] at hc
    linear_combination c17581.trans k2 - hc

theorem publicBatchWrapper86_f2433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 55) = bor (a (.wire 53 39)) (a (.virt 18533)) := by
  have c17582 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17583 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17584 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17585 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17586 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17587 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_13
  simp only [← c17585, ← c17586, k3, ← c17587] at e_262_13
  have e_267_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_5
  simp only [← c17582, ← c17583, ← c17584] at e_267_5
  have hr := e_262_13
  simp only [e_267_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 55) = a (.virt 201) := by
  have c17588 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17588

theorem publicBatchWrapper86_f2435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10605)) (a (.wire 165 15)) (a (.virt 18535)) (a (.virt 18536)) := by
  have c17589 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17590 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17591 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17592 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17593 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17594 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17595 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17596 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17597 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17598 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17599 := (publicBatchWrapper86_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17600 := (publicBatchWrapper86_copies550 a h).1
  have c17601 := (publicBatchWrapper86_copies550 a h).2.1
  have c17602 := (publicBatchWrapper86_copies550 a h).2.2.1
  have c17603 := (publicBatchWrapper86_copies550 a h).2.2.2.1
  have c17604 := (publicBatchWrapper86_copies550 a h).2.2.2.2.1
  have c17605 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_268_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_16
  simp only [← c17589, k3, ← c17590, k3, ← c17591] at e_268_16
  have e_268_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_17
  simp only [← c17592, ← c17593, k3, ← c17594] at e_268_17
  have e_268_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_18
  simp only [← c17601, ← c17602, k3, ← c17603] at e_268_18
  have e_269_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_6
  simp only [← c17595, ← c17596, ← c17597] at e_269_6
  have e_269_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_7
  simp only [← c17598, ← c17599, ← c17600] at e_269_7
  refine ⟨?_, ?_⟩
  · have hc := e_269_6
    simp only [e_268_17] at hc
    linear_combination c17604.trans k2 - hc
  · have hc := e_268_18
    simp only [e_268_16, e_269_7, e_268_17] at hc
    linear_combination c17605.trans k2 - hc

theorem publicBatchWrapper86_f2436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10606)) (a (.wire 165 23)) (a (.virt 18537)) (a (.virt 18538)) := by
  have c17606 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.1
  have c17607 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.1
  have c17608 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.1
  have c17609 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.1
  have c17610 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17611 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17612 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17613 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17614 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17615 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17616 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17617 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17618 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17619 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17620 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17621 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17622 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_268_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_19
  simp only [← c17606, k3, ← c17607, k3, ← c17608] at e_268_19
  have e_269_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_8
  simp only [← c17612, ← c17613, ← c17614] at e_269_8
  have e_269_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_9
  simp only [← c17615, ← c17616, ← c17617] at e_269_9
  have e_270_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_0
  simp only [← c17609, ← c17610, k3, ← c17611] at e_270_0
  have e_270_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_1
  simp only [← c17618, ← c17619, k3, ← c17620] at e_270_1
  refine ⟨?_, ?_⟩
  · have hc := e_269_8
    simp only [e_270_0] at hc
    linear_combination c17621.trans k2 - hc
  · have hc := e_270_1
    simp only [e_268_19, e_269_9, e_270_0] at hc
    linear_combination c17622.trans k2 - hc

theorem publicBatchWrapper86_f2437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10607)) (a (.wire 165 31)) (a (.virt 18539)) (a (.virt 18540)) := by
  have c17623 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17624 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17625 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17626 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17627 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17628 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17629 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17630 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17631 := (publicBatchWrapper86_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17632 := (publicBatchWrapper86_copies551 a h).1
  have c17633 := (publicBatchWrapper86_copies551 a h).2.1
  have c17634 := (publicBatchWrapper86_copies551 a h).2.2.1
  have c17635 := (publicBatchWrapper86_copies551 a h).2.2.2.1
  have c17636 := (publicBatchWrapper86_copies551 a h).2.2.2.2.1
  have c17637 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.1
  have c17638 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.1
  have c17639 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_269_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_10
  simp only [← c17629, ← c17630, ← c17631] at e_269_10
  have e_269_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_11
  simp only [← c17632, ← c17633, ← c17634] at e_269_11
  have e_270_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_2
  simp only [← c17623, k3, ← c17624, k3, ← c17625] at e_270_2
  have e_270_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_3
  simp only [← c17626, ← c17627, k3, ← c17628] at e_270_3
  have e_270_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_4
  simp only [← c17635, ← c17636, k3, ← c17637] at e_270_4
  refine ⟨?_, ?_⟩
  · have hc := e_269_10
    simp only [e_270_3] at hc
    linear_combination c17638.trans k2 - hc
  · have hc := e_270_4
    simp only [e_270_2, e_269_11, e_270_3] at hc
    linear_combination c17639.trans k2 - hc

theorem publicBatchWrapper86_f2438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10608)) (a (.wire 165 39)) (a (.virt 18541)) (a (.virt 18542)) := by
  have c17640 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.1
  have c17641 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.1
  have c17642 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17643 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17644 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17645 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17646 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17647 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17648 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17649 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17650 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17651 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17652 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17653 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17654 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17655 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17656 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_269_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_12
  simp only [← c17646, ← c17647, ← c17648] at e_269_12
  have e_269_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_13
  simp only [← c17649, ← c17650, ← c17651] at e_269_13
  have e_270_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_5
  simp only [← c17640, k3, ← c17641, k3, ← c17642] at e_270_5
  have e_270_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_6
  simp only [← c17643, ← c17644, k3, ← c17645] at e_270_6
  have e_270_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_7
  simp only [← c17652, ← c17653, k3, ← c17654] at e_270_7
  refine ⟨?_, ?_⟩
  · have hc := e_269_12
    simp only [e_270_6] at hc
    linear_combination c17655.trans k2 - hc
  · have hc := e_270_7
    simp only [e_270_5, e_269_13, e_270_6] at hc
    linear_combination c17656.trans k2 - hc

theorem publicBatchWrapper86_f2439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 269 59) = band (a (.virt 18535)) (a (.virt 18537)) := by
  have c17657 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17658 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17659 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_269_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_14
  simp only [← c17657, ← c17658, ← c17659] at e_269_14
  have hr := e_269_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 269 63) = band (a (.virt 18539)) (a (.virt 18541)) := by
  have c17660 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17661 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17662 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_269_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_15
  simp only [← c17660, ← c17661, ← c17662] at e_269_15
  have hr := e_269_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 269 67) = band (a (.wire 269 59)) (a (.wire 269 63)) := by
  have c17663 := (publicBatchWrapper86_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17664 := (publicBatchWrapper86_copies552 a h).1
  have c17665 := (publicBatchWrapper86_copies552 a h).2.1
  have e_269_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_16
  simp only [← c17663, ← c17664, ← c17665] at e_269_16
  have hr := e_269_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 59) = bor (a (.wire 53 39)) (a (.wire 269 67)) := by
  have c17666 := (publicBatchWrapper86_copies552 a h).2.2.1
  have c17667 := (publicBatchWrapper86_copies552 a h).2.2.2.1
  have c17668 := (publicBatchWrapper86_copies552 a h).2.2.2.2.1
  have c17669 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.1
  have c17670 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.1
  have c17671 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_14
  simp only [← c17669, ← c17670, k3, ← c17671] at e_262_14
  have e_267_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_6
  simp only [← c17666, ← c17667, ← c17668] at e_267_6
  have hr := e_262_14
  simp only [e_267_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 59) = a (.virt 201) := by
  have c17672 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.1
  exact c17672

theorem publicBatchWrapper86_f2444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10803)) (a (.wire 165 55)) (a (.virt 18543)) (a (.virt 18544)) := by
  have c17673 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.1
  have c17674 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17675 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17676 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17677 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17678 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17679 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17680 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17681 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17682 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17683 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17684 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17685 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17686 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17687 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17688 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17689 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_269_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_17
  simp only [← c17679, ← c17680, ← c17681] at e_269_17
  have e_269_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_18
  simp only [← c17682, ← c17683, ← c17684] at e_269_18
  have e_270_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_8
  simp only [← c17673, k3, ← c17674, k3, ← c17675] at e_270_8
  have e_270_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_9
  simp only [← c17676, ← c17677, k3, ← c17678] at e_270_9
  have e_270_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_10
  simp only [← c17685, ← c17686, k3, ← c17687] at e_270_10
  refine ⟨?_, ?_⟩
  · have hc := e_269_17
    simp only [e_270_9] at hc
    linear_combination c17688.trans k2 - hc
  · have hc := e_270_10
    simp only [e_270_8, e_269_18, e_270_9] at hc
    linear_combination c17689.trans k2 - hc

theorem publicBatchWrapper86_f2445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 63) = bor (a (.wire 54 3)) (a (.virt 18543)) := by
  have c17690 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17691 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17692 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17693 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17694 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17695 := (publicBatchWrapper86_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_15
  simp only [← c17693, ← c17694, k3, ← c17695] at e_262_15
  have e_267_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_7
  simp only [← c17690, ← c17691, ← c17692] at e_267_7
  have hr := e_262_15
  simp only [e_267_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 63) = a (.virt 201) := by
  have c17696 := (publicBatchWrapper86_copies553 a h).1
  exact c17696

theorem publicBatchWrapper86_f2447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10804)) (a (.wire 165 63)) (a (.virt 18545)) (a (.virt 18546)) := by
  have c17697 := (publicBatchWrapper86_copies553 a h).2.1
  have c17698 := (publicBatchWrapper86_copies553 a h).2.2.1
  have c17699 := (publicBatchWrapper86_copies553 a h).2.2.2.1
  have c17700 := (publicBatchWrapper86_copies553 a h).2.2.2.2.1
  have c17701 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.1
  have c17702 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.1
  have c17703 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.1
  have c17704 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.1
  have c17705 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.1
  have c17706 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17707 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17708 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17709 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17710 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17711 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17712 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17713 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_269_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_19
  simp only [← c17703, ← c17704, ← c17705] at e_269_19
  have e_270_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_11
  simp only [← c17697, k3, ← c17698, k3, ← c17699] at e_270_11
  have e_270_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_12
  simp only [← c17700, ← c17701, k3, ← c17702] at e_270_12
  have e_270_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_13
  simp only [← c17709, ← c17710, k3, ← c17711] at e_270_13
  have e_271_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_0
  simp only [← c17706, ← c17707, ← c17708] at e_271_0
  refine ⟨?_, ?_⟩
  · have hc := e_269_19
    simp only [e_270_12] at hc
    linear_combination c17712.trans k2 - hc
  · have hc := e_270_13
    simp only [e_270_11, e_271_0, e_270_12] at hc
    linear_combination c17713.trans k2 - hc

theorem publicBatchWrapper86_f2448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 67) = bor (a (.wire 54 3)) (a (.virt 18545)) := by
  have c17714 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17715 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17716 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17717 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17718 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17719 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_16
  simp only [← c17717, ← c17718, k3, ← c17719] at e_262_16
  have e_267_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_8
  simp only [← c17714, ← c17715, ← c17716] at e_267_8
  have hr := e_262_16
  simp only [e_267_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 67) = a (.virt 201) := by
  have c17720 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17720

theorem publicBatchWrapper86_f2450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10805)) (a (.wire 165 15)) (a (.virt 18547)) (a (.virt 18548)) := by
  have c17721 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17722 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17723 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17724 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17725 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17726 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17727 := (publicBatchWrapper86_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17728 := (publicBatchWrapper86_copies554 a h).1
  have c17729 := (publicBatchWrapper86_copies554 a h).2.1
  have c17730 := (publicBatchWrapper86_copies554 a h).2.2.1
  have c17731 := (publicBatchWrapper86_copies554 a h).2.2.2.1
  have c17732 := (publicBatchWrapper86_copies554 a h).2.2.2.2.1
  have c17733 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.1
  have c17734 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.1
  have c17735 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.1
  have c17736 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.1
  have c17737 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_270_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_14
  simp only [← c17721, k3, ← c17722, k3, ← c17723] at e_270_14
  have e_270_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_15
  simp only [← c17724, ← c17725, k3, ← c17726] at e_270_15
  have e_270_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_16
  simp only [← c17733, ← c17734, k3, ← c17735] at e_270_16
  have e_271_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_1
  simp only [← c17727, ← c17728, ← c17729] at e_271_1
  have e_271_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_2
  simp only [← c17730, ← c17731, ← c17732] at e_271_2
  refine ⟨?_, ?_⟩
  · have hc := e_271_1
    simp only [e_270_15] at hc
    linear_combination c17736.trans k2 - hc
  · have hc := e_270_16
    simp only [e_270_14, e_271_2, e_270_15] at hc
    linear_combination c17737.trans k2 - hc

theorem publicBatchWrapper86_f2451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10806)) (a (.wire 165 23)) (a (.virt 18549)) (a (.virt 18550)) := by
  have c17738 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17739 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17740 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17741 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17742 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17743 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17744 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17745 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17746 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17747 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17748 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17749 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17750 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17751 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17752 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17753 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17754 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_270_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_17
  simp only [← c17738, k3, ← c17739, k3, ← c17740] at e_270_17
  have e_270_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_18
  simp only [← c17741, ← c17742, k3, ← c17743] at e_270_18
  have e_270_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_19
  simp only [← c17750, ← c17751, k3, ← c17752] at e_270_19
  have e_271_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_3
  simp only [← c17744, ← c17745, ← c17746] at e_271_3
  have e_271_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_4
  simp only [← c17747, ← c17748, ← c17749] at e_271_4
  refine ⟨?_, ?_⟩
  · have hc := e_271_3
    simp only [e_270_18] at hc
    linear_combination c17753.trans k2 - hc
  · have hc := e_270_19
    simp only [e_270_17, e_271_4, e_270_18] at hc
    linear_combination c17754.trans k2 - hc

theorem publicBatchWrapper86_f2452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10807)) (a (.wire 165 31)) (a (.virt 18551)) (a (.virt 18552)) := by
  have c17755 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17756 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17757 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17758 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17759 := (publicBatchWrapper86_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17760 := (publicBatchWrapper86_copies555 a h).1
  have c17761 := (publicBatchWrapper86_copies555 a h).2.1
  have c17762 := (publicBatchWrapper86_copies555 a h).2.2.1
  have c17763 := (publicBatchWrapper86_copies555 a h).2.2.2.1
  have c17764 := (publicBatchWrapper86_copies555 a h).2.2.2.2.1
  have c17765 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.1
  have c17766 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.1
  have c17767 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.1
  have c17768 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.1
  have c17769 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.1
  have c17770 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17771 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_271_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_5
  simp only [← c17761, ← c17762, ← c17763] at e_271_5
  have e_271_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_6
  simp only [← c17764, ← c17765, ← c17766] at e_271_6
  have e_272_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_0
  simp only [← c17755, k3, ← c17756, k3, ← c17757] at e_272_0
  have e_272_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_1
  simp only [← c17758, ← c17759, k3, ← c17760] at e_272_1
  have e_272_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_2
  simp only [← c17767, ← c17768, k3, ← c17769] at e_272_2
  refine ⟨?_, ?_⟩
  · have hc := e_271_5
    simp only [e_272_1] at hc
    linear_combination c17770.trans k2 - hc
  · have hc := e_272_2
    simp only [e_272_0, e_271_6, e_272_1] at hc
    linear_combination c17771.trans k2 - hc

theorem publicBatchWrapper86_f2453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10808)) (a (.wire 165 39)) (a (.virt 18553)) (a (.virt 18554)) := by
  have c17772 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17773 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17774 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17775 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17776 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17777 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17778 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17779 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17780 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17781 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17782 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17783 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17784 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17785 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17786 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17787 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17788 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_271_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_7
  simp only [← c17778, ← c17779, ← c17780] at e_271_7
  have e_271_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_8
  simp only [← c17781, ← c17782, ← c17783] at e_271_8
  have e_272_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_3
  simp only [← c17772, k3, ← c17773, k3, ← c17774] at e_272_3
  have e_272_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_4
  simp only [← c17775, ← c17776, k3, ← c17777] at e_272_4
  have e_272_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_5
  simp only [← c17784, ← c17785, k3, ← c17786] at e_272_5
  refine ⟨?_, ?_⟩
  · have hc := e_271_7
    simp only [e_272_4] at hc
    linear_combination c17787.trans k2 - hc
  · have hc := e_272_5
    simp only [e_272_3, e_271_8, e_272_4] at hc
    linear_combination c17788.trans k2 - hc

theorem publicBatchWrapper86_f2454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 271 39) = band (a (.virt 18547)) (a (.virt 18549)) := by
  have c17789 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17790 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17791 := (publicBatchWrapper86_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_271_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_9
  simp only [← c17789, ← c17790, ← c17791] at e_271_9
  have hr := e_271_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 271 43) = band (a (.virt 18551)) (a (.virt 18553)) := by
  have c17792 := (publicBatchWrapper86_copies556 a h).1
  have c17793 := (publicBatchWrapper86_copies556 a h).2.1
  have c17794 := (publicBatchWrapper86_copies556 a h).2.2.1
  have e_271_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_10
  simp only [← c17792, ← c17793, ← c17794] at e_271_10
  have hr := e_271_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 271 47) = band (a (.wire 271 39)) (a (.wire 271 43)) := by
  have c17795 := (publicBatchWrapper86_copies556 a h).2.2.2.1
  have c17796 := (publicBatchWrapper86_copies556 a h).2.2.2.2.1
  have c17797 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.1
  have e_271_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_11
  simp only [← c17795, ← c17796, ← c17797] at e_271_11
  have hr := e_271_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 71) = bor (a (.wire 54 3)) (a (.wire 271 47)) := by
  have c17798 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.1
  have c17799 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.1
  have c17800 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.1
  have c17801 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.1
  have c17802 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17803 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_17
  simp only [← c17801, ← c17802, k3, ← c17803] at e_262_17
  have e_267_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_9
  simp only [← c17798, ← c17799, ← c17800] at e_267_9
  have hr := e_262_17
  simp only [e_267_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 71) = a (.virt 201) := by
  have c17804 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17804

theorem publicBatchWrapper86_f2459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11003)) (a (.wire 165 55)) (a (.virt 18555)) (a (.virt 18556)) := by
  have c17805 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17806 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17807 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17808 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17809 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17810 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17811 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17812 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17813 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17814 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17815 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17816 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17817 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17818 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17819 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17820 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17821 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_271_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_12
  simp only [← c17811, ← c17812, ← c17813] at e_271_12
  have e_271_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_13
  simp only [← c17814, ← c17815, ← c17816] at e_271_13
  have e_272_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_6
  simp only [← c17805, k3, ← c17806, k3, ← c17807] at e_272_6
  have e_272_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_7
  simp only [← c17808, ← c17809, k3, ← c17810] at e_272_7
  have e_272_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_8
  simp only [← c17817, ← c17818, k3, ← c17819] at e_272_8
  refine ⟨?_, ?_⟩
  · have hc := e_271_12
    simp only [e_272_7] at hc
    linear_combination c17820.trans k2 - hc
  · have hc := e_272_8
    simp only [e_272_6, e_271_13, e_272_7] at hc
    linear_combination c17821.trans k2 - hc

theorem publicBatchWrapper86_f2460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 75) = bor (a (.wire 54 47)) (a (.virt 18555)) := by
  have c17822 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17823 := (publicBatchWrapper86_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17824 := (publicBatchWrapper86_copies557 a h).1
  have c17825 := (publicBatchWrapper86_copies557 a h).2.1
  have c17826 := (publicBatchWrapper86_copies557 a h).2.2.1
  have c17827 := (publicBatchWrapper86_copies557 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_18
  simp only [← c17825, ← c17826, k3, ← c17827] at e_262_18
  have e_267_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_10
  simp only [← c17822, ← c17823, ← c17824] at e_267_10
  have hr := e_262_18
  simp only [e_267_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 75) = a (.virt 201) := by
  have c17828 := (publicBatchWrapper86_copies557 a h).2.2.2.2.1
  exact c17828

theorem publicBatchWrapper86_f2462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11004)) (a (.wire 165 63)) (a (.virt 18557)) (a (.virt 18558)) := by
  have c17829 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.1
  have c17830 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.1
  have c17831 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.1
  have c17832 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.1
  have c17833 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.1
  have c17834 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17835 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17836 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17837 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17838 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17839 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17840 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17841 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17842 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17843 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17844 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17845 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_271_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_14
  simp only [← c17835, ← c17836, ← c17837] at e_271_14
  have e_271_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_15
  simp only [← c17838, ← c17839, ← c17840] at e_271_15
  have e_272_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_9
  simp only [← c17829, k3, ← c17830, k3, ← c17831] at e_272_9
  have e_272_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_10
  simp only [← c17832, ← c17833, k3, ← c17834] at e_272_10
  have e_272_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_11
  simp only [← c17841, ← c17842, k3, ← c17843] at e_272_11
  refine ⟨?_, ?_⟩
  · have hc := e_271_14
    simp only [e_272_10] at hc
    linear_combination c17844.trans k2 - hc
  · have hc := e_272_11
    simp only [e_272_9, e_271_15, e_272_10] at hc
    linear_combination c17845.trans k2 - hc

theorem publicBatchWrapper86_f2463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 79) = bor (a (.wire 54 47)) (a (.virt 18557)) := by
  have c17846 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17847 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17848 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17849 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17850 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17851 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_262_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_19
  simp only [← c17849, ← c17850, k3, ← c17851] at e_262_19
  have e_267_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_11
  simp only [← c17846, ← c17847, ← c17848] at e_267_11
  have hr := e_262_19
  simp only [e_267_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 262 79) = a (.virt 201) := by
  have c17852 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17852

theorem publicBatchWrapper86_f2465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11005)) (a (.wire 165 15)) (a (.virt 18559)) (a (.virt 18560)) := by
  have c17853 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17854 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17855 := (publicBatchWrapper86_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17856 := (publicBatchWrapper86_copies558 a h).1
  have c17857 := (publicBatchWrapper86_copies558 a h).2.1
  have c17858 := (publicBatchWrapper86_copies558 a h).2.2.1
  have c17859 := (publicBatchWrapper86_copies558 a h).2.2.2.1
  have c17860 := (publicBatchWrapper86_copies558 a h).2.2.2.2.1
  have c17861 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.1
  have c17862 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.1
  have c17863 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.1
  have c17864 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.1
  have c17865 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.1
  have c17866 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17867 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17868 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17869 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_271_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_16
  simp only [← c17859, ← c17860, ← c17861] at e_271_16
  have e_271_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_17
  simp only [← c17862, ← c17863, ← c17864] at e_271_17
  have e_272_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_12
  simp only [← c17853, k3, ← c17854, k3, ← c17855] at e_272_12
  have e_272_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_13
  simp only [← c17856, ← c17857, k3, ← c17858] at e_272_13
  have e_272_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_14
  simp only [← c17865, ← c17866, k3, ← c17867] at e_272_14
  refine ⟨?_, ?_⟩
  · have hc := e_271_16
    simp only [e_272_13] at hc
    linear_combination c17868.trans k2 - hc
  · have hc := e_272_14
    simp only [e_272_12, e_271_17, e_272_13] at hc
    linear_combination c17869.trans k2 - hc

theorem publicBatchWrapper86_f2466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11006)) (a (.wire 165 23)) (a (.virt 18561)) (a (.virt 18562)) := by
  have c17870 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17871 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17872 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17873 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17874 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17875 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17876 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17877 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17878 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17879 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17880 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17881 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17882 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17883 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17884 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17885 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17886 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_271_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_18
  simp only [← c17876, ← c17877, ← c17878] at e_271_18
  have e_271_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_19
  simp only [← c17879, ← c17880, ← c17881] at e_271_19
  have e_272_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_15
  simp only [← c17870, k3, ← c17871, k3, ← c17872] at e_272_15
  have e_272_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_16
  simp only [← c17873, ← c17874, k3, ← c17875] at e_272_16
  have e_272_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_17
  simp only [← c17882, ← c17883, k3, ← c17884] at e_272_17
  refine ⟨?_, ?_⟩
  · have hc := e_271_18
    simp only [e_272_16] at hc
    linear_combination c17885.trans k2 - hc
  · have hc := e_272_17
    simp only [e_272_15, e_271_19, e_272_16] at hc
    linear_combination c17886.trans k2 - hc

theorem publicBatchWrapper86_f2467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11007)) (a (.wire 165 31)) (a (.virt 18563)) (a (.virt 18564)) := by
  have c17887 := (publicBatchWrapper86_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17888 := (publicBatchWrapper86_copies559 a h).1
  have c17889 := (publicBatchWrapper86_copies559 a h).2.1
  have c17890 := (publicBatchWrapper86_copies559 a h).2.2.1
  have c17891 := (publicBatchWrapper86_copies559 a h).2.2.2.1
  have c17892 := (publicBatchWrapper86_copies559 a h).2.2.2.2.1
  have c17893 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.1
  have c17894 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.1
  have c17895 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.1
  have c17896 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.1
  have c17897 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.1
  have c17898 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17899 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17900 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17901 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17902 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17903 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_272_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_18
  simp only [← c17887, k3, ← c17888, k3, ← c17889] at e_272_18
  have e_272_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_19
  simp only [← c17890, ← c17891, k3, ← c17892] at e_272_19
  have e_273_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_0
  simp only [← c17893, ← c17894, ← c17895] at e_273_0
  have e_273_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_1
  simp only [← c17896, ← c17897, ← c17898] at e_273_1
  have e_274_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_0
  simp only [← c17899, ← c17900, k3, ← c17901] at e_274_0
  refine ⟨?_, ?_⟩
  · have hc := e_273_0
    simp only [e_272_19] at hc
    linear_combination c17902.trans k2 - hc
  · have hc := e_274_0
    simp only [e_272_18, e_273_1, e_272_19] at hc
    linear_combination c17903.trans k2 - hc

theorem publicBatchWrapper86_f2468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11008)) (a (.wire 165 39)) (a (.virt 18565)) (a (.virt 18566)) := by
  have c17904 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17905 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17906 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17907 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17908 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17909 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17910 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17911 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17912 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17913 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17914 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17915 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17916 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17917 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17918 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17919 := (publicBatchWrapper86_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17920 := (publicBatchWrapper86_copies560 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_273_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_2
  simp only [← c17910, ← c17911, ← c17912] at e_273_2
  have e_273_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_3
  simp only [← c17913, ← c17914, ← c17915] at e_273_3
  have e_274_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_1
  simp only [← c17904, k3, ← c17905, k3, ← c17906] at e_274_1
  have e_274_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_2
  simp only [← c17907, ← c17908, k3, ← c17909] at e_274_2
  have e_274_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_3
  simp only [← c17916, ← c17917, k3, ← c17918] at e_274_3
  refine ⟨?_, ?_⟩
  · have hc := e_273_2
    simp only [e_274_2] at hc
    linear_combination c17919.trans k2 - hc
  · have hc := e_274_3
    simp only [e_274_1, e_273_3, e_274_2] at hc
    linear_combination c17920.trans k2 - hc

theorem publicBatchWrapper86_f2469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 273 19) = band (a (.virt 18559)) (a (.virt 18561)) := by
  have c17921 := (publicBatchWrapper86_copies560 a h).2.1
  have c17922 := (publicBatchWrapper86_copies560 a h).2.2.1
  have c17923 := (publicBatchWrapper86_copies560 a h).2.2.2.1
  have e_273_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_4
  simp only [← c17921, ← c17922, ← c17923] at e_273_4
  have hr := e_273_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 273 23) = band (a (.virt 18563)) (a (.virt 18565)) := by
  have c17924 := (publicBatchWrapper86_copies560 a h).2.2.2.2.1
  have c17925 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.1
  have c17926 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.1
  have e_273_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_5
  simp only [← c17924, ← c17925, ← c17926] at e_273_5
  have hr := e_273_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 273 27) = band (a (.wire 273 19)) (a (.wire 273 23)) := by
  have c17927 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.1
  have c17928 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.1
  have c17929 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.1
  have e_273_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_6
  simp only [← c17927, ← c17928, ← c17929] at e_273_6
  have hr := e_273_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 3) = bor (a (.wire 54 47)) (a (.wire 273 27)) := by
  have c17930 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17931 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17932 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17933 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17934 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17935 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_267_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_12
  simp only [← c17930, ← c17931, ← c17932] at e_267_12
  have e_275_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_0
  simp only [← c17933, ← c17934, k3, ← c17935] at e_275_0
  have hr := e_275_0
  simp only [e_267_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 3) = a (.virt 201) := by
  have c17936 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17936

theorem publicBatchWrapper86_f2474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11203)) (a (.wire 165 55)) (a (.virt 18567)) (a (.virt 18568)) := by
  have c17937 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17938 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17939 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17940 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17941 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17942 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17943 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17944 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17945 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17946 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17947 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17948 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17949 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17950 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17951 := (publicBatchWrapper86_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17952 := (publicBatchWrapper86_copies561 a h).1
  have c17953 := (publicBatchWrapper86_copies561 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_273_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_7
  simp only [← c17943, ← c17944, ← c17945] at e_273_7
  have e_273_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_8
  simp only [← c17946, ← c17947, ← c17948] at e_273_8
  have e_274_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_4
  simp only [← c17937, k3, ← c17938, k3, ← c17939] at e_274_4
  have e_274_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_5
  simp only [← c17940, ← c17941, k3, ← c17942] at e_274_5
  have e_274_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_6
  simp only [← c17949, ← c17950, k3, ← c17951] at e_274_6
  refine ⟨?_, ?_⟩
  · have hc := e_273_7
    simp only [e_274_5] at hc
    linear_combination c17952.trans k2 - hc
  · have hc := e_274_6
    simp only [e_274_4, e_273_8, e_274_5] at hc
    linear_combination c17953.trans k2 - hc

theorem publicBatchWrapper86_f2475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 7) = bor (a (.wire 56 11)) (a (.virt 18567)) := by
  have c17954 := (publicBatchWrapper86_copies561 a h).2.2.1
  have c17955 := (publicBatchWrapper86_copies561 a h).2.2.2.1
  have c17956 := (publicBatchWrapper86_copies561 a h).2.2.2.2.1
  have c17957 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.1
  have c17958 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.1
  have c17959 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_267_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_13
  simp only [← c17954, ← c17955, ← c17956] at e_267_13
  have e_275_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_1
  simp only [← c17957, ← c17958, k3, ← c17959] at e_275_1
  have hr := e_275_1
  simp only [e_267_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 7) = a (.virt 201) := by
  have c17960 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.1
  exact c17960

theorem publicBatchWrapper86_f2477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11204)) (a (.wire 165 63)) (a (.virt 18569)) (a (.virt 18570)) := by
  have c17961 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.1
  have c17962 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17963 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17964 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17965 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17966 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17967 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17968 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17969 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17970 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17971 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17972 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17973 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17974 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17975 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17976 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17977 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_273_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_9
  simp only [← c17967, ← c17968, ← c17969] at e_273_9
  have e_273_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_10
  simp only [← c17970, ← c17971, ← c17972] at e_273_10
  have e_274_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_7
  simp only [← c17961, k3, ← c17962, k3, ← c17963] at e_274_7
  have e_274_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_8
  simp only [← c17964, ← c17965, k3, ← c17966] at e_274_8
  have e_274_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_9
  simp only [← c17973, ← c17974, k3, ← c17975] at e_274_9
  refine ⟨?_, ?_⟩
  · have hc := e_273_9
    simp only [e_274_8] at hc
    linear_combination c17976.trans k2 - hc
  · have hc := e_274_9
    simp only [e_274_7, e_273_10, e_274_8] at hc
    linear_combination c17977.trans k2 - hc

theorem publicBatchWrapper86_f2478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 11) = bor (a (.wire 56 11)) (a (.virt 18569)) := by
  have c17978 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17979 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17980 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17981 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17982 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17983 := (publicBatchWrapper86_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_267_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_14
  simp only [← c17978, ← c17979, ← c17980] at e_267_14
  have e_275_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_2
  simp only [← c17981, ← c17982, k3, ← c17983] at e_275_2
  have hr := e_275_2
  simp only [e_267_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 11) = a (.virt 201) := by
  have c17984 := (publicBatchWrapper86_copies562 a h).1
  exact c17984

theorem publicBatchWrapper86_f2480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11205)) (a (.wire 165 15)) (a (.virt 18571)) (a (.virt 18572)) := by
  have c17985 := (publicBatchWrapper86_copies562 a h).2.1
  have c17986 := (publicBatchWrapper86_copies562 a h).2.2.1
  have c17987 := (publicBatchWrapper86_copies562 a h).2.2.2.1
  have c17988 := (publicBatchWrapper86_copies562 a h).2.2.2.2.1
  have c17989 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.1
  have c17990 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.1
  have c17991 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.1
  have c17992 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.1
  have c17993 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.1
  have c17994 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17995 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17996 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17997 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17998 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17999 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18000 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18001 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_273_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_11
  simp only [← c17991, ← c17992, ← c17993] at e_273_11
  have e_273_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_12
  simp only [← c17994, ← c17995, ← c17996] at e_273_12
  have e_274_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_10
  simp only [← c17985, k3, ← c17986, k3, ← c17987] at e_274_10
  have e_274_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_11
  simp only [← c17988, ← c17989, k3, ← c17990] at e_274_11
  have e_274_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_12
  simp only [← c17997, ← c17998, k3, ← c17999] at e_274_12
  refine ⟨?_, ?_⟩
  · have hc := e_273_11
    simp only [e_274_11] at hc
    linear_combination c18000.trans k2 - hc
  · have hc := e_274_12
    simp only [e_274_10, e_273_12, e_274_11] at hc
    linear_combination c18001.trans k2 - hc

theorem publicBatchWrapper86_f2481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11206)) (a (.wire 165 23)) (a (.virt 18573)) (a (.virt 18574)) := by
  have c18002 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18003 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18004 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18005 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18006 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18007 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18008 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18009 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18010 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18011 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18012 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18013 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18014 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18015 := (publicBatchWrapper86_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18016 := (publicBatchWrapper86_copies563 a h).1
  have c18017 := (publicBatchWrapper86_copies563 a h).2.1
  have c18018 := (publicBatchWrapper86_copies563 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_273_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_13
  simp only [← c18008, ← c18009, ← c18010] at e_273_13
  have e_273_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_14
  simp only [← c18011, ← c18012, ← c18013] at e_273_14
  have e_274_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_13
  simp only [← c18002, k3, ← c18003, k3, ← c18004] at e_274_13
  have e_274_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_14
  simp only [← c18005, ← c18006, k3, ← c18007] at e_274_14
  have e_274_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_15
  simp only [← c18014, ← c18015, k3, ← c18016] at e_274_15
  refine ⟨?_, ?_⟩
  · have hc := e_273_13
    simp only [e_274_14] at hc
    linear_combination c18017.trans k2 - hc
  · have hc := e_274_15
    simp only [e_274_13, e_273_14, e_274_14] at hc
    linear_combination c18018.trans k2 - hc

theorem publicBatchWrapper86_f2482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11207)) (a (.wire 165 31)) (a (.virt 18575)) (a (.virt 18576)) := by
  have c18019 := (publicBatchWrapper86_copies563 a h).2.2.2.1
  have c18020 := (publicBatchWrapper86_copies563 a h).2.2.2.2.1
  have c18021 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.1
  have c18022 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.1
  have c18023 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.1
  have c18024 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.1
  have c18025 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.1
  have c18026 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18027 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18028 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18029 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18030 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18031 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18032 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18033 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18034 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18035 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_273_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_15
  simp only [← c18025, ← c18026, ← c18027] at e_273_15
  have e_273_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_16
  simp only [← c18028, ← c18029, ← c18030] at e_273_16
  have e_274_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_16
  simp only [← c18019, k3, ← c18020, k3, ← c18021] at e_274_16
  have e_274_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_17
  simp only [← c18022, ← c18023, k3, ← c18024] at e_274_17
  have e_274_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_18
  simp only [← c18031, ← c18032, k3, ← c18033] at e_274_18
  refine ⟨?_, ?_⟩
  · have hc := e_273_15
    simp only [e_274_17] at hc
    linear_combination c18034.trans k2 - hc
  · have hc := e_274_18
    simp only [e_274_16, e_273_16, e_274_17] at hc
    linear_combination c18035.trans k2 - hc

theorem publicBatchWrapper86_f2483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11208)) (a (.wire 165 39)) (a (.virt 18577)) (a (.virt 18578)) := by
  have c18036 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18037 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18038 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18039 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18040 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18041 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18042 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18043 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18044 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18045 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18046 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18047 := (publicBatchWrapper86_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18048 := (publicBatchWrapper86_copies564 a h).1
  have c18049 := (publicBatchWrapper86_copies564 a h).2.1
  have c18050 := (publicBatchWrapper86_copies564 a h).2.2.1
  have c18051 := (publicBatchWrapper86_copies564 a h).2.2.2.1
  have c18052 := (publicBatchWrapper86_copies564 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_273_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_17
  simp only [← c18042, ← c18043, ← c18044] at e_273_17
  have e_273_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_18
  simp only [← c18045, ← c18046, ← c18047] at e_273_18
  have e_274_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_19
  simp only [← c18036, k3, ← c18037, k3, ← c18038] at e_274_19
  have e_276_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_0
  simp only [← c18039, ← c18040, k3, ← c18041] at e_276_0
  have e_276_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_1
  simp only [← c18048, ← c18049, k3, ← c18050] at e_276_1
  refine ⟨?_, ?_⟩
  · have hc := e_273_17
    simp only [e_276_0] at hc
    linear_combination c18051.trans k2 - hc
  · have hc := e_276_1
    simp only [e_274_19, e_273_18, e_276_0] at hc
    linear_combination c18052.trans k2 - hc

theorem publicBatchWrapper86_f2484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 273 79) = band (a (.virt 18571)) (a (.virt 18573)) := by
  have c18053 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.1
  have c18054 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.1
  have c18055 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.1
  have e_273_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_19
  simp only [← c18053, ← c18054, ← c18055] at e_273_19
  have hr := e_273_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 277 3) = band (a (.virt 18575)) (a (.virt 18577)) := by
  have c18056 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.1
  have c18057 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.1
  have c18058 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_277_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_0
  simp only [← c18056, ← c18057, ← c18058] at e_277_0
  have hr := e_277_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 277 7) = band (a (.wire 273 79)) (a (.wire 277 3)) := by
  have c18059 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18060 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18061 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_277_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_1
  simp only [← c18059, ← c18060, ← c18061] at e_277_1
  have hr := e_277_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 15) = bor (a (.wire 56 11)) (a (.wire 277 7)) := by
  have c18062 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18063 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18064 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18065 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18066 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18067 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_267_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_15
  simp only [← c18062, ← c18063, ← c18064] at e_267_15
  have e_275_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_3
  simp only [← c18065, ← c18066, k3, ← c18067] at e_275_3
  have hr := e_275_3
  simp only [e_267_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 15) = a (.virt 201) := by
  have c18068 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18068

theorem publicBatchWrapper86_f2489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11403)) (a (.wire 165 55)) (a (.virt 18579)) (a (.virt 18580)) := by
  have c18069 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18070 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18071 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18072 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18073 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18074 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18075 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18076 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18077 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18078 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18079 := (publicBatchWrapper86_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18080 := (publicBatchWrapper86_copies565 a h).1
  have c18081 := (publicBatchWrapper86_copies565 a h).2.1
  have c18082 := (publicBatchWrapper86_copies565 a h).2.2.1
  have c18083 := (publicBatchWrapper86_copies565 a h).2.2.2.1
  have c18084 := (publicBatchWrapper86_copies565 a h).2.2.2.2.1
  have c18085 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_276_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_2
  simp only [← c18069, k3, ← c18070, k3, ← c18071] at e_276_2
  have e_276_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_3
  simp only [← c18072, ← c18073, k3, ← c18074] at e_276_3
  have e_276_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_4
  simp only [← c18081, ← c18082, k3, ← c18083] at e_276_4
  have e_277_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_2
  simp only [← c18075, ← c18076, ← c18077] at e_277_2
  have e_277_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_3
  simp only [← c18078, ← c18079, ← c18080] at e_277_3
  refine ⟨?_, ?_⟩
  · have hc := e_277_2
    simp only [e_276_3] at hc
    linear_combination c18084.trans k2 - hc
  · have hc := e_276_4
    simp only [e_276_2, e_277_3, e_276_3] at hc
    linear_combination c18085.trans k2 - hc

theorem publicBatchWrapper86_f2490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 19) = bor (a (.wire 56 55)) (a (.virt 18579)) := by
  have c18086 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.1
  have c18087 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.1
  have c18088 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.1
  have c18089 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.1
  have c18090 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18091 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_267_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_16
  simp only [← c18086, ← c18087, ← c18088] at e_267_16
  have e_275_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_4
  simp only [← c18089, ← c18090, k3, ← c18091] at e_275_4
  have hr := e_275_4
  simp only [e_267_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 19) = a (.virt 201) := by
  have c18092 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18092

theorem publicBatchWrapper86_f2492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11404)) (a (.wire 165 63)) (a (.virt 18581)) (a (.virt 18582)) := by
  have c18093 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18094 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18095 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18096 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18097 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18098 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18099 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18100 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18101 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18102 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18103 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18104 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18105 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18106 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18107 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18108 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18109 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_276_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_5
  simp only [← c18093, k3, ← c18094, k3, ← c18095] at e_276_5
  have e_276_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_6
  simp only [← c18096, ← c18097, k3, ← c18098] at e_276_6
  have e_276_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_7
  simp only [← c18105, ← c18106, k3, ← c18107] at e_276_7
  have e_277_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_4
  simp only [← c18099, ← c18100, ← c18101] at e_277_4
  have e_277_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_5
  simp only [← c18102, ← c18103, ← c18104] at e_277_5
  refine ⟨?_, ?_⟩
  · have hc := e_277_4
    simp only [e_276_6] at hc
    linear_combination c18108.trans k2 - hc
  · have hc := e_276_7
    simp only [e_276_5, e_277_5, e_276_6] at hc
    linear_combination c18109.trans k2 - hc

theorem publicBatchWrapper86_f2493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 23) = bor (a (.wire 56 55)) (a (.virt 18581)) := by
  have c18110 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18111 := (publicBatchWrapper86_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18112 := (publicBatchWrapper86_copies566 a h).1
  have c18113 := (publicBatchWrapper86_copies566 a h).2.1
  have c18114 := (publicBatchWrapper86_copies566 a h).2.2.1
  have c18115 := (publicBatchWrapper86_copies566 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_267_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_17
  simp only [← c18110, ← c18111, ← c18112] at e_267_17
  have e_275_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_5
  simp only [← c18113, ← c18114, k3, ← c18115] at e_275_5
  have hr := e_275_5
  simp only [e_267_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 23) = a (.virt 201) := by
  have c18116 := (publicBatchWrapper86_copies566 a h).2.2.2.2.1
  exact c18116

theorem publicBatchWrapper86_f2495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11405)) (a (.wire 165 15)) (a (.virt 18583)) (a (.virt 18584)) := by
  have c18117 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.1
  have c18118 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.1
  have c18119 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.1
  have c18120 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.1
  have c18121 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.1
  have c18122 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18123 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18124 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18125 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18126 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18127 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18128 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18129 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18130 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18131 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18132 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18133 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_276_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_8
  simp only [← c18117, k3, ← c18118, k3, ← c18119] at e_276_8
  have e_276_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_9
  simp only [← c18120, ← c18121, k3, ← c18122] at e_276_9
  have e_276_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_10
  simp only [← c18129, ← c18130, k3, ← c18131] at e_276_10
  have e_277_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_6
  simp only [← c18123, ← c18124, ← c18125] at e_277_6
  have e_277_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_7
  simp only [← c18126, ← c18127, ← c18128] at e_277_7
  refine ⟨?_, ?_⟩
  · have hc := e_277_6
    simp only [e_276_9] at hc
    linear_combination c18132.trans k2 - hc
  · have hc := e_276_10
    simp only [e_276_8, e_277_7, e_276_9] at hc
    linear_combination c18133.trans k2 - hc

theorem publicBatchWrapper86_f2496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11406)) (a (.wire 165 23)) (a (.virt 18585)) (a (.virt 18586)) := by
  have c18134 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18135 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18136 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18137 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18138 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18139 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18140 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18141 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18142 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18143 := (publicBatchWrapper86_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18144 := (publicBatchWrapper86_copies567 a h).1
  have c18145 := (publicBatchWrapper86_copies567 a h).2.1
  have c18146 := (publicBatchWrapper86_copies567 a h).2.2.1
  have c18147 := (publicBatchWrapper86_copies567 a h).2.2.2.1
  have c18148 := (publicBatchWrapper86_copies567 a h).2.2.2.2.1
  have c18149 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.1
  have c18150 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_276_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_11
  simp only [← c18134, k3, ← c18135, k3, ← c18136] at e_276_11
  have e_276_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_12
  simp only [← c18137, ← c18138, k3, ← c18139] at e_276_12
  have e_276_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_13
  simp only [← c18146, ← c18147, k3, ← c18148] at e_276_13
  have e_277_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_8
  simp only [← c18140, ← c18141, ← c18142] at e_277_8
  have e_277_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_9
  simp only [← c18143, ← c18144, ← c18145] at e_277_9
  refine ⟨?_, ?_⟩
  · have hc := e_277_8
    simp only [e_276_12] at hc
    linear_combination c18149.trans k2 - hc
  · have hc := e_276_13
    simp only [e_276_11, e_277_9, e_276_12] at hc
    linear_combination c18150.trans k2 - hc

theorem publicBatchWrapper86_f2497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11407)) (a (.wire 165 31)) (a (.virt 18587)) (a (.virt 18588)) := by
  have c18151 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.1
  have c18152 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.1
  have c18153 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.1
  have c18154 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18155 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18156 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18157 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18158 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18159 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18160 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18161 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18162 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18163 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18164 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18165 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18166 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18167 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_276_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_14
  simp only [← c18151, k3, ← c18152, k3, ← c18153] at e_276_14
  have e_276_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_15
  simp only [← c18154, ← c18155, k3, ← c18156] at e_276_15
  have e_276_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_16
  simp only [← c18163, ← c18164, k3, ← c18165] at e_276_16
  have e_277_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_10
  simp only [← c18157, ← c18158, ← c18159] at e_277_10
  have e_277_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_11
  simp only [← c18160, ← c18161, ← c18162] at e_277_11
  refine ⟨?_, ?_⟩
  · have hc := e_277_10
    simp only [e_276_15] at hc
    linear_combination c18166.trans k2 - hc
  · have hc := e_276_16
    simp only [e_276_14, e_277_11, e_276_15] at hc
    linear_combination c18167.trans k2 - hc

theorem publicBatchWrapper86_f2498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11408)) (a (.wire 165 39)) (a (.virt 18589)) (a (.virt 18590)) := by
  have c18168 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18169 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18170 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18171 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18172 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18173 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18174 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18175 := (publicBatchWrapper86_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18176 := (publicBatchWrapper86_copies568 a h).1
  have c18177 := (publicBatchWrapper86_copies568 a h).2.1
  have c18178 := (publicBatchWrapper86_copies568 a h).2.2.1
  have c18179 := (publicBatchWrapper86_copies568 a h).2.2.2.1
  have c18180 := (publicBatchWrapper86_copies568 a h).2.2.2.2.1
  have c18181 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.1
  have c18182 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.1
  have c18183 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.1
  have c18184 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_276_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_17
  simp only [← c18168, k3, ← c18169, k3, ← c18170] at e_276_17
  have e_276_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_18
  simp only [← c18171, ← c18172, k3, ← c18173] at e_276_18
  have e_276_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_19
  simp only [← c18180, ← c18181, k3, ← c18182] at e_276_19
  have e_277_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_12
  simp only [← c18174, ← c18175, ← c18176] at e_277_12
  have e_277_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_13
  simp only [← c18177, ← c18178, ← c18179] at e_277_13
  refine ⟨?_, ?_⟩
  · have hc := e_277_12
    simp only [e_276_18] at hc
    linear_combination c18183.trans k2 - hc
  · have hc := e_276_19
    simp only [e_276_17, e_277_13, e_276_18] at hc
    linear_combination c18184.trans k2 - hc

theorem publicBatchWrapper86_f2499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 277 59) = band (a (.virt 18583)) (a (.virt 18585)) := by
  have c18185 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.1
  have c18186 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18187 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_277_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_14
  simp only [← c18185, ← c18186, ← c18187] at e_277_14
  have hr := e_277_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 277 63) = band (a (.virt 18587)) (a (.virt 18589)) := by
  have c18188 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18189 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18190 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_277_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_15
  simp only [← c18188, ← c18189, ← c18190] at e_277_15
  have hr := e_277_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 277 67) = band (a (.wire 277 59)) (a (.wire 277 63)) := by
  have c18191 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18192 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18193 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_277_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_16
  simp only [← c18191, ← c18192, ← c18193] at e_277_16
  have hr := e_277_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 27) = bor (a (.wire 56 55)) (a (.wire 277 67)) := by
  have c18194 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18195 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18196 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18197 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18198 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18199 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_267_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_18
  simp only [← c18194, ← c18195, ← c18196] at e_267_18
  have e_275_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_6
  simp only [← c18197, ← c18198, k3, ← c18199] at e_275_6
  have hr := e_275_6
  simp only [e_267_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 27) = a (.virt 201) := by
  have c18200 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18200

theorem publicBatchWrapper86_f2504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11603)) (a (.wire 165 55)) (a (.virt 18591)) (a (.virt 18592)) := by
  have c18201 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18202 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18203 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18204 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18205 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18206 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18207 := (publicBatchWrapper86_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18208 := (publicBatchWrapper86_copies569 a h).1
  have c18209 := (publicBatchWrapper86_copies569 a h).2.1
  have c18210 := (publicBatchWrapper86_copies569 a h).2.2.1
  have c18211 := (publicBatchWrapper86_copies569 a h).2.2.2.1
  have c18212 := (publicBatchWrapper86_copies569 a h).2.2.2.2.1
  have c18213 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.1
  have c18214 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.1
  have c18215 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.1
  have c18216 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.1
  have c18217 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_277_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_17
  simp only [← c18207, ← c18208, ← c18209] at e_277_17
  have e_277_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_18
  simp only [← c18210, ← c18211, ← c18212] at e_277_18
  have e_278_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_0
  simp only [← c18201, k3, ← c18202, k3, ← c18203] at e_278_0
  have e_278_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_1
  simp only [← c18204, ← c18205, k3, ← c18206] at e_278_1
  have e_278_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_2
  simp only [← c18213, ← c18214, k3, ← c18215] at e_278_2
  refine ⟨?_, ?_⟩
  · have hc := e_277_17
    simp only [e_278_1] at hc
    linear_combination c18216.trans k2 - hc
  · have hc := e_278_2
    simp only [e_278_0, e_277_18, e_278_1] at hc
    linear_combination c18217.trans k2 - hc

theorem publicBatchWrapper86_f2505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 31) = bor (a (.wire 58 19)) (a (.virt 18591)) := by
  have c18218 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18219 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18220 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18221 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18222 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18223 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_267_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_19
  simp only [← c18218, ← c18219, ← c18220] at e_267_19
  have e_275_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_7
  simp only [← c18221, ← c18222, k3, ← c18223] at e_275_7
  have hr := e_275_7
  simp only [e_267_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 31) = a (.virt 201) := by
  have c18224 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18224

theorem publicBatchWrapper86_f2507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11604)) (a (.wire 165 63)) (a (.virt 18593)) (a (.virt 18594)) := by
  have c18225 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18226 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18227 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18228 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18229 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18230 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18231 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18232 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18233 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18234 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18235 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18236 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18237 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18238 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18239 := (publicBatchWrapper86_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18240 := (publicBatchWrapper86_copies570 a h).1
  have c18241 := (publicBatchWrapper86_copies570 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_277_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_19
  simp only [← c18231, ← c18232, ← c18233] at e_277_19
  have e_278_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_3
  simp only [← c18225, k3, ← c18226, k3, ← c18227] at e_278_3
  have e_278_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_4
  simp only [← c18228, ← c18229, k3, ← c18230] at e_278_4
  have e_278_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_5
  simp only [← c18237, ← c18238, k3, ← c18239] at e_278_5
  have e_279_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_0
  simp only [← c18234, ← c18235, ← c18236] at e_279_0
  refine ⟨?_, ?_⟩
  · have hc := e_277_19
    simp only [e_278_4] at hc
    linear_combination c18240.trans k2 - hc
  · have hc := e_278_5
    simp only [e_278_3, e_279_0, e_278_4] at hc
    linear_combination c18241.trans k2 - hc

theorem publicBatchWrapper86_f2508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 35) = bor (a (.wire 58 19)) (a (.virt 18593)) := by
  have c18242 := (publicBatchWrapper86_copies570 a h).2.2.1
  have c18243 := (publicBatchWrapper86_copies570 a h).2.2.2.1
  have c18244 := (publicBatchWrapper86_copies570 a h).2.2.2.2.1
  have c18245 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.1
  have c18246 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.1
  have c18247 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_8
  simp only [← c18245, ← c18246, k3, ← c18247] at e_275_8
  have e_280_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_0
  simp only [← c18242, ← c18243, ← c18244] at e_280_0
  have hr := e_275_8
  simp only [e_280_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 35) = a (.virt 201) := by
  have c18248 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.1
  exact c18248

theorem publicBatchWrapper86_f2510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11605)) (a (.wire 165 15)) (a (.virt 18595)) (a (.virt 18596)) := by
  have c18249 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.1
  have c18250 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18251 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18252 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18253 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18254 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18255 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18256 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18257 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18258 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18259 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18260 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18261 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18262 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18263 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18264 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18265 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_278_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_6
  simp only [← c18249, k3, ← c18250, k3, ← c18251] at e_278_6
  have e_278_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_7
  simp only [← c18252, ← c18253, k3, ← c18254] at e_278_7
  have e_278_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_8
  simp only [← c18261, ← c18262, k3, ← c18263] at e_278_8
  have e_279_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_1
  simp only [← c18255, ← c18256, ← c18257] at e_279_1
  have e_279_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_2
  simp only [← c18258, ← c18259, ← c18260] at e_279_2
  refine ⟨?_, ?_⟩
  · have hc := e_279_1
    simp only [e_278_7] at hc
    linear_combination c18264.trans k2 - hc
  · have hc := e_278_8
    simp only [e_278_6, e_279_2, e_278_7] at hc
    linear_combination c18265.trans k2 - hc

theorem publicBatchWrapper86_f2511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11606)) (a (.wire 165 23)) (a (.virt 18597)) (a (.virt 18598)) := by
  have c18266 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18267 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18268 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18269 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18270 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18271 := (publicBatchWrapper86_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18272 := (publicBatchWrapper86_copies571 a h).1
  have c18273 := (publicBatchWrapper86_copies571 a h).2.1
  have c18274 := (publicBatchWrapper86_copies571 a h).2.2.1
  have c18275 := (publicBatchWrapper86_copies571 a h).2.2.2.1
  have c18276 := (publicBatchWrapper86_copies571 a h).2.2.2.2.1
  have c18277 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.1
  have c18278 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.1
  have c18279 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.1
  have c18280 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.1
  have c18281 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.1
  have c18282 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_278_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_9
  simp only [← c18266, k3, ← c18267, k3, ← c18268] at e_278_9
  have e_278_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_10
  simp only [← c18269, ← c18270, k3, ← c18271] at e_278_10
  have e_278_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_11
  simp only [← c18278, ← c18279, k3, ← c18280] at e_278_11
  have e_279_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_3
  simp only [← c18272, ← c18273, ← c18274] at e_279_3
  have e_279_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_4
  simp only [← c18275, ← c18276, ← c18277] at e_279_4
  refine ⟨?_, ?_⟩
  · have hc := e_279_3
    simp only [e_278_10] at hc
    linear_combination c18281.trans k2 - hc
  · have hc := e_278_11
    simp only [e_278_9, e_279_4, e_278_10] at hc
    linear_combination c18282.trans k2 - hc

theorem publicBatchWrapper86_f2512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11607)) (a (.wire 165 31)) (a (.virt 18599)) (a (.virt 18600)) := by
  have c18283 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18284 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18285 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18286 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18287 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18288 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18289 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18290 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18291 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18292 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18293 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18294 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18295 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18296 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18297 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18298 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18299 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_278_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_12
  simp only [← c18283, k3, ← c18284, k3, ← c18285] at e_278_12
  have e_278_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_13
  simp only [← c18286, ← c18287, k3, ← c18288] at e_278_13
  have e_278_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_14
  simp only [← c18295, ← c18296, k3, ← c18297] at e_278_14
  have e_279_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_5
  simp only [← c18289, ← c18290, ← c18291] at e_279_5
  have e_279_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_6
  simp only [← c18292, ← c18293, ← c18294] at e_279_6
  refine ⟨?_, ?_⟩
  · have hc := e_279_5
    simp only [e_278_13] at hc
    linear_combination c18298.trans k2 - hc
  · have hc := e_278_14
    simp only [e_278_12, e_279_6, e_278_13] at hc
    linear_combination c18299.trans k2 - hc

theorem publicBatchWrapper86_f2513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11608)) (a (.wire 165 39)) (a (.virt 18601)) (a (.virt 18602)) := by
  have c18300 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18301 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18302 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18303 := (publicBatchWrapper86_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18304 := (publicBatchWrapper86_copies572 a h).1
  have c18305 := (publicBatchWrapper86_copies572 a h).2.1
  have c18306 := (publicBatchWrapper86_copies572 a h).2.2.1
  have c18307 := (publicBatchWrapper86_copies572 a h).2.2.2.1
  have c18308 := (publicBatchWrapper86_copies572 a h).2.2.2.2.1
  have c18309 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.1
  have c18310 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.1
  have c18311 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.1
  have c18312 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.1
  have c18313 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.1
  have c18314 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18315 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18316 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_278_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_15
  simp only [← c18300, k3, ← c18301, k3, ← c18302] at e_278_15
  have e_278_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_16
  simp only [← c18303, ← c18304, k3, ← c18305] at e_278_16
  have e_278_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_17
  simp only [← c18312, ← c18313, k3, ← c18314] at e_278_17
  have e_279_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_7
  simp only [← c18306, ← c18307, ← c18308] at e_279_7
  have e_279_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_8
  simp only [← c18309, ← c18310, ← c18311] at e_279_8
  refine ⟨?_, ?_⟩
  · have hc := e_279_7
    simp only [e_278_16] at hc
    linear_combination c18315.trans k2 - hc
  · have hc := e_278_17
    simp only [e_278_15, e_279_8, e_278_16] at hc
    linear_combination c18316.trans k2 - hc

theorem publicBatchWrapper86_f2514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 279 39) = band (a (.virt 18595)) (a (.virt 18597)) := by
  have c18317 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18318 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18319 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_279_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_9
  simp only [← c18317, ← c18318, ← c18319] at e_279_9
  have hr := e_279_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 279 43) = band (a (.virt 18599)) (a (.virt 18601)) := by
  have c18320 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18321 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18322 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_279_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_10
  simp only [← c18320, ← c18321, ← c18322] at e_279_10
  have hr := e_279_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 279 47) = band (a (.wire 279 39)) (a (.wire 279 43)) := by
  have c18323 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18324 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18325 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_279_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_11
  simp only [← c18323, ← c18324, ← c18325] at e_279_11
  have hr := e_279_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 39) = bor (a (.wire 58 19)) (a (.wire 279 47)) := by
  have c18326 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18327 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18328 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18329 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18330 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18331 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_9
  simp only [← c18329, ← c18330, k3, ← c18331] at e_275_9
  have e_280_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_1
  simp only [← c18326, ← c18327, ← c18328] at e_280_1
  have hr := e_275_9
  simp only [e_280_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 39) = a (.virt 201) := by
  have c18332 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18332

theorem publicBatchWrapper86_f2519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11803)) (a (.wire 165 55)) (a (.virt 18603)) (a (.virt 18604)) := by
  have c18333 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18334 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18335 := (publicBatchWrapper86_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18336 := (publicBatchWrapper86_copies573 a h).1
  have c18337 := (publicBatchWrapper86_copies573 a h).2.1
  have c18338 := (publicBatchWrapper86_copies573 a h).2.2.1
  have c18339 := (publicBatchWrapper86_copies573 a h).2.2.2.1
  have c18340 := (publicBatchWrapper86_copies573 a h).2.2.2.2.1
  have c18341 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.1
  have c18342 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.1
  have c18343 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.1
  have c18344 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.1
  have c18345 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.1
  have c18346 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18347 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18348 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18349 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_278_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_18
  simp only [← c18333, k3, ← c18334, k3, ← c18335] at e_278_18
  have e_278_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_19
  simp only [← c18336, ← c18337, k3, ← c18338] at e_278_19
  have e_279_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_12
  simp only [← c18339, ← c18340, ← c18341] at e_279_12
  have e_279_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_13
  simp only [← c18342, ← c18343, ← c18344] at e_279_13
  have e_281_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_0
  simp only [← c18345, ← c18346, k3, ← c18347] at e_281_0
  refine ⟨?_, ?_⟩
  · have hc := e_279_12
    simp only [e_278_19] at hc
    linear_combination c18348.trans k2 - hc
  · have hc := e_281_0
    simp only [e_278_18, e_279_13, e_278_19] at hc
    linear_combination c18349.trans k2 - hc

theorem publicBatchWrapper86_f2520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 43) = bor (a (.wire 58 63)) (a (.virt 18603)) := by
  have c18350 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18351 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18352 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18353 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18354 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18355 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_10
  simp only [← c18353, ← c18354, k3, ← c18355] at e_275_10
  have e_280_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_2
  simp only [← c18350, ← c18351, ← c18352] at e_280_2
  have hr := e_275_10
  simp only [e_280_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 43) = a (.virt 201) := by
  have c18356 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18356

theorem publicBatchWrapper86_f2522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11804)) (a (.wire 165 63)) (a (.virt 18605)) (a (.virt 18606)) := by
  have c18357 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18358 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18359 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18360 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18361 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18362 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18363 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18364 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18365 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18366 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18367 := (publicBatchWrapper86_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18368 := (publicBatchWrapper86_copies574 a h).1
  have c18369 := (publicBatchWrapper86_copies574 a h).2.1
  have c18370 := (publicBatchWrapper86_copies574 a h).2.2.1
  have c18371 := (publicBatchWrapper86_copies574 a h).2.2.2.1
  have c18372 := (publicBatchWrapper86_copies574 a h).2.2.2.2.1
  have c18373 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_279_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_14
  simp only [← c18363, ← c18364, ← c18365] at e_279_14
  have e_279_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_15
  simp only [← c18366, ← c18367, ← c18368] at e_279_15
  have e_281_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_1
  simp only [← c18357, k3, ← c18358, k3, ← c18359] at e_281_1
  have e_281_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_2
  simp only [← c18360, ← c18361, k3, ← c18362] at e_281_2
  have e_281_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_3
  simp only [← c18369, ← c18370, k3, ← c18371] at e_281_3
  refine ⟨?_, ?_⟩
  · have hc := e_279_14
    simp only [e_281_2] at hc
    linear_combination c18372.trans k2 - hc
  · have hc := e_281_3
    simp only [e_281_1, e_279_15, e_281_2] at hc
    linear_combination c18373.trans k2 - hc

theorem publicBatchWrapper86_f2523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 47) = bor (a (.wire 58 63)) (a (.virt 18605)) := by
  have c18374 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.1
  have c18375 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.1
  have c18376 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.1
  have c18377 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.1
  have c18378 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18379 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_11
  simp only [← c18377, ← c18378, k3, ← c18379] at e_275_11
  have e_280_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_3
  simp only [← c18374, ← c18375, ← c18376] at e_280_3
  have hr := e_275_11
  simp only [e_280_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 47) = a (.virt 201) := by
  have c18380 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18380

theorem publicBatchWrapper86_f2525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11805)) (a (.wire 165 15)) (a (.virt 18607)) (a (.virt 18608)) := by
  have c18381 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18382 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18383 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18384 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18385 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18386 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18387 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18388 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18389 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18390 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18391 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18392 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18393 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18394 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18395 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18396 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18397 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_279_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_16
  simp only [← c18387, ← c18388, ← c18389] at e_279_16
  have e_279_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_17
  simp only [← c18390, ← c18391, ← c18392] at e_279_17
  have e_281_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_4
  simp only [← c18381, k3, ← c18382, k3, ← c18383] at e_281_4
  have e_281_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_5
  simp only [← c18384, ← c18385, k3, ← c18386] at e_281_5
  have e_281_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_6
  simp only [← c18393, ← c18394, k3, ← c18395] at e_281_6
  refine ⟨?_, ?_⟩
  · have hc := e_279_16
    simp only [e_281_5] at hc
    linear_combination c18396.trans k2 - hc
  · have hc := e_281_6
    simp only [e_281_4, e_279_17, e_281_5] at hc
    linear_combination c18397.trans k2 - hc

theorem publicBatchWrapper86_f2526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11806)) (a (.wire 165 23)) (a (.virt 18609)) (a (.virt 18610)) := by
  have c18398 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18399 := (publicBatchWrapper86_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18400 := (publicBatchWrapper86_copies575 a h).1
  have c18401 := (publicBatchWrapper86_copies575 a h).2.1
  have c18402 := (publicBatchWrapper86_copies575 a h).2.2.1
  have c18403 := (publicBatchWrapper86_copies575 a h).2.2.2.1
  have c18404 := (publicBatchWrapper86_copies575 a h).2.2.2.2.1
  have c18405 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.1
  have c18406 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.1
  have c18407 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.1
  have c18408 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.1
  have c18409 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.1
  have c18410 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18411 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18412 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18413 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18414 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_279_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_18
  simp only [← c18404, ← c18405, ← c18406] at e_279_18
  have e_279_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_19
  simp only [← c18407, ← c18408, ← c18409] at e_279_19
  have e_281_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_7
  simp only [← c18398, k3, ← c18399, k3, ← c18400] at e_281_7
  have e_281_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_8
  simp only [← c18401, ← c18402, k3, ← c18403] at e_281_8
  have e_281_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_9
  simp only [← c18410, ← c18411, k3, ← c18412] at e_281_9
  refine ⟨?_, ?_⟩
  · have hc := e_279_18
    simp only [e_281_8] at hc
    linear_combination c18413.trans k2 - hc
  · have hc := e_281_9
    simp only [e_281_7, e_279_19, e_281_8] at hc
    linear_combination c18414.trans k2 - hc

theorem publicBatchWrapper86_f2527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11807)) (a (.wire 165 31)) (a (.virt 18611)) (a (.virt 18612)) := by
  have c18415 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18416 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18417 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18418 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18419 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18420 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18421 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18422 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18423 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18424 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18425 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18426 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18427 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18428 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18429 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18430 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18431 := (publicBatchWrapper86_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_281_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_10
  simp only [← c18415, k3, ← c18416, k3, ← c18417] at e_281_10
  have e_281_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_11
  simp only [← c18418, ← c18419, k3, ← c18420] at e_281_11
  have e_281_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_12
  simp only [← c18427, ← c18428, k3, ← c18429] at e_281_12
  have e_282_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_0
  simp only [← c18421, ← c18422, ← c18423] at e_282_0
  have e_282_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_1
  simp only [← c18424, ← c18425, ← c18426] at e_282_1
  refine ⟨?_, ?_⟩
  · have hc := e_282_0
    simp only [e_281_11] at hc
    linear_combination c18430.trans k2 - hc
  · have hc := e_281_12
    simp only [e_281_10, e_282_1, e_281_11] at hc
    linear_combination c18431.trans k2 - hc

theorem publicBatchWrapper86_f2528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11808)) (a (.wire 165 39)) (a (.virt 18613)) (a (.virt 18614)) := by
  have c18432 := (publicBatchWrapper86_copies576 a h).1
  have c18433 := (publicBatchWrapper86_copies576 a h).2.1
  have c18434 := (publicBatchWrapper86_copies576 a h).2.2.1
  have c18435 := (publicBatchWrapper86_copies576 a h).2.2.2.1
  have c18436 := (publicBatchWrapper86_copies576 a h).2.2.2.2.1
  have c18437 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.1
  have c18438 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.1
  have c18439 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.1
  have c18440 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.1
  have c18441 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.1
  have c18442 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18443 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18444 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18445 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18446 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18447 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18448 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_281_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_13
  simp only [← c18432, k3, ← c18433, k3, ← c18434] at e_281_13
  have e_281_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_14
  simp only [← c18435, ← c18436, k3, ← c18437] at e_281_14
  have e_281_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_15
  simp only [← c18444, ← c18445, k3, ← c18446] at e_281_15
  have e_282_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_2
  simp only [← c18438, ← c18439, ← c18440] at e_282_2
  have e_282_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_3
  simp only [← c18441, ← c18442, ← c18443] at e_282_3
  refine ⟨?_, ?_⟩
  · have hc := e_282_2
    simp only [e_281_14] at hc
    linear_combination c18447.trans k2 - hc
  · have hc := e_281_15
    simp only [e_281_13, e_282_3, e_281_14] at hc
    linear_combination c18448.trans k2 - hc

theorem publicBatchWrapper86_f2529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 282 19) = band (a (.virt 18607)) (a (.virt 18609)) := by
  have c18449 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18450 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18451 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_282_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_4
  simp only [← c18449, ← c18450, ← c18451] at e_282_4
  have hr := e_282_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 282 23) = band (a (.virt 18611)) (a (.virt 18613)) := by
  have c18452 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18453 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18454 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_282_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_5
  simp only [← c18452, ← c18453, ← c18454] at e_282_5
  have hr := e_282_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 282 27) = band (a (.wire 282 19)) (a (.wire 282 23)) := by
  have c18455 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18456 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18457 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_282_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_6
  simp only [← c18455, ← c18456, ← c18457] at e_282_6
  have hr := e_282_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 51) = bor (a (.wire 58 63)) (a (.wire 282 27)) := by
  have c18458 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18459 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18460 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18461 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18462 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18463 := (publicBatchWrapper86_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_12
  simp only [← c18461, ← c18462, k3, ← c18463] at e_275_12
  have e_280_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_4
  simp only [← c18458, ← c18459, ← c18460] at e_280_4
  have hr := e_275_12
  simp only [e_280_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 51) = a (.virt 201) := by
  have c18464 := (publicBatchWrapper86_copies577 a h).1
  exact c18464

theorem publicBatchWrapper86_f2534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12003)) (a (.wire 165 55)) (a (.virt 18615)) (a (.virt 18616)) := by
  have c18465 := (publicBatchWrapper86_copies577 a h).2.1
  have c18466 := (publicBatchWrapper86_copies577 a h).2.2.1
  have c18467 := (publicBatchWrapper86_copies577 a h).2.2.2.1
  have c18468 := (publicBatchWrapper86_copies577 a h).2.2.2.2.1
  have c18469 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.1
  have c18470 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.1
  have c18471 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.1
  have c18472 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.1
  have c18473 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.1
  have c18474 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18475 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18476 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18477 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18478 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18479 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18480 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18481 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_281_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_16
  simp only [← c18465, k3, ← c18466, k3, ← c18467] at e_281_16
  have e_281_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_17
  simp only [← c18468, ← c18469, k3, ← c18470] at e_281_17
  have e_281_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_18
  simp only [← c18477, ← c18478, k3, ← c18479] at e_281_18
  have e_282_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_7
  simp only [← c18471, ← c18472, ← c18473] at e_282_7
  have e_282_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_8
  simp only [← c18474, ← c18475, ← c18476] at e_282_8
  refine ⟨?_, ?_⟩
  · have hc := e_282_7
    simp only [e_281_17] at hc
    linear_combination c18480.trans k2 - hc
  · have hc := e_281_18
    simp only [e_281_16, e_282_8, e_281_17] at hc
    linear_combination c18481.trans k2 - hc

theorem publicBatchWrapper86_f2535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 55) = bor (a (.wire 60 27)) (a (.virt 18615)) := by
  have c18482 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18483 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18484 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18485 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18486 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18487 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_13
  simp only [← c18485, ← c18486, k3, ← c18487] at e_275_13
  have e_280_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_5
  simp only [← c18482, ← c18483, ← c18484] at e_280_5
  have hr := e_275_13
  simp only [e_280_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 55) = a (.virt 201) := by
  have c18488 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18488

theorem publicBatchWrapper86_f2537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12004)) (a (.wire 165 63)) (a (.virt 18617)) (a (.virt 18618)) := by
  have c18489 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18490 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18491 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18492 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18493 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18494 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18495 := (publicBatchWrapper86_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18496 := (publicBatchWrapper86_copies578 a h).1
  have c18497 := (publicBatchWrapper86_copies578 a h).2.1
  have c18498 := (publicBatchWrapper86_copies578 a h).2.2.1
  have c18499 := (publicBatchWrapper86_copies578 a h).2.2.2.1
  have c18500 := (publicBatchWrapper86_copies578 a h).2.2.2.2.1
  have c18501 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.1
  have c18502 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.1
  have c18503 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.1
  have c18504 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.1
  have c18505 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_281_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_19
  simp only [← c18489, k3, ← c18490, k3, ← c18491] at e_281_19
  have e_282_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_9
  simp only [← c18495, ← c18496, ← c18497] at e_282_9
  have e_282_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_10
  simp only [← c18498, ← c18499, ← c18500] at e_282_10
  have e_283_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_0
  simp only [← c18492, ← c18493, k3, ← c18494] at e_283_0
  have e_283_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_1
  simp only [← c18501, ← c18502, k3, ← c18503] at e_283_1
  refine ⟨?_, ?_⟩
  · have hc := e_282_9
    simp only [e_283_0] at hc
    linear_combination c18504.trans k2 - hc
  · have hc := e_283_1
    simp only [e_281_19, e_282_10, e_283_0] at hc
    linear_combination c18505.trans k2 - hc

theorem publicBatchWrapper86_f2538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 59) = bor (a (.wire 60 27)) (a (.virt 18617)) := by
  have c18506 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18507 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18508 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18509 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18510 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18511 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_14
  simp only [← c18509, ← c18510, k3, ← c18511] at e_275_14
  have e_280_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_6
  simp only [← c18506, ← c18507, ← c18508] at e_280_6
  have hr := e_275_14
  simp only [e_280_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 59) = a (.virt 201) := by
  have c18512 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18512

theorem publicBatchWrapper86_f2540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12005)) (a (.wire 165 15)) (a (.virt 18619)) (a (.virt 18620)) := by
  have c18513 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18514 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18515 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18516 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18517 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18518 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18519 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18520 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18521 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18522 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18523 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18524 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18525 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18526 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18527 := (publicBatchWrapper86_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18528 := (publicBatchWrapper86_copies579 a h).1
  have c18529 := (publicBatchWrapper86_copies579 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_282_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_11
  simp only [← c18519, ← c18520, ← c18521] at e_282_11
  have e_282_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_12
  simp only [← c18522, ← c18523, ← c18524] at e_282_12
  have e_283_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_2
  simp only [← c18513, k3, ← c18514, k3, ← c18515] at e_283_2
  have e_283_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_3
  simp only [← c18516, ← c18517, k3, ← c18518] at e_283_3
  have e_283_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_4
  simp only [← c18525, ← c18526, k3, ← c18527] at e_283_4
  refine ⟨?_, ?_⟩
  · have hc := e_282_11
    simp only [e_283_3] at hc
    linear_combination c18528.trans k2 - hc
  · have hc := e_283_4
    simp only [e_283_2, e_282_12, e_283_3] at hc
    linear_combination c18529.trans k2 - hc

theorem publicBatchWrapper86_f2541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12006)) (a (.wire 165 23)) (a (.virt 18621)) (a (.virt 18622)) := by
  have c18530 := (publicBatchWrapper86_copies579 a h).2.2.1
  have c18531 := (publicBatchWrapper86_copies579 a h).2.2.2.1
  have c18532 := (publicBatchWrapper86_copies579 a h).2.2.2.2.1
  have c18533 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.1
  have c18534 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.1
  have c18535 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.1
  have c18536 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.1
  have c18537 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.1
  have c18538 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18539 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18540 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18541 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18542 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18543 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18544 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18545 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18546 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_282_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_13
  simp only [← c18536, ← c18537, ← c18538] at e_282_13
  have e_282_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_14
  simp only [← c18539, ← c18540, ← c18541] at e_282_14
  have e_283_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_5
  simp only [← c18530, k3, ← c18531, k3, ← c18532] at e_283_5
  have e_283_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_6
  simp only [← c18533, ← c18534, k3, ← c18535] at e_283_6
  have e_283_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_7
  simp only [← c18542, ← c18543, k3, ← c18544] at e_283_7
  refine ⟨?_, ?_⟩
  · have hc := e_282_13
    simp only [e_283_6] at hc
    linear_combination c18545.trans k2 - hc
  · have hc := e_283_7
    simp only [e_283_5, e_282_14, e_283_6] at hc
    linear_combination c18546.trans k2 - hc

theorem publicBatchWrapper86_f2542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12007)) (a (.wire 165 31)) (a (.virt 18623)) (a (.virt 18624)) := by
  have c18547 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18548 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18549 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18550 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18551 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18552 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18553 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18554 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18555 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18556 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18557 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18558 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18559 := (publicBatchWrapper86_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18560 := (publicBatchWrapper86_copies580 a h).1
  have c18561 := (publicBatchWrapper86_copies580 a h).2.1
  have c18562 := (publicBatchWrapper86_copies580 a h).2.2.1
  have c18563 := (publicBatchWrapper86_copies580 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_282_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_15
  simp only [← c18553, ← c18554, ← c18555] at e_282_15
  have e_282_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_16
  simp only [← c18556, ← c18557, ← c18558] at e_282_16
  have e_283_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_8
  simp only [← c18547, k3, ← c18548, k3, ← c18549] at e_283_8
  have e_283_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_9
  simp only [← c18550, ← c18551, k3, ← c18552] at e_283_9
  have e_283_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_10
  simp only [← c18559, ← c18560, k3, ← c18561] at e_283_10
  refine ⟨?_, ?_⟩
  · have hc := e_282_15
    simp only [e_283_9] at hc
    linear_combination c18562.trans k2 - hc
  · have hc := e_283_10
    simp only [e_283_8, e_282_16, e_283_9] at hc
    linear_combination c18563.trans k2 - hc

theorem publicBatchWrapper86_f2543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12008)) (a (.wire 165 39)) (a (.virt 18625)) (a (.virt 18626)) := by
  have c18564 := (publicBatchWrapper86_copies580 a h).2.2.2.2.1
  have c18565 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.1
  have c18566 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.1
  have c18567 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.1
  have c18568 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.1
  have c18569 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.1
  have c18570 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18571 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18572 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18573 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18574 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18575 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18576 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18577 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18578 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18579 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18580 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_282_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_17
  simp only [← c18570, ← c18571, ← c18572] at e_282_17
  have e_282_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_18
  simp only [← c18573, ← c18574, ← c18575] at e_282_18
  have e_283_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_11
  simp only [← c18564, k3, ← c18565, k3, ← c18566] at e_283_11
  have e_283_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_12
  simp only [← c18567, ← c18568, k3, ← c18569] at e_283_12
  have e_283_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_13
  simp only [← c18576, ← c18577, k3, ← c18578] at e_283_13
  refine ⟨?_, ?_⟩
  · have hc := e_282_17
    simp only [e_283_12] at hc
    linear_combination c18579.trans k2 - hc
  · have hc := e_283_13
    simp only [e_283_11, e_282_18, e_283_12] at hc
    linear_combination c18580.trans k2 - hc

theorem publicBatchWrapper86_f2544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 282 79) = band (a (.virt 18619)) (a (.virt 18621)) := by
  have c18581 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18582 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18583 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_282_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_19
  simp only [← c18581, ← c18582, ← c18583] at e_282_19
  have hr := e_282_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 284 3) = band (a (.virt 18623)) (a (.virt 18625)) := by
  have c18584 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18585 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18586 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_284_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_0
  simp only [← c18584, ← c18585, ← c18586] at e_284_0
  have hr := e_284_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 284 7) = band (a (.wire 282 79)) (a (.wire 284 3)) := by
  have c18587 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18588 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18589 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_284_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_1
  simp only [← c18587, ← c18588, ← c18589] at e_284_1
  have hr := e_284_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 63) = bor (a (.wire 60 27)) (a (.wire 284 7)) := by
  have c18590 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18591 := (publicBatchWrapper86_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18592 := (publicBatchWrapper86_copies581 a h).1
  have c18593 := (publicBatchWrapper86_copies581 a h).2.1
  have c18594 := (publicBatchWrapper86_copies581 a h).2.2.1
  have c18595 := (publicBatchWrapper86_copies581 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_15
  simp only [← c18593, ← c18594, k3, ← c18595] at e_275_15
  have e_280_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_7
  simp only [← c18590, ← c18591, ← c18592] at e_280_7
  have hr := e_275_15
  simp only [e_280_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 63) = a (.virt 201) := by
  have c18596 := (publicBatchWrapper86_copies581 a h).2.2.2.2.1
  exact c18596

theorem publicBatchWrapper86_f2549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12203)) (a (.wire 165 55)) (a (.virt 18627)) (a (.virt 18628)) := by
  have c18597 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.1
  have c18598 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.1
  have c18599 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.1
  have c18600 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.1
  have c18601 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.1
  have c18602 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18603 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18604 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18605 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18606 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18607 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18608 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18609 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18610 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18611 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18612 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18613 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_283_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_14
  simp only [← c18597, k3, ← c18598, k3, ← c18599] at e_283_14
  have e_283_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_15
  simp only [← c18600, ← c18601, k3, ← c18602] at e_283_15
  have e_283_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_16
  simp only [← c18609, ← c18610, k3, ← c18611] at e_283_16
  have e_284_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_2
  simp only [← c18603, ← c18604, ← c18605] at e_284_2
  have e_284_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_3
  simp only [← c18606, ← c18607, ← c18608] at e_284_3
  refine ⟨?_, ?_⟩
  · have hc := e_284_2
    simp only [e_283_15] at hc
    linear_combination c18612.trans k2 - hc
  · have hc := e_283_16
    simp only [e_283_14, e_284_3, e_283_15] at hc
    linear_combination c18613.trans k2 - hc

theorem publicBatchWrapper86_f2550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 67) = bor (a (.wire 60 71)) (a (.virt 18627)) := by
  have c18614 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18615 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18616 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18617 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18618 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18619 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_16
  simp only [← c18617, ← c18618, k3, ← c18619] at e_275_16
  have e_280_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_8
  simp only [← c18614, ← c18615, ← c18616] at e_280_8
  have hr := e_275_16
  simp only [e_280_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 67) = a (.virt 201) := by
  have c18620 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18620

theorem publicBatchWrapper86_f2552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12204)) (a (.wire 165 63)) (a (.virt 18629)) (a (.virt 18630)) := by
  have c18621 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18622 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18623 := (publicBatchWrapper86_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18624 := (publicBatchWrapper86_copies582 a h).1
  have c18625 := (publicBatchWrapper86_copies582 a h).2.1
  have c18626 := (publicBatchWrapper86_copies582 a h).2.2.1
  have c18627 := (publicBatchWrapper86_copies582 a h).2.2.2.1
  have c18628 := (publicBatchWrapper86_copies582 a h).2.2.2.2.1
  have c18629 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.1
  have c18630 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.1
  have c18631 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.1
  have c18632 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.1
  have c18633 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.1
  have c18634 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18635 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18636 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18637 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_283_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_17
  simp only [← c18621, k3, ← c18622, k3, ← c18623] at e_283_17
  have e_283_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_18
  simp only [← c18624, ← c18625, k3, ← c18626] at e_283_18
  have e_283_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_19
  simp only [← c18633, ← c18634, k3, ← c18635] at e_283_19
  have e_284_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_4
  simp only [← c18627, ← c18628, ← c18629] at e_284_4
  have e_284_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_5
  simp only [← c18630, ← c18631, ← c18632] at e_284_5
  refine ⟨?_, ?_⟩
  · have hc := e_284_4
    simp only [e_283_18] at hc
    linear_combination c18636.trans k2 - hc
  · have hc := e_283_19
    simp only [e_283_17, e_284_5, e_283_18] at hc
    linear_combination c18637.trans k2 - hc

theorem publicBatchWrapper86_f2553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 71) = bor (a (.wire 60 71)) (a (.virt 18629)) := by
  have c18638 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18639 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18640 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18641 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18642 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18643 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_17
  simp only [← c18641, ← c18642, k3, ← c18643] at e_275_17
  have e_280_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_9
  simp only [← c18638, ← c18639, ← c18640] at e_280_9
  have hr := e_275_17
  simp only [e_280_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 71) = a (.virt 201) := by
  have c18644 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18644

theorem publicBatchWrapper86_f2555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12205)) (a (.wire 165 15)) (a (.virt 18631)) (a (.virt 18632)) := by
  have c18645 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18646 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18647 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18648 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18649 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18650 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18651 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18652 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18653 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18654 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18655 := (publicBatchWrapper86_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18656 := (publicBatchWrapper86_copies583 a h).1
  have c18657 := (publicBatchWrapper86_copies583 a h).2.1
  have c18658 := (publicBatchWrapper86_copies583 a h).2.2.1
  have c18659 := (publicBatchWrapper86_copies583 a h).2.2.2.1
  have c18660 := (publicBatchWrapper86_copies583 a h).2.2.2.2.1
  have c18661 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_284_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_6
  simp only [← c18651, ← c18652, ← c18653] at e_284_6
  have e_284_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_7
  simp only [← c18654, ← c18655, ← c18656] at e_284_7
  have e_285_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_0
  simp only [← c18645, k3, ← c18646, k3, ← c18647] at e_285_0
  have e_285_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_1
  simp only [← c18648, ← c18649, k3, ← c18650] at e_285_1
  have e_285_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_2
  simp only [← c18657, ← c18658, k3, ← c18659] at e_285_2
  refine ⟨?_, ?_⟩
  · have hc := e_284_6
    simp only [e_285_1] at hc
    linear_combination c18660.trans k2 - hc
  · have hc := e_285_2
    simp only [e_285_0, e_284_7, e_285_1] at hc
    linear_combination c18661.trans k2 - hc

theorem publicBatchWrapper86_f2556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12206)) (a (.wire 165 23)) (a (.virt 18633)) (a (.virt 18634)) := by
  have c18662 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.1
  have c18663 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.1
  have c18664 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.1
  have c18665 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.1
  have c18666 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18667 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18668 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18669 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18670 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18671 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18672 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18673 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18674 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18675 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18676 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18677 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18678 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_284_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_8
  simp only [← c18668, ← c18669, ← c18670] at e_284_8
  have e_284_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_9
  simp only [← c18671, ← c18672, ← c18673] at e_284_9
  have e_285_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_3
  simp only [← c18662, k3, ← c18663, k3, ← c18664] at e_285_3
  have e_285_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_4
  simp only [← c18665, ← c18666, k3, ← c18667] at e_285_4
  have e_285_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_5
  simp only [← c18674, ← c18675, k3, ← c18676] at e_285_5
  refine ⟨?_, ?_⟩
  · have hc := e_284_8
    simp only [e_285_4] at hc
    linear_combination c18677.trans k2 - hc
  · have hc := e_285_5
    simp only [e_285_3, e_284_9, e_285_4] at hc
    linear_combination c18678.trans k2 - hc

theorem publicBatchWrapper86_f2557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12207)) (a (.wire 165 31)) (a (.virt 18635)) (a (.virt 18636)) := by
  have c18679 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18680 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18681 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18682 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18683 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18684 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18685 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18686 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18687 := (publicBatchWrapper86_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18688 := (publicBatchWrapper86_copies584 a h).1
  have c18689 := (publicBatchWrapper86_copies584 a h).2.1
  have c18690 := (publicBatchWrapper86_copies584 a h).2.2.1
  have c18691 := (publicBatchWrapper86_copies584 a h).2.2.2.1
  have c18692 := (publicBatchWrapper86_copies584 a h).2.2.2.2.1
  have c18693 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.1
  have c18694 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.1
  have c18695 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_284_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_10
  simp only [← c18685, ← c18686, ← c18687] at e_284_10
  have e_284_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_11
  simp only [← c18688, ← c18689, ← c18690] at e_284_11
  have e_285_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_6
  simp only [← c18679, k3, ← c18680, k3, ← c18681] at e_285_6
  have e_285_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_7
  simp only [← c18682, ← c18683, k3, ← c18684] at e_285_7
  have e_285_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_8
  simp only [← c18691, ← c18692, k3, ← c18693] at e_285_8
  refine ⟨?_, ?_⟩
  · have hc := e_284_10
    simp only [e_285_7] at hc
    linear_combination c18694.trans k2 - hc
  · have hc := e_285_8
    simp only [e_285_6, e_284_11, e_285_7] at hc
    linear_combination c18695.trans k2 - hc

theorem publicBatchWrapper86_f2558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12208)) (a (.wire 165 39)) (a (.virt 18637)) (a (.virt 18638)) := by
  have c18696 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.1
  have c18697 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.1
  have c18698 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18699 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18700 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18701 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18702 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18703 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18704 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18705 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18706 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18707 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18708 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18709 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18710 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18711 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18712 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_284_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_12
  simp only [← c18702, ← c18703, ← c18704] at e_284_12
  have e_284_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_13
  simp only [← c18705, ← c18706, ← c18707] at e_284_13
  have e_285_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_9
  simp only [← c18696, k3, ← c18697, k3, ← c18698] at e_285_9
  have e_285_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_10
  simp only [← c18699, ← c18700, k3, ← c18701] at e_285_10
  have e_285_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_11
  simp only [← c18708, ← c18709, k3, ← c18710] at e_285_11
  refine ⟨?_, ?_⟩
  · have hc := e_284_12
    simp only [e_285_10] at hc
    linear_combination c18711.trans k2 - hc
  · have hc := e_285_11
    simp only [e_285_9, e_284_13, e_285_10] at hc
    linear_combination c18712.trans k2 - hc

theorem publicBatchWrapper86_f2559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 284 59) = band (a (.virt 18631)) (a (.virt 18633)) := by
  have c18713 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18714 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18715 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_284_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_14
  simp only [← c18713, ← c18714, ← c18715] at e_284_14
  have hr := e_284_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 284 63) = band (a (.virt 18635)) (a (.virt 18637)) := by
  have c18716 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18717 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18718 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_284_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_15
  simp only [← c18716, ← c18717, ← c18718] at e_284_15
  have hr := e_284_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 284 67) = band (a (.wire 284 59)) (a (.wire 284 63)) := by
  have c18719 := (publicBatchWrapper86_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18720 := (publicBatchWrapper86_copies585 a h).1
  have c18721 := (publicBatchWrapper86_copies585 a h).2.1
  have e_284_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_16
  simp only [← c18719, ← c18720, ← c18721] at e_284_16
  have hr := e_284_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 75) = bor (a (.wire 60 71)) (a (.wire 284 67)) := by
  have c18722 := (publicBatchWrapper86_copies585 a h).2.2.1
  have c18723 := (publicBatchWrapper86_copies585 a h).2.2.2.1
  have c18724 := (publicBatchWrapper86_copies585 a h).2.2.2.2.1
  have c18725 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.1
  have c18726 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.1
  have c18727 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_18
  simp only [← c18725, ← c18726, k3, ← c18727] at e_275_18
  have e_280_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_10
  simp only [← c18722, ← c18723, ← c18724] at e_280_10
  have hr := e_275_18
  simp only [e_280_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 75) = a (.virt 201) := by
  have c18728 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.1
  exact c18728

theorem publicBatchWrapper86_f2564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12403)) (a (.wire 165 55)) (a (.virt 18639)) (a (.virt 18640)) := by
  have c18729 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.1
  have c18730 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18731 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18732 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18733 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18734 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18735 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18736 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18737 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18738 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18739 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18740 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18741 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18742 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18743 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18744 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18745 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_284_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_17
  simp only [← c18735, ← c18736, ← c18737] at e_284_17
  have e_284_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_18
  simp only [← c18738, ← c18739, ← c18740] at e_284_18
  have e_285_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_12
  simp only [← c18729, k3, ← c18730, k3, ← c18731] at e_285_12
  have e_285_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_13
  simp only [← c18732, ← c18733, k3, ← c18734] at e_285_13
  have e_285_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_14
  simp only [← c18741, ← c18742, k3, ← c18743] at e_285_14
  refine ⟨?_, ?_⟩
  · have hc := e_284_17
    simp only [e_285_13] at hc
    linear_combination c18744.trans k2 - hc
  · have hc := e_285_14
    simp only [e_285_12, e_284_18, e_285_13] at hc
    linear_combination c18745.trans k2 - hc

theorem publicBatchWrapper86_f2565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 79) = bor (a (.wire 61 35)) (a (.virt 18639)) := by
  have c18746 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18747 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18748 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18749 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18750 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18751 := (publicBatchWrapper86_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_275_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_19
  simp only [← c18749, ← c18750, k3, ← c18751] at e_275_19
  have e_280_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_11
  simp only [← c18746, ← c18747, ← c18748] at e_280_11
  have hr := e_275_19
  simp only [e_280_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 275 79) = a (.virt 201) := by
  have c18752 := (publicBatchWrapper86_copies586 a h).1
  exact c18752

theorem publicBatchWrapper86_f2567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12404)) (a (.wire 165 63)) (a (.virt 18641)) (a (.virt 18642)) := by
  have c18753 := (publicBatchWrapper86_copies586 a h).2.1
  have c18754 := (publicBatchWrapper86_copies586 a h).2.2.1
  have c18755 := (publicBatchWrapper86_copies586 a h).2.2.2.1
  have c18756 := (publicBatchWrapper86_copies586 a h).2.2.2.2.1
  have c18757 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.1
  have c18758 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.1
  have c18759 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.1
  have c18760 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.1
  have c18761 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.1
  have c18762 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18763 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18764 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18765 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18766 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18767 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18768 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18769 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_284_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_19
  simp only [← c18759, ← c18760, ← c18761] at e_284_19
  have e_285_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_15
  simp only [← c18753, k3, ← c18754, k3, ← c18755] at e_285_15
  have e_285_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_16
  simp only [← c18756, ← c18757, k3, ← c18758] at e_285_16
  have e_285_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_17
  simp only [← c18765, ← c18766, k3, ← c18767] at e_285_17
  have e_286_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_0
  simp only [← c18762, ← c18763, ← c18764] at e_286_0
  refine ⟨?_, ?_⟩
  · have hc := e_284_19
    simp only [e_285_16] at hc
    linear_combination c18768.trans k2 - hc
  · have hc := e_285_17
    simp only [e_285_15, e_286_0, e_285_16] at hc
    linear_combination c18769.trans k2 - hc

theorem publicBatchWrapper86_f2568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 3) = bor (a (.wire 61 35)) (a (.virt 18641)) := by
  have c18770 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18771 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18772 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18773 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18774 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18775 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_280_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_12
  simp only [← c18770, ← c18771, ← c18772] at e_280_12
  have e_287_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_0
  simp only [← c18773, ← c18774, k3, ← c18775] at e_287_0
  have hr := e_287_0
  simp only [e_280_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 3) = a (.virt 201) := by
  have c18776 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18776

theorem publicBatchWrapper86_f2570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12405)) (a (.wire 165 15)) (a (.virt 18643)) (a (.virt 18644)) := by
  have c18777 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18778 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18779 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18780 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18781 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18782 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18783 := (publicBatchWrapper86_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18784 := (publicBatchWrapper86_copies587 a h).1
  have c18785 := (publicBatchWrapper86_copies587 a h).2.1
  have c18786 := (publicBatchWrapper86_copies587 a h).2.2.1
  have c18787 := (publicBatchWrapper86_copies587 a h).2.2.2.1
  have c18788 := (publicBatchWrapper86_copies587 a h).2.2.2.2.1
  have c18789 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.1
  have c18790 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.1
  have c18791 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.1
  have c18792 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.1
  have c18793 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_285_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_18
  simp only [← c18777, k3, ← c18778, k3, ← c18779] at e_285_18
  have e_285_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_19
  simp only [← c18780, ← c18781, k3, ← c18782] at e_285_19
  have e_286_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_1
  simp only [← c18783, ← c18784, ← c18785] at e_286_1
  have e_286_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_2
  simp only [← c18786, ← c18787, ← c18788] at e_286_2
  have e_288_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_0
  simp only [← c18789, ← c18790, k3, ← c18791] at e_288_0
  refine ⟨?_, ?_⟩
  · have hc := e_286_1
    simp only [e_285_19] at hc
    linear_combination c18792.trans k2 - hc
  · have hc := e_288_0
    simp only [e_285_18, e_286_2, e_285_19] at hc
    linear_combination c18793.trans k2 - hc

theorem publicBatchWrapper86_f2571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12406)) (a (.wire 165 23)) (a (.virt 18645)) (a (.virt 18646)) := by
  have c18794 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18795 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18796 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18797 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18798 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18799 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18800 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18801 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18802 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18803 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18804 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18805 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18806 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18807 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18808 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18809 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18810 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_286_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_3
  simp only [← c18800, ← c18801, ← c18802] at e_286_3
  have e_286_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_4
  simp only [← c18803, ← c18804, ← c18805] at e_286_4
  have e_288_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_1
  simp only [← c18794, k3, ← c18795, k3, ← c18796] at e_288_1
  have e_288_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_2
  simp only [← c18797, ← c18798, k3, ← c18799] at e_288_2
  have e_288_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_3
  simp only [← c18806, ← c18807, k3, ← c18808] at e_288_3
  refine ⟨?_, ?_⟩
  · have hc := e_286_3
    simp only [e_288_2] at hc
    linear_combination c18809.trans k2 - hc
  · have hc := e_288_3
    simp only [e_288_1, e_286_4, e_288_2] at hc
    linear_combination c18810.trans k2 - hc

theorem publicBatchWrapper86_f2572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12407)) (a (.wire 165 31)) (a (.virt 18647)) (a (.virt 18648)) := by
  have c18811 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18812 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18813 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18814 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18815 := (publicBatchWrapper86_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18816 := (publicBatchWrapper86_copies588 a h).1
  have c18817 := (publicBatchWrapper86_copies588 a h).2.1
  have c18818 := (publicBatchWrapper86_copies588 a h).2.2.1
  have c18819 := (publicBatchWrapper86_copies588 a h).2.2.2.1
  have c18820 := (publicBatchWrapper86_copies588 a h).2.2.2.2.1
  have c18821 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.1
  have c18822 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.1
  have c18823 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.1
  have c18824 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.1
  have c18825 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.1
  have c18826 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18827 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_286_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_5
  simp only [← c18817, ← c18818, ← c18819] at e_286_5
  have e_286_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_6
  simp only [← c18820, ← c18821, ← c18822] at e_286_6
  have e_288_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_4
  simp only [← c18811, k3, ← c18812, k3, ← c18813] at e_288_4
  have e_288_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_5
  simp only [← c18814, ← c18815, k3, ← c18816] at e_288_5
  have e_288_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_6
  simp only [← c18823, ← c18824, k3, ← c18825] at e_288_6
  refine ⟨?_, ?_⟩
  · have hc := e_286_5
    simp only [e_288_5] at hc
    linear_combination c18826.trans k2 - hc
  · have hc := e_288_6
    simp only [e_288_4, e_286_6, e_288_5] at hc
    linear_combination c18827.trans k2 - hc

theorem publicBatchWrapper86_f2573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12408)) (a (.wire 165 39)) (a (.virt 18649)) (a (.virt 18650)) := by
  have c18828 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18829 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18830 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18831 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18832 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18833 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18834 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18835 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18836 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18837 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18838 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18839 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18840 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18841 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18842 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18843 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18844 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_286_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_7
  simp only [← c18834, ← c18835, ← c18836] at e_286_7
  have e_286_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_8
  simp only [← c18837, ← c18838, ← c18839] at e_286_8
  have e_288_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_7
  simp only [← c18828, k3, ← c18829, k3, ← c18830] at e_288_7
  have e_288_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_8
  simp only [← c18831, ← c18832, k3, ← c18833] at e_288_8
  have e_288_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_9
  simp only [← c18840, ← c18841, k3, ← c18842] at e_288_9
  refine ⟨?_, ?_⟩
  · have hc := e_286_7
    simp only [e_288_8] at hc
    linear_combination c18843.trans k2 - hc
  · have hc := e_288_9
    simp only [e_288_7, e_286_8, e_288_8] at hc
    linear_combination c18844.trans k2 - hc

theorem publicBatchWrapper86_f2574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 286 39) = band (a (.virt 18643)) (a (.virt 18645)) := by
  have c18845 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18846 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18847 := (publicBatchWrapper86_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_286_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_9
  simp only [← c18845, ← c18846, ← c18847] at e_286_9
  have hr := e_286_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 286 43) = band (a (.virt 18647)) (a (.virt 18649)) := by
  have c18848 := (publicBatchWrapper86_copies589 a h).1
  have c18849 := (publicBatchWrapper86_copies589 a h).2.1
  have c18850 := (publicBatchWrapper86_copies589 a h).2.2.1
  have e_286_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_10
  simp only [← c18848, ← c18849, ← c18850] at e_286_10
  have hr := e_286_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 286 47) = band (a (.wire 286 39)) (a (.wire 286 43)) := by
  have c18851 := (publicBatchWrapper86_copies589 a h).2.2.2.1
  have c18852 := (publicBatchWrapper86_copies589 a h).2.2.2.2.1
  have c18853 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.1
  have e_286_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_11
  simp only [← c18851, ← c18852, ← c18853] at e_286_11
  have hr := e_286_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 7) = bor (a (.wire 61 35)) (a (.wire 286 47)) := by
  have c18854 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.1
  have c18855 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.1
  have c18856 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.1
  have c18857 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.1
  have c18858 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18859 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_280_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_13
  simp only [← c18854, ← c18855, ← c18856] at e_280_13
  have e_287_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_1
  simp only [← c18857, ← c18858, k3, ← c18859] at e_287_1
  have hr := e_287_1
  simp only [e_280_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 7) = a (.virt 201) := by
  have c18860 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18860

theorem publicBatchWrapper86_f2579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12603)) (a (.wire 165 55)) (a (.virt 18651)) (a (.virt 18652)) := by
  have c18861 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18862 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18863 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18864 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18865 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18866 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18867 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18868 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18869 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18870 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18871 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18872 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18873 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18874 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18875 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18876 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18877 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_286_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_12
  simp only [← c18867, ← c18868, ← c18869] at e_286_12
  have e_286_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_13
  simp only [← c18870, ← c18871, ← c18872] at e_286_13
  have e_288_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_10
  simp only [← c18861, k3, ← c18862, k3, ← c18863] at e_288_10
  have e_288_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_11
  simp only [← c18864, ← c18865, k3, ← c18866] at e_288_11
  have e_288_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_12
  simp only [← c18873, ← c18874, k3, ← c18875] at e_288_12
  refine ⟨?_, ?_⟩
  · have hc := e_286_12
    simp only [e_288_11] at hc
    linear_combination c18876.trans k2 - hc
  · have hc := e_288_12
    simp only [e_288_10, e_286_13, e_288_11] at hc
    linear_combination c18877.trans k2 - hc

theorem publicBatchWrapper86_f2580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 11) = bor (a (.wire 61 79)) (a (.virt 18651)) := by
  have c18878 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18879 := (publicBatchWrapper86_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18880 := (publicBatchWrapper86_copies590 a h).1
  have c18881 := (publicBatchWrapper86_copies590 a h).2.1
  have c18882 := (publicBatchWrapper86_copies590 a h).2.2.1
  have c18883 := (publicBatchWrapper86_copies590 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_280_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_14
  simp only [← c18878, ← c18879, ← c18880] at e_280_14
  have e_287_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_2
  simp only [← c18881, ← c18882, k3, ← c18883] at e_287_2
  have hr := e_287_2
  simp only [e_280_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 11) = a (.virt 201) := by
  have c18884 := (publicBatchWrapper86_copies590 a h).2.2.2.2.1
  exact c18884

theorem publicBatchWrapper86_f2582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12604)) (a (.wire 165 63)) (a (.virt 18653)) (a (.virt 18654)) := by
  have c18885 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.1
  have c18886 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.1
  have c18887 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.1
  have c18888 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.1
  have c18889 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.1
  have c18890 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18891 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18892 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18893 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18894 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18895 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18896 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18897 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18898 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18899 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18900 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18901 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_286_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_14
  simp only [← c18891, ← c18892, ← c18893] at e_286_14
  have e_286_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_15
  simp only [← c18894, ← c18895, ← c18896] at e_286_15
  have e_288_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_13
  simp only [← c18885, k3, ← c18886, k3, ← c18887] at e_288_13
  have e_288_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_14
  simp only [← c18888, ← c18889, k3, ← c18890] at e_288_14
  have e_288_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_15
  simp only [← c18897, ← c18898, k3, ← c18899] at e_288_15
  refine ⟨?_, ?_⟩
  · have hc := e_286_14
    simp only [e_288_14] at hc
    linear_combination c18900.trans k2 - hc
  · have hc := e_288_15
    simp only [e_288_13, e_286_15, e_288_14] at hc
    linear_combination c18901.trans k2 - hc

theorem publicBatchWrapper86_f2583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 15) = bor (a (.wire 61 79)) (a (.virt 18653)) := by
  have c18902 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18903 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18904 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18905 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18906 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18907 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_280_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_15
  simp only [← c18902, ← c18903, ← c18904] at e_280_15
  have e_287_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_3
  simp only [← c18905, ← c18906, k3, ← c18907] at e_287_3
  have hr := e_287_3
  simp only [e_280_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 15) = a (.virt 201) := by
  have c18908 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18908

theorem publicBatchWrapper86_f2585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12605)) (a (.wire 165 15)) (a (.virt 18655)) (a (.virt 18656)) := by
  have c18909 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18910 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18911 := (publicBatchWrapper86_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18912 := (publicBatchWrapper86_copies591 a h).1
  have c18913 := (publicBatchWrapper86_copies591 a h).2.1
  have c18914 := (publicBatchWrapper86_copies591 a h).2.2.1
  have c18915 := (publicBatchWrapper86_copies591 a h).2.2.2.1
  have c18916 := (publicBatchWrapper86_copies591 a h).2.2.2.2.1
  have c18917 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.1
  have c18918 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.1
  have c18919 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.1
  have c18920 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.1
  have c18921 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.1
  have c18922 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18923 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18924 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18925 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_286_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_16
  simp only [← c18915, ← c18916, ← c18917] at e_286_16
  have e_286_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_17
  simp only [← c18918, ← c18919, ← c18920] at e_286_17
  have e_288_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_16
  simp only [← c18909, k3, ← c18910, k3, ← c18911] at e_288_16
  have e_288_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_17
  simp only [← c18912, ← c18913, k3, ← c18914] at e_288_17
  have e_288_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_18
  simp only [← c18921, ← c18922, k3, ← c18923] at e_288_18
  refine ⟨?_, ?_⟩
  · have hc := e_286_16
    simp only [e_288_17] at hc
    linear_combination c18924.trans k2 - hc
  · have hc := e_288_18
    simp only [e_288_16, e_286_17, e_288_17] at hc
    linear_combination c18925.trans k2 - hc

theorem publicBatchWrapper86_f2586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12606)) (a (.wire 165 23)) (a (.virt 18657)) (a (.virt 18658)) := by
  have c18926 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18927 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18928 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18929 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18930 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18931 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18932 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18933 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18934 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18935 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18936 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18937 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18938 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18939 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18940 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18941 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18942 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_286_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_18
  simp only [← c18932, ← c18933, ← c18934] at e_286_18
  have e_286_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_19
  simp only [← c18935, ← c18936, ← c18937] at e_286_19
  have e_288_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_19
  simp only [← c18926, k3, ← c18927, k3, ← c18928] at e_288_19
  have e_289_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_0
  simp only [← c18929, ← c18930, k3, ← c18931] at e_289_0
  have e_289_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_1
  simp only [← c18938, ← c18939, k3, ← c18940] at e_289_1
  refine ⟨?_, ?_⟩
  · have hc := e_286_18
    simp only [e_289_0] at hc
    linear_combination c18941.trans k2 - hc
  · have hc := e_289_1
    simp only [e_288_19, e_286_19, e_289_0] at hc
    linear_combination c18942.trans k2 - hc

theorem publicBatchWrapper86_f2587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12607)) (a (.wire 165 31)) (a (.virt 18659)) (a (.virt 18660)) := by
  have c18943 := (publicBatchWrapper86_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18944 := (publicBatchWrapper86_copies592 a h).1
  have c18945 := (publicBatchWrapper86_copies592 a h).2.1
  have c18946 := (publicBatchWrapper86_copies592 a h).2.2.1
  have c18947 := (publicBatchWrapper86_copies592 a h).2.2.2.1
  have c18948 := (publicBatchWrapper86_copies592 a h).2.2.2.2.1
  have c18949 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.1
  have c18950 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.1
  have c18951 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.1
  have c18952 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.1
  have c18953 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.1
  have c18954 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18955 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18956 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18957 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18958 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18959 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_289_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_2
  simp only [← c18943, k3, ← c18944, k3, ← c18945] at e_289_2
  have e_289_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_3
  simp only [← c18946, ← c18947, k3, ← c18948] at e_289_3
  have e_289_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_4
  simp only [← c18955, ← c18956, k3, ← c18957] at e_289_4
  have e_290_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_0
  simp only [← c18949, ← c18950, ← c18951] at e_290_0
  have e_290_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_1
  simp only [← c18952, ← c18953, ← c18954] at e_290_1
  refine ⟨?_, ?_⟩
  · have hc := e_290_0
    simp only [e_289_3] at hc
    linear_combination c18958.trans k2 - hc
  · have hc := e_289_4
    simp only [e_289_2, e_290_1, e_289_3] at hc
    linear_combination c18959.trans k2 - hc

theorem publicBatchWrapper86_f2588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12608)) (a (.wire 165 39)) (a (.virt 18661)) (a (.virt 18662)) := by
  have c18960 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18961 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18962 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18963 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18964 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18965 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18966 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18967 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18968 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18969 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18970 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18971 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18972 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18973 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18974 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18975 := (publicBatchWrapper86_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18976 := (publicBatchWrapper86_copies593 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_289_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_5
  simp only [← c18960, k3, ← c18961, k3, ← c18962] at e_289_5
  have e_289_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_6
  simp only [← c18963, ← c18964, k3, ← c18965] at e_289_6
  have e_289_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_7
  simp only [← c18972, ← c18973, k3, ← c18974] at e_289_7
  have e_290_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_2
  simp only [← c18966, ← c18967, ← c18968] at e_290_2
  have e_290_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_3
  simp only [← c18969, ← c18970, ← c18971] at e_290_3
  refine ⟨?_, ?_⟩
  · have hc := e_290_2
    simp only [e_289_6] at hc
    linear_combination c18975.trans k2 - hc
  · have hc := e_289_7
    simp only [e_289_5, e_290_3, e_289_6] at hc
    linear_combination c18976.trans k2 - hc

theorem publicBatchWrapper86_f2589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 290 19) = band (a (.virt 18655)) (a (.virt 18657)) := by
  have c18977 := (publicBatchWrapper86_copies593 a h).2.1
  have c18978 := (publicBatchWrapper86_copies593 a h).2.2.1
  have c18979 := (publicBatchWrapper86_copies593 a h).2.2.2.1
  have e_290_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_4
  simp only [← c18977, ← c18978, ← c18979] at e_290_4
  have hr := e_290_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 290 23) = band (a (.virt 18659)) (a (.virt 18661)) := by
  have c18980 := (publicBatchWrapper86_copies593 a h).2.2.2.2.1
  have c18981 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.1
  have c18982 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.1
  have e_290_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_5
  simp only [← c18980, ← c18981, ← c18982] at e_290_5
  have hr := e_290_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 290 27) = band (a (.wire 290 19)) (a (.wire 290 23)) := by
  have c18983 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.1
  have c18984 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.1
  have c18985 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.1
  have e_290_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_6
  simp only [← c18983, ← c18984, ← c18985] at e_290_6
  have hr := e_290_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 19) = bor (a (.wire 61 79)) (a (.wire 290 27)) := by
  have c18986 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18987 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18988 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18989 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18990 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18991 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_280_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_16
  simp only [← c18986, ← c18987, ← c18988] at e_280_16
  have e_287_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_4
  simp only [← c18989, ← c18990, k3, ← c18991] at e_287_4
  have hr := e_287_4
  simp only [e_280_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 19) = a (.virt 201) := by
  have c18992 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18992

theorem publicBatchWrapper86_f2594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12803)) (a (.wire 165 55)) (a (.virt 18663)) (a (.virt 18664)) := by
  have c18993 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18994 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18995 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18996 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18997 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18998 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18999 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19000 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19001 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19002 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19003 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19004 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19005 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19006 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19007 := (publicBatchWrapper86_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19008 := (publicBatchWrapper86_copies594 a h).1
  have c19009 := (publicBatchWrapper86_copies594 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_289_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_8
  simp only [← c18993, k3, ← c18994, k3, ← c18995] at e_289_8
  have e_289_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_9
  simp only [← c18996, ← c18997, k3, ← c18998] at e_289_9
  have e_289_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_10
  simp only [← c19005, ← c19006, k3, ← c19007] at e_289_10
  have e_290_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_7
  simp only [← c18999, ← c19000, ← c19001] at e_290_7
  have e_290_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_8
  simp only [← c19002, ← c19003, ← c19004] at e_290_8
  refine ⟨?_, ?_⟩
  · have hc := e_290_7
    simp only [e_289_9] at hc
    linear_combination c19008.trans k2 - hc
  · have hc := e_289_10
    simp only [e_289_8, e_290_8, e_289_9] at hc
    linear_combination c19009.trans k2 - hc

theorem publicBatchWrapper86_f2595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 23) = bor (a (.wire 63 43)) (a (.virt 18663)) := by
  have c19010 := (publicBatchWrapper86_copies594 a h).2.2.1
  have c19011 := (publicBatchWrapper86_copies594 a h).2.2.2.1
  have c19012 := (publicBatchWrapper86_copies594 a h).2.2.2.2.1
  have c19013 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.1
  have c19014 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.1
  have c19015 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_280_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_17
  simp only [← c19010, ← c19011, ← c19012] at e_280_17
  have e_287_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_5
  simp only [← c19013, ← c19014, k3, ← c19015] at e_287_5
  have hr := e_287_5
  simp only [e_280_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 23) = a (.virt 201) := by
  have c19016 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.1
  exact c19016

theorem publicBatchWrapper86_f2597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12804)) (a (.wire 165 63)) (a (.virt 18665)) (a (.virt 18666)) := by
  have c19017 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.1
  have c19018 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19019 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19020 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19021 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19022 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19023 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19024 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19025 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19026 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19027 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19028 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19029 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19030 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19031 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19032 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19033 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_289_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_11
  simp only [← c19017, k3, ← c19018, k3, ← c19019] at e_289_11
  have e_289_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_12
  simp only [← c19020, ← c19021, k3, ← c19022] at e_289_12
  have e_289_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_13
  simp only [← c19029, ← c19030, k3, ← c19031] at e_289_13
  have e_290_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_9
  simp only [← c19023, ← c19024, ← c19025] at e_290_9
  have e_290_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_10
  simp only [← c19026, ← c19027, ← c19028] at e_290_10
  refine ⟨?_, ?_⟩
  · have hc := e_290_9
    simp only [e_289_12] at hc
    linear_combination c19032.trans k2 - hc
  · have hc := e_289_13
    simp only [e_289_11, e_290_10, e_289_12] at hc
    linear_combination c19033.trans k2 - hc

theorem publicBatchWrapper86_f2598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 27) = bor (a (.wire 63 43)) (a (.virt 18665)) := by
  have c19034 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19035 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19036 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19037 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19038 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19039 := (publicBatchWrapper86_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_280_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_18
  simp only [← c19034, ← c19035, ← c19036] at e_280_18
  have e_287_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_6
  simp only [← c19037, ← c19038, k3, ← c19039] at e_287_6
  have hr := e_287_6
  simp only [e_280_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 27) = a (.virt 201) := by
  have c19040 := (publicBatchWrapper86_copies595 a h).1
  exact c19040

theorem publicBatchWrapper86_f2600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12805)) (a (.wire 165 15)) (a (.virt 18667)) (a (.virt 18668)) := by
  have c19041 := (publicBatchWrapper86_copies595 a h).2.1
  have c19042 := (publicBatchWrapper86_copies595 a h).2.2.1
  have c19043 := (publicBatchWrapper86_copies595 a h).2.2.2.1
  have c19044 := (publicBatchWrapper86_copies595 a h).2.2.2.2.1
  have c19045 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.1
  have c19046 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.1
  have c19047 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.1
  have c19048 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.1
  have c19049 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.1
  have c19050 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19051 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19052 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19053 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19054 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19055 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19056 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19057 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_289_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_14
  simp only [← c19041, k3, ← c19042, k3, ← c19043] at e_289_14
  have e_289_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_15
  simp only [← c19044, ← c19045, k3, ← c19046] at e_289_15
  have e_289_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_16
  simp only [← c19053, ← c19054, k3, ← c19055] at e_289_16
  have e_290_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_11
  simp only [← c19047, ← c19048, ← c19049] at e_290_11
  have e_290_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_12
  simp only [← c19050, ← c19051, ← c19052] at e_290_12
  refine ⟨?_, ?_⟩
  · have hc := e_290_11
    simp only [e_289_15] at hc
    linear_combination c19056.trans k2 - hc
  · have hc := e_289_16
    simp only [e_289_14, e_290_12, e_289_15] at hc
    linear_combination c19057.trans k2 - hc

theorem publicBatchWrapper86_f2601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12806)) (a (.wire 165 23)) (a (.virt 18669)) (a (.virt 18670)) := by
  have c19058 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19059 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19060 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19061 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19062 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19063 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19064 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19065 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19066 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19067 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19068 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19069 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19070 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19071 := (publicBatchWrapper86_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19072 := (publicBatchWrapper86_copies596 a h).1
  have c19073 := (publicBatchWrapper86_copies596 a h).2.1
  have c19074 := (publicBatchWrapper86_copies596 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_289_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_17
  simp only [← c19058, k3, ← c19059, k3, ← c19060] at e_289_17
  have e_289_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_18
  simp only [← c19061, ← c19062, k3, ← c19063] at e_289_18
  have e_289_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_19
  simp only [← c19070, ← c19071, k3, ← c19072] at e_289_19
  have e_290_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_13
  simp only [← c19064, ← c19065, ← c19066] at e_290_13
  have e_290_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_14
  simp only [← c19067, ← c19068, ← c19069] at e_290_14
  refine ⟨?_, ?_⟩
  · have hc := e_290_13
    simp only [e_289_18] at hc
    linear_combination c19073.trans k2 - hc
  · have hc := e_289_19
    simp only [e_289_17, e_290_14, e_289_18] at hc
    linear_combination c19074.trans k2 - hc

theorem publicBatchWrapper86_f2602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12807)) (a (.wire 165 31)) (a (.virt 18671)) (a (.virt 18672)) := by
  have c19075 := (publicBatchWrapper86_copies596 a h).2.2.2.1
  have c19076 := (publicBatchWrapper86_copies596 a h).2.2.2.2.1
  have c19077 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.1
  have c19078 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.1
  have c19079 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.1
  have c19080 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.1
  have c19081 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.1
  have c19082 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19083 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19084 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19085 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19086 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19087 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19088 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19089 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19090 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19091 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_290_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_15
  simp only [← c19081, ← c19082, ← c19083] at e_290_15
  have e_290_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_16
  simp only [← c19084, ← c19085, ← c19086] at e_290_16
  have e_291_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_0
  simp only [← c19075, k3, ← c19076, k3, ← c19077] at e_291_0
  have e_291_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_1
  simp only [← c19078, ← c19079, k3, ← c19080] at e_291_1
  have e_291_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_2
  simp only [← c19087, ← c19088, k3, ← c19089] at e_291_2
  refine ⟨?_, ?_⟩
  · have hc := e_290_15
    simp only [e_291_1] at hc
    linear_combination c19090.trans k2 - hc
  · have hc := e_291_2
    simp only [e_291_0, e_290_16, e_291_1] at hc
    linear_combination c19091.trans k2 - hc

theorem publicBatchWrapper86_f2603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12808)) (a (.wire 165 39)) (a (.virt 18673)) (a (.virt 18674)) := by
  have c19092 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19093 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19094 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19095 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19096 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19097 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19098 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19099 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19100 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19101 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19102 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19103 := (publicBatchWrapper86_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19104 := (publicBatchWrapper86_copies597 a h).1
  have c19105 := (publicBatchWrapper86_copies597 a h).2.1
  have c19106 := (publicBatchWrapper86_copies597 a h).2.2.1
  have c19107 := (publicBatchWrapper86_copies597 a h).2.2.2.1
  have c19108 := (publicBatchWrapper86_copies597 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_290_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_17
  simp only [← c19098, ← c19099, ← c19100] at e_290_17
  have e_290_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_18
  simp only [← c19101, ← c19102, ← c19103] at e_290_18
  have e_291_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_3
  simp only [← c19092, k3, ← c19093, k3, ← c19094] at e_291_3
  have e_291_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_4
  simp only [← c19095, ← c19096, k3, ← c19097] at e_291_4
  have e_291_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_5
  simp only [← c19104, ← c19105, k3, ← c19106] at e_291_5
  refine ⟨?_, ?_⟩
  · have hc := e_290_17
    simp only [e_291_4] at hc
    linear_combination c19107.trans k2 - hc
  · have hc := e_291_5
    simp only [e_291_3, e_290_18, e_291_4] at hc
    linear_combination c19108.trans k2 - hc

theorem publicBatchWrapper86_f2604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 290 79) = band (a (.virt 18667)) (a (.virt 18669)) := by
  have c19109 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.1
  have c19110 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.1
  have c19111 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.1
  have e_290_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_19
  simp only [← c19109, ← c19110, ← c19111] at e_290_19
  have hr := e_290_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 292 3) = band (a (.virt 18671)) (a (.virt 18673)) := by
  have c19112 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.1
  have c19113 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.1
  have c19114 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_292_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_0
  simp only [← c19112, ← c19113, ← c19114] at e_292_0
  have hr := e_292_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 292 7) = band (a (.wire 290 79)) (a (.wire 292 3)) := by
  have c19115 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19116 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19117 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_292_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_1
  simp only [← c19115, ← c19116, ← c19117] at e_292_1
  have hr := e_292_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 31) = bor (a (.wire 63 43)) (a (.wire 292 7)) := by
  have c19118 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19119 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19120 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19121 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19122 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19123 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_280_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_19
  simp only [← c19118, ← c19119, ← c19120] at e_280_19
  have e_287_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_7
  simp only [← c19121, ← c19122, k3, ← c19123] at e_287_7
  have hr := e_287_7
  simp only [e_280_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 31) = a (.virt 201) := by
  have c19124 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19124

theorem publicBatchWrapper86_f2609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13003)) (a (.wire 165 55)) (a (.virt 18675)) (a (.virt 18676)) := by
  have c19125 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19126 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19127 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19128 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19129 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19130 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19131 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19132 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19133 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19134 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19135 := (publicBatchWrapper86_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19136 := (publicBatchWrapper86_copies598 a h).1
  have c19137 := (publicBatchWrapper86_copies598 a h).2.1
  have c19138 := (publicBatchWrapper86_copies598 a h).2.2.1
  have c19139 := (publicBatchWrapper86_copies598 a h).2.2.2.1
  have c19140 := (publicBatchWrapper86_copies598 a h).2.2.2.2.1
  have c19141 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_291_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_6
  simp only [← c19125, k3, ← c19126, k3, ← c19127] at e_291_6
  have e_291_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_7
  simp only [← c19128, ← c19129, k3, ← c19130] at e_291_7
  have e_291_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_8
  simp only [← c19137, ← c19138, k3, ← c19139] at e_291_8
  have e_292_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_2
  simp only [← c19131, ← c19132, ← c19133] at e_292_2
  have e_292_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_3
  simp only [← c19134, ← c19135, ← c19136] at e_292_3
  refine ⟨?_, ?_⟩
  · have hc := e_292_2
    simp only [e_291_7] at hc
    linear_combination c19140.trans k2 - hc
  · have hc := e_291_8
    simp only [e_291_6, e_292_3, e_291_7] at hc
    linear_combination c19141.trans k2 - hc

theorem publicBatchWrapper86_f2610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 35) = bor (a (.wire 65 7)) (a (.virt 18675)) := by
  have c19142 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.1
  have c19143 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.1
  have c19144 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.1
  have c19145 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.1
  have c19146 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19147 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_8
  simp only [← c19145, ← c19146, k3, ← c19147] at e_287_8
  have e_293_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_0
  simp only [← c19142, ← c19143, ← c19144] at e_293_0
  have hr := e_287_8
  simp only [e_293_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 35) = a (.virt 201) := by
  have c19148 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19148

theorem publicBatchWrapper86_f2612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13004)) (a (.wire 165 63)) (a (.virt 18677)) (a (.virt 18678)) := by
  have c19149 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19150 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19151 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19152 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19153 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19154 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19155 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19156 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19157 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19158 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19159 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19160 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19161 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19162 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19163 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19164 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19165 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_291_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_9
  simp only [← c19149, k3, ← c19150, k3, ← c19151] at e_291_9
  have e_291_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_10
  simp only [← c19152, ← c19153, k3, ← c19154] at e_291_10
  have e_291_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_11
  simp only [← c19161, ← c19162, k3, ← c19163] at e_291_11
  have e_292_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_4
  simp only [← c19155, ← c19156, ← c19157] at e_292_4
  have e_292_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_5
  simp only [← c19158, ← c19159, ← c19160] at e_292_5
  refine ⟨?_, ?_⟩
  · have hc := e_292_4
    simp only [e_291_10] at hc
    linear_combination c19164.trans k2 - hc
  · have hc := e_291_11
    simp only [e_291_9, e_292_5, e_291_10] at hc
    linear_combination c19165.trans k2 - hc

theorem publicBatchWrapper86_f2613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 39) = bor (a (.wire 65 7)) (a (.virt 18677)) := by
  have c19166 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19167 := (publicBatchWrapper86_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19168 := (publicBatchWrapper86_copies599 a h).1
  have c19169 := (publicBatchWrapper86_copies599 a h).2.1
  have c19170 := (publicBatchWrapper86_copies599 a h).2.2.1
  have c19171 := (publicBatchWrapper86_copies599 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_9
  simp only [← c19169, ← c19170, k3, ← c19171] at e_287_9
  have e_293_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_1
  simp only [← c19166, ← c19167, ← c19168] at e_293_1
  have hr := e_287_9
  simp only [e_293_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 39) = a (.virt 201) := by
  have c19172 := (publicBatchWrapper86_copies599 a h).2.2.2.2.1
  exact c19172

theorem publicBatchWrapper86_f2615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13005)) (a (.wire 165 15)) (a (.virt 18679)) (a (.virt 18680)) := by
  have c19173 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.1
  have c19174 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.1
  have c19175 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.1
  have c19176 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.1
  have c19177 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.1
  have c19178 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19179 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19180 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19181 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19182 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19183 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19184 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19185 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19186 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19187 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19188 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19189 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_291_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_12
  simp only [← c19173, k3, ← c19174, k3, ← c19175] at e_291_12
  have e_291_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_13
  simp only [← c19176, ← c19177, k3, ← c19178] at e_291_13
  have e_291_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_14
  simp only [← c19185, ← c19186, k3, ← c19187] at e_291_14
  have e_292_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_6
  simp only [← c19179, ← c19180, ← c19181] at e_292_6
  have e_292_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_7
  simp only [← c19182, ← c19183, ← c19184] at e_292_7
  refine ⟨?_, ?_⟩
  · have hc := e_292_6
    simp only [e_291_13] at hc
    linear_combination c19188.trans k2 - hc
  · have hc := e_291_14
    simp only [e_291_12, e_292_7, e_291_13] at hc
    linear_combination c19189.trans k2 - hc

theorem publicBatchWrapper86_f2616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13006)) (a (.wire 165 23)) (a (.virt 18681)) (a (.virt 18682)) := by
  have c19190 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19191 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19192 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19193 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19194 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19195 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19196 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19197 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19198 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19199 := (publicBatchWrapper86_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19200 := (publicBatchWrapper86_copies600 a h).1
  have c19201 := (publicBatchWrapper86_copies600 a h).2.1
  have c19202 := (publicBatchWrapper86_copies600 a h).2.2.1
  have c19203 := (publicBatchWrapper86_copies600 a h).2.2.2.1
  have c19204 := (publicBatchWrapper86_copies600 a h).2.2.2.2.1
  have c19205 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.1
  have c19206 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_291_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_15
  simp only [← c19190, k3, ← c19191, k3, ← c19192] at e_291_15
  have e_291_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_16
  simp only [← c19193, ← c19194, k3, ← c19195] at e_291_16
  have e_291_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_17
  simp only [← c19202, ← c19203, k3, ← c19204] at e_291_17
  have e_292_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_8
  simp only [← c19196, ← c19197, ← c19198] at e_292_8
  have e_292_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_9
  simp only [← c19199, ← c19200, ← c19201] at e_292_9
  refine ⟨?_, ?_⟩
  · have hc := e_292_8
    simp only [e_291_16] at hc
    linear_combination c19205.trans k2 - hc
  · have hc := e_291_17
    simp only [e_291_15, e_292_9, e_291_16] at hc
    linear_combination c19206.trans k2 - hc

theorem publicBatchWrapper86_f2617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13007)) (a (.wire 165 31)) (a (.virt 18683)) (a (.virt 18684)) := by
  have c19207 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.1
  have c19208 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.1
  have c19209 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.1
  have c19210 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19211 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19212 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19213 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19214 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19215 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19216 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19217 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19218 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19219 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19220 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19221 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19222 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19223 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_291_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_18
  simp only [← c19207, k3, ← c19208, k3, ← c19209] at e_291_18
  have e_291_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_19
  simp only [← c19210, ← c19211, k3, ← c19212] at e_291_19
  have e_292_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_10
  simp only [← c19213, ← c19214, ← c19215] at e_292_10
  have e_292_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_11
  simp only [← c19216, ← c19217, ← c19218] at e_292_11
  have e_294_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_0
  simp only [← c19219, ← c19220, k3, ← c19221] at e_294_0
  refine ⟨?_, ?_⟩
  · have hc := e_292_10
    simp only [e_291_19] at hc
    linear_combination c19222.trans k2 - hc
  · have hc := e_294_0
    simp only [e_291_18, e_292_11, e_291_19] at hc
    linear_combination c19223.trans k2 - hc

theorem publicBatchWrapper86_f2618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13008)) (a (.wire 165 39)) (a (.virt 18685)) (a (.virt 18686)) := by
  have c19224 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19225 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19226 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19227 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19228 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19229 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19230 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19231 := (publicBatchWrapper86_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19232 := (publicBatchWrapper86_copies601 a h).1
  have c19233 := (publicBatchWrapper86_copies601 a h).2.1
  have c19234 := (publicBatchWrapper86_copies601 a h).2.2.1
  have c19235 := (publicBatchWrapper86_copies601 a h).2.2.2.1
  have c19236 := (publicBatchWrapper86_copies601 a h).2.2.2.2.1
  have c19237 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.1
  have c19238 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.1
  have c19239 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.1
  have c19240 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_292_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_12
  simp only [← c19230, ← c19231, ← c19232] at e_292_12
  have e_292_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_13
  simp only [← c19233, ← c19234, ← c19235] at e_292_13
  have e_294_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_1
  simp only [← c19224, k3, ← c19225, k3, ← c19226] at e_294_1
  have e_294_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_2
  simp only [← c19227, ← c19228, k3, ← c19229] at e_294_2
  have e_294_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_3
  simp only [← c19236, ← c19237, k3, ← c19238] at e_294_3
  refine ⟨?_, ?_⟩
  · have hc := e_292_12
    simp only [e_294_2] at hc
    linear_combination c19239.trans k2 - hc
  · have hc := e_294_3
    simp only [e_294_1, e_292_13, e_294_2] at hc
    linear_combination c19240.trans k2 - hc

theorem publicBatchWrapper86_f2619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 292 59) = band (a (.virt 18679)) (a (.virt 18681)) := by
  have c19241 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.1
  have c19242 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19243 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_292_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_14
  simp only [← c19241, ← c19242, ← c19243] at e_292_14
  have hr := e_292_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 292 63) = band (a (.virt 18683)) (a (.virt 18685)) := by
  have c19244 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19245 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19246 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_292_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_15
  simp only [← c19244, ← c19245, ← c19246] at e_292_15
  have hr := e_292_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 292 67) = band (a (.wire 292 59)) (a (.wire 292 63)) := by
  have c19247 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19248 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19249 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_292_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_16
  simp only [← c19247, ← c19248, ← c19249] at e_292_16
  have hr := e_292_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 43) = bor (a (.wire 65 7)) (a (.wire 292 67)) := by
  have c19250 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19251 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19252 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19253 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19254 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19255 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_10
  simp only [← c19253, ← c19254, k3, ← c19255] at e_287_10
  have e_293_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_2
  simp only [← c19250, ← c19251, ← c19252] at e_293_2
  have hr := e_287_10
  simp only [e_293_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 43) = a (.virt 201) := by
  have c19256 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19256

theorem publicBatchWrapper86_f2624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13203)) (a (.wire 165 55)) (a (.virt 18687)) (a (.virt 18688)) := by
  have c19257 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19258 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19259 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19260 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19261 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19262 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19263 := (publicBatchWrapper86_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19264 := (publicBatchWrapper86_copies602 a h).1
  have c19265 := (publicBatchWrapper86_copies602 a h).2.1
  have c19266 := (publicBatchWrapper86_copies602 a h).2.2.1
  have c19267 := (publicBatchWrapper86_copies602 a h).2.2.2.1
  have c19268 := (publicBatchWrapper86_copies602 a h).2.2.2.2.1
  have c19269 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.1
  have c19270 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.1
  have c19271 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.1
  have c19272 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.1
  have c19273 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_292_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_17
  simp only [← c19263, ← c19264, ← c19265] at e_292_17
  have e_292_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_18
  simp only [← c19266, ← c19267, ← c19268] at e_292_18
  have e_294_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_4
  simp only [← c19257, k3, ← c19258, k3, ← c19259] at e_294_4
  have e_294_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_5
  simp only [← c19260, ← c19261, k3, ← c19262] at e_294_5
  have e_294_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_6
  simp only [← c19269, ← c19270, k3, ← c19271] at e_294_6
  refine ⟨?_, ?_⟩
  · have hc := e_292_17
    simp only [e_294_5] at hc
    linear_combination c19272.trans k2 - hc
  · have hc := e_294_6
    simp only [e_294_4, e_292_18, e_294_5] at hc
    linear_combination c19273.trans k2 - hc

theorem publicBatchWrapper86_f2625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 47) = bor (a (.wire 65 51)) (a (.virt 18687)) := by
  have c19274 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19275 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19276 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19277 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19278 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19279 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_11
  simp only [← c19277, ← c19278, k3, ← c19279] at e_287_11
  have e_293_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_3
  simp only [← c19274, ← c19275, ← c19276] at e_293_3
  have hr := e_287_11
  simp only [e_293_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 47) = a (.virt 201) := by
  have c19280 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19280

theorem publicBatchWrapper86_f2627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13204)) (a (.wire 165 63)) (a (.virt 18689)) (a (.virt 18690)) := by
  have c19281 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19282 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19283 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19284 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19285 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19286 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19287 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19288 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19289 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19290 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19291 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19292 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19293 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19294 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19295 := (publicBatchWrapper86_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19296 := (publicBatchWrapper86_copies603 a h).1
  have c19297 := (publicBatchWrapper86_copies603 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_292_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_19
  simp only [← c19287, ← c19288, ← c19289] at e_292_19
  have e_294_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_7
  simp only [← c19281, k3, ← c19282, k3, ← c19283] at e_294_7
  have e_294_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_8
  simp only [← c19284, ← c19285, k3, ← c19286] at e_294_8
  have e_294_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_9
  simp only [← c19293, ← c19294, k3, ← c19295] at e_294_9
  have e_295_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_0
  simp only [← c19290, ← c19291, ← c19292] at e_295_0
  refine ⟨?_, ?_⟩
  · have hc := e_292_19
    simp only [e_294_8] at hc
    linear_combination c19296.trans k2 - hc
  · have hc := e_294_9
    simp only [e_294_7, e_295_0, e_294_8] at hc
    linear_combination c19297.trans k2 - hc

theorem publicBatchWrapper86_f2628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 51) = bor (a (.wire 65 51)) (a (.virt 18689)) := by
  have c19298 := (publicBatchWrapper86_copies603 a h).2.2.1
  have c19299 := (publicBatchWrapper86_copies603 a h).2.2.2.1
  have c19300 := (publicBatchWrapper86_copies603 a h).2.2.2.2.1
  have c19301 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.1
  have c19302 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.1
  have c19303 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_12
  simp only [← c19301, ← c19302, k3, ← c19303] at e_287_12
  have e_293_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_4
  simp only [← c19298, ← c19299, ← c19300] at e_293_4
  have hr := e_287_12
  simp only [e_293_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 51) = a (.virt 201) := by
  have c19304 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.1
  exact c19304

theorem publicBatchWrapper86_f2630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13205)) (a (.wire 165 15)) (a (.virt 18691)) (a (.virt 18692)) := by
  have c19305 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.1
  have c19306 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19307 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19308 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19309 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19310 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19311 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19312 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19313 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19314 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19315 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19316 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19317 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19318 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19319 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19320 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19321 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_294_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_10
  simp only [← c19305, k3, ← c19306, k3, ← c19307] at e_294_10
  have e_294_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_11
  simp only [← c19308, ← c19309, k3, ← c19310] at e_294_11
  have e_294_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_12
  simp only [← c19317, ← c19318, k3, ← c19319] at e_294_12
  have e_295_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_1
  simp only [← c19311, ← c19312, ← c19313] at e_295_1
  have e_295_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_2
  simp only [← c19314, ← c19315, ← c19316] at e_295_2
  refine ⟨?_, ?_⟩
  · have hc := e_295_1
    simp only [e_294_11] at hc
    linear_combination c19320.trans k2 - hc
  · have hc := e_294_12
    simp only [e_294_10, e_295_2, e_294_11] at hc
    linear_combination c19321.trans k2 - hc

theorem publicBatchWrapper86_f2631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13206)) (a (.wire 165 23)) (a (.virt 18693)) (a (.virt 18694)) := by
  have c19322 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19323 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19324 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19325 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19326 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19327 := (publicBatchWrapper86_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19328 := (publicBatchWrapper86_copies604 a h).1
  have c19329 := (publicBatchWrapper86_copies604 a h).2.1
  have c19330 := (publicBatchWrapper86_copies604 a h).2.2.1
  have c19331 := (publicBatchWrapper86_copies604 a h).2.2.2.1
  have c19332 := (publicBatchWrapper86_copies604 a h).2.2.2.2.1
  have c19333 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.1
  have c19334 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.1
  have c19335 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.1
  have c19336 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.1
  have c19337 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.1
  have c19338 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_294_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_13
  simp only [← c19322, k3, ← c19323, k3, ← c19324] at e_294_13
  have e_294_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_14
  simp only [← c19325, ← c19326, k3, ← c19327] at e_294_14
  have e_294_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_15
  simp only [← c19334, ← c19335, k3, ← c19336] at e_294_15
  have e_295_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_3
  simp only [← c19328, ← c19329, ← c19330] at e_295_3
  have e_295_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_4
  simp only [← c19331, ← c19332, ← c19333] at e_295_4
  refine ⟨?_, ?_⟩
  · have hc := e_295_3
    simp only [e_294_14] at hc
    linear_combination c19337.trans k2 - hc
  · have hc := e_294_15
    simp only [e_294_13, e_295_4, e_294_14] at hc
    linear_combination c19338.trans k2 - hc

theorem publicBatchWrapper86_f2632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13207)) (a (.wire 165 31)) (a (.virt 18695)) (a (.virt 18696)) := by
  have c19339 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19340 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19341 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19342 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19343 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19344 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19345 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19346 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19347 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19348 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19349 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19350 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19351 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19352 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19353 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19354 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19355 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_294_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_16
  simp only [← c19339, k3, ← c19340, k3, ← c19341] at e_294_16
  have e_294_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_17
  simp only [← c19342, ← c19343, k3, ← c19344] at e_294_17
  have e_294_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_18
  simp only [← c19351, ← c19352, k3, ← c19353] at e_294_18
  have e_295_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_5
  simp only [← c19345, ← c19346, ← c19347] at e_295_5
  have e_295_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_6
  simp only [← c19348, ← c19349, ← c19350] at e_295_6
  refine ⟨?_, ?_⟩
  · have hc := e_295_5
    simp only [e_294_17] at hc
    linear_combination c19354.trans k2 - hc
  · have hc := e_294_18
    simp only [e_294_16, e_295_6, e_294_17] at hc
    linear_combination c19355.trans k2 - hc

theorem publicBatchWrapper86_f2633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13208)) (a (.wire 165 39)) (a (.virt 18697)) (a (.virt 18698)) := by
  have c19356 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19357 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19358 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19359 := (publicBatchWrapper86_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19360 := (publicBatchWrapper86_copies605 a h).1
  have c19361 := (publicBatchWrapper86_copies605 a h).2.1
  have c19362 := (publicBatchWrapper86_copies605 a h).2.2.1
  have c19363 := (publicBatchWrapper86_copies605 a h).2.2.2.1
  have c19364 := (publicBatchWrapper86_copies605 a h).2.2.2.2.1
  have c19365 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.1
  have c19366 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.1
  have c19367 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.1
  have c19368 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.1
  have c19369 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.1
  have c19370 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19371 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19372 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_294_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_19
  simp only [← c19356, k3, ← c19357, k3, ← c19358] at e_294_19
  have e_295_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_7
  simp only [← c19362, ← c19363, ← c19364] at e_295_7
  have e_295_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_8
  simp only [← c19365, ← c19366, ← c19367] at e_295_8
  have e_296_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_0
  simp only [← c19359, ← c19360, k3, ← c19361] at e_296_0
  have e_296_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_1
  simp only [← c19368, ← c19369, k3, ← c19370] at e_296_1
  refine ⟨?_, ?_⟩
  · have hc := e_295_7
    simp only [e_296_0] at hc
    linear_combination c19371.trans k2 - hc
  · have hc := e_296_1
    simp only [e_294_19, e_295_8, e_296_0] at hc
    linear_combination c19372.trans k2 - hc

theorem publicBatchWrapper86_f2634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 295 39) = band (a (.virt 18691)) (a (.virt 18693)) := by
  have c19373 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19374 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19375 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_295_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_9
  simp only [← c19373, ← c19374, ← c19375] at e_295_9
  have hr := e_295_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 295 43) = band (a (.virt 18695)) (a (.virt 18697)) := by
  have c19376 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19377 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19378 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_295_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_10
  simp only [← c19376, ← c19377, ← c19378] at e_295_10
  have hr := e_295_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 295 47) = band (a (.wire 295 39)) (a (.wire 295 43)) := by
  have c19379 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19380 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19381 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_295_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_11
  simp only [← c19379, ← c19380, ← c19381] at e_295_11
  have hr := e_295_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 55) = bor (a (.wire 65 51)) (a (.wire 295 47)) := by
  have c19382 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19383 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19384 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19385 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19386 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19387 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_13
  simp only [← c19385, ← c19386, k3, ← c19387] at e_287_13
  have e_293_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_5
  simp only [← c19382, ← c19383, ← c19384] at e_293_5
  have hr := e_287_13
  simp only [e_293_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 55) = a (.virt 201) := by
  have c19388 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19388

theorem publicBatchWrapper86_f2639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13403)) (a (.wire 165 55)) (a (.virt 18699)) (a (.virt 18700)) := by
  have c19389 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19390 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19391 := (publicBatchWrapper86_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19392 := (publicBatchWrapper86_copies606 a h).1
  have c19393 := (publicBatchWrapper86_copies606 a h).2.1
  have c19394 := (publicBatchWrapper86_copies606 a h).2.2.1
  have c19395 := (publicBatchWrapper86_copies606 a h).2.2.2.1
  have c19396 := (publicBatchWrapper86_copies606 a h).2.2.2.2.1
  have c19397 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.1
  have c19398 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.1
  have c19399 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.1
  have c19400 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.1
  have c19401 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.1
  have c19402 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19403 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19404 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19405 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_295_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_12
  simp only [← c19395, ← c19396, ← c19397] at e_295_12
  have e_295_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_13
  simp only [← c19398, ← c19399, ← c19400] at e_295_13
  have e_296_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_2
  simp only [← c19389, k3, ← c19390, k3, ← c19391] at e_296_2
  have e_296_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_3
  simp only [← c19392, ← c19393, k3, ← c19394] at e_296_3
  have e_296_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_4
  simp only [← c19401, ← c19402, k3, ← c19403] at e_296_4
  refine ⟨?_, ?_⟩
  · have hc := e_295_12
    simp only [e_296_3] at hc
    linear_combination c19404.trans k2 - hc
  · have hc := e_296_4
    simp only [e_296_2, e_295_13, e_296_3] at hc
    linear_combination c19405.trans k2 - hc

theorem publicBatchWrapper86_f2640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 59) = bor (a (.wire 67 15)) (a (.virt 18699)) := by
  have c19406 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19407 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19408 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19409 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19410 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19411 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_14
  simp only [← c19409, ← c19410, k3, ← c19411] at e_287_14
  have e_293_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_6
  simp only [← c19406, ← c19407, ← c19408] at e_293_6
  have hr := e_287_14
  simp only [e_293_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 59) = a (.virt 201) := by
  have c19412 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19412

theorem publicBatchWrapper86_f2642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13404)) (a (.wire 165 63)) (a (.virt 18701)) (a (.virt 18702)) := by
  have c19413 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19414 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19415 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19416 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19417 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19418 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19419 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19420 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19421 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19422 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19423 := (publicBatchWrapper86_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19424 := (publicBatchWrapper86_copies607 a h).1
  have c19425 := (publicBatchWrapper86_copies607 a h).2.1
  have c19426 := (publicBatchWrapper86_copies607 a h).2.2.1
  have c19427 := (publicBatchWrapper86_copies607 a h).2.2.2.1
  have c19428 := (publicBatchWrapper86_copies607 a h).2.2.2.2.1
  have c19429 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_295_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_14
  simp only [← c19419, ← c19420, ← c19421] at e_295_14
  have e_295_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_15
  simp only [← c19422, ← c19423, ← c19424] at e_295_15
  have e_296_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_5
  simp only [← c19413, k3, ← c19414, k3, ← c19415] at e_296_5
  have e_296_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_6
  simp only [← c19416, ← c19417, k3, ← c19418] at e_296_6
  have e_296_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_7
  simp only [← c19425, ← c19426, k3, ← c19427] at e_296_7
  refine ⟨?_, ?_⟩
  · have hc := e_295_14
    simp only [e_296_6] at hc
    linear_combination c19428.trans k2 - hc
  · have hc := e_296_7
    simp only [e_296_5, e_295_15, e_296_6] at hc
    linear_combination c19429.trans k2 - hc

theorem publicBatchWrapper86_f2643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 63) = bor (a (.wire 67 15)) (a (.virt 18701)) := by
  have c19430 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.1
  have c19431 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.1
  have c19432 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.1
  have c19433 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.1
  have c19434 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19435 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_15
  simp only [← c19433, ← c19434, k3, ← c19435] at e_287_15
  have e_293_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_7
  simp only [← c19430, ← c19431, ← c19432] at e_293_7
  have hr := e_287_15
  simp only [e_293_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 63) = a (.virt 201) := by
  have c19436 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19436

theorem publicBatchWrapper86_f2645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13405)) (a (.wire 165 15)) (a (.virt 18703)) (a (.virt 18704)) := by
  have c19437 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19438 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19439 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19440 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19441 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19442 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19443 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19444 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19445 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19446 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19447 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19448 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19449 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19450 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19451 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19452 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19453 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_295_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_16
  simp only [← c19443, ← c19444, ← c19445] at e_295_16
  have e_295_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_17
  simp only [← c19446, ← c19447, ← c19448] at e_295_17
  have e_296_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_8
  simp only [← c19437, k3, ← c19438, k3, ← c19439] at e_296_8
  have e_296_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_9
  simp only [← c19440, ← c19441, k3, ← c19442] at e_296_9
  have e_296_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_10
  simp only [← c19449, ← c19450, k3, ← c19451] at e_296_10
  refine ⟨?_, ?_⟩
  · have hc := e_295_16
    simp only [e_296_9] at hc
    linear_combination c19452.trans k2 - hc
  · have hc := e_296_10
    simp only [e_296_8, e_295_17, e_296_9] at hc
    linear_combination c19453.trans k2 - hc

theorem publicBatchWrapper86_f2646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13406)) (a (.wire 165 23)) (a (.virt 18705)) (a (.virt 18706)) := by
  have c19454 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19455 := (publicBatchWrapper86_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19456 := (publicBatchWrapper86_copies608 a h).1
  have c19457 := (publicBatchWrapper86_copies608 a h).2.1
  have c19458 := (publicBatchWrapper86_copies608 a h).2.2.1
  have c19459 := (publicBatchWrapper86_copies608 a h).2.2.2.1
  have c19460 := (publicBatchWrapper86_copies608 a h).2.2.2.2.1
  have c19461 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.1
  have c19462 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.1
  have c19463 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.1
  have c19464 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.1
  have c19465 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.1
  have c19466 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19467 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19468 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19469 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19470 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_295_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_18
  simp only [← c19460, ← c19461, ← c19462] at e_295_18
  have e_295_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_19
  simp only [← c19463, ← c19464, ← c19465] at e_295_19
  have e_296_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_11
  simp only [← c19454, k3, ← c19455, k3, ← c19456] at e_296_11
  have e_296_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_12
  simp only [← c19457, ← c19458, k3, ← c19459] at e_296_12
  have e_296_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_13
  simp only [← c19466, ← c19467, k3, ← c19468] at e_296_13
  refine ⟨?_, ?_⟩
  · have hc := e_295_18
    simp only [e_296_12] at hc
    linear_combination c19469.trans k2 - hc
  · have hc := e_296_13
    simp only [e_296_11, e_295_19, e_296_12] at hc
    linear_combination c19470.trans k2 - hc

theorem publicBatchWrapper86_f2647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13407)) (a (.wire 165 31)) (a (.virt 18707)) (a (.virt 18708)) := by
  have c19471 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19472 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19473 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19474 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19475 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19476 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19477 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19478 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19479 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19480 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19481 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19482 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19483 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19484 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19485 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19486 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19487 := (publicBatchWrapper86_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_296_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_14
  simp only [← c19471, k3, ← c19472, k3, ← c19473] at e_296_14
  have e_296_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_15
  simp only [← c19474, ← c19475, k3, ← c19476] at e_296_15
  have e_296_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_16
  simp only [← c19483, ← c19484, k3, ← c19485] at e_296_16
  have e_297_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_0
  simp only [← c19477, ← c19478, ← c19479] at e_297_0
  have e_297_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_1
  simp only [← c19480, ← c19481, ← c19482] at e_297_1
  refine ⟨?_, ?_⟩
  · have hc := e_297_0
    simp only [e_296_15] at hc
    linear_combination c19486.trans k2 - hc
  · have hc := e_296_16
    simp only [e_296_14, e_297_1, e_296_15] at hc
    linear_combination c19487.trans k2 - hc

theorem publicBatchWrapper86_f2648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13408)) (a (.wire 165 39)) (a (.virt 18709)) (a (.virt 18710)) := by
  have c19488 := (publicBatchWrapper86_copies609 a h).1
  have c19489 := (publicBatchWrapper86_copies609 a h).2.1
  have c19490 := (publicBatchWrapper86_copies609 a h).2.2.1
  have c19491 := (publicBatchWrapper86_copies609 a h).2.2.2.1
  have c19492 := (publicBatchWrapper86_copies609 a h).2.2.2.2.1
  have c19493 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.1
  have c19494 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.1
  have c19495 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.1
  have c19496 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.1
  have c19497 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.1
  have c19498 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19499 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19500 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19501 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19502 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19503 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19504 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_296_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_17
  simp only [← c19488, k3, ← c19489, k3, ← c19490] at e_296_17
  have e_296_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_18
  simp only [← c19491, ← c19492, k3, ← c19493] at e_296_18
  have e_296_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_19
  simp only [← c19500, ← c19501, k3, ← c19502] at e_296_19
  have e_297_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_2
  simp only [← c19494, ← c19495, ← c19496] at e_297_2
  have e_297_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_3
  simp only [← c19497, ← c19498, ← c19499] at e_297_3
  refine ⟨?_, ?_⟩
  · have hc := e_297_2
    simp only [e_296_18] at hc
    linear_combination c19503.trans k2 - hc
  · have hc := e_296_19
    simp only [e_296_17, e_297_3, e_296_18] at hc
    linear_combination c19504.trans k2 - hc

theorem publicBatchWrapper86_f2649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 297 19) = band (a (.virt 18703)) (a (.virt 18705)) := by
  have c19505 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19506 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19507 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_297_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_4
  simp only [← c19505, ← c19506, ← c19507] at e_297_4
  have hr := e_297_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 297 23) = band (a (.virt 18707)) (a (.virt 18709)) := by
  have c19508 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19509 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19510 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_297_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_5
  simp only [← c19508, ← c19509, ← c19510] at e_297_5
  have hr := e_297_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 297 27) = band (a (.wire 297 19)) (a (.wire 297 23)) := by
  have c19511 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19512 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19513 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_297_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_6
  simp only [← c19511, ← c19512, ← c19513] at e_297_6
  have hr := e_297_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 67) = bor (a (.wire 67 15)) (a (.wire 297 27)) := by
  have c19514 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19515 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19516 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19517 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19518 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19519 := (publicBatchWrapper86_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_16
  simp only [← c19517, ← c19518, k3, ← c19519] at e_287_16
  have e_293_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_8
  simp only [← c19514, ← c19515, ← c19516] at e_293_8
  have hr := e_287_16
  simp only [e_293_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 67) = a (.virt 201) := by
  have c19520 := (publicBatchWrapper86_copies610 a h).1
  exact c19520

theorem publicBatchWrapper86_f2654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13603)) (a (.wire 165 55)) (a (.virt 18711)) (a (.virt 18712)) := by
  have c19521 := (publicBatchWrapper86_copies610 a h).2.1
  have c19522 := (publicBatchWrapper86_copies610 a h).2.2.1
  have c19523 := (publicBatchWrapper86_copies610 a h).2.2.2.1
  have c19524 := (publicBatchWrapper86_copies610 a h).2.2.2.2.1
  have c19525 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.1
  have c19526 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.1
  have c19527 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.1
  have c19528 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.1
  have c19529 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.1
  have c19530 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19531 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19532 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19533 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19534 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19535 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19536 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19537 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_297_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_7
  simp only [← c19527, ← c19528, ← c19529] at e_297_7
  have e_297_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_8
  simp only [← c19530, ← c19531, ← c19532] at e_297_8
  have e_298_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_0
  simp only [← c19521, k3, ← c19522, k3, ← c19523] at e_298_0
  have e_298_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_1
  simp only [← c19524, ← c19525, k3, ← c19526] at e_298_1
  have e_298_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_2
  simp only [← c19533, ← c19534, k3, ← c19535] at e_298_2
  refine ⟨?_, ?_⟩
  · have hc := e_297_7
    simp only [e_298_1] at hc
    linear_combination c19536.trans k2 - hc
  · have hc := e_298_2
    simp only [e_298_0, e_297_8, e_298_1] at hc
    linear_combination c19537.trans k2 - hc

theorem publicBatchWrapper86_f2655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 71) = bor (a (.wire 67 59)) (a (.virt 18711)) := by
  have c19538 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19539 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19540 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19541 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19542 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19543 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_17
  simp only [← c19541, ← c19542, k3, ← c19543] at e_287_17
  have e_293_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_9
  simp only [← c19538, ← c19539, ← c19540] at e_293_9
  have hr := e_287_17
  simp only [e_293_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 71) = a (.virt 201) := by
  have c19544 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19544

theorem publicBatchWrapper86_f2657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13604)) (a (.wire 165 63)) (a (.virt 18713)) (a (.virt 18714)) := by
  have c19545 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19546 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19547 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19548 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19549 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19550 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19551 := (publicBatchWrapper86_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19552 := (publicBatchWrapper86_copies611 a h).1
  have c19553 := (publicBatchWrapper86_copies611 a h).2.1
  have c19554 := (publicBatchWrapper86_copies611 a h).2.2.1
  have c19555 := (publicBatchWrapper86_copies611 a h).2.2.2.1
  have c19556 := (publicBatchWrapper86_copies611 a h).2.2.2.2.1
  have c19557 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.1
  have c19558 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.1
  have c19559 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.1
  have c19560 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.1
  have c19561 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_297_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_9
  simp only [← c19551, ← c19552, ← c19553] at e_297_9
  have e_297_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_10
  simp only [← c19554, ← c19555, ← c19556] at e_297_10
  have e_298_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_3
  simp only [← c19545, k3, ← c19546, k3, ← c19547] at e_298_3
  have e_298_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_4
  simp only [← c19548, ← c19549, k3, ← c19550] at e_298_4
  have e_298_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_5
  simp only [← c19557, ← c19558, k3, ← c19559] at e_298_5
  refine ⟨?_, ?_⟩
  · have hc := e_297_9
    simp only [e_298_4] at hc
    linear_combination c19560.trans k2 - hc
  · have hc := e_298_5
    simp only [e_298_3, e_297_10, e_298_4] at hc
    linear_combination c19561.trans k2 - hc

theorem publicBatchWrapper86_f2658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 75) = bor (a (.wire 67 59)) (a (.virt 18713)) := by
  have c19562 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19563 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19564 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19565 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19566 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19567 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_18
  simp only [← c19565, ← c19566, k3, ← c19567] at e_287_18
  have e_293_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_10
  simp only [← c19562, ← c19563, ← c19564] at e_293_10
  have hr := e_287_18
  simp only [e_293_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 75) = a (.virt 201) := by
  have c19568 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19568

theorem publicBatchWrapper86_f2660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13605)) (a (.wire 165 15)) (a (.virt 18715)) (a (.virt 18716)) := by
  have c19569 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19570 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19571 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19572 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19573 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19574 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19575 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19576 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19577 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19578 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19579 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19580 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19581 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19582 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19583 := (publicBatchWrapper86_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19584 := (publicBatchWrapper86_copies612 a h).1
  have c19585 := (publicBatchWrapper86_copies612 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_297_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_11
  simp only [← c19575, ← c19576, ← c19577] at e_297_11
  have e_297_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_12
  simp only [← c19578, ← c19579, ← c19580] at e_297_12
  have e_298_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_6
  simp only [← c19569, k3, ← c19570, k3, ← c19571] at e_298_6
  have e_298_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_7
  simp only [← c19572, ← c19573, k3, ← c19574] at e_298_7
  have e_298_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_8
  simp only [← c19581, ← c19582, k3, ← c19583] at e_298_8
  refine ⟨?_, ?_⟩
  · have hc := e_297_11
    simp only [e_298_7] at hc
    linear_combination c19584.trans k2 - hc
  · have hc := e_298_8
    simp only [e_298_6, e_297_12, e_298_7] at hc
    linear_combination c19585.trans k2 - hc

theorem publicBatchWrapper86_f2661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13606)) (a (.wire 165 23)) (a (.virt 18717)) (a (.virt 18718)) := by
  have c19586 := (publicBatchWrapper86_copies612 a h).2.2.1
  have c19587 := (publicBatchWrapper86_copies612 a h).2.2.2.1
  have c19588 := (publicBatchWrapper86_copies612 a h).2.2.2.2.1
  have c19589 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.1
  have c19590 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.1
  have c19591 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.1
  have c19592 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.1
  have c19593 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.1
  have c19594 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19595 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19596 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19597 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19598 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19599 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19600 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19601 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19602 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_297_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_13
  simp only [← c19592, ← c19593, ← c19594] at e_297_13
  have e_297_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_14
  simp only [← c19595, ← c19596, ← c19597] at e_297_14
  have e_298_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_9
  simp only [← c19586, k3, ← c19587, k3, ← c19588] at e_298_9
  have e_298_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_10
  simp only [← c19589, ← c19590, k3, ← c19591] at e_298_10
  have e_298_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_11
  simp only [← c19598, ← c19599, k3, ← c19600] at e_298_11
  refine ⟨?_, ?_⟩
  · have hc := e_297_13
    simp only [e_298_10] at hc
    linear_combination c19601.trans k2 - hc
  · have hc := e_298_11
    simp only [e_298_9, e_297_14, e_298_10] at hc
    linear_combination c19602.trans k2 - hc

theorem publicBatchWrapper86_f2662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13607)) (a (.wire 165 31)) (a (.virt 18719)) (a (.virt 18720)) := by
  have c19603 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19604 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19605 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19606 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19607 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19608 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19609 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19610 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19611 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19612 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19613 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19614 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19615 := (publicBatchWrapper86_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19616 := (publicBatchWrapper86_copies613 a h).1
  have c19617 := (publicBatchWrapper86_copies613 a h).2.1
  have c19618 := (publicBatchWrapper86_copies613 a h).2.2.1
  have c19619 := (publicBatchWrapper86_copies613 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_297_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_15
  simp only [← c19609, ← c19610, ← c19611] at e_297_15
  have e_297_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_16
  simp only [← c19612, ← c19613, ← c19614] at e_297_16
  have e_298_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_12
  simp only [← c19603, k3, ← c19604, k3, ← c19605] at e_298_12
  have e_298_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_13
  simp only [← c19606, ← c19607, k3, ← c19608] at e_298_13
  have e_298_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_14
  simp only [← c19615, ← c19616, k3, ← c19617] at e_298_14
  refine ⟨?_, ?_⟩
  · have hc := e_297_15
    simp only [e_298_13] at hc
    linear_combination c19618.trans k2 - hc
  · have hc := e_298_14
    simp only [e_298_12, e_297_16, e_298_13] at hc
    linear_combination c19619.trans k2 - hc

theorem publicBatchWrapper86_f2663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13608)) (a (.wire 165 39)) (a (.virt 18721)) (a (.virt 18722)) := by
  have c19620 := (publicBatchWrapper86_copies613 a h).2.2.2.2.1
  have c19621 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.1
  have c19622 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.1
  have c19623 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.1
  have c19624 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.1
  have c19625 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.1
  have c19626 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19627 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19628 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19629 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19630 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19631 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19632 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19633 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19634 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19635 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19636 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_297_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_17
  simp only [← c19626, ← c19627, ← c19628] at e_297_17
  have e_297_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_18
  simp only [← c19629, ← c19630, ← c19631] at e_297_18
  have e_298_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_15
  simp only [← c19620, k3, ← c19621, k3, ← c19622] at e_298_15
  have e_298_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_16
  simp only [← c19623, ← c19624, k3, ← c19625] at e_298_16
  have e_298_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_17
  simp only [← c19632, ← c19633, k3, ← c19634] at e_298_17
  refine ⟨?_, ?_⟩
  · have hc := e_297_17
    simp only [e_298_16] at hc
    linear_combination c19635.trans k2 - hc
  · have hc := e_298_17
    simp only [e_298_15, e_297_18, e_298_16] at hc
    linear_combination c19636.trans k2 - hc

theorem publicBatchWrapper86_f2664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 297 79) = band (a (.virt 18715)) (a (.virt 18717)) := by
  have c19637 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19638 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19639 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_297_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_19
  simp only [← c19637, ← c19638, ← c19639] at e_297_19
  have hr := e_297_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 299 3) = band (a (.virt 18719)) (a (.virt 18721)) := by
  have c19640 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19641 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19642 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_299_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_0
  simp only [← c19640, ← c19641, ← c19642] at e_299_0
  have hr := e_299_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 299 7) = band (a (.wire 297 79)) (a (.wire 299 3)) := by
  have c19643 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19644 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19645 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_299_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_1
  simp only [← c19643, ← c19644, ← c19645] at e_299_1
  have hr := e_299_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 79) = bor (a (.wire 67 59)) (a (.wire 299 7)) := by
  have c19646 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19647 := (publicBatchWrapper86_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19648 := (publicBatchWrapper86_copies614 a h).1
  have c19649 := (publicBatchWrapper86_copies614 a h).2.1
  have c19650 := (publicBatchWrapper86_copies614 a h).2.2.1
  have c19651 := (publicBatchWrapper86_copies614 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_287_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_19
  simp only [← c19649, ← c19650, k3, ← c19651] at e_287_19
  have e_293_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_11
  simp only [← c19646, ← c19647, ← c19648] at e_293_11
  have hr := e_287_19
  simp only [e_293_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 287 79) = a (.virt 201) := by
  have c19652 := (publicBatchWrapper86_copies614 a h).2.2.2.2.1
  exact c19652

theorem publicBatchWrapper86_f2669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13803)) (a (.wire 165 55)) (a (.virt 18723)) (a (.virt 18724)) := by
  have c19653 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.1
  have c19654 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.1
  have c19655 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.1
  have c19656 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.1
  have c19657 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.1
  have c19658 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19659 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19660 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19661 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19662 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19663 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19664 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19665 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19666 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19667 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19668 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19669 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_298_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_18
  simp only [← c19653, k3, ← c19654, k3, ← c19655] at e_298_18
  have e_298_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_19
  simp only [← c19656, ← c19657, k3, ← c19658] at e_298_19
  have e_299_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_2
  simp only [← c19659, ← c19660, ← c19661] at e_299_2
  have e_299_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_3
  simp only [← c19662, ← c19663, ← c19664] at e_299_3
  have e_300_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_0
  simp only [← c19665, ← c19666, k3, ← c19667] at e_300_0
  refine ⟨?_, ?_⟩
  · have hc := e_299_2
    simp only [e_298_19] at hc
    linear_combination c19668.trans k2 - hc
  · have hc := e_300_0
    simp only [e_298_18, e_299_3, e_298_19] at hc
    linear_combination c19669.trans k2 - hc

theorem publicBatchWrapper86_f2670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 3) = bor (a (.wire 68 23)) (a (.virt 18723)) := by
  have c19670 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19671 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19672 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19673 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19674 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19675 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_293_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_12
  simp only [← c19670, ← c19671, ← c19672] at e_293_12
  have e_301_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_0
  simp only [← c19673, ← c19674, k3, ← c19675] at e_301_0
  have hr := e_301_0
  simp only [e_293_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 3) = a (.virt 201) := by
  have c19676 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19676

theorem publicBatchWrapper86_f2672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13804)) (a (.wire 165 63)) (a (.virt 18725)) (a (.virt 18726)) := by
  have c19677 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19678 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19679 := (publicBatchWrapper86_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19680 := (publicBatchWrapper86_copies615 a h).1
  have c19681 := (publicBatchWrapper86_copies615 a h).2.1
  have c19682 := (publicBatchWrapper86_copies615 a h).2.2.1
  have c19683 := (publicBatchWrapper86_copies615 a h).2.2.2.1
  have c19684 := (publicBatchWrapper86_copies615 a h).2.2.2.2.1
  have c19685 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.1
  have c19686 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.1
  have c19687 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.1
  have c19688 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.1
  have c19689 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.1
  have c19690 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19691 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19692 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19693 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_299_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_4
  simp only [← c19683, ← c19684, ← c19685] at e_299_4
  have e_299_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_5
  simp only [← c19686, ← c19687, ← c19688] at e_299_5
  have e_300_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_1
  simp only [← c19677, k3, ← c19678, k3, ← c19679] at e_300_1
  have e_300_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_2
  simp only [← c19680, ← c19681, k3, ← c19682] at e_300_2
  have e_300_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_3
  simp only [← c19689, ← c19690, k3, ← c19691] at e_300_3
  refine ⟨?_, ?_⟩
  · have hc := e_299_4
    simp only [e_300_2] at hc
    linear_combination c19692.trans k2 - hc
  · have hc := e_300_3
    simp only [e_300_1, e_299_5, e_300_2] at hc
    linear_combination c19693.trans k2 - hc

theorem publicBatchWrapper86_f2673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 7) = bor (a (.wire 68 23)) (a (.virt 18725)) := by
  have c19694 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19695 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19696 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19697 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19698 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19699 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_293_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_13
  simp only [← c19694, ← c19695, ← c19696] at e_293_13
  have e_301_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_1
  simp only [← c19697, ← c19698, k3, ← c19699] at e_301_1
  have hr := e_301_1
  simp only [e_293_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 7) = a (.virt 201) := by
  have c19700 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19700

theorem publicBatchWrapper86_f2675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13805)) (a (.wire 165 15)) (a (.virt 18727)) (a (.virt 18728)) := by
  have c19701 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19702 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19703 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19704 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19705 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19706 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19707 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19708 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19709 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19710 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19711 := (publicBatchWrapper86_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19712 := (publicBatchWrapper86_copies616 a h).1
  have c19713 := (publicBatchWrapper86_copies616 a h).2.1
  have c19714 := (publicBatchWrapper86_copies616 a h).2.2.1
  have c19715 := (publicBatchWrapper86_copies616 a h).2.2.2.1
  have c19716 := (publicBatchWrapper86_copies616 a h).2.2.2.2.1
  have c19717 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_299_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_6
  simp only [← c19707, ← c19708, ← c19709] at e_299_6
  have e_299_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_7
  simp only [← c19710, ← c19711, ← c19712] at e_299_7
  have e_300_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_4
  simp only [← c19701, k3, ← c19702, k3, ← c19703] at e_300_4
  have e_300_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_5
  simp only [← c19704, ← c19705, k3, ← c19706] at e_300_5
  have e_300_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_6
  simp only [← c19713, ← c19714, k3, ← c19715] at e_300_6
  refine ⟨?_, ?_⟩
  · have hc := e_299_6
    simp only [e_300_5] at hc
    linear_combination c19716.trans k2 - hc
  · have hc := e_300_6
    simp only [e_300_4, e_299_7, e_300_5] at hc
    linear_combination c19717.trans k2 - hc

theorem publicBatchWrapper86_f2676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13806)) (a (.wire 165 23)) (a (.virt 18729)) (a (.virt 18730)) := by
  have c19718 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.1
  have c19719 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.1
  have c19720 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.1
  have c19721 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.1
  have c19722 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19723 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19724 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19725 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19726 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19727 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19728 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19729 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19730 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19731 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19732 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19733 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19734 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_299_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_8
  simp only [← c19724, ← c19725, ← c19726] at e_299_8
  have e_299_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_9
  simp only [← c19727, ← c19728, ← c19729] at e_299_9
  have e_300_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_7
  simp only [← c19718, k3, ← c19719, k3, ← c19720] at e_300_7
  have e_300_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_8
  simp only [← c19721, ← c19722, k3, ← c19723] at e_300_8
  have e_300_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_9
  simp only [← c19730, ← c19731, k3, ← c19732] at e_300_9
  refine ⟨?_, ?_⟩
  · have hc := e_299_8
    simp only [e_300_8] at hc
    linear_combination c19733.trans k2 - hc
  · have hc := e_300_9
    simp only [e_300_7, e_299_9, e_300_8] at hc
    linear_combination c19734.trans k2 - hc

theorem publicBatchWrapper86_f2677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13807)) (a (.wire 165 31)) (a (.virt 18731)) (a (.virt 18732)) := by
  have c19735 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19736 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19737 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19738 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19739 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19740 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19741 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19742 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19743 := (publicBatchWrapper86_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19744 := (publicBatchWrapper86_copies617 a h).1
  have c19745 := (publicBatchWrapper86_copies617 a h).2.1
  have c19746 := (publicBatchWrapper86_copies617 a h).2.2.1
  have c19747 := (publicBatchWrapper86_copies617 a h).2.2.2.1
  have c19748 := (publicBatchWrapper86_copies617 a h).2.2.2.2.1
  have c19749 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.1
  have c19750 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.1
  have c19751 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_299_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_10
  simp only [← c19741, ← c19742, ← c19743] at e_299_10
  have e_299_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_11
  simp only [← c19744, ← c19745, ← c19746] at e_299_11
  have e_300_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_10
  simp only [← c19735, k3, ← c19736, k3, ← c19737] at e_300_10
  have e_300_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_11
  simp only [← c19738, ← c19739, k3, ← c19740] at e_300_11
  have e_300_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_12
  simp only [← c19747, ← c19748, k3, ← c19749] at e_300_12
  refine ⟨?_, ?_⟩
  · have hc := e_299_10
    simp only [e_300_11] at hc
    linear_combination c19750.trans k2 - hc
  · have hc := e_300_12
    simp only [e_300_10, e_299_11, e_300_11] at hc
    linear_combination c19751.trans k2 - hc

theorem publicBatchWrapper86_f2678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13808)) (a (.wire 165 39)) (a (.virt 18733)) (a (.virt 18734)) := by
  have c19752 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.1
  have c19753 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.1
  have c19754 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19755 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19756 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19757 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19758 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19759 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19760 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19761 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19762 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19763 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19764 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19765 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19766 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19767 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19768 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_299_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_12
  simp only [← c19758, ← c19759, ← c19760] at e_299_12
  have e_299_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_13
  simp only [← c19761, ← c19762, ← c19763] at e_299_13
  have e_300_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_13
  simp only [← c19752, k3, ← c19753, k3, ← c19754] at e_300_13
  have e_300_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_14
  simp only [← c19755, ← c19756, k3, ← c19757] at e_300_14
  have e_300_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_15
  simp only [← c19764, ← c19765, k3, ← c19766] at e_300_15
  refine ⟨?_, ?_⟩
  · have hc := e_299_12
    simp only [e_300_14] at hc
    linear_combination c19767.trans k2 - hc
  · have hc := e_300_15
    simp only [e_300_13, e_299_13, e_300_14] at hc
    linear_combination c19768.trans k2 - hc

theorem publicBatchWrapper86_f2679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 299 59) = band (a (.virt 18727)) (a (.virt 18729)) := by
  have c19769 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19770 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19771 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_299_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_14
  simp only [← c19769, ← c19770, ← c19771] at e_299_14
  have hr := e_299_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 299 63) = band (a (.virt 18731)) (a (.virt 18733)) := by
  have c19772 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19773 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19774 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_299_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_15
  simp only [← c19772, ← c19773, ← c19774] at e_299_15
  have hr := e_299_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 299 67) = band (a (.wire 299 59)) (a (.wire 299 63)) := by
  have c19775 := (publicBatchWrapper86_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19776 := (publicBatchWrapper86_copies618 a h).1
  have c19777 := (publicBatchWrapper86_copies618 a h).2.1
  have e_299_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_16
  simp only [← c19775, ← c19776, ← c19777] at e_299_16
  have hr := e_299_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 11) = bor (a (.wire 68 23)) (a (.wire 299 67)) := by
  have c19778 := (publicBatchWrapper86_copies618 a h).2.2.1
  have c19779 := (publicBatchWrapper86_copies618 a h).2.2.2.1
  have c19780 := (publicBatchWrapper86_copies618 a h).2.2.2.2.1
  have c19781 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.1
  have c19782 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.1
  have c19783 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_293_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_14
  simp only [← c19778, ← c19779, ← c19780] at e_293_14
  have e_301_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_2
  simp only [← c19781, ← c19782, k3, ← c19783] at e_301_2
  have hr := e_301_2
  simp only [e_293_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 11) = a (.virt 201) := by
  have c19784 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.1
  exact c19784

theorem publicBatchWrapper86_f2684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14003)) (a (.wire 165 55)) (a (.virt 18735)) (a (.virt 18736)) := by
  have c19785 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.1
  have c19786 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19787 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19788 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19789 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19790 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19791 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19792 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19793 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19794 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19795 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19796 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19797 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19798 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19799 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19800 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19801 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_299_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_17
  simp only [← c19791, ← c19792, ← c19793] at e_299_17
  have e_299_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_18
  simp only [← c19794, ← c19795, ← c19796] at e_299_18
  have e_300_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_16
  simp only [← c19785, k3, ← c19786, k3, ← c19787] at e_300_16
  have e_300_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_17
  simp only [← c19788, ← c19789, k3, ← c19790] at e_300_17
  have e_300_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_18
  simp only [← c19797, ← c19798, k3, ← c19799] at e_300_18
  refine ⟨?_, ?_⟩
  · have hc := e_299_17
    simp only [e_300_17] at hc
    linear_combination c19800.trans k2 - hc
  · have hc := e_300_18
    simp only [e_300_16, e_299_18, e_300_17] at hc
    linear_combination c19801.trans k2 - hc

theorem publicBatchWrapper86_f2685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 15) = bor (a (.wire 68 67)) (a (.virt 18735)) := by
  have c19802 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19803 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19804 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19805 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19806 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19807 := (publicBatchWrapper86_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_293_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_15
  simp only [← c19802, ← c19803, ← c19804] at e_293_15
  have e_301_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_3
  simp only [← c19805, ← c19806, k3, ← c19807] at e_301_3
  have hr := e_301_3
  simp only [e_293_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 15) = a (.virt 201) := by
  have c19808 := (publicBatchWrapper86_copies619 a h).1
  exact c19808

theorem publicBatchWrapper86_f2687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14004)) (a (.wire 165 63)) (a (.virt 18737)) (a (.virt 18738)) := by
  have c19809 := (publicBatchWrapper86_copies619 a h).2.1
  have c19810 := (publicBatchWrapper86_copies619 a h).2.2.1
  have c19811 := (publicBatchWrapper86_copies619 a h).2.2.2.1
  have c19812 := (publicBatchWrapper86_copies619 a h).2.2.2.2.1
  have c19813 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.1
  have c19814 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.1
  have c19815 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.1
  have c19816 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.1
  have c19817 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.1
  have c19818 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19819 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19820 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19821 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19822 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19823 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19824 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19825 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_299_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_19
  simp only [← c19815, ← c19816, ← c19817] at e_299_19
  have e_300_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_19
  simp only [← c19809, k3, ← c19810, k3, ← c19811] at e_300_19
  have e_302_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_0
  simp only [← c19812, ← c19813, k3, ← c19814] at e_302_0
  have e_302_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_1
  simp only [← c19821, ← c19822, k3, ← c19823] at e_302_1
  have e_303_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_0
  simp only [← c19818, ← c19819, ← c19820] at e_303_0
  refine ⟨?_, ?_⟩
  · have hc := e_299_19
    simp only [e_302_0] at hc
    linear_combination c19824.trans k2 - hc
  · have hc := e_302_1
    simp only [e_300_19, e_303_0, e_302_0] at hc
    linear_combination c19825.trans k2 - hc

theorem publicBatchWrapper86_f2688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 19) = bor (a (.wire 68 67)) (a (.virt 18737)) := by
  have c19826 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19827 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19828 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19829 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19830 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19831 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_293_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_16
  simp only [← c19826, ← c19827, ← c19828] at e_293_16
  have e_301_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_4
  simp only [← c19829, ← c19830, k3, ← c19831] at e_301_4
  have hr := e_301_4
  simp only [e_293_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 19) = a (.virt 201) := by
  have c19832 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19832

theorem publicBatchWrapper86_f2690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14005)) (a (.wire 165 15)) (a (.virt 18739)) (a (.virt 18740)) := by
  have c19833 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19834 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19835 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19836 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19837 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19838 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19839 := (publicBatchWrapper86_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19840 := (publicBatchWrapper86_copies620 a h).1
  have c19841 := (publicBatchWrapper86_copies620 a h).2.1
  have c19842 := (publicBatchWrapper86_copies620 a h).2.2.1
  have c19843 := (publicBatchWrapper86_copies620 a h).2.2.2.1
  have c19844 := (publicBatchWrapper86_copies620 a h).2.2.2.2.1
  have c19845 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.1
  have c19846 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.1
  have c19847 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.1
  have c19848 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.1
  have c19849 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_302_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_2
  simp only [← c19833, k3, ← c19834, k3, ← c19835] at e_302_2
  have e_302_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_3
  simp only [← c19836, ← c19837, k3, ← c19838] at e_302_3
  have e_302_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_4
  simp only [← c19845, ← c19846, k3, ← c19847] at e_302_4
  have e_303_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_1
  simp only [← c19839, ← c19840, ← c19841] at e_303_1
  have e_303_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_2
  simp only [← c19842, ← c19843, ← c19844] at e_303_2
  refine ⟨?_, ?_⟩
  · have hc := e_303_1
    simp only [e_302_3] at hc
    linear_combination c19848.trans k2 - hc
  · have hc := e_302_4
    simp only [e_302_2, e_303_2, e_302_3] at hc
    linear_combination c19849.trans k2 - hc

theorem publicBatchWrapper86_f2691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14006)) (a (.wire 165 23)) (a (.virt 18741)) (a (.virt 18742)) := by
  have c19850 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19851 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19852 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19853 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19854 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19855 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19856 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19857 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19858 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19859 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19860 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19861 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19862 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19863 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19864 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19865 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19866 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_302_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_5
  simp only [← c19850, k3, ← c19851, k3, ← c19852] at e_302_5
  have e_302_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_6
  simp only [← c19853, ← c19854, k3, ← c19855] at e_302_6
  have e_302_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_7
  simp only [← c19862, ← c19863, k3, ← c19864] at e_302_7
  have e_303_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_3
  simp only [← c19856, ← c19857, ← c19858] at e_303_3
  have e_303_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_4
  simp only [← c19859, ← c19860, ← c19861] at e_303_4
  refine ⟨?_, ?_⟩
  · have hc := e_303_3
    simp only [e_302_6] at hc
    linear_combination c19865.trans k2 - hc
  · have hc := e_302_7
    simp only [e_302_5, e_303_4, e_302_6] at hc
    linear_combination c19866.trans k2 - hc

theorem publicBatchWrapper86_f2692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14007)) (a (.wire 165 31)) (a (.virt 18743)) (a (.virt 18744)) := by
  have c19867 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19868 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19869 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19870 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19871 := (publicBatchWrapper86_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19872 := (publicBatchWrapper86_copies621 a h).1
  have c19873 := (publicBatchWrapper86_copies621 a h).2.1
  have c19874 := (publicBatchWrapper86_copies621 a h).2.2.1
  have c19875 := (publicBatchWrapper86_copies621 a h).2.2.2.1
  have c19876 := (publicBatchWrapper86_copies621 a h).2.2.2.2.1
  have c19877 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.1
  have c19878 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.1
  have c19879 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.1
  have c19880 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.1
  have c19881 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.1
  have c19882 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19883 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_302_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_8
  simp only [← c19867, k3, ← c19868, k3, ← c19869] at e_302_8
  have e_302_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_9
  simp only [← c19870, ← c19871, k3, ← c19872] at e_302_9
  have e_302_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_10
  simp only [← c19879, ← c19880, k3, ← c19881] at e_302_10
  have e_303_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_5
  simp only [← c19873, ← c19874, ← c19875] at e_303_5
  have e_303_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_6
  simp only [← c19876, ← c19877, ← c19878] at e_303_6
  refine ⟨?_, ?_⟩
  · have hc := e_303_5
    simp only [e_302_9] at hc
    linear_combination c19882.trans k2 - hc
  · have hc := e_302_10
    simp only [e_302_8, e_303_6, e_302_9] at hc
    linear_combination c19883.trans k2 - hc

theorem publicBatchWrapper86_f2693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14008)) (a (.wire 165 39)) (a (.virt 18745)) (a (.virt 18746)) := by
  have c19884 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19885 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19886 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19887 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19888 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19889 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19890 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19891 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19892 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19893 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19894 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19895 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19896 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19897 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19898 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19899 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19900 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_302_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_11
  simp only [← c19884, k3, ← c19885, k3, ← c19886] at e_302_11
  have e_302_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_12
  simp only [← c19887, ← c19888, k3, ← c19889] at e_302_12
  have e_302_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_13
  simp only [← c19896, ← c19897, k3, ← c19898] at e_302_13
  have e_303_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_7
  simp only [← c19890, ← c19891, ← c19892] at e_303_7
  have e_303_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_8
  simp only [← c19893, ← c19894, ← c19895] at e_303_8
  refine ⟨?_, ?_⟩
  · have hc := e_303_7
    simp only [e_302_12] at hc
    linear_combination c19899.trans k2 - hc
  · have hc := e_302_13
    simp only [e_302_11, e_303_8, e_302_12] at hc
    linear_combination c19900.trans k2 - hc

theorem publicBatchWrapper86_f2694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 303 39) = band (a (.virt 18739)) (a (.virt 18741)) := by
  have c19901 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19902 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19903 := (publicBatchWrapper86_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_303_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_9
  simp only [← c19901, ← c19902, ← c19903] at e_303_9
  have hr := e_303_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 303 43) = band (a (.virt 18743)) (a (.virt 18745)) := by
  have c19904 := (publicBatchWrapper86_copies622 a h).1
  have c19905 := (publicBatchWrapper86_copies622 a h).2.1
  have c19906 := (publicBatchWrapper86_copies622 a h).2.2.1
  have e_303_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_10
  simp only [← c19904, ← c19905, ← c19906] at e_303_10
  have hr := e_303_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 303 47) = band (a (.wire 303 39)) (a (.wire 303 43)) := by
  have c19907 := (publicBatchWrapper86_copies622 a h).2.2.2.1
  have c19908 := (publicBatchWrapper86_copies622 a h).2.2.2.2.1
  have c19909 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.1
  have e_303_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_11
  simp only [← c19907, ← c19908, ← c19909] at e_303_11
  have hr := e_303_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 23) = bor (a (.wire 68 67)) (a (.wire 303 47)) := by
  have c19910 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.1
  have c19911 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.1
  have c19912 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.1
  have c19913 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.1
  have c19914 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19915 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_293_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_17
  simp only [← c19910, ← c19911, ← c19912] at e_293_17
  have e_301_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_5
  simp only [← c19913, ← c19914, k3, ← c19915] at e_301_5
  have hr := e_301_5
  simp only [e_293_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 23) = a (.virt 201) := by
  have c19916 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19916

theorem publicBatchWrapper86_f2699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14203)) (a (.wire 165 55)) (a (.virt 18747)) (a (.virt 18748)) := by
  have c19917 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19918 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19919 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19920 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19921 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19922 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19923 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19924 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19925 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19926 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19927 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19928 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19929 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19930 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19931 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19932 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19933 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_302_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_14
  simp only [← c19917, k3, ← c19918, k3, ← c19919] at e_302_14
  have e_302_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_15
  simp only [← c19920, ← c19921, k3, ← c19922] at e_302_15
  have e_302_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_16
  simp only [← c19929, ← c19930, k3, ← c19931] at e_302_16
  have e_303_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_12
  simp only [← c19923, ← c19924, ← c19925] at e_303_12
  have e_303_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_13
  simp only [← c19926, ← c19927, ← c19928] at e_303_13
  refine ⟨?_, ?_⟩
  · have hc := e_303_12
    simp only [e_302_15] at hc
    linear_combination c19932.trans k2 - hc
  · have hc := e_302_16
    simp only [e_302_14, e_303_13, e_302_15] at hc
    linear_combination c19933.trans k2 - hc

theorem publicBatchWrapper86_f2700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 27) = bor (a (.wire 70 31)) (a (.virt 18747)) := by
  have c19934 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19935 := (publicBatchWrapper86_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19936 := (publicBatchWrapper86_copies623 a h).1
  have c19937 := (publicBatchWrapper86_copies623 a h).2.1
  have c19938 := (publicBatchWrapper86_copies623 a h).2.2.1
  have c19939 := (publicBatchWrapper86_copies623 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_293_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_18
  simp only [← c19934, ← c19935, ← c19936] at e_293_18
  have e_301_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_6
  simp only [← c19937, ← c19938, k3, ← c19939] at e_301_6
  have hr := e_301_6
  simp only [e_293_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 27) = a (.virt 201) := by
  have c19940 := (publicBatchWrapper86_copies623 a h).2.2.2.2.1
  exact c19940

theorem publicBatchWrapper86_f2702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14204)) (a (.wire 165 63)) (a (.virt 18749)) (a (.virt 18750)) := by
  have c19941 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.1
  have c19942 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.1
  have c19943 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.1
  have c19944 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.1
  have c19945 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.1
  have c19946 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19947 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19948 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19949 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19950 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19951 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19952 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19953 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19954 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19955 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19956 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19957 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_302_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_17
  simp only [← c19941, k3, ← c19942, k3, ← c19943] at e_302_17
  have e_302_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_18
  simp only [← c19944, ← c19945, k3, ← c19946] at e_302_18
  have e_302_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_19
  simp only [← c19953, ← c19954, k3, ← c19955] at e_302_19
  have e_303_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_14
  simp only [← c19947, ← c19948, ← c19949] at e_303_14
  have e_303_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_15
  simp only [← c19950, ← c19951, ← c19952] at e_303_15
  refine ⟨?_, ?_⟩
  · have hc := e_303_14
    simp only [e_302_18] at hc
    linear_combination c19956.trans k2 - hc
  · have hc := e_302_19
    simp only [e_302_17, e_303_15, e_302_18] at hc
    linear_combination c19957.trans k2 - hc

theorem publicBatchWrapper86_f2703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 31) = bor (a (.wire 70 31)) (a (.virt 18749)) := by
  have c19958 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19959 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19960 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19961 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19962 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19963 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_293_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_19
  simp only [← c19958, ← c19959, ← c19960] at e_293_19
  have e_301_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_7
  simp only [← c19961, ← c19962, k3, ← c19963] at e_301_7
  have hr := e_301_7
  simp only [e_293_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 31) = a (.virt 201) := by
  have c19964 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19964

theorem publicBatchWrapper86_f2705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14205)) (a (.wire 165 15)) (a (.virt 18751)) (a (.virt 18752)) := by
  have c19965 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19966 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19967 := (publicBatchWrapper86_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19968 := (publicBatchWrapper86_copies624 a h).1
  have c19969 := (publicBatchWrapper86_copies624 a h).2.1
  have c19970 := (publicBatchWrapper86_copies624 a h).2.2.1
  have c19971 := (publicBatchWrapper86_copies624 a h).2.2.2.1
  have c19972 := (publicBatchWrapper86_copies624 a h).2.2.2.2.1
  have c19973 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.1
  have c19974 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.1
  have c19975 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.1
  have c19976 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.1
  have c19977 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.1
  have c19978 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19979 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19980 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19981 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_303_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_16
  simp only [← c19971, ← c19972, ← c19973] at e_303_16
  have e_303_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_17
  simp only [← c19974, ← c19975, ← c19976] at e_303_17
  have e_304_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_0
  simp only [← c19965, k3, ← c19966, k3, ← c19967] at e_304_0
  have e_304_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_1
  simp only [← c19968, ← c19969, k3, ← c19970] at e_304_1
  have e_304_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_2
  simp only [← c19977, ← c19978, k3, ← c19979] at e_304_2
  refine ⟨?_, ?_⟩
  · have hc := e_303_16
    simp only [e_304_1] at hc
    linear_combination c19980.trans k2 - hc
  · have hc := e_304_2
    simp only [e_304_0, e_303_17, e_304_1] at hc
    linear_combination c19981.trans k2 - hc

theorem publicBatchWrapper86_f2706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14206)) (a (.wire 165 23)) (a (.virt 18753)) (a (.virt 18754)) := by
  have c19982 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19983 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19984 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19985 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19986 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19987 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19988 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19989 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19990 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19991 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19992 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19993 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19994 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19995 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19996 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19997 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19998 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_303_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_18
  simp only [← c19988, ← c19989, ← c19990] at e_303_18
  have e_303_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_19
  simp only [← c19991, ← c19992, ← c19993] at e_303_19
  have e_304_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_3
  simp only [← c19982, k3, ← c19983, k3, ← c19984] at e_304_3
  have e_304_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_4
  simp only [← c19985, ← c19986, k3, ← c19987] at e_304_4
  have e_304_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_5
  simp only [← c19994, ← c19995, k3, ← c19996] at e_304_5
  refine ⟨?_, ?_⟩
  · have hc := e_303_18
    simp only [e_304_4] at hc
    linear_combination c19997.trans k2 - hc
  · have hc := e_304_5
    simp only [e_304_3, e_303_19, e_304_4] at hc
    linear_combination c19998.trans k2 - hc

theorem publicBatchWrapper86_f2707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14207)) (a (.wire 165 31)) (a (.virt 18755)) (a (.virt 18756)) := by
  have c19999 := (publicBatchWrapper86_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20000 := (publicBatchWrapper86_copies625 a h).1
  have c20001 := (publicBatchWrapper86_copies625 a h).2.1
  have c20002 := (publicBatchWrapper86_copies625 a h).2.2.1
  have c20003 := (publicBatchWrapper86_copies625 a h).2.2.2.1
  have c20004 := (publicBatchWrapper86_copies625 a h).2.2.2.2.1
  have c20005 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.1
  have c20006 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.1
  have c20007 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.1
  have c20008 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.1
  have c20009 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.1
  have c20010 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20011 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20012 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20013 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20014 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20015 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_304_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_6
  simp only [← c19999, k3, ← c20000, k3, ← c20001] at e_304_6
  have e_304_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_7
  simp only [← c20002, ← c20003, k3, ← c20004] at e_304_7
  have e_304_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_8
  simp only [← c20011, ← c20012, k3, ← c20013] at e_304_8
  have e_305_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_0
  simp only [← c20005, ← c20006, ← c20007] at e_305_0
  have e_305_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_1
  simp only [← c20008, ← c20009, ← c20010] at e_305_1
  refine ⟨?_, ?_⟩
  · have hc := e_305_0
    simp only [e_304_7] at hc
    linear_combination c20014.trans k2 - hc
  · have hc := e_304_8
    simp only [e_304_6, e_305_1, e_304_7] at hc
    linear_combination c20015.trans k2 - hc

theorem publicBatchWrapper86_f2708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14208)) (a (.wire 165 39)) (a (.virt 18757)) (a (.virt 18758)) := by
  have c20016 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20017 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20018 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20019 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20020 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20021 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20022 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20023 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20024 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20025 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20026 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20027 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20028 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20029 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20030 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20031 := (publicBatchWrapper86_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20032 := (publicBatchWrapper86_copies626 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_304_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_9
  simp only [← c20016, k3, ← c20017, k3, ← c20018] at e_304_9
  have e_304_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_10
  simp only [← c20019, ← c20020, k3, ← c20021] at e_304_10
  have e_304_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_11
  simp only [← c20028, ← c20029, k3, ← c20030] at e_304_11
  have e_305_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_2
  simp only [← c20022, ← c20023, ← c20024] at e_305_2
  have e_305_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_3
  simp only [← c20025, ← c20026, ← c20027] at e_305_3
  refine ⟨?_, ?_⟩
  · have hc := e_305_2
    simp only [e_304_10] at hc
    linear_combination c20031.trans k2 - hc
  · have hc := e_304_11
    simp only [e_304_9, e_305_3, e_304_10] at hc
    linear_combination c20032.trans k2 - hc

theorem publicBatchWrapper86_f2709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 305 19) = band (a (.virt 18751)) (a (.virt 18753)) := by
  have c20033 := (publicBatchWrapper86_copies626 a h).2.1
  have c20034 := (publicBatchWrapper86_copies626 a h).2.2.1
  have c20035 := (publicBatchWrapper86_copies626 a h).2.2.2.1
  have e_305_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_4
  simp only [← c20033, ← c20034, ← c20035] at e_305_4
  have hr := e_305_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 305 23) = band (a (.virt 18755)) (a (.virt 18757)) := by
  have c20036 := (publicBatchWrapper86_copies626 a h).2.2.2.2.1
  have c20037 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.1
  have c20038 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.1
  have e_305_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_5
  simp only [← c20036, ← c20037, ← c20038] at e_305_5
  have hr := e_305_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 305 27) = band (a (.wire 305 19)) (a (.wire 305 23)) := by
  have c20039 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.1
  have c20040 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.1
  have c20041 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.1
  have e_305_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_6
  simp only [← c20039, ← c20040, ← c20041] at e_305_6
  have hr := e_305_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 35) = bor (a (.wire 70 31)) (a (.wire 305 27)) := by
  have c20042 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20043 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20044 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20045 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20046 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20047 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_8
  simp only [← c20045, ← c20046, k3, ← c20047] at e_301_8
  have e_306_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_0
  simp only [← c20042, ← c20043, ← c20044] at e_306_0
  have hr := e_301_8
  simp only [e_306_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 35) = a (.virt 201) := by
  have c20048 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20048

theorem publicBatchWrapper86_f2714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14403)) (a (.wire 165 55)) (a (.virt 18759)) (a (.virt 18760)) := by
  have c20049 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20050 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20051 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20052 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20053 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20054 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20055 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20056 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20057 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20058 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20059 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20060 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20061 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20062 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20063 := (publicBatchWrapper86_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20064 := (publicBatchWrapper86_copies627 a h).1
  have c20065 := (publicBatchWrapper86_copies627 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_304_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_12
  simp only [← c20049, k3, ← c20050, k3, ← c20051] at e_304_12
  have e_304_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_13
  simp only [← c20052, ← c20053, k3, ← c20054] at e_304_13
  have e_304_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_14
  simp only [← c20061, ← c20062, k3, ← c20063] at e_304_14
  have e_305_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_7
  simp only [← c20055, ← c20056, ← c20057] at e_305_7
  have e_305_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_8
  simp only [← c20058, ← c20059, ← c20060] at e_305_8
  refine ⟨?_, ?_⟩
  · have hc := e_305_7
    simp only [e_304_13] at hc
    linear_combination c20064.trans k2 - hc
  · have hc := e_304_14
    simp only [e_304_12, e_305_8, e_304_13] at hc
    linear_combination c20065.trans k2 - hc

theorem publicBatchWrapper86_f2715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 39) = bor (a (.wire 70 75)) (a (.virt 18759)) := by
  have c20066 := (publicBatchWrapper86_copies627 a h).2.2.1
  have c20067 := (publicBatchWrapper86_copies627 a h).2.2.2.1
  have c20068 := (publicBatchWrapper86_copies627 a h).2.2.2.2.1
  have c20069 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.1
  have c20070 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.1
  have c20071 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_9
  simp only [← c20069, ← c20070, k3, ← c20071] at e_301_9
  have e_306_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_1
  simp only [← c20066, ← c20067, ← c20068] at e_306_1
  have hr := e_301_9
  simp only [e_306_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 39) = a (.virt 201) := by
  have c20072 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.1
  exact c20072

theorem publicBatchWrapper86_f2717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14404)) (a (.wire 165 63)) (a (.virt 18761)) (a (.virt 18762)) := by
  have c20073 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.1
  have c20074 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20075 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20076 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20077 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20078 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20079 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20080 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20081 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20082 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20083 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20084 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20085 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20086 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20087 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20088 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20089 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_304_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_15
  simp only [← c20073, k3, ← c20074, k3, ← c20075] at e_304_15
  have e_304_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_16
  simp only [← c20076, ← c20077, k3, ← c20078] at e_304_16
  have e_304_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_17
  simp only [← c20085, ← c20086, k3, ← c20087] at e_304_17
  have e_305_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_9
  simp only [← c20079, ← c20080, ← c20081] at e_305_9
  have e_305_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_10
  simp only [← c20082, ← c20083, ← c20084] at e_305_10
  refine ⟨?_, ?_⟩
  · have hc := e_305_9
    simp only [e_304_16] at hc
    linear_combination c20088.trans k2 - hc
  · have hc := e_304_17
    simp only [e_304_15, e_305_10, e_304_16] at hc
    linear_combination c20089.trans k2 - hc

theorem publicBatchWrapper86_f2718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 43) = bor (a (.wire 70 75)) (a (.virt 18761)) := by
  have c20090 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20091 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20092 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20093 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20094 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20095 := (publicBatchWrapper86_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_10
  simp only [← c20093, ← c20094, k3, ← c20095] at e_301_10
  have e_306_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_2
  simp only [← c20090, ← c20091, ← c20092] at e_306_2
  have hr := e_301_10
  simp only [e_306_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 43) = a (.virt 201) := by
  have c20096 := (publicBatchWrapper86_copies628 a h).1
  exact c20096

theorem publicBatchWrapper86_f2720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14405)) (a (.wire 165 15)) (a (.virt 18763)) (a (.virt 18764)) := by
  have c20097 := (publicBatchWrapper86_copies628 a h).2.1
  have c20098 := (publicBatchWrapper86_copies628 a h).2.2.1
  have c20099 := (publicBatchWrapper86_copies628 a h).2.2.2.1
  have c20100 := (publicBatchWrapper86_copies628 a h).2.2.2.2.1
  have c20101 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.1
  have c20102 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.1
  have c20103 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.1
  have c20104 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.1
  have c20105 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.1
  have c20106 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20107 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20108 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20109 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20110 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20111 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20112 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20113 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_304_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_18
  simp only [← c20097, k3, ← c20098, k3, ← c20099] at e_304_18
  have e_304_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_19
  simp only [← c20100, ← c20101, k3, ← c20102] at e_304_19
  have e_305_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_11
  simp only [← c20103, ← c20104, ← c20105] at e_305_11
  have e_305_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_12
  simp only [← c20106, ← c20107, ← c20108] at e_305_12
  have e_307_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_0
  simp only [← c20109, ← c20110, k3, ← c20111] at e_307_0
  refine ⟨?_, ?_⟩
  · have hc := e_305_11
    simp only [e_304_19] at hc
    linear_combination c20112.trans k2 - hc
  · have hc := e_307_0
    simp only [e_304_18, e_305_12, e_304_19] at hc
    linear_combination c20113.trans k2 - hc

theorem publicBatchWrapper86_f2721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14406)) (a (.wire 165 23)) (a (.virt 18765)) (a (.virt 18766)) := by
  have c20114 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20115 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20116 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20117 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20118 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20119 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20120 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20121 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20122 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20123 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20124 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20125 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20126 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20127 := (publicBatchWrapper86_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20128 := (publicBatchWrapper86_copies629 a h).1
  have c20129 := (publicBatchWrapper86_copies629 a h).2.1
  have c20130 := (publicBatchWrapper86_copies629 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_305_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_13
  simp only [← c20120, ← c20121, ← c20122] at e_305_13
  have e_305_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_14
  simp only [← c20123, ← c20124, ← c20125] at e_305_14
  have e_307_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_1
  simp only [← c20114, k3, ← c20115, k3, ← c20116] at e_307_1
  have e_307_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_2
  simp only [← c20117, ← c20118, k3, ← c20119] at e_307_2
  have e_307_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_3
  simp only [← c20126, ← c20127, k3, ← c20128] at e_307_3
  refine ⟨?_, ?_⟩
  · have hc := e_305_13
    simp only [e_307_2] at hc
    linear_combination c20129.trans k2 - hc
  · have hc := e_307_3
    simp only [e_307_1, e_305_14, e_307_2] at hc
    linear_combination c20130.trans k2 - hc

theorem publicBatchWrapper86_f2722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14407)) (a (.wire 165 31)) (a (.virt 18767)) (a (.virt 18768)) := by
  have c20131 := (publicBatchWrapper86_copies629 a h).2.2.2.1
  have c20132 := (publicBatchWrapper86_copies629 a h).2.2.2.2.1
  have c20133 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.1
  have c20134 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.1
  have c20135 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.1
  have c20136 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.1
  have c20137 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.1
  have c20138 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20139 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20140 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20141 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20142 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20143 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20144 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20145 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20146 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20147 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_305_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_15
  simp only [← c20137, ← c20138, ← c20139] at e_305_15
  have e_305_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_16
  simp only [← c20140, ← c20141, ← c20142] at e_305_16
  have e_307_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_4
  simp only [← c20131, k3, ← c20132, k3, ← c20133] at e_307_4
  have e_307_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_5
  simp only [← c20134, ← c20135, k3, ← c20136] at e_307_5
  have e_307_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_6
  simp only [← c20143, ← c20144, k3, ← c20145] at e_307_6
  refine ⟨?_, ?_⟩
  · have hc := e_305_15
    simp only [e_307_5] at hc
    linear_combination c20146.trans k2 - hc
  · have hc := e_307_6
    simp only [e_307_4, e_305_16, e_307_5] at hc
    linear_combination c20147.trans k2 - hc

theorem publicBatchWrapper86_f2723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14408)) (a (.wire 165 39)) (a (.virt 18769)) (a (.virt 18770)) := by
  have c20148 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20149 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20150 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20151 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20152 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20153 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20154 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20155 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20156 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20157 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20158 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20159 := (publicBatchWrapper86_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20160 := (publicBatchWrapper86_copies630 a h).1
  have c20161 := (publicBatchWrapper86_copies630 a h).2.1
  have c20162 := (publicBatchWrapper86_copies630 a h).2.2.1
  have c20163 := (publicBatchWrapper86_copies630 a h).2.2.2.1
  have c20164 := (publicBatchWrapper86_copies630 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_305_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_17
  simp only [← c20154, ← c20155, ← c20156] at e_305_17
  have e_305_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_18
  simp only [← c20157, ← c20158, ← c20159] at e_305_18
  have e_307_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_7
  simp only [← c20148, k3, ← c20149, k3, ← c20150] at e_307_7
  have e_307_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_8
  simp only [← c20151, ← c20152, k3, ← c20153] at e_307_8
  have e_307_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_9
  simp only [← c20160, ← c20161, k3, ← c20162] at e_307_9
  refine ⟨?_, ?_⟩
  · have hc := e_305_17
    simp only [e_307_8] at hc
    linear_combination c20163.trans k2 - hc
  · have hc := e_307_9
    simp only [e_307_7, e_305_18, e_307_8] at hc
    linear_combination c20164.trans k2 - hc

theorem publicBatchWrapper86_f2724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 305 79) = band (a (.virt 18763)) (a (.virt 18765)) := by
  have c20165 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.1
  have c20166 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.1
  have c20167 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.1
  have e_305_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_19
  simp only [← c20165, ← c20166, ← c20167] at e_305_19
  have hr := e_305_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 308 3) = band (a (.virt 18767)) (a (.virt 18769)) := by
  have c20168 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.1
  have c20169 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.1
  have c20170 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_308_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_0
  simp only [← c20168, ← c20169, ← c20170] at e_308_0
  have hr := e_308_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 308 7) = band (a (.wire 305 79)) (a (.wire 308 3)) := by
  have c20171 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20172 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20173 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_308_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_1
  simp only [← c20171, ← c20172, ← c20173] at e_308_1
  have hr := e_308_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 47) = bor (a (.wire 70 75)) (a (.wire 308 7)) := by
  have c20174 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20175 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20176 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20177 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20178 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20179 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_11
  simp only [← c20177, ← c20178, k3, ← c20179] at e_301_11
  have e_306_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_3
  simp only [← c20174, ← c20175, ← c20176] at e_306_3
  have hr := e_301_11
  simp only [e_306_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 47) = a (.virt 201) := by
  have c20180 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20180

theorem publicBatchWrapper86_f2729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14603)) (a (.wire 165 55)) (a (.virt 18771)) (a (.virt 18772)) := by
  have c20181 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20182 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20183 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20184 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20185 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20186 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20187 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20188 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20189 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20190 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20191 := (publicBatchWrapper86_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20192 := (publicBatchWrapper86_copies631 a h).1
  have c20193 := (publicBatchWrapper86_copies631 a h).2.1
  have c20194 := (publicBatchWrapper86_copies631 a h).2.2.1
  have c20195 := (publicBatchWrapper86_copies631 a h).2.2.2.1
  have c20196 := (publicBatchWrapper86_copies631 a h).2.2.2.2.1
  have c20197 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_307_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_10
  simp only [← c20181, k3, ← c20182, k3, ← c20183] at e_307_10
  have e_307_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_11
  simp only [← c20184, ← c20185, k3, ← c20186] at e_307_11
  have e_307_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_12
  simp only [← c20193, ← c20194, k3, ← c20195] at e_307_12
  have e_308_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_2
  simp only [← c20187, ← c20188, ← c20189] at e_308_2
  have e_308_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_3
  simp only [← c20190, ← c20191, ← c20192] at e_308_3
  refine ⟨?_, ?_⟩
  · have hc := e_308_2
    simp only [e_307_11] at hc
    linear_combination c20196.trans k2 - hc
  · have hc := e_307_12
    simp only [e_307_10, e_308_3, e_307_11] at hc
    linear_combination c20197.trans k2 - hc

theorem publicBatchWrapper86_f2730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 51) = bor (a (.wire 72 39)) (a (.virt 18771)) := by
  have c20198 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.1
  have c20199 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.1
  have c20200 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.1
  have c20201 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.1
  have c20202 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20203 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_12
  simp only [← c20201, ← c20202, k3, ← c20203] at e_301_12
  have e_306_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_4
  simp only [← c20198, ← c20199, ← c20200] at e_306_4
  have hr := e_301_12
  simp only [e_306_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 51) = a (.virt 201) := by
  have c20204 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20204

theorem publicBatchWrapper86_f2732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14604)) (a (.wire 165 63)) (a (.virt 18773)) (a (.virt 18774)) := by
  have c20205 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20206 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20207 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20208 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20209 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20210 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20211 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20212 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20213 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20214 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20215 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20216 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20217 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20218 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20219 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20220 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20221 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_307_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_13
  simp only [← c20205, k3, ← c20206, k3, ← c20207] at e_307_13
  have e_307_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_14
  simp only [← c20208, ← c20209, k3, ← c20210] at e_307_14
  have e_307_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_15
  simp only [← c20217, ← c20218, k3, ← c20219] at e_307_15
  have e_308_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_4
  simp only [← c20211, ← c20212, ← c20213] at e_308_4
  have e_308_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_5
  simp only [← c20214, ← c20215, ← c20216] at e_308_5
  refine ⟨?_, ?_⟩
  · have hc := e_308_4
    simp only [e_307_14] at hc
    linear_combination c20220.trans k2 - hc
  · have hc := e_307_15
    simp only [e_307_13, e_308_5, e_307_14] at hc
    linear_combination c20221.trans k2 - hc

theorem publicBatchWrapper86_f2733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 55) = bor (a (.wire 72 39)) (a (.virt 18773)) := by
  have c20222 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20223 := (publicBatchWrapper86_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20224 := (publicBatchWrapper86_copies632 a h).1
  have c20225 := (publicBatchWrapper86_copies632 a h).2.1
  have c20226 := (publicBatchWrapper86_copies632 a h).2.2.1
  have c20227 := (publicBatchWrapper86_copies632 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_13
  simp only [← c20225, ← c20226, k3, ← c20227] at e_301_13
  have e_306_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_5
  simp only [← c20222, ← c20223, ← c20224] at e_306_5
  have hr := e_301_13
  simp only [e_306_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 55) = a (.virt 201) := by
  have c20228 := (publicBatchWrapper86_copies632 a h).2.2.2.2.1
  exact c20228

theorem publicBatchWrapper86_f2735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14605)) (a (.wire 165 15)) (a (.virt 18775)) (a (.virt 18776)) := by
  have c20229 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.1
  have c20230 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.1
  have c20231 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.1
  have c20232 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.1
  have c20233 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.1
  have c20234 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20235 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20236 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20237 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20238 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20239 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20240 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20241 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20242 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20243 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20244 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20245 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_307_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_16
  simp only [← c20229, k3, ← c20230, k3, ← c20231] at e_307_16
  have e_307_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_17
  simp only [← c20232, ← c20233, k3, ← c20234] at e_307_17
  have e_307_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_18
  simp only [← c20241, ← c20242, k3, ← c20243] at e_307_18
  have e_308_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_6
  simp only [← c20235, ← c20236, ← c20237] at e_308_6
  have e_308_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_7
  simp only [← c20238, ← c20239, ← c20240] at e_308_7
  refine ⟨?_, ?_⟩
  · have hc := e_308_6
    simp only [e_307_17] at hc
    linear_combination c20244.trans k2 - hc
  · have hc := e_307_18
    simp only [e_307_16, e_308_7, e_307_17] at hc
    linear_combination c20245.trans k2 - hc

theorem publicBatchWrapper86_f2736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14606)) (a (.wire 165 23)) (a (.virt 18777)) (a (.virt 18778)) := by
  have c20246 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20247 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20248 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20249 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20250 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20251 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20252 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20253 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20254 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20255 := (publicBatchWrapper86_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20256 := (publicBatchWrapper86_copies633 a h).1
  have c20257 := (publicBatchWrapper86_copies633 a h).2.1
  have c20258 := (publicBatchWrapper86_copies633 a h).2.2.1
  have c20259 := (publicBatchWrapper86_copies633 a h).2.2.2.1
  have c20260 := (publicBatchWrapper86_copies633 a h).2.2.2.2.1
  have c20261 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.1
  have c20262 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_307_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_19
  simp only [← c20246, k3, ← c20247, k3, ← c20248] at e_307_19
  have e_308_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_8
  simp only [← c20252, ← c20253, ← c20254] at e_308_8
  have e_308_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_9
  simp only [← c20255, ← c20256, ← c20257] at e_308_9
  have e_309_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_0
  simp only [← c20249, ← c20250, k3, ← c20251] at e_309_0
  have e_309_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_1
  simp only [← c20258, ← c20259, k3, ← c20260] at e_309_1
  refine ⟨?_, ?_⟩
  · have hc := e_308_8
    simp only [e_309_0] at hc
    linear_combination c20261.trans k2 - hc
  · have hc := e_309_1
    simp only [e_307_19, e_308_9, e_309_0] at hc
    linear_combination c20262.trans k2 - hc

theorem publicBatchWrapper86_f2737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14607)) (a (.wire 165 31)) (a (.virt 18779)) (a (.virt 18780)) := by
  have c20263 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.1
  have c20264 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.1
  have c20265 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.1
  have c20266 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20267 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20268 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20269 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20270 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20271 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20272 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20273 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20274 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20275 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20276 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20277 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20278 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20279 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_308_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_10
  simp only [← c20269, ← c20270, ← c20271] at e_308_10
  have e_308_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_11
  simp only [← c20272, ← c20273, ← c20274] at e_308_11
  have e_309_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_2
  simp only [← c20263, k3, ← c20264, k3, ← c20265] at e_309_2
  have e_309_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_3
  simp only [← c20266, ← c20267, k3, ← c20268] at e_309_3
  have e_309_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_4
  simp only [← c20275, ← c20276, k3, ← c20277] at e_309_4
  refine ⟨?_, ?_⟩
  · have hc := e_308_10
    simp only [e_309_3] at hc
    linear_combination c20278.trans k2 - hc
  · have hc := e_309_4
    simp only [e_309_2, e_308_11, e_309_3] at hc
    linear_combination c20279.trans k2 - hc

theorem publicBatchWrapper86_f2738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14608)) (a (.wire 165 39)) (a (.virt 18781)) (a (.virt 18782)) := by
  have c20280 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20281 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20282 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20283 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20284 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20285 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20286 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20287 := (publicBatchWrapper86_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20288 := (publicBatchWrapper86_copies634 a h).1
  have c20289 := (publicBatchWrapper86_copies634 a h).2.1
  have c20290 := (publicBatchWrapper86_copies634 a h).2.2.1
  have c20291 := (publicBatchWrapper86_copies634 a h).2.2.2.1
  have c20292 := (publicBatchWrapper86_copies634 a h).2.2.2.2.1
  have c20293 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.1
  have c20294 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.1
  have c20295 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.1
  have c20296 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_308_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_12
  simp only [← c20286, ← c20287, ← c20288] at e_308_12
  have e_308_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_13
  simp only [← c20289, ← c20290, ← c20291] at e_308_13
  have e_309_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_5
  simp only [← c20280, k3, ← c20281, k3, ← c20282] at e_309_5
  have e_309_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_6
  simp only [← c20283, ← c20284, k3, ← c20285] at e_309_6
  have e_309_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_7
  simp only [← c20292, ← c20293, k3, ← c20294] at e_309_7
  refine ⟨?_, ?_⟩
  · have hc := e_308_12
    simp only [e_309_6] at hc
    linear_combination c20295.trans k2 - hc
  · have hc := e_309_7
    simp only [e_309_5, e_308_13, e_309_6] at hc
    linear_combination c20296.trans k2 - hc

theorem publicBatchWrapper86_f2739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 308 59) = band (a (.virt 18775)) (a (.virt 18777)) := by
  have c20297 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.1
  have c20298 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20299 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_308_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_14
  simp only [← c20297, ← c20298, ← c20299] at e_308_14
  have hr := e_308_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 308 63) = band (a (.virt 18779)) (a (.virt 18781)) := by
  have c20300 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20301 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20302 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_308_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_15
  simp only [← c20300, ← c20301, ← c20302] at e_308_15
  have hr := e_308_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 308 67) = band (a (.wire 308 59)) (a (.wire 308 63)) := by
  have c20303 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20304 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20305 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_308_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_16
  simp only [← c20303, ← c20304, ← c20305] at e_308_16
  have hr := e_308_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 59) = bor (a (.wire 72 39)) (a (.wire 308 67)) := by
  have c20306 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20307 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20308 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20309 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20310 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20311 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_14
  simp only [← c20309, ← c20310, k3, ← c20311] at e_301_14
  have e_306_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_6
  simp only [← c20306, ← c20307, ← c20308] at e_306_6
  have hr := e_301_14
  simp only [e_306_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 59) = a (.virt 201) := by
  have c20312 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20312

theorem publicBatchWrapper86_f2744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14803)) (a (.wire 165 55)) (a (.virt 18783)) (a (.virt 18784)) := by
  have c20313 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20314 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20315 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20316 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20317 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20318 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20319 := (publicBatchWrapper86_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20320 := (publicBatchWrapper86_copies635 a h).1
  have c20321 := (publicBatchWrapper86_copies635 a h).2.1
  have c20322 := (publicBatchWrapper86_copies635 a h).2.2.1
  have c20323 := (publicBatchWrapper86_copies635 a h).2.2.2.1
  have c20324 := (publicBatchWrapper86_copies635 a h).2.2.2.2.1
  have c20325 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.1
  have c20326 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.1
  have c20327 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.1
  have c20328 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.1
  have c20329 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_308_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_17
  simp only [← c20319, ← c20320, ← c20321] at e_308_17
  have e_308_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_18
  simp only [← c20322, ← c20323, ← c20324] at e_308_18
  have e_309_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_8
  simp only [← c20313, k3, ← c20314, k3, ← c20315] at e_309_8
  have e_309_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_9
  simp only [← c20316, ← c20317, k3, ← c20318] at e_309_9
  have e_309_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_10
  simp only [← c20325, ← c20326, k3, ← c20327] at e_309_10
  refine ⟨?_, ?_⟩
  · have hc := e_308_17
    simp only [e_309_9] at hc
    linear_combination c20328.trans k2 - hc
  · have hc := e_309_10
    simp only [e_309_8, e_308_18, e_309_9] at hc
    linear_combination c20329.trans k2 - hc

theorem publicBatchWrapper86_f2745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 63) = bor (a (.wire 73 3)) (a (.virt 18783)) := by
  have c20330 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20331 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20332 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20333 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20334 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20335 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_15
  simp only [← c20333, ← c20334, k3, ← c20335] at e_301_15
  have e_306_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_7
  simp only [← c20330, ← c20331, ← c20332] at e_306_7
  have hr := e_301_15
  simp only [e_306_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 63) = a (.virt 201) := by
  have c20336 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20336

theorem publicBatchWrapper86_f2747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14804)) (a (.wire 165 63)) (a (.virt 18785)) (a (.virt 18786)) := by
  have c20337 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20338 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20339 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20340 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20341 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20342 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20343 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20344 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20345 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20346 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20347 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20348 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20349 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20350 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20351 := (publicBatchWrapper86_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20352 := (publicBatchWrapper86_copies636 a h).1
  have c20353 := (publicBatchWrapper86_copies636 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_308_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_19
  simp only [← c20343, ← c20344, ← c20345] at e_308_19
  have e_309_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_11
  simp only [← c20337, k3, ← c20338, k3, ← c20339] at e_309_11
  have e_309_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_12
  simp only [← c20340, ← c20341, k3, ← c20342] at e_309_12
  have e_309_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_13
  simp only [← c20349, ← c20350, k3, ← c20351] at e_309_13
  have e_310_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_0
  simp only [← c20346, ← c20347, ← c20348] at e_310_0
  refine ⟨?_, ?_⟩
  · have hc := e_308_19
    simp only [e_309_12] at hc
    linear_combination c20352.trans k2 - hc
  · have hc := e_309_13
    simp only [e_309_11, e_310_0, e_309_12] at hc
    linear_combination c20353.trans k2 - hc

theorem publicBatchWrapper86_f2748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 67) = bor (a (.wire 73 3)) (a (.virt 18785)) := by
  have c20354 := (publicBatchWrapper86_copies636 a h).2.2.1
  have c20355 := (publicBatchWrapper86_copies636 a h).2.2.2.1
  have c20356 := (publicBatchWrapper86_copies636 a h).2.2.2.2.1
  have c20357 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.1
  have c20358 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.1
  have c20359 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_16
  simp only [← c20357, ← c20358, k3, ← c20359] at e_301_16
  have e_306_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_8
  simp only [← c20354, ← c20355, ← c20356] at e_306_8
  have hr := e_301_16
  simp only [e_306_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 67) = a (.virt 201) := by
  have c20360 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.1
  exact c20360

theorem publicBatchWrapper86_f2750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14805)) (a (.wire 165 15)) (a (.virt 18787)) (a (.virt 18788)) := by
  have c20361 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.1
  have c20362 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20363 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20364 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20365 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20366 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20367 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20368 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20369 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20370 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20371 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20372 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20373 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20374 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20375 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20376 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20377 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_309_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_14
  simp only [← c20361, k3, ← c20362, k3, ← c20363] at e_309_14
  have e_309_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_15
  simp only [← c20364, ← c20365, k3, ← c20366] at e_309_15
  have e_309_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_16
  simp only [← c20373, ← c20374, k3, ← c20375] at e_309_16
  have e_310_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_1
  simp only [← c20367, ← c20368, ← c20369] at e_310_1
  have e_310_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_2
  simp only [← c20370, ← c20371, ← c20372] at e_310_2
  refine ⟨?_, ?_⟩
  · have hc := e_310_1
    simp only [e_309_15] at hc
    linear_combination c20376.trans k2 - hc
  · have hc := e_309_16
    simp only [e_309_14, e_310_2, e_309_15] at hc
    linear_combination c20377.trans k2 - hc

theorem publicBatchWrapper86_f2751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14806)) (a (.wire 165 23)) (a (.virt 18789)) (a (.virt 18790)) := by
  have c20378 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20379 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20380 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20381 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20382 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20383 := (publicBatchWrapper86_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20384 := (publicBatchWrapper86_copies637 a h).1
  have c20385 := (publicBatchWrapper86_copies637 a h).2.1
  have c20386 := (publicBatchWrapper86_copies637 a h).2.2.1
  have c20387 := (publicBatchWrapper86_copies637 a h).2.2.2.1
  have c20388 := (publicBatchWrapper86_copies637 a h).2.2.2.2.1
  have c20389 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.1
  have c20390 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.1
  have c20391 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.1
  have c20392 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.1
  have c20393 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.1
  have c20394 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_309_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_17
  simp only [← c20378, k3, ← c20379, k3, ← c20380] at e_309_17
  have e_309_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_18
  simp only [← c20381, ← c20382, k3, ← c20383] at e_309_18
  have e_309_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_19
  simp only [← c20390, ← c20391, k3, ← c20392] at e_309_19
  have e_310_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_3
  simp only [← c20384, ← c20385, ← c20386] at e_310_3
  have e_310_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_4
  simp only [← c20387, ← c20388, ← c20389] at e_310_4
  refine ⟨?_, ?_⟩
  · have hc := e_310_3
    simp only [e_309_18] at hc
    linear_combination c20393.trans k2 - hc
  · have hc := e_309_19
    simp only [e_309_17, e_310_4, e_309_18] at hc
    linear_combination c20394.trans k2 - hc

theorem publicBatchWrapper86_f2752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14807)) (a (.wire 165 31)) (a (.virt 18791)) (a (.virt 18792)) := by
  have c20395 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20396 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20397 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20398 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20399 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20400 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20401 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20402 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20403 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20404 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20405 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20406 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20407 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20408 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20409 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20410 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20411 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_310_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_5
  simp only [← c20401, ← c20402, ← c20403] at e_310_5
  have e_310_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_6
  simp only [← c20404, ← c20405, ← c20406] at e_310_6
  have e_311_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_0
  simp only [← c20395, k3, ← c20396, k3, ← c20397] at e_311_0
  have e_311_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_1
  simp only [← c20398, ← c20399, k3, ← c20400] at e_311_1
  have e_311_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_2
  simp only [← c20407, ← c20408, k3, ← c20409] at e_311_2
  refine ⟨?_, ?_⟩
  · have hc := e_310_5
    simp only [e_311_1] at hc
    linear_combination c20410.trans k2 - hc
  · have hc := e_311_2
    simp only [e_311_0, e_310_6, e_311_1] at hc
    linear_combination c20411.trans k2 - hc

theorem publicBatchWrapper86_f2753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14808)) (a (.wire 165 39)) (a (.virt 18793)) (a (.virt 18794)) := by
  have c20412 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20413 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20414 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20415 := (publicBatchWrapper86_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20416 := (publicBatchWrapper86_copies638 a h).1
  have c20417 := (publicBatchWrapper86_copies638 a h).2.1
  have c20418 := (publicBatchWrapper86_copies638 a h).2.2.1
  have c20419 := (publicBatchWrapper86_copies638 a h).2.2.2.1
  have c20420 := (publicBatchWrapper86_copies638 a h).2.2.2.2.1
  have c20421 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.1
  have c20422 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.1
  have c20423 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.1
  have c20424 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.1
  have c20425 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.1
  have c20426 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20427 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20428 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_310_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_7
  simp only [← c20418, ← c20419, ← c20420] at e_310_7
  have e_310_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_8
  simp only [← c20421, ← c20422, ← c20423] at e_310_8
  have e_311_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_3
  simp only [← c20412, k3, ← c20413, k3, ← c20414] at e_311_3
  have e_311_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_4
  simp only [← c20415, ← c20416, k3, ← c20417] at e_311_4
  have e_311_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_5
  simp only [← c20424, ← c20425, k3, ← c20426] at e_311_5
  refine ⟨?_, ?_⟩
  · have hc := e_310_7
    simp only [e_311_4] at hc
    linear_combination c20427.trans k2 - hc
  · have hc := e_311_5
    simp only [e_311_3, e_310_8, e_311_4] at hc
    linear_combination c20428.trans k2 - hc

theorem publicBatchWrapper86_f2754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 310 39) = band (a (.virt 18787)) (a (.virt 18789)) := by
  have c20429 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20430 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20431 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_310_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_9
  simp only [← c20429, ← c20430, ← c20431] at e_310_9
  have hr := e_310_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 310 43) = band (a (.virt 18791)) (a (.virt 18793)) := by
  have c20432 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20433 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20434 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_310_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_10
  simp only [← c20432, ← c20433, ← c20434] at e_310_10
  have hr := e_310_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 310 47) = band (a (.wire 310 39)) (a (.wire 310 43)) := by
  have c20435 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20436 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20437 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_310_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_11
  simp only [← c20435, ← c20436, ← c20437] at e_310_11
  have hr := e_310_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 71) = bor (a (.wire 73 3)) (a (.wire 310 47)) := by
  have c20438 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20439 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20440 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20441 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20442 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20443 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_17
  simp only [← c20441, ← c20442, k3, ← c20443] at e_301_17
  have e_306_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_9
  simp only [← c20438, ← c20439, ← c20440] at e_306_9
  have hr := e_301_17
  simp only [e_306_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 71) = a (.virt 201) := by
  have c20444 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20444

theorem publicBatchWrapper86_f2759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15003)) (a (.wire 165 55)) (a (.virt 18795)) (a (.virt 18796)) := by
  have c20445 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20446 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20447 := (publicBatchWrapper86_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20448 := (publicBatchWrapper86_copies639 a h).1
  have c20449 := (publicBatchWrapper86_copies639 a h).2.1
  have c20450 := (publicBatchWrapper86_copies639 a h).2.2.1
  have c20451 := (publicBatchWrapper86_copies639 a h).2.2.2.1
  have c20452 := (publicBatchWrapper86_copies639 a h).2.2.2.2.1
  have c20453 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.1
  have c20454 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.1
  have c20455 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.1
  have c20456 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.1
  have c20457 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.1
  have c20458 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20459 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20460 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20461 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_310_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_12
  simp only [← c20451, ← c20452, ← c20453] at e_310_12
  have e_310_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_13
  simp only [← c20454, ← c20455, ← c20456] at e_310_13
  have e_311_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_6
  simp only [← c20445, k3, ← c20446, k3, ← c20447] at e_311_6
  have e_311_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_7
  simp only [← c20448, ← c20449, k3, ← c20450] at e_311_7
  have e_311_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_8
  simp only [← c20457, ← c20458, k3, ← c20459] at e_311_8
  refine ⟨?_, ?_⟩
  · have hc := e_310_12
    simp only [e_311_7] at hc
    linear_combination c20460.trans k2 - hc
  · have hc := e_311_8
    simp only [e_311_6, e_310_13, e_311_7] at hc
    linear_combination c20461.trans k2 - hc

theorem publicBatchWrapper86_f2760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 75) = bor (a (.wire 73 47)) (a (.virt 18795)) := by
  have c20462 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20463 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20464 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20465 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20466 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20467 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_18
  simp only [← c20465, ← c20466, k3, ← c20467] at e_301_18
  have e_306_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_10
  simp only [← c20462, ← c20463, ← c20464] at e_306_10
  have hr := e_301_18
  simp only [e_306_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 75) = a (.virt 201) := by
  have c20468 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20468

theorem publicBatchWrapper86_f2762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15004)) (a (.wire 165 63)) (a (.virt 18797)) (a (.virt 18798)) := by
  have c20469 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20470 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20471 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20472 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20473 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20474 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20475 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20476 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20477 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20478 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20479 := (publicBatchWrapper86_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20480 := (publicBatchWrapper86_copies640 a h).1
  have c20481 := (publicBatchWrapper86_copies640 a h).2.1
  have c20482 := (publicBatchWrapper86_copies640 a h).2.2.1
  have c20483 := (publicBatchWrapper86_copies640 a h).2.2.2.1
  have c20484 := (publicBatchWrapper86_copies640 a h).2.2.2.2.1
  have c20485 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_310_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_14
  simp only [← c20475, ← c20476, ← c20477] at e_310_14
  have e_310_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_15
  simp only [← c20478, ← c20479, ← c20480] at e_310_15
  have e_311_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_9
  simp only [← c20469, k3, ← c20470, k3, ← c20471] at e_311_9
  have e_311_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_10
  simp only [← c20472, ← c20473, k3, ← c20474] at e_311_10
  have e_311_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_11
  simp only [← c20481, ← c20482, k3, ← c20483] at e_311_11
  refine ⟨?_, ?_⟩
  · have hc := e_310_14
    simp only [e_311_10] at hc
    linear_combination c20484.trans k2 - hc
  · have hc := e_311_11
    simp only [e_311_9, e_310_15, e_311_10] at hc
    linear_combination c20485.trans k2 - hc

theorem publicBatchWrapper86_f2763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 79) = bor (a (.wire 73 47)) (a (.virt 18797)) := by
  have c20486 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.1
  have c20487 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.1
  have c20488 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.1
  have c20489 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.1
  have c20490 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20491 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_301_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_19
  simp only [← c20489, ← c20490, k3, ← c20491] at e_301_19
  have e_306_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_11
  simp only [← c20486, ← c20487, ← c20488] at e_306_11
  have hr := e_301_19
  simp only [e_306_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 301 79) = a (.virt 201) := by
  have c20492 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20492

theorem publicBatchWrapper86_f2765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15005)) (a (.wire 165 15)) (a (.virt 18799)) (a (.virt 18800)) := by
  have c20493 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20494 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20495 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20496 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20497 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20498 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20499 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20500 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20501 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20502 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20503 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20504 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20505 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20506 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20507 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20508 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20509 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_310_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_16
  simp only [← c20499, ← c20500, ← c20501] at e_310_16
  have e_310_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_17
  simp only [← c20502, ← c20503, ← c20504] at e_310_17
  have e_311_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_12
  simp only [← c20493, k3, ← c20494, k3, ← c20495] at e_311_12
  have e_311_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_13
  simp only [← c20496, ← c20497, k3, ← c20498] at e_311_13
  have e_311_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_14
  simp only [← c20505, ← c20506, k3, ← c20507] at e_311_14
  refine ⟨?_, ?_⟩
  · have hc := e_310_16
    simp only [e_311_13] at hc
    linear_combination c20508.trans k2 - hc
  · have hc := e_311_14
    simp only [e_311_12, e_310_17, e_311_13] at hc
    linear_combination c20509.trans k2 - hc

theorem publicBatchWrapper86_f2766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15006)) (a (.wire 165 23)) (a (.virt 18801)) (a (.virt 18802)) := by
  have c20510 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20511 := (publicBatchWrapper86_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20512 := (publicBatchWrapper86_copies641 a h).1
  have c20513 := (publicBatchWrapper86_copies641 a h).2.1
  have c20514 := (publicBatchWrapper86_copies641 a h).2.2.1
  have c20515 := (publicBatchWrapper86_copies641 a h).2.2.2.1
  have c20516 := (publicBatchWrapper86_copies641 a h).2.2.2.2.1
  have c20517 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.1
  have c20518 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.1
  have c20519 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.1
  have c20520 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.1
  have c20521 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.1
  have c20522 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20523 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20524 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20525 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20526 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_310_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_18
  simp only [← c20516, ← c20517, ← c20518] at e_310_18
  have e_310_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_19
  simp only [← c20519, ← c20520, ← c20521] at e_310_19
  have e_311_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_15
  simp only [← c20510, k3, ← c20511, k3, ← c20512] at e_311_15
  have e_311_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_16
  simp only [← c20513, ← c20514, k3, ← c20515] at e_311_16
  have e_311_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_17
  simp only [← c20522, ← c20523, k3, ← c20524] at e_311_17
  refine ⟨?_, ?_⟩
  · have hc := e_310_18
    simp only [e_311_16] at hc
    linear_combination c20525.trans k2 - hc
  · have hc := e_311_17
    simp only [e_311_15, e_310_19, e_311_16] at hc
    linear_combination c20526.trans k2 - hc

theorem publicBatchWrapper86_f2767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15007)) (a (.wire 165 31)) (a (.virt 18803)) (a (.virt 18804)) := by
  have c20527 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20528 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20529 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20530 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20531 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20532 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20533 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20534 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20535 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20536 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20537 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20538 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20539 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20540 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20541 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20542 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20543 := (publicBatchWrapper86_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_311_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_18
  simp only [← c20527, k3, ← c20528, k3, ← c20529] at e_311_18
  have e_311_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_19
  simp only [← c20530, ← c20531, k3, ← c20532] at e_311_19
  have e_312_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_0
  simp only [← c20533, ← c20534, ← c20535] at e_312_0
  have e_312_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_1
  simp only [← c20536, ← c20537, ← c20538] at e_312_1
  have e_313_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_0
  simp only [← c20539, ← c20540, k3, ← c20541] at e_313_0
  refine ⟨?_, ?_⟩
  · have hc := e_312_0
    simp only [e_311_19] at hc
    linear_combination c20542.trans k2 - hc
  · have hc := e_313_0
    simp only [e_311_18, e_312_1, e_311_19] at hc
    linear_combination c20543.trans k2 - hc

theorem publicBatchWrapper86_f2768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15008)) (a (.wire 165 39)) (a (.virt 18805)) (a (.virt 18806)) := by
  have c20544 := (publicBatchWrapper86_copies642 a h).1
  have c20545 := (publicBatchWrapper86_copies642 a h).2.1
  have c20546 := (publicBatchWrapper86_copies642 a h).2.2.1
  have c20547 := (publicBatchWrapper86_copies642 a h).2.2.2.1
  have c20548 := (publicBatchWrapper86_copies642 a h).2.2.2.2.1
  have c20549 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.1
  have c20550 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.1
  have c20551 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.1
  have c20552 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.1
  have c20553 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.1
  have c20554 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20555 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20556 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20557 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20558 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20559 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20560 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_312_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_2
  simp only [← c20550, ← c20551, ← c20552] at e_312_2
  have e_312_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_3
  simp only [← c20553, ← c20554, ← c20555] at e_312_3
  have e_313_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_1
  simp only [← c20544, k3, ← c20545, k3, ← c20546] at e_313_1
  have e_313_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_2
  simp only [← c20547, ← c20548, k3, ← c20549] at e_313_2
  have e_313_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_3
  simp only [← c20556, ← c20557, k3, ← c20558] at e_313_3
  refine ⟨?_, ?_⟩
  · have hc := e_312_2
    simp only [e_313_2] at hc
    linear_combination c20559.trans k2 - hc
  · have hc := e_313_3
    simp only [e_313_1, e_312_3, e_313_2] at hc
    linear_combination c20560.trans k2 - hc

theorem publicBatchWrapper86_f2769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 312 19) = band (a (.virt 18799)) (a (.virt 18801)) := by
  have c20561 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20562 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20563 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_312_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_4
  simp only [← c20561, ← c20562, ← c20563] at e_312_4
  have hr := e_312_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 312 23) = band (a (.virt 18803)) (a (.virt 18805)) := by
  have c20564 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20565 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20566 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_312_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_5
  simp only [← c20564, ← c20565, ← c20566] at e_312_5
  have hr := e_312_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 312 27) = band (a (.wire 312 19)) (a (.wire 312 23)) := by
  have c20567 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20568 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20569 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_312_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_6
  simp only [← c20567, ← c20568, ← c20569] at e_312_6
  have hr := e_312_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 3) = bor (a (.wire 73 47)) (a (.wire 312 27)) := by
  have c20570 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20571 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20572 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20573 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20574 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20575 := (publicBatchWrapper86_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_306_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_12
  simp only [← c20570, ← c20571, ← c20572] at e_306_12
  have e_314_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_0
  simp only [← c20573, ← c20574, k3, ← c20575] at e_314_0
  have hr := e_314_0
  simp only [e_306_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 3) = a (.virt 201) := by
  have c20576 := (publicBatchWrapper86_copies643 a h).1
  exact c20576

theorem publicBatchWrapper86_f2774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15203)) (a (.wire 165 55)) (a (.virt 18807)) (a (.virt 18808)) := by
  have c20577 := (publicBatchWrapper86_copies643 a h).2.1
  have c20578 := (publicBatchWrapper86_copies643 a h).2.2.1
  have c20579 := (publicBatchWrapper86_copies643 a h).2.2.2.1
  have c20580 := (publicBatchWrapper86_copies643 a h).2.2.2.2.1
  have c20581 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.1
  have c20582 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.1
  have c20583 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.1
  have c20584 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.1
  have c20585 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.1
  have c20586 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20587 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20588 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20589 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20590 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20591 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20592 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20593 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_312_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_7
  simp only [← c20583, ← c20584, ← c20585] at e_312_7
  have e_312_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_8
  simp only [← c20586, ← c20587, ← c20588] at e_312_8
  have e_313_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_4
  simp only [← c20577, k3, ← c20578, k3, ← c20579] at e_313_4
  have e_313_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_5
  simp only [← c20580, ← c20581, k3, ← c20582] at e_313_5
  have e_313_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_6
  simp only [← c20589, ← c20590, k3, ← c20591] at e_313_6
  refine ⟨?_, ?_⟩
  · have hc := e_312_7
    simp only [e_313_5] at hc
    linear_combination c20592.trans k2 - hc
  · have hc := e_313_6
    simp only [e_313_4, e_312_8, e_313_5] at hc
    linear_combination c20593.trans k2 - hc

theorem publicBatchWrapper86_f2775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 7) = bor (a (.wire 75 11)) (a (.virt 18807)) := by
  have c20594 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20595 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20596 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20597 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20598 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20599 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_306_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_13
  simp only [← c20594, ← c20595, ← c20596] at e_306_13
  have e_314_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_1
  simp only [← c20597, ← c20598, k3, ← c20599] at e_314_1
  have hr := e_314_1
  simp only [e_306_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 7) = a (.virt 201) := by
  have c20600 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20600

theorem publicBatchWrapper86_f2777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15204)) (a (.wire 165 63)) (a (.virt 18809)) (a (.virt 18810)) := by
  have c20601 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20602 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20603 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20604 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20605 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20606 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20607 := (publicBatchWrapper86_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20608 := (publicBatchWrapper86_copies644 a h).1
  have c20609 := (publicBatchWrapper86_copies644 a h).2.1
  have c20610 := (publicBatchWrapper86_copies644 a h).2.2.1
  have c20611 := (publicBatchWrapper86_copies644 a h).2.2.2.1
  have c20612 := (publicBatchWrapper86_copies644 a h).2.2.2.2.1
  have c20613 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.1
  have c20614 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.1
  have c20615 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.1
  have c20616 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.1
  have c20617 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_312_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_9
  simp only [← c20607, ← c20608, ← c20609] at e_312_9
  have e_312_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_10
  simp only [← c20610, ← c20611, ← c20612] at e_312_10
  have e_313_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_7
  simp only [← c20601, k3, ← c20602, k3, ← c20603] at e_313_7
  have e_313_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_8
  simp only [← c20604, ← c20605, k3, ← c20606] at e_313_8
  have e_313_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_9
  simp only [← c20613, ← c20614, k3, ← c20615] at e_313_9
  refine ⟨?_, ?_⟩
  · have hc := e_312_9
    simp only [e_313_8] at hc
    linear_combination c20616.trans k2 - hc
  · have hc := e_313_9
    simp only [e_313_7, e_312_10, e_313_8] at hc
    linear_combination c20617.trans k2 - hc

theorem publicBatchWrapper86_f2778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 11) = bor (a (.wire 75 11)) (a (.virt 18809)) := by
  have c20618 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20619 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20620 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20621 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20622 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20623 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_306_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_14
  simp only [← c20618, ← c20619, ← c20620] at e_306_14
  have e_314_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_2
  simp only [← c20621, ← c20622, k3, ← c20623] at e_314_2
  have hr := e_314_2
  simp only [e_306_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 11) = a (.virt 201) := by
  have c20624 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20624

theorem publicBatchWrapper86_f2780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15205)) (a (.wire 165 15)) (a (.virt 18811)) (a (.virt 18812)) := by
  have c20625 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20626 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20627 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20628 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20629 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20630 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20631 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20632 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20633 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20634 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20635 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20636 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20637 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20638 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20639 := (publicBatchWrapper86_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20640 := (publicBatchWrapper86_copies645 a h).1
  have c20641 := (publicBatchWrapper86_copies645 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_312_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_11
  simp only [← c20631, ← c20632, ← c20633] at e_312_11
  have e_312_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_12
  simp only [← c20634, ← c20635, ← c20636] at e_312_12
  have e_313_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_10
  simp only [← c20625, k3, ← c20626, k3, ← c20627] at e_313_10
  have e_313_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_11
  simp only [← c20628, ← c20629, k3, ← c20630] at e_313_11
  have e_313_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_12
  simp only [← c20637, ← c20638, k3, ← c20639] at e_313_12
  refine ⟨?_, ?_⟩
  · have hc := e_312_11
    simp only [e_313_11] at hc
    linear_combination c20640.trans k2 - hc
  · have hc := e_313_12
    simp only [e_313_10, e_312_12, e_313_11] at hc
    linear_combination c20641.trans k2 - hc

theorem publicBatchWrapper86_f2781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15206)) (a (.wire 165 23)) (a (.virt 18813)) (a (.virt 18814)) := by
  have c20642 := (publicBatchWrapper86_copies645 a h).2.2.1
  have c20643 := (publicBatchWrapper86_copies645 a h).2.2.2.1
  have c20644 := (publicBatchWrapper86_copies645 a h).2.2.2.2.1
  have c20645 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.1
  have c20646 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.1
  have c20647 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.1
  have c20648 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.1
  have c20649 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.1
  have c20650 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20651 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20652 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20653 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20654 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20655 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20656 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20657 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20658 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_312_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_13
  simp only [← c20648, ← c20649, ← c20650] at e_312_13
  have e_312_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_14
  simp only [← c20651, ← c20652, ← c20653] at e_312_14
  have e_313_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_13
  simp only [← c20642, k3, ← c20643, k3, ← c20644] at e_313_13
  have e_313_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_14
  simp only [← c20645, ← c20646, k3, ← c20647] at e_313_14
  have e_313_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_15
  simp only [← c20654, ← c20655, k3, ← c20656] at e_313_15
  refine ⟨?_, ?_⟩
  · have hc := e_312_13
    simp only [e_313_14] at hc
    linear_combination c20657.trans k2 - hc
  · have hc := e_313_15
    simp only [e_313_13, e_312_14, e_313_14] at hc
    linear_combination c20658.trans k2 - hc

theorem publicBatchWrapper86_f2782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15207)) (a (.wire 165 31)) (a (.virt 18815)) (a (.virt 18816)) := by
  have c20659 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20660 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20661 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20662 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20663 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20664 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20665 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20666 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20667 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20668 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20669 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20670 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20671 := (publicBatchWrapper86_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20672 := (publicBatchWrapper86_copies646 a h).1
  have c20673 := (publicBatchWrapper86_copies646 a h).2.1
  have c20674 := (publicBatchWrapper86_copies646 a h).2.2.1
  have c20675 := (publicBatchWrapper86_copies646 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_312_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_15
  simp only [← c20665, ← c20666, ← c20667] at e_312_15
  have e_312_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_16
  simp only [← c20668, ← c20669, ← c20670] at e_312_16
  have e_313_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_16
  simp only [← c20659, k3, ← c20660, k3, ← c20661] at e_313_16
  have e_313_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_17
  simp only [← c20662, ← c20663, k3, ← c20664] at e_313_17
  have e_313_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_18
  simp only [← c20671, ← c20672, k3, ← c20673] at e_313_18
  refine ⟨?_, ?_⟩
  · have hc := e_312_15
    simp only [e_313_17] at hc
    linear_combination c20674.trans k2 - hc
  · have hc := e_313_18
    simp only [e_313_16, e_312_16, e_313_17] at hc
    linear_combination c20675.trans k2 - hc

theorem publicBatchWrapper86_f2783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15208)) (a (.wire 165 39)) (a (.virt 18817)) (a (.virt 18818)) := by
  have c20676 := (publicBatchWrapper86_copies646 a h).2.2.2.2.1
  have c20677 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.1
  have c20678 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.1
  have c20679 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.1
  have c20680 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.1
  have c20681 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.1
  have c20682 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20683 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20684 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20685 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20686 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20687 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20688 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20689 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20690 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20691 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20692 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_312_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_17
  simp only [← c20682, ← c20683, ← c20684] at e_312_17
  have e_312_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_18
  simp only [← c20685, ← c20686, ← c20687] at e_312_18
  have e_313_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_19
  simp only [← c20676, k3, ← c20677, k3, ← c20678] at e_313_19
  have e_315_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_0
  simp only [← c20679, ← c20680, k3, ← c20681] at e_315_0
  have e_315_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_1
  simp only [← c20688, ← c20689, k3, ← c20690] at e_315_1
  refine ⟨?_, ?_⟩
  · have hc := e_312_17
    simp only [e_315_0] at hc
    linear_combination c20691.trans k2 - hc
  · have hc := e_315_1
    simp only [e_313_19, e_312_18, e_315_0] at hc
    linear_combination c20692.trans k2 - hc

theorem publicBatchWrapper86_f2784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 312 79) = band (a (.virt 18811)) (a (.virt 18813)) := by
  have c20693 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20694 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20695 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_312_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_19
  simp only [← c20693, ← c20694, ← c20695] at e_312_19
  have hr := e_312_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 316 3) = band (a (.virt 18815)) (a (.virt 18817)) := by
  have c20696 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20697 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20698 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_316_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_0
  simp only [← c20696, ← c20697, ← c20698] at e_316_0
  have hr := e_316_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 316 7) = band (a (.wire 312 79)) (a (.wire 316 3)) := by
  have c20699 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20700 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20701 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_316_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_1
  simp only [← c20699, ← c20700, ← c20701] at e_316_1
  have hr := e_316_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 15) = bor (a (.wire 75 11)) (a (.wire 316 7)) := by
  have c20702 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20703 := (publicBatchWrapper86_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20704 := (publicBatchWrapper86_copies647 a h).1
  have c20705 := (publicBatchWrapper86_copies647 a h).2.1
  have c20706 := (publicBatchWrapper86_copies647 a h).2.2.1
  have c20707 := (publicBatchWrapper86_copies647 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_306_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_15
  simp only [← c20702, ← c20703, ← c20704] at e_306_15
  have e_314_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_3
  simp only [← c20705, ← c20706, k3, ← c20707] at e_314_3
  have hr := e_314_3
  simp only [e_306_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 15) = a (.virt 201) := by
  have c20708 := (publicBatchWrapper86_copies647 a h).2.2.2.2.1
  exact c20708

theorem publicBatchWrapper86_f2789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15403)) (a (.wire 165 55)) (a (.virt 18819)) (a (.virt 18820)) := by
  have c20709 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.1
  have c20710 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.1
  have c20711 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.1
  have c20712 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.1
  have c20713 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.1
  have c20714 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20715 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20716 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20717 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20718 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20719 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20720 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20721 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20722 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20723 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20724 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20725 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_315_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_2
  simp only [← c20709, k3, ← c20710, k3, ← c20711] at e_315_2
  have e_315_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_3
  simp only [← c20712, ← c20713, k3, ← c20714] at e_315_3
  have e_315_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_4
  simp only [← c20721, ← c20722, k3, ← c20723] at e_315_4
  have e_316_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_2
  simp only [← c20715, ← c20716, ← c20717] at e_316_2
  have e_316_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_3
  simp only [← c20718, ← c20719, ← c20720] at e_316_3
  refine ⟨?_, ?_⟩
  · have hc := e_316_2
    simp only [e_315_3] at hc
    linear_combination c20724.trans k2 - hc
  · have hc := e_315_4
    simp only [e_315_2, e_316_3, e_315_3] at hc
    linear_combination c20725.trans k2 - hc

theorem publicBatchWrapper86_f2790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 19) = bor (a (.wire 75 55)) (a (.virt 18819)) := by
  have c20726 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20727 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20728 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20729 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20730 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20731 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_306_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_16
  simp only [← c20726, ← c20727, ← c20728] at e_306_16
  have e_314_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_4
  simp only [← c20729, ← c20730, k3, ← c20731] at e_314_4
  have hr := e_314_4
  simp only [e_306_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 19) = a (.virt 201) := by
  have c20732 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20732

theorem publicBatchWrapper86_f2792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15404)) (a (.wire 165 63)) (a (.virt 18821)) (a (.virt 18822)) := by
  have c20733 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20734 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20735 := (publicBatchWrapper86_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20736 := (publicBatchWrapper86_copies648 a h).1
  have c20737 := (publicBatchWrapper86_copies648 a h).2.1
  have c20738 := (publicBatchWrapper86_copies648 a h).2.2.1
  have c20739 := (publicBatchWrapper86_copies648 a h).2.2.2.1
  have c20740 := (publicBatchWrapper86_copies648 a h).2.2.2.2.1
  have c20741 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.1
  have c20742 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.1
  have c20743 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.1
  have c20744 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.1
  have c20745 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.1
  have c20746 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20747 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20748 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20749 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_315_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_5
  simp only [← c20733, k3, ← c20734, k3, ← c20735] at e_315_5
  have e_315_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_6
  simp only [← c20736, ← c20737, k3, ← c20738] at e_315_6
  have e_315_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_7
  simp only [← c20745, ← c20746, k3, ← c20747] at e_315_7
  have e_316_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_4
  simp only [← c20739, ← c20740, ← c20741] at e_316_4
  have e_316_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_5
  simp only [← c20742, ← c20743, ← c20744] at e_316_5
  refine ⟨?_, ?_⟩
  · have hc := e_316_4
    simp only [e_315_6] at hc
    linear_combination c20748.trans k2 - hc
  · have hc := e_315_7
    simp only [e_315_5, e_316_5, e_315_6] at hc
    linear_combination c20749.trans k2 - hc

theorem publicBatchWrapper86_f2793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 23) = bor (a (.wire 75 55)) (a (.virt 18821)) := by
  have c20750 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20751 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20752 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20753 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20754 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20755 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_306_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_17
  simp only [← c20750, ← c20751, ← c20752] at e_306_17
  have e_314_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_5
  simp only [← c20753, ← c20754, k3, ← c20755] at e_314_5
  have hr := e_314_5
  simp only [e_306_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 23) = a (.virt 201) := by
  have c20756 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20756

theorem publicBatchWrapper86_f2795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15405)) (a (.wire 165 15)) (a (.virt 18823)) (a (.virt 18824)) := by
  have c20757 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20758 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20759 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20760 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20761 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20762 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20763 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20764 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20765 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20766 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20767 := (publicBatchWrapper86_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20768 := (publicBatchWrapper86_copies649 a h).1
  have c20769 := (publicBatchWrapper86_copies649 a h).2.1
  have c20770 := (publicBatchWrapper86_copies649 a h).2.2.1
  have c20771 := (publicBatchWrapper86_copies649 a h).2.2.2.1
  have c20772 := (publicBatchWrapper86_copies649 a h).2.2.2.2.1
  have c20773 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_315_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_8
  simp only [← c20757, k3, ← c20758, k3, ← c20759] at e_315_8
  have e_315_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_9
  simp only [← c20760, ← c20761, k3, ← c20762] at e_315_9
  have e_315_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_10
  simp only [← c20769, ← c20770, k3, ← c20771] at e_315_10
  have e_316_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_6
  simp only [← c20763, ← c20764, ← c20765] at e_316_6
  have e_316_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_7
  simp only [← c20766, ← c20767, ← c20768] at e_316_7
  refine ⟨?_, ?_⟩
  · have hc := e_316_6
    simp only [e_315_9] at hc
    linear_combination c20772.trans k2 - hc
  · have hc := e_315_10
    simp only [e_315_8, e_316_7, e_315_9] at hc
    linear_combination c20773.trans k2 - hc

theorem publicBatchWrapper86_f2796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15406)) (a (.wire 165 23)) (a (.virt 18825)) (a (.virt 18826)) := by
  have c20774 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.1
  have c20775 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.1
  have c20776 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.1
  have c20777 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.1
  have c20778 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20779 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20780 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20781 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20782 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20783 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20784 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20785 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20786 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20787 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20788 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20789 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20790 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_315_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_11
  simp only [← c20774, k3, ← c20775, k3, ← c20776] at e_315_11
  have e_315_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_12
  simp only [← c20777, ← c20778, k3, ← c20779] at e_315_12
  have e_315_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_13
  simp only [← c20786, ← c20787, k3, ← c20788] at e_315_13
  have e_316_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_8
  simp only [← c20780, ← c20781, ← c20782] at e_316_8
  have e_316_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_9
  simp only [← c20783, ← c20784, ← c20785] at e_316_9
  refine ⟨?_, ?_⟩
  · have hc := e_316_8
    simp only [e_315_12] at hc
    linear_combination c20789.trans k2 - hc
  · have hc := e_315_13
    simp only [e_315_11, e_316_9, e_315_12] at hc
    linear_combination c20790.trans k2 - hc

theorem publicBatchWrapper86_f2797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15407)) (a (.wire 165 31)) (a (.virt 18827)) (a (.virt 18828)) := by
  have c20791 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20792 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20793 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20794 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20795 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20796 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20797 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20798 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20799 := (publicBatchWrapper86_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20800 := (publicBatchWrapper86_copies650 a h).1
  have c20801 := (publicBatchWrapper86_copies650 a h).2.1
  have c20802 := (publicBatchWrapper86_copies650 a h).2.2.1
  have c20803 := (publicBatchWrapper86_copies650 a h).2.2.2.1
  have c20804 := (publicBatchWrapper86_copies650 a h).2.2.2.2.1
  have c20805 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.1
  have c20806 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.1
  have c20807 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_315_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_14
  simp only [← c20791, k3, ← c20792, k3, ← c20793] at e_315_14
  have e_315_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_15
  simp only [← c20794, ← c20795, k3, ← c20796] at e_315_15
  have e_315_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_16
  simp only [← c20803, ← c20804, k3, ← c20805] at e_315_16
  have e_316_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_10
  simp only [← c20797, ← c20798, ← c20799] at e_316_10
  have e_316_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_11
  simp only [← c20800, ← c20801, ← c20802] at e_316_11
  refine ⟨?_, ?_⟩
  · have hc := e_316_10
    simp only [e_315_15] at hc
    linear_combination c20806.trans k2 - hc
  · have hc := e_315_16
    simp only [e_315_14, e_316_11, e_315_15] at hc
    linear_combination c20807.trans k2 - hc

theorem publicBatchWrapper86_f2798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15408)) (a (.wire 165 39)) (a (.virt 18829)) (a (.virt 18830)) := by
  have c20808 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.1
  have c20809 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.1
  have c20810 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20811 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20812 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20813 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20814 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20815 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20816 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20817 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20818 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20819 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20820 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20821 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20822 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20823 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20824 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_315_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_17
  simp only [← c20808, k3, ← c20809, k3, ← c20810] at e_315_17
  have e_315_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_18
  simp only [← c20811, ← c20812, k3, ← c20813] at e_315_18
  have e_315_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_19
  simp only [← c20820, ← c20821, k3, ← c20822] at e_315_19
  have e_316_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_12
  simp only [← c20814, ← c20815, ← c20816] at e_316_12
  have e_316_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_13
  simp only [← c20817, ← c20818, ← c20819] at e_316_13
  refine ⟨?_, ?_⟩
  · have hc := e_316_12
    simp only [e_315_18] at hc
    linear_combination c20823.trans k2 - hc
  · have hc := e_315_19
    simp only [e_315_17, e_316_13, e_315_18] at hc
    linear_combination c20824.trans k2 - hc

theorem publicBatchWrapper86_f2799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 316 59) = band (a (.virt 18823)) (a (.virt 18825)) := by
  have c20825 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20826 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20827 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_316_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_14
  simp only [← c20825, ← c20826, ← c20827] at e_316_14
  have hr := e_316_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 316 63) = band (a (.virt 18827)) (a (.virt 18829)) := by
  have c20828 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20829 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20830 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_316_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_15
  simp only [← c20828, ← c20829, ← c20830] at e_316_15
  have hr := e_316_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 316 67) = band (a (.wire 316 59)) (a (.wire 316 63)) := by
  have c20831 := (publicBatchWrapper86_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20832 := (publicBatchWrapper86_copies651 a h).1
  have c20833 := (publicBatchWrapper86_copies651 a h).2.1
  have e_316_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_16
  simp only [← c20831, ← c20832, ← c20833] at e_316_16
  have hr := e_316_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 27) = bor (a (.wire 75 55)) (a (.wire 316 67)) := by
  have c20834 := (publicBatchWrapper86_copies651 a h).2.2.1
  have c20835 := (publicBatchWrapper86_copies651 a h).2.2.2.1
  have c20836 := (publicBatchWrapper86_copies651 a h).2.2.2.2.1
  have c20837 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.1
  have c20838 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.1
  have c20839 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_306_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_18
  simp only [← c20834, ← c20835, ← c20836] at e_306_18
  have e_314_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_6
  simp only [← c20837, ← c20838, k3, ← c20839] at e_314_6
  have hr := e_314_6
  simp only [e_306_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 27) = a (.virt 201) := by
  have c20840 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.1
  exact c20840

theorem publicBatchWrapper86_f2804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15603)) (a (.wire 165 55)) (a (.virt 18831)) (a (.virt 18832)) := by
  have c20841 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.1
  have c20842 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20843 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20844 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20845 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20846 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20847 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20848 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20849 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20850 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20851 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20852 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20853 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20854 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20855 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20856 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20857 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_316_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_17
  simp only [← c20847, ← c20848, ← c20849] at e_316_17
  have e_316_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_18
  simp only [← c20850, ← c20851, ← c20852] at e_316_18
  have e_317_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_0
  simp only [← c20841, k3, ← c20842, k3, ← c20843] at e_317_0
  have e_317_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_1
  simp only [← c20844, ← c20845, k3, ← c20846] at e_317_1
  have e_317_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_2
  simp only [← c20853, ← c20854, k3, ← c20855] at e_317_2
  refine ⟨?_, ?_⟩
  · have hc := e_316_17
    simp only [e_317_1] at hc
    linear_combination c20856.trans k2 - hc
  · have hc := e_317_2
    simp only [e_317_0, e_316_18, e_317_1] at hc
    linear_combination c20857.trans k2 - hc

theorem publicBatchWrapper86_f2805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 31) = bor (a (.wire 77 19)) (a (.virt 18831)) := by
  have c20858 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20859 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20860 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20861 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20862 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20863 := (publicBatchWrapper86_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_306_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_19
  simp only [← c20858, ← c20859, ← c20860] at e_306_19
  have e_314_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_7
  simp only [← c20861, ← c20862, k3, ← c20863] at e_314_7
  have hr := e_314_7
  simp only [e_306_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 31) = a (.virt 201) := by
  have c20864 := (publicBatchWrapper86_copies652 a h).1
  exact c20864

theorem publicBatchWrapper86_f2807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15604)) (a (.wire 165 63)) (a (.virt 18833)) (a (.virt 18834)) := by
  have c20865 := (publicBatchWrapper86_copies652 a h).2.1
  have c20866 := (publicBatchWrapper86_copies652 a h).2.2.1
  have c20867 := (publicBatchWrapper86_copies652 a h).2.2.2.1
  have c20868 := (publicBatchWrapper86_copies652 a h).2.2.2.2.1
  have c20869 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.1
  have c20870 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.1
  have c20871 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.1
  have c20872 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.1
  have c20873 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.1
  have c20874 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20875 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20876 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20877 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20878 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20879 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20880 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20881 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_316_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_19
  simp only [← c20871, ← c20872, ← c20873] at e_316_19
  have e_317_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_3
  simp only [← c20865, k3, ← c20866, k3, ← c20867] at e_317_3
  have e_317_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_4
  simp only [← c20868, ← c20869, k3, ← c20870] at e_317_4
  have e_317_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_5
  simp only [← c20877, ← c20878, k3, ← c20879] at e_317_5
  have e_318_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_0
  simp only [← c20874, ← c20875, ← c20876] at e_318_0
  refine ⟨?_, ?_⟩
  · have hc := e_316_19
    simp only [e_317_4] at hc
    linear_combination c20880.trans k2 - hc
  · have hc := e_317_5
    simp only [e_317_3, e_318_0, e_317_4] at hc
    linear_combination c20881.trans k2 - hc

theorem publicBatchWrapper86_f2808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 35) = bor (a (.wire 77 19)) (a (.virt 18833)) := by
  have c20882 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20883 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20884 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20885 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20886 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20887 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_314_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_8
  simp only [← c20885, ← c20886, k3, ← c20887] at e_314_8
  have e_319_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_0
  simp only [← c20882, ← c20883, ← c20884] at e_319_0
  have hr := e_314_8
  simp only [e_319_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 35) = a (.virt 201) := by
  have c20888 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20888

theorem publicBatchWrapper86_f2810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15605)) (a (.wire 165 15)) (a (.virt 18835)) (a (.virt 18836)) := by
  have c20889 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20890 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20891 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20892 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20893 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20894 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20895 := (publicBatchWrapper86_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20896 := (publicBatchWrapper86_copies653 a h).1
  have c20897 := (publicBatchWrapper86_copies653 a h).2.1
  have c20898 := (publicBatchWrapper86_copies653 a h).2.2.1
  have c20899 := (publicBatchWrapper86_copies653 a h).2.2.2.1
  have c20900 := (publicBatchWrapper86_copies653 a h).2.2.2.2.1
  have c20901 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.1
  have c20902 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.1
  have c20903 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.1
  have c20904 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.1
  have c20905 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_317_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_6
  simp only [← c20889, k3, ← c20890, k3, ← c20891] at e_317_6
  have e_317_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_7
  simp only [← c20892, ← c20893, k3, ← c20894] at e_317_7
  have e_317_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_8
  simp only [← c20901, ← c20902, k3, ← c20903] at e_317_8
  have e_318_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_1
  simp only [← c20895, ← c20896, ← c20897] at e_318_1
  have e_318_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_2
  simp only [← c20898, ← c20899, ← c20900] at e_318_2
  refine ⟨?_, ?_⟩
  · have hc := e_318_1
    simp only [e_317_7] at hc
    linear_combination c20904.trans k2 - hc
  · have hc := e_317_8
    simp only [e_317_6, e_318_2, e_317_7] at hc
    linear_combination c20905.trans k2 - hc

theorem publicBatchWrapper86_f2811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15606)) (a (.wire 165 23)) (a (.virt 18837)) (a (.virt 18838)) := by
  have c20906 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20907 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20908 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20909 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20910 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20911 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20912 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20913 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20914 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20915 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20916 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20917 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20918 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20919 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20920 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20921 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20922 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_317_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_9
  simp only [← c20906, k3, ← c20907, k3, ← c20908] at e_317_9
  have e_317_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_10
  simp only [← c20909, ← c20910, k3, ← c20911] at e_317_10
  have e_317_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_11
  simp only [← c20918, ← c20919, k3, ← c20920] at e_317_11
  have e_318_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_3
  simp only [← c20912, ← c20913, ← c20914] at e_318_3
  have e_318_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_4
  simp only [← c20915, ← c20916, ← c20917] at e_318_4
  refine ⟨?_, ?_⟩
  · have hc := e_318_3
    simp only [e_317_10] at hc
    linear_combination c20921.trans k2 - hc
  · have hc := e_317_11
    simp only [e_317_9, e_318_4, e_317_10] at hc
    linear_combination c20922.trans k2 - hc

theorem publicBatchWrapper86_f2812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15607)) (a (.wire 165 31)) (a (.virt 18839)) (a (.virt 18840)) := by
  have c20923 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20924 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20925 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20926 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20927 := (publicBatchWrapper86_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20928 := (publicBatchWrapper86_copies654 a h).1
  have c20929 := (publicBatchWrapper86_copies654 a h).2.1
  have c20930 := (publicBatchWrapper86_copies654 a h).2.2.1
  have c20931 := (publicBatchWrapper86_copies654 a h).2.2.2.1
  have c20932 := (publicBatchWrapper86_copies654 a h).2.2.2.2.1
  have c20933 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.1
  have c20934 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.1
  have c20935 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.1
  have c20936 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.1
  have c20937 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.1
  have c20938 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20939 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_317_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_12
  simp only [← c20923, k3, ← c20924, k3, ← c20925] at e_317_12
  have e_317_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_13
  simp only [← c20926, ← c20927, k3, ← c20928] at e_317_13
  have e_317_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_14
  simp only [← c20935, ← c20936, k3, ← c20937] at e_317_14
  have e_318_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_5
  simp only [← c20929, ← c20930, ← c20931] at e_318_5
  have e_318_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_6
  simp only [← c20932, ← c20933, ← c20934] at e_318_6
  refine ⟨?_, ?_⟩
  · have hc := e_318_5
    simp only [e_317_13] at hc
    linear_combination c20938.trans k2 - hc
  · have hc := e_317_14
    simp only [e_317_12, e_318_6, e_317_13] at hc
    linear_combination c20939.trans k2 - hc

theorem publicBatchWrapper86_f2813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15608)) (a (.wire 165 39)) (a (.virt 18841)) (a (.virt 18842)) := by
  have c20940 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20941 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20942 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20943 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20944 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20945 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20946 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20947 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20948 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20949 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20950 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20951 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20952 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20953 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20954 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20955 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20956 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_317_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_15
  simp only [← c20940, k3, ← c20941, k3, ← c20942] at e_317_15
  have e_317_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_16
  simp only [← c20943, ← c20944, k3, ← c20945] at e_317_16
  have e_317_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_17
  simp only [← c20952, ← c20953, k3, ← c20954] at e_317_17
  have e_318_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_7
  simp only [← c20946, ← c20947, ← c20948] at e_318_7
  have e_318_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_8
  simp only [← c20949, ← c20950, ← c20951] at e_318_8
  refine ⟨?_, ?_⟩
  · have hc := e_318_7
    simp only [e_317_16] at hc
    linear_combination c20955.trans k2 - hc
  · have hc := e_317_17
    simp only [e_317_15, e_318_8, e_317_16] at hc
    linear_combination c20956.trans k2 - hc

theorem publicBatchWrapper86_f2814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 318 39) = band (a (.virt 18835)) (a (.virt 18837)) := by
  have c20957 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20958 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20959 := (publicBatchWrapper86_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_318_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_9
  simp only [← c20957, ← c20958, ← c20959] at e_318_9
  have hr := e_318_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 318 43) = band (a (.virt 18839)) (a (.virt 18841)) := by
  have c20960 := (publicBatchWrapper86_copies655 a h).1
  have c20961 := (publicBatchWrapper86_copies655 a h).2.1
  have c20962 := (publicBatchWrapper86_copies655 a h).2.2.1
  have e_318_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_10
  simp only [← c20960, ← c20961, ← c20962] at e_318_10
  have hr := e_318_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 318 47) = band (a (.wire 318 39)) (a (.wire 318 43)) := by
  have c20963 := (publicBatchWrapper86_copies655 a h).2.2.2.1
  have c20964 := (publicBatchWrapper86_copies655 a h).2.2.2.2.1
  have c20965 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.1
  have e_318_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_11
  simp only [← c20963, ← c20964, ← c20965] at e_318_11
  have hr := e_318_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 39) = bor (a (.wire 77 19)) (a (.wire 318 47)) := by
  have c20966 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.1
  have c20967 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.1
  have c20968 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.1
  have c20969 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.1
  have c20970 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20971 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_314_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_9
  simp only [← c20969, ← c20970, k3, ← c20971] at e_314_9
  have e_319_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_1
  simp only [← c20966, ← c20967, ← c20968] at e_319_1
  have hr := e_314_9
  simp only [e_319_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 39) = a (.virt 201) := by
  have c20972 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20972

theorem publicBatchWrapper86_f2819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15803)) (a (.wire 165 55)) (a (.virt 18843)) (a (.virt 18844)) := by
  have c20973 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20974 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20975 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20976 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20977 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20978 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20979 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20980 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20981 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20982 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20983 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20984 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20985 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20986 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20987 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20988 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20989 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_317_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_18
  simp only [← c20973, k3, ← c20974, k3, ← c20975] at e_317_18
  have e_317_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_19
  simp only [← c20976, ← c20977, k3, ← c20978] at e_317_19
  have e_318_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_12
  simp only [← c20979, ← c20980, ← c20981] at e_318_12
  have e_318_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_13
  simp only [← c20982, ← c20983, ← c20984] at e_318_13
  have e_320_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_320_0
  simp only [← c20985, ← c20986, k3, ← c20987] at e_320_0
  refine ⟨?_, ?_⟩
  · have hc := e_318_12
    simp only [e_317_19] at hc
    linear_combination c20988.trans k2 - hc
  · have hc := e_320_0
    simp only [e_317_18, e_318_13, e_317_19] at hc
    linear_combination c20989.trans k2 - hc

theorem publicBatchWrapper86_f2820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 43) = bor (a (.wire 77 63)) (a (.virt 18843)) := by
  have c20990 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20991 := (publicBatchWrapper86_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20992 := (publicBatchWrapper86_copies656 a h).1
  have c20993 := (publicBatchWrapper86_copies656 a h).2.1
  have c20994 := (publicBatchWrapper86_copies656 a h).2.2.1
  have c20995 := (publicBatchWrapper86_copies656 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_314_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_10
  simp only [← c20993, ← c20994, k3, ← c20995] at e_314_10
  have e_319_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_2
  simp only [← c20990, ← c20991, ← c20992] at e_319_2
  have hr := e_314_10
  simp only [e_319_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 43) = a (.virt 201) := by
  have c20996 := (publicBatchWrapper86_copies656 a h).2.2.2.2.1
  exact c20996

theorem publicBatchWrapper86_f2822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15804)) (a (.wire 165 63)) (a (.virt 18845)) (a (.virt 18846)) := by
  have c20997 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.1
  have c20998 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.1
  have c20999 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.1
  have c21000 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.1
  have c21001 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.1
  have c21002 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21003 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21004 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21005 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21006 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21007 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21008 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21009 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21010 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21011 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21012 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21013 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_318_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_14
  simp only [← c21003, ← c21004, ← c21005] at e_318_14
  have e_318_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_15
  simp only [← c21006, ← c21007, ← c21008] at e_318_15
  have e_320_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_320_1
  simp only [← c20997, k3, ← c20998, k3, ← c20999] at e_320_1
  have e_320_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_320_2
  simp only [← c21000, ← c21001, k3, ← c21002] at e_320_2
  have e_320_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_320_3
  simp only [← c21009, ← c21010, k3, ← c21011] at e_320_3
  refine ⟨?_, ?_⟩
  · have hc := e_318_14
    simp only [e_320_2] at hc
    linear_combination c21012.trans k2 - hc
  · have hc := e_320_3
    simp only [e_320_1, e_318_15, e_320_2] at hc
    linear_combination c21013.trans k2 - hc

theorem publicBatchWrapper86_f2823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 47) = bor (a (.wire 77 63)) (a (.virt 18845)) := by
  have c21014 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21015 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21016 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21017 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21018 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21019 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_314_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_11
  simp only [← c21017, ← c21018, k3, ← c21019] at e_314_11
  have e_319_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_3
  simp only [← c21014, ← c21015, ← c21016] at e_319_3
  have hr := e_314_11
  simp only [e_319_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 314 47) = a (.virt 201) := by
  have c21020 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21020

theorem publicBatchWrapper86_f2825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15805)) (a (.wire 165 15)) (a (.virt 18847)) (a (.virt 18848)) := by
  have c21021 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21022 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21023 := (publicBatchWrapper86_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21024 := (publicBatchWrapper86_copies657 a h).1
  have c21025 := (publicBatchWrapper86_copies657 a h).2.1
  have c21026 := (publicBatchWrapper86_copies657 a h).2.2.1
  have c21027 := (publicBatchWrapper86_copies657 a h).2.2.2.1
  have c21028 := (publicBatchWrapper86_copies657 a h).2.2.2.2.1
  have c21029 := (publicBatchWrapper86_copies657 a h).2.2.2.2.2.1
  have c21030 := (publicBatchWrapper86_copies657 a h).2.2.2.2.2.2.1
  have c21031 := (publicBatchWrapper86_copies657 a h).2.2.2.2.2.2.2.1
  have c21032 := (publicBatchWrapper86_copies657 a h).2.2.2.2.2.2.2.2.1
  have c21033 := (publicBatchWrapper86_copies657 a h).2.2.2.2.2.2.2.2.2.1
  have c21034 := (publicBatchWrapper86_copies657 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21035 := (publicBatchWrapper86_copies657 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21036 := (publicBatchWrapper86_copies657 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21037 := (publicBatchWrapper86_copies657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_318_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_16
  simp only [← c21027, ← c21028, ← c21029] at e_318_16
  have e_318_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_17
  simp only [← c21030, ← c21031, ← c21032] at e_318_17
  have e_320_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_320_4
  simp only [← c21021, k3, ← c21022, k3, ← c21023] at e_320_4
  have e_320_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_320_5
  simp only [← c21024, ← c21025, k3, ← c21026] at e_320_5
  have e_320_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row320 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_320_6
  simp only [← c21033, ← c21034, k3, ← c21035] at e_320_6
  refine ⟨?_, ?_⟩
  · have hc := e_318_16
    simp only [e_320_5] at hc
    linear_combination c21036.trans k2 - hc
  · have hc := e_320_6
    simp only [e_320_4, e_318_17, e_320_5] at hc
    linear_combination c21037.trans k2 - hc

end Plonky2Spec.Generated
