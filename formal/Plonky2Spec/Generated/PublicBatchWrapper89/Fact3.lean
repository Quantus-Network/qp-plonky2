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

theorem publicBatchWrapper89_f2228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 240 43) = band (a (.wire 240 35)) (a (.wire 240 39)) := by
  have c15656 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.1
  have c15657 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.1
  have c15658 := (publicBatchWrapper89_copies489 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_240_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row240 (by decide)
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
  have e_231_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_15
  simp only [← c15659, ← c15660, ← c15661] at e_231_15
  have e_236_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row236 (by decide)
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
  have e_239_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_16
  simp only [← c15666, k3, ← c15667, k3, ← c15668] at e_239_16
  have e_239_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_17
  simp only [← c15669, ← c15670, k3, ← c15671] at e_239_17
  have e_239_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_18
  simp only [← c15678, ← c15679, k3, ← c15680] at e_239_18
  have e_240_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_11
  simp only [← c15672, ← c15673, ← c15674] at e_240_11
  have e_240_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row240 (by decide)
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
  have e_231_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_16
  simp only [← c15683, ← c15684, ← c15685] at e_231_16
  have e_236_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row236 (by decide)
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
  have e_239_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_239_19
  simp only [← c15690, k3, ← c15691, k3, ← c15692] at e_239_19
  have e_240_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_13
  simp only [← c15696, ← c15697, ← c15698] at e_240_13
  have e_240_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_14
  simp only [← c15699, ← c15700, ← c15701] at e_240_14
  have e_241_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_0
  simp only [← c15693, ← c15694, k3, ← c15695] at e_241_0
  have e_241_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row241 (by decide)
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
  have e_231_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_17
  simp only [← c15707, ← c15708, ← c15709] at e_231_17
  have e_236_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row236 (by decide)
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
  have e_240_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_15
  simp only [← c15720, ← c15721, ← c15722] at e_240_15
  have e_240_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_16
  simp only [← c15723, ← c15724, ← c15725] at e_240_16
  have e_241_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_2
  simp only [← c15714, k3, ← c15715, k3, ← c15716] at e_241_2
  have e_241_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_3
  simp only [← c15717, ← c15718, k3, ← c15719] at e_241_3
  have e_241_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row241 (by decide)
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
  have e_240_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_17
  simp only [← c15737, ← c15738, ← c15739] at e_240_17
  have e_240_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_18
  simp only [← c15740, ← c15741, ← c15742] at e_240_18
  have e_241_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_5
  simp only [← c15731, k3, ← c15732, k3, ← c15733] at e_241_5
  have e_241_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_6
  simp only [← c15734, ← c15735, k3, ← c15736] at e_241_6
  have e_241_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row241 (by decide)
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
  have e_240_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_240_19
  simp only [← c15754, ← c15755, ← c15756] at e_240_19
  have e_241_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_8
  simp only [← c15748, k3, ← c15749, k3, ← c15750] at e_241_8
  have e_241_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_9
  simp only [← c15751, ← c15752, k3, ← c15753] at e_241_9
  have e_241_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_10
  simp only [← c15760, ← c15761, k3, ← c15762] at e_241_10
  have e_242_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row242 (by decide)
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
  have e_241_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_11
  simp only [← c15765, k3, ← c15766, k3, ← c15767] at e_241_11
  have e_241_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_12
  simp only [← c15768, ← c15769, k3, ← c15770] at e_241_12
  have e_241_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_13
  simp only [← c15777, ← c15778, k3, ← c15779] at e_241_13
  have e_242_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_1
  simp only [← c15771, ← c15772, ← c15773] at e_242_1
  have e_242_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row242 (by decide)
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
  have e_242_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row242 (by decide)
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
  have e_242_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row242 (by decide)
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
  have e_242_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row242 (by decide)
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
  have e_231_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_18
  simp only [← c15791, ← c15792, ← c15793] at e_231_18
  have e_236_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row236 (by decide)
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
  have e_241_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_14
  simp only [← c15798, k3, ← c15799, k3, ← c15800] at e_241_14
  have e_241_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_15
  simp only [← c15801, ← c15802, k3, ← c15803] at e_241_15
  have e_241_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_16
  simp only [← c15810, ← c15811, k3, ← c15812] at e_241_16
  have e_242_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_6
  simp only [← c15804, ← c15805, ← c15806] at e_242_6
  have e_242_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row242 (by decide)
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
  have e_231_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_231_19
  simp only [← c15815, ← c15816, ← c15817] at e_231_19
  have e_236_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row236 (by decide)
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
  have e_241_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_17
  simp only [← c15822, k3, ← c15823, k3, ← c15824] at e_241_17
  have e_241_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_18
  simp only [← c15825, ← c15826, k3, ← c15827] at e_241_18
  have e_241_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_241_19
  simp only [← c15834, ← c15835, k3, ← c15836] at e_241_19
  have e_242_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_8
  simp only [← c15828, ← c15829, ← c15830] at e_242_8
  have e_242_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row242 (by decide)
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
  have e_236_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_12
  simp only [← c15842, ← c15843, k3, ← c15844] at e_236_12
  have e_243_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row243 (by decide)
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
  have e_242_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_10
  simp only [← c15852, ← c15853, ← c15854] at e_242_10
  have e_242_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_11
  simp only [← c15855, ← c15856, ← c15857] at e_242_11
  have e_244_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_0
  simp only [← c15846, k3, ← c15847, k3, ← c15848] at e_244_0
  have e_244_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_1
  simp only [← c15849, ← c15850, k3, ← c15851] at e_244_1
  have e_244_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row244 (by decide)
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
  have e_242_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_12
  simp only [← c15869, ← c15870, ← c15871] at e_242_12
  have e_242_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_13
  simp only [← c15872, ← c15873, ← c15874] at e_242_13
  have e_244_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_3
  simp only [← c15863, k3, ← c15864, k3, ← c15865] at e_244_3
  have e_244_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_4
  simp only [← c15866, ← c15867, k3, ← c15868] at e_244_4
  have e_244_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row244 (by decide)
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
  have e_242_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_14
  simp only [← c15886, ← c15887, ← c15888] at e_242_14
  have e_242_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_15
  simp only [← c15889, ← c15890, ← c15891] at e_242_15
  have e_244_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_6
  simp only [← c15880, k3, ← c15881, k3, ← c15882] at e_244_6
  have e_244_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_7
  simp only [← c15883, ← c15884, k3, ← c15885] at e_244_7
  have e_244_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row244 (by decide)
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
  have e_242_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_16
  simp only [← c15903, ← c15904, ← c15905] at e_242_16
  have e_242_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_242_17
  simp only [← c15906, ← c15907, ← c15908] at e_242_17
  have e_244_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_9
  simp only [← c15897, k3, ← c15898, k3, ← c15899] at e_244_9
  have e_244_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_10
  simp only [← c15900, ← c15901, k3, ← c15902] at e_244_10
  have e_244_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row244 (by decide)
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
  have e_242_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row242 (by decide)
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
  have e_242_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row242 (by decide)
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
  have e_245_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row245 (by decide)
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
  have e_236_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_13
  simp only [← c15926, ← c15927, k3, ← c15928] at e_236_13
  have e_243_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row243 (by decide)
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

theorem publicBatchWrapper89_f2261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7603)) (a (.wire 170 23)) (a (.virt 18975)) (a (.virt 18976)) := by
  have c15930 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15931 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15932 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15933 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15934 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15935 := (publicBatchWrapper89_copies497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15936 := (publicBatchWrapper89_copies498 a h).1
  have c15937 := (publicBatchWrapper89_copies498 a h).2.1
  have c15938 := (publicBatchWrapper89_copies498 a h).2.2.1
  have c15939 := (publicBatchWrapper89_copies498 a h).2.2.2.1
  have c15940 := (publicBatchWrapper89_copies498 a h).2.2.2.2.1
  have c15941 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.1
  have c15942 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.1
  have c15943 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.1
  have c15944 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.1
  have c15945 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.1
  have c15946 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_244_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_12
  simp only [← c15930, k3, ← c15931, k3, ← c15932] at e_244_12
  have e_244_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_13
  simp only [← c15933, ← c15934, k3, ← c15935] at e_244_13
  have e_244_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_14
  simp only [← c15942, ← c15943, k3, ← c15944] at e_244_14
  have e_245_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_1
  simp only [← c15936, ← c15937, ← c15938] at e_245_1
  have e_245_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_2
  simp only [← c15939, ← c15940, ← c15941] at e_245_2
  refine ⟨?_, ?_⟩
  · have hc := e_245_1
    simp only [e_244_13] at hc
    linear_combination c15945.trans k2 - hc
  · have hc := e_244_14
    simp only [e_244_12, e_245_2, e_244_13] at hc
    linear_combination c15946.trans k2 - hc

theorem publicBatchWrapper89_f2262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 59) = bor (a (.wire 38 51)) (a (.virt 18975)) := by
  have c15947 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15948 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15949 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15950 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15951 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15952 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_236_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_14
  simp only [← c15950, ← c15951, k3, ← c15952] at e_236_14
  have e_243_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_2
  simp only [← c15947, ← c15948, ← c15949] at e_243_2
  have hr := e_236_14
  simp only [e_243_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 59) = a (.virt 201) := by
  have c15953 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15953

theorem publicBatchWrapper89_f2264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7604)) (a (.wire 170 31)) (a (.virt 18977)) (a (.virt 18978)) := by
  have c15954 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15955 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15956 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15957 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15958 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15959 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15960 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15961 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15962 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15963 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15964 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15965 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15966 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15967 := (publicBatchWrapper89_copies498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c15968 := (publicBatchWrapper89_copies499 a h).1
  have c15969 := (publicBatchWrapper89_copies499 a h).2.1
  have c15970 := (publicBatchWrapper89_copies499 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_244_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_15
  simp only [← c15954, k3, ← c15955, k3, ← c15956] at e_244_15
  have e_244_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_16
  simp only [← c15957, ← c15958, k3, ← c15959] at e_244_16
  have e_244_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_17
  simp only [← c15966, ← c15967, k3, ← c15968] at e_244_17
  have e_245_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_3
  simp only [← c15960, ← c15961, ← c15962] at e_245_3
  have e_245_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_4
  simp only [← c15963, ← c15964, ← c15965] at e_245_4
  refine ⟨?_, ?_⟩
  · have hc := e_245_3
    simp only [e_244_16] at hc
    linear_combination c15969.trans k2 - hc
  · have hc := e_244_17
    simp only [e_244_15, e_245_4, e_244_16] at hc
    linear_combination c15970.trans k2 - hc

theorem publicBatchWrapper89_f2265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 63) = bor (a (.wire 38 51)) (a (.virt 18977)) := by
  have c15971 := (publicBatchWrapper89_copies499 a h).2.2.2.1
  have c15972 := (publicBatchWrapper89_copies499 a h).2.2.2.2.1
  have c15973 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.1
  have c15974 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.1
  have c15975 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.1
  have c15976 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_236_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_15
  simp only [← c15974, ← c15975, k3, ← c15976] at e_236_15
  have e_243_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_3
  simp only [← c15971, ← c15972, ← c15973] at e_243_3
  have hr := e_236_15
  simp only [e_243_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 63) = a (.virt 201) := by
  have c15977 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.1
  exact c15977

theorem publicBatchWrapper89_f2267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7605)) (a (.wire 169 63)) (a (.virt 18979)) (a (.virt 18980)) := by
  have c15978 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.1
  have c15979 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c15980 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15981 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15982 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15983 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15984 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15985 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15986 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15987 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15988 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15989 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15990 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15991 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15992 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15993 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15994 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_244_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_18
  simp only [← c15978, k3, ← c15979, k3, ← c15980] at e_244_18
  have e_244_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_244_19
  simp only [← c15981, ← c15982, k3, ← c15983] at e_244_19
  have e_245_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_5
  simp only [← c15984, ← c15985, ← c15986] at e_245_5
  have e_245_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_6
  simp only [← c15987, ← c15988, ← c15989] at e_245_6
  have e_246_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_0
  simp only [← c15990, ← c15991, k3, ← c15992] at e_246_0
  refine ⟨?_, ?_⟩
  · have hc := e_245_5
    simp only [e_244_19] at hc
    linear_combination c15993.trans k2 - hc
  · have hc := e_246_0
    simp only [e_244_18, e_245_6, e_244_19] at hc
    linear_combination c15994.trans k2 - hc

theorem publicBatchWrapper89_f2268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7606)) (a (.wire 169 71)) (a (.virt 18981)) (a (.virt 18982)) := by
  have c15995 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15996 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15997 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15998 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15999 := (publicBatchWrapper89_copies499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16000 := (publicBatchWrapper89_copies500 a h).1
  have c16001 := (publicBatchWrapper89_copies500 a h).2.1
  have c16002 := (publicBatchWrapper89_copies500 a h).2.2.1
  have c16003 := (publicBatchWrapper89_copies500 a h).2.2.2.1
  have c16004 := (publicBatchWrapper89_copies500 a h).2.2.2.2.1
  have c16005 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.1
  have c16006 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.1
  have c16007 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.1
  have c16008 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.1
  have c16009 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.1
  have c16010 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16011 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_245_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_7
  simp only [← c16001, ← c16002, ← c16003] at e_245_7
  have e_245_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_8
  simp only [← c16004, ← c16005, ← c16006] at e_245_8
  have e_246_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_1
  simp only [← c15995, k3, ← c15996, k3, ← c15997] at e_246_1
  have e_246_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_2
  simp only [← c15998, ← c15999, k3, ← c16000] at e_246_2
  have e_246_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_3
  simp only [← c16007, ← c16008, k3, ← c16009] at e_246_3
  refine ⟨?_, ?_⟩
  · have hc := e_245_7
    simp only [e_246_2] at hc
    linear_combination c16010.trans k2 - hc
  · have hc := e_246_3
    simp only [e_246_1, e_245_8, e_246_2] at hc
    linear_combination c16011.trans k2 - hc

theorem publicBatchWrapper89_f2269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7607)) (a (.wire 169 79)) (a (.virt 18983)) (a (.virt 18984)) := by
  have c16012 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16013 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16014 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16015 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16016 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16017 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16018 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16019 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16020 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16021 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16022 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16023 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16024 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16025 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16026 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16027 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16028 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_245_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_9
  simp only [← c16018, ← c16019, ← c16020] at e_245_9
  have e_245_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_10
  simp only [← c16021, ← c16022, ← c16023] at e_245_10
  have e_246_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_4
  simp only [← c16012, k3, ← c16013, k3, ← c16014] at e_246_4
  have e_246_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_5
  simp only [← c16015, ← c16016, k3, ← c16017] at e_246_5
  have e_246_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_6
  simp only [← c16024, ← c16025, k3, ← c16026] at e_246_6
  refine ⟨?_, ?_⟩
  · have hc := e_245_9
    simp only [e_246_5] at hc
    linear_combination c16027.trans k2 - hc
  · have hc := e_246_6
    simp only [e_246_4, e_245_10, e_246_5] at hc
    linear_combination c16028.trans k2 - hc

theorem publicBatchWrapper89_f2270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7608)) (a (.wire 170 7)) (a (.virt 18985)) (a (.virt 18986)) := by
  have c16029 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16030 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16031 := (publicBatchWrapper89_copies500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16032 := (publicBatchWrapper89_copies501 a h).1
  have c16033 := (publicBatchWrapper89_copies501 a h).2.1
  have c16034 := (publicBatchWrapper89_copies501 a h).2.2.1
  have c16035 := (publicBatchWrapper89_copies501 a h).2.2.2.1
  have c16036 := (publicBatchWrapper89_copies501 a h).2.2.2.2.1
  have c16037 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.1
  have c16038 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.1
  have c16039 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.1
  have c16040 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.1
  have c16041 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.1
  have c16042 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16043 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16044 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16045 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_245_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_11
  simp only [← c16035, ← c16036, ← c16037] at e_245_11
  have e_245_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_12
  simp only [← c16038, ← c16039, ← c16040] at e_245_12
  have e_246_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_7
  simp only [← c16029, k3, ← c16030, k3, ← c16031] at e_246_7
  have e_246_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_8
  simp only [← c16032, ← c16033, k3, ← c16034] at e_246_8
  have e_246_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_9
  simp only [← c16041, ← c16042, k3, ← c16043] at e_246_9
  refine ⟨?_, ?_⟩
  · have hc := e_245_11
    simp only [e_246_8] at hc
    linear_combination c16044.trans k2 - hc
  · have hc := e_246_9
    simp only [e_246_7, e_245_12, e_246_8] at hc
    linear_combination c16045.trans k2 - hc

theorem publicBatchWrapper89_f2271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 245 55) = band (a (.virt 18979)) (a (.virt 18981)) := by
  have c16046 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16047 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16048 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_245_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_13
  simp only [← c16046, ← c16047, ← c16048] at e_245_13
  have hr := e_245_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 245 59) = band (a (.virt 18983)) (a (.virt 18985)) := by
  have c16049 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16050 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16051 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_245_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_14
  simp only [← c16049, ← c16050, ← c16051] at e_245_14
  have hr := e_245_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 245 63) = band (a (.wire 245 55)) (a (.wire 245 59)) := by
  have c16052 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16053 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16054 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_245_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_15
  simp only [← c16052, ← c16053, ← c16054] at e_245_15
  have hr := e_245_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 67) = bor (a (.wire 38 51)) (a (.wire 245 63)) := by
  have c16055 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16056 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16057 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16058 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16059 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16060 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_236_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_16
  simp only [← c16058, ← c16059, k3, ← c16060] at e_236_16
  have e_243_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_4
  simp only [← c16055, ← c16056, ← c16057] at e_243_4
  have hr := e_236_16
  simp only [e_243_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 67) = a (.virt 201) := by
  have c16061 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16061

theorem publicBatchWrapper89_f2276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7803)) (a (.wire 170 23)) (a (.virt 18987)) (a (.virt 18988)) := by
  have c16062 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16063 := (publicBatchWrapper89_copies501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16064 := (publicBatchWrapper89_copies502 a h).1
  have c16065 := (publicBatchWrapper89_copies502 a h).2.1
  have c16066 := (publicBatchWrapper89_copies502 a h).2.2.1
  have c16067 := (publicBatchWrapper89_copies502 a h).2.2.2.1
  have c16068 := (publicBatchWrapper89_copies502 a h).2.2.2.2.1
  have c16069 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.1
  have c16070 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.1
  have c16071 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.1
  have c16072 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.1
  have c16073 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.1
  have c16074 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16075 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16076 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16077 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16078 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_245_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_16
  simp only [← c16068, ← c16069, ← c16070] at e_245_16
  have e_245_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_17
  simp only [← c16071, ← c16072, ← c16073] at e_245_17
  have e_246_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_10
  simp only [← c16062, k3, ← c16063, k3, ← c16064] at e_246_10
  have e_246_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_11
  simp only [← c16065, ← c16066, k3, ← c16067] at e_246_11
  have e_246_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_12
  simp only [← c16074, ← c16075, k3, ← c16076] at e_246_12
  refine ⟨?_, ?_⟩
  · have hc := e_245_16
    simp only [e_246_11] at hc
    linear_combination c16077.trans k2 - hc
  · have hc := e_246_12
    simp only [e_246_10, e_245_17, e_246_11] at hc
    linear_combination c16078.trans k2 - hc

theorem publicBatchWrapper89_f2277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 71) = bor (a (.wire 39 15)) (a (.virt 18987)) := by
  have c16079 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16080 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16081 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16082 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16083 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16084 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_236_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_17
  simp only [← c16082, ← c16083, k3, ← c16084] at e_236_17
  have e_243_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_5
  simp only [← c16079, ← c16080, ← c16081] at e_243_5
  have hr := e_236_17
  simp only [e_243_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 71) = a (.virt 201) := by
  have c16085 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16085

theorem publicBatchWrapper89_f2279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7804)) (a (.wire 170 31)) (a (.virt 18989)) (a (.virt 18990)) := by
  have c16086 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16087 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16088 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16089 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16090 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16091 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16092 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16093 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16094 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16095 := (publicBatchWrapper89_copies502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16096 := (publicBatchWrapper89_copies503 a h).1
  have c16097 := (publicBatchWrapper89_copies503 a h).2.1
  have c16098 := (publicBatchWrapper89_copies503 a h).2.2.1
  have c16099 := (publicBatchWrapper89_copies503 a h).2.2.2.1
  have c16100 := (publicBatchWrapper89_copies503 a h).2.2.2.2.1
  have c16101 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.1
  have c16102 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_245_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_18
  simp only [← c16092, ← c16093, ← c16094] at e_245_18
  have e_245_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_245_19
  simp only [← c16095, ← c16096, ← c16097] at e_245_19
  have e_246_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_13
  simp only [← c16086, k3, ← c16087, k3, ← c16088] at e_246_13
  have e_246_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_14
  simp only [← c16089, ← c16090, k3, ← c16091] at e_246_14
  have e_246_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_15
  simp only [← c16098, ← c16099, k3, ← c16100] at e_246_15
  refine ⟨?_, ?_⟩
  · have hc := e_245_18
    simp only [e_246_14] at hc
    linear_combination c16101.trans k2 - hc
  · have hc := e_246_15
    simp only [e_246_13, e_245_19, e_246_14] at hc
    linear_combination c16102.trans k2 - hc

theorem publicBatchWrapper89_f2280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 75) = bor (a (.wire 39 15)) (a (.virt 18989)) := by
  have c16103 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.1
  have c16104 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.1
  have c16105 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.1
  have c16106 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16107 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16108 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_236_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_18
  simp only [← c16106, ← c16107, k3, ← c16108] at e_236_18
  have e_243_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_6
  simp only [← c16103, ← c16104, ← c16105] at e_243_6
  have hr := e_236_18
  simp only [e_243_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 75) = a (.virt 201) := by
  have c16109 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16109

theorem publicBatchWrapper89_f2282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7805)) (a (.wire 169 63)) (a (.virt 18991)) (a (.virt 18992)) := by
  have c16110 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16111 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16112 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16113 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16114 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16115 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16116 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16117 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16118 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16119 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16120 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16121 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16122 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16123 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16124 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16125 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16126 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_246_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_16
  simp only [← c16110, k3, ← c16111, k3, ← c16112] at e_246_16
  have e_246_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_17
  simp only [← c16113, ← c16114, k3, ← c16115] at e_246_17
  have e_246_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_18
  simp only [← c16122, ← c16123, k3, ← c16124] at e_246_18
  have e_247_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_0
  simp only [← c16116, ← c16117, ← c16118] at e_247_0
  have e_247_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_1
  simp only [← c16119, ← c16120, ← c16121] at e_247_1
  refine ⟨?_, ?_⟩
  · have hc := e_247_0
    simp only [e_246_17] at hc
    linear_combination c16125.trans k2 - hc
  · have hc := e_246_18
    simp only [e_246_16, e_247_1, e_246_17] at hc
    linear_combination c16126.trans k2 - hc

theorem publicBatchWrapper89_f2283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7806)) (a (.wire 169 71)) (a (.virt 18993)) (a (.virt 18994)) := by
  have c16127 := (publicBatchWrapper89_copies503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16128 := (publicBatchWrapper89_copies504 a h).1
  have c16129 := (publicBatchWrapper89_copies504 a h).2.1
  have c16130 := (publicBatchWrapper89_copies504 a h).2.2.1
  have c16131 := (publicBatchWrapper89_copies504 a h).2.2.2.1
  have c16132 := (publicBatchWrapper89_copies504 a h).2.2.2.2.1
  have c16133 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.1
  have c16134 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.1
  have c16135 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.1
  have c16136 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.1
  have c16137 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.1
  have c16138 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16139 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16140 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16141 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16142 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16143 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_246_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_246_19
  simp only [← c16127, k3, ← c16128, k3, ← c16129] at e_246_19
  have e_247_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_2
  simp only [← c16133, ← c16134, ← c16135] at e_247_2
  have e_247_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_3
  simp only [← c16136, ← c16137, ← c16138] at e_247_3
  have e_248_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_0
  simp only [← c16130, ← c16131, k3, ← c16132] at e_248_0
  have e_248_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_1
  simp only [← c16139, ← c16140, k3, ← c16141] at e_248_1
  refine ⟨?_, ?_⟩
  · have hc := e_247_2
    simp only [e_248_0] at hc
    linear_combination c16142.trans k2 - hc
  · have hc := e_248_1
    simp only [e_246_19, e_247_3, e_248_0] at hc
    linear_combination c16143.trans k2 - hc

theorem publicBatchWrapper89_f2284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7807)) (a (.wire 169 79)) (a (.virt 18995)) (a (.virt 18996)) := by
  have c16144 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16145 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16146 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16147 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16148 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16149 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16150 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16151 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16152 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16153 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16154 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16155 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16156 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16157 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16158 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16159 := (publicBatchWrapper89_copies504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16160 := (publicBatchWrapper89_copies505 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_247_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_4
  simp only [← c16150, ← c16151, ← c16152] at e_247_4
  have e_247_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_5
  simp only [← c16153, ← c16154, ← c16155] at e_247_5
  have e_248_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_2
  simp only [← c16144, k3, ← c16145, k3, ← c16146] at e_248_2
  have e_248_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_3
  simp only [← c16147, ← c16148, k3, ← c16149] at e_248_3
  have e_248_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_4
  simp only [← c16156, ← c16157, k3, ← c16158] at e_248_4
  refine ⟨?_, ?_⟩
  · have hc := e_247_4
    simp only [e_248_3] at hc
    linear_combination c16159.trans k2 - hc
  · have hc := e_248_4
    simp only [e_248_2, e_247_5, e_248_3] at hc
    linear_combination c16160.trans k2 - hc

theorem publicBatchWrapper89_f2285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7808)) (a (.wire 170 7)) (a (.virt 18997)) (a (.virt 18998)) := by
  have c16161 := (publicBatchWrapper89_copies505 a h).2.1
  have c16162 := (publicBatchWrapper89_copies505 a h).2.2.1
  have c16163 := (publicBatchWrapper89_copies505 a h).2.2.2.1
  have c16164 := (publicBatchWrapper89_copies505 a h).2.2.2.2.1
  have c16165 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.1
  have c16166 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.1
  have c16167 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.1
  have c16168 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.1
  have c16169 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.1
  have c16170 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16171 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16172 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16173 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16174 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16175 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16176 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16177 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_247_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_6
  simp only [← c16167, ← c16168, ← c16169] at e_247_6
  have e_247_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_7
  simp only [← c16170, ← c16171, ← c16172] at e_247_7
  have e_248_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_5
  simp only [← c16161, k3, ← c16162, k3, ← c16163] at e_248_5
  have e_248_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_6
  simp only [← c16164, ← c16165, k3, ← c16166] at e_248_6
  have e_248_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_7
  simp only [← c16173, ← c16174, k3, ← c16175] at e_248_7
  refine ⟨?_, ?_⟩
  · have hc := e_247_6
    simp only [e_248_6] at hc
    linear_combination c16176.trans k2 - hc
  · have hc := e_248_7
    simp only [e_248_5, e_247_7, e_248_6] at hc
    linear_combination c16177.trans k2 - hc

theorem publicBatchWrapper89_f2286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 247 35) = band (a (.virt 18991)) (a (.virt 18993)) := by
  have c16178 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16179 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16180 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_247_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_8
  simp only [← c16178, ← c16179, ← c16180] at e_247_8
  have hr := e_247_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 247 39) = band (a (.virt 18995)) (a (.virt 18997)) := by
  have c16181 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16182 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16183 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_247_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_9
  simp only [← c16181, ← c16182, ← c16183] at e_247_9
  have hr := e_247_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 247 43) = band (a (.wire 247 35)) (a (.wire 247 39)) := by
  have c16184 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16185 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16186 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_247_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_10
  simp only [← c16184, ← c16185, ← c16186] at e_247_10
  have hr := e_247_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 79) = bor (a (.wire 39 15)) (a (.wire 247 43)) := by
  have c16187 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16188 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16189 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16190 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16191 := (publicBatchWrapper89_copies505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16192 := (publicBatchWrapper89_copies506 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_236_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_236_19
  simp only [← c16190, ← c16191, k3, ← c16192] at e_236_19
  have e_243_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_7
  simp only [← c16187, ← c16188, ← c16189] at e_243_7
  have hr := e_236_19
  simp only [e_243_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 236 79) = a (.virt 201) := by
  have c16193 := (publicBatchWrapper89_copies506 a h).2.1
  exact c16193

theorem publicBatchWrapper89_f2291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8003)) (a (.wire 170 23)) (a (.virt 18999)) (a (.virt 19000)) := by
  have c16194 := (publicBatchWrapper89_copies506 a h).2.2.1
  have c16195 := (publicBatchWrapper89_copies506 a h).2.2.2.1
  have c16196 := (publicBatchWrapper89_copies506 a h).2.2.2.2.1
  have c16197 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.1
  have c16198 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.1
  have c16199 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.1
  have c16200 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.1
  have c16201 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.1
  have c16202 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16203 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16204 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16205 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16206 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16207 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16208 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16209 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16210 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_247_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_11
  simp only [← c16200, ← c16201, ← c16202] at e_247_11
  have e_247_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_12
  simp only [← c16203, ← c16204, ← c16205] at e_247_12
  have e_248_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_8
  simp only [← c16194, k3, ← c16195, k3, ← c16196] at e_248_8
  have e_248_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_9
  simp only [← c16197, ← c16198, k3, ← c16199] at e_248_9
  have e_248_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_10
  simp only [← c16206, ← c16207, k3, ← c16208] at e_248_10
  refine ⟨?_, ?_⟩
  · have hc := e_247_11
    simp only [e_248_9] at hc
    linear_combination c16209.trans k2 - hc
  · have hc := e_248_10
    simp only [e_248_8, e_247_12, e_248_9] at hc
    linear_combination c16210.trans k2 - hc

theorem publicBatchWrapper89_f2292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 3) = bor (a (.wire 39 59)) (a (.virt 18999)) := by
  have c16211 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16212 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16213 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16214 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16215 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16216 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_8
  simp only [← c16211, ← c16212, ← c16213] at e_243_8
  have e_249_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_0
  simp only [← c16214, ← c16215, k3, ← c16216] at e_249_0
  have hr := e_249_0
  simp only [e_243_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 3) = a (.virt 201) := by
  have c16217 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16217

theorem publicBatchWrapper89_f2294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8004)) (a (.wire 170 31)) (a (.virt 19001)) (a (.virt 19002)) := by
  have c16218 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16219 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16220 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16221 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16222 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16223 := (publicBatchWrapper89_copies506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16224 := (publicBatchWrapper89_copies507 a h).1
  have c16225 := (publicBatchWrapper89_copies507 a h).2.1
  have c16226 := (publicBatchWrapper89_copies507 a h).2.2.1
  have c16227 := (publicBatchWrapper89_copies507 a h).2.2.2.1
  have c16228 := (publicBatchWrapper89_copies507 a h).2.2.2.2.1
  have c16229 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.1
  have c16230 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.1
  have c16231 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.1
  have c16232 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.1
  have c16233 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.1
  have c16234 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_247_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_13
  simp only [← c16224, ← c16225, ← c16226] at e_247_13
  have e_247_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_14
  simp only [← c16227, ← c16228, ← c16229] at e_247_14
  have e_248_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_11
  simp only [← c16218, k3, ← c16219, k3, ← c16220] at e_248_11
  have e_248_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_12
  simp only [← c16221, ← c16222, k3, ← c16223] at e_248_12
  have e_248_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_13
  simp only [← c16230, ← c16231, k3, ← c16232] at e_248_13
  refine ⟨?_, ?_⟩
  · have hc := e_247_13
    simp only [e_248_12] at hc
    linear_combination c16233.trans k2 - hc
  · have hc := e_248_13
    simp only [e_248_11, e_247_14, e_248_12] at hc
    linear_combination c16234.trans k2 - hc

theorem publicBatchWrapper89_f2295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 7) = bor (a (.wire 39 59)) (a (.virt 19001)) := by
  have c16235 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16236 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16237 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16238 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16239 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16240 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_9
  simp only [← c16235, ← c16236, ← c16237] at e_243_9
  have e_249_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_1
  simp only [← c16238, ← c16239, k3, ← c16240] at e_249_1
  have hr := e_249_1
  simp only [e_243_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 7) = a (.virt 201) := by
  have c16241 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16241

theorem publicBatchWrapper89_f2297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8005)) (a (.wire 169 63)) (a (.virt 19003)) (a (.virt 19004)) := by
  have c16242 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16243 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16244 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16245 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16246 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16247 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16248 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16249 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16250 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16251 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16252 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16253 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16254 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16255 := (publicBatchWrapper89_copies507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16256 := (publicBatchWrapper89_copies508 a h).1
  have c16257 := (publicBatchWrapper89_copies508 a h).2.1
  have c16258 := (publicBatchWrapper89_copies508 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_247_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_15
  simp only [← c16248, ← c16249, ← c16250] at e_247_15
  have e_247_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_16
  simp only [← c16251, ← c16252, ← c16253] at e_247_16
  have e_248_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_14
  simp only [← c16242, k3, ← c16243, k3, ← c16244] at e_248_14
  have e_248_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_15
  simp only [← c16245, ← c16246, k3, ← c16247] at e_248_15
  have e_248_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_16
  simp only [← c16254, ← c16255, k3, ← c16256] at e_248_16
  refine ⟨?_, ?_⟩
  · have hc := e_247_15
    simp only [e_248_15] at hc
    linear_combination c16257.trans k2 - hc
  · have hc := e_248_16
    simp only [e_248_14, e_247_16, e_248_15] at hc
    linear_combination c16258.trans k2 - hc

theorem publicBatchWrapper89_f2298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8006)) (a (.wire 169 71)) (a (.virt 19005)) (a (.virt 19006)) := by
  have c16259 := (publicBatchWrapper89_copies508 a h).2.2.2.1
  have c16260 := (publicBatchWrapper89_copies508 a h).2.2.2.2.1
  have c16261 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.1
  have c16262 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.1
  have c16263 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.1
  have c16264 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.1
  have c16265 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.1
  have c16266 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16267 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16268 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16269 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16270 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16271 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16272 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16273 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16274 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16275 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_247_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_17
  simp only [← c16265, ← c16266, ← c16267] at e_247_17
  have e_247_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_18
  simp only [← c16268, ← c16269, ← c16270] at e_247_18
  have e_248_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_17
  simp only [← c16259, k3, ← c16260, k3, ← c16261] at e_248_17
  have e_248_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_18
  simp only [← c16262, ← c16263, k3, ← c16264] at e_248_18
  have e_248_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_248_19
  simp only [← c16271, ← c16272, k3, ← c16273] at e_248_19
  refine ⟨?_, ?_⟩
  · have hc := e_247_17
    simp only [e_248_18] at hc
    linear_combination c16274.trans k2 - hc
  · have hc := e_248_19
    simp only [e_248_17, e_247_18, e_248_18] at hc
    linear_combination c16275.trans k2 - hc

theorem publicBatchWrapper89_f2299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8007)) (a (.wire 169 79)) (a (.virt 19007)) (a (.virt 19008)) := by
  have c16276 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16277 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16278 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16279 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16280 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16281 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16282 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16283 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16284 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16285 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16286 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16287 := (publicBatchWrapper89_copies508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16288 := (publicBatchWrapper89_copies509 a h).1
  have c16289 := (publicBatchWrapper89_copies509 a h).2.1
  have c16290 := (publicBatchWrapper89_copies509 a h).2.2.1
  have c16291 := (publicBatchWrapper89_copies509 a h).2.2.2.1
  have c16292 := (publicBatchWrapper89_copies509 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_247_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_247_19
  simp only [← c16282, ← c16283, ← c16284] at e_247_19
  have e_250_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_0
  simp only [← c16276, k3, ← c16277, k3, ← c16278] at e_250_0
  have e_250_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_1
  simp only [← c16279, ← c16280, k3, ← c16281] at e_250_1
  have e_250_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_2
  simp only [← c16288, ← c16289, k3, ← c16290] at e_250_2
  have e_251_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_0
  simp only [← c16285, ← c16286, ← c16287] at e_251_0
  refine ⟨?_, ?_⟩
  · have hc := e_247_19
    simp only [e_250_1] at hc
    linear_combination c16291.trans k2 - hc
  · have hc := e_250_2
    simp only [e_250_0, e_251_0, e_250_1] at hc
    linear_combination c16292.trans k2 - hc

theorem publicBatchWrapper89_f2300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8008)) (a (.wire 170 7)) (a (.virt 19009)) (a (.virt 19010)) := by
  have c16293 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.1
  have c16294 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.1
  have c16295 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.1
  have c16296 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.1
  have c16297 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.1
  have c16298 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16299 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16300 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16301 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16302 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16303 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16304 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16305 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16306 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16307 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16308 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16309 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_250_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_3
  simp only [← c16293, k3, ← c16294, k3, ← c16295] at e_250_3
  have e_250_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_4
  simp only [← c16296, ← c16297, k3, ← c16298] at e_250_4
  have e_250_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_5
  simp only [← c16305, ← c16306, k3, ← c16307] at e_250_5
  have e_251_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_1
  simp only [← c16299, ← c16300, ← c16301] at e_251_1
  have e_251_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_2
  simp only [← c16302, ← c16303, ← c16304] at e_251_2
  refine ⟨?_, ?_⟩
  · have hc := e_251_1
    simp only [e_250_4] at hc
    linear_combination c16308.trans k2 - hc
  · have hc := e_250_5
    simp only [e_250_3, e_251_2, e_250_4] at hc
    linear_combination c16309.trans k2 - hc

theorem publicBatchWrapper89_f2301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 251 15) = band (a (.virt 19003)) (a (.virt 19005)) := by
  have c16310 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16311 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16312 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_251_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_3
  simp only [← c16310, ← c16311, ← c16312] at e_251_3
  have hr := e_251_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 251 19) = band (a (.virt 19007)) (a (.virt 19009)) := by
  have c16313 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16314 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16315 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_251_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_4
  simp only [← c16313, ← c16314, ← c16315] at e_251_4
  have hr := e_251_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 251 23) = band (a (.wire 251 15)) (a (.wire 251 19)) := by
  have c16316 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16317 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16318 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_251_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_5
  simp only [← c16316, ← c16317, ← c16318] at e_251_5
  have hr := e_251_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 11) = bor (a (.wire 39 59)) (a (.wire 251 23)) := by
  have c16319 := (publicBatchWrapper89_copies509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16320 := (publicBatchWrapper89_copies510 a h).1
  have c16321 := (publicBatchWrapper89_copies510 a h).2.1
  have c16322 := (publicBatchWrapper89_copies510 a h).2.2.1
  have c16323 := (publicBatchWrapper89_copies510 a h).2.2.2.1
  have c16324 := (publicBatchWrapper89_copies510 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_10
  simp only [← c16319, ← c16320, ← c16321] at e_243_10
  have e_249_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_2
  simp only [← c16322, ← c16323, k3, ← c16324] at e_249_2
  have hr := e_249_2
  simp only [e_243_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 11) = a (.virt 201) := by
  have c16325 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.1
  exact c16325

theorem publicBatchWrapper89_f2306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8203)) (a (.wire 170 23)) (a (.virt 19011)) (a (.virt 19012)) := by
  have c16326 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.1
  have c16327 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.1
  have c16328 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.1
  have c16329 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.1
  have c16330 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16331 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16332 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16333 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16334 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16335 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16336 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16337 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16338 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16339 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16340 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16341 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16342 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_250_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_6
  simp only [← c16326, k3, ← c16327, k3, ← c16328] at e_250_6
  have e_250_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_7
  simp only [← c16329, ← c16330, k3, ← c16331] at e_250_7
  have e_250_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_8
  simp only [← c16338, ← c16339, k3, ← c16340] at e_250_8
  have e_251_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_6
  simp only [← c16332, ← c16333, ← c16334] at e_251_6
  have e_251_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_7
  simp only [← c16335, ← c16336, ← c16337] at e_251_7
  refine ⟨?_, ?_⟩
  · have hc := e_251_6
    simp only [e_250_7] at hc
    linear_combination c16341.trans k2 - hc
  · have hc := e_250_8
    simp only [e_250_6, e_251_7, e_250_7] at hc
    linear_combination c16342.trans k2 - hc

theorem publicBatchWrapper89_f2307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 15) = bor (a (.wire 41 23)) (a (.virt 19011)) := by
  have c16343 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16344 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16345 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16346 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16347 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16348 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_11
  simp only [← c16343, ← c16344, ← c16345] at e_243_11
  have e_249_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_3
  simp only [← c16346, ← c16347, k3, ← c16348] at e_249_3
  have hr := e_249_3
  simp only [e_243_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 15) = a (.virt 201) := by
  have c16349 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16349

theorem publicBatchWrapper89_f2309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8204)) (a (.wire 170 31)) (a (.virt 19013)) (a (.virt 19014)) := by
  have c16350 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16351 := (publicBatchWrapper89_copies510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16352 := (publicBatchWrapper89_copies511 a h).1
  have c16353 := (publicBatchWrapper89_copies511 a h).2.1
  have c16354 := (publicBatchWrapper89_copies511 a h).2.2.1
  have c16355 := (publicBatchWrapper89_copies511 a h).2.2.2.1
  have c16356 := (publicBatchWrapper89_copies511 a h).2.2.2.2.1
  have c16357 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.1
  have c16358 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.1
  have c16359 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.1
  have c16360 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.1
  have c16361 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.1
  have c16362 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16363 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16364 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16365 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16366 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_250_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_9
  simp only [← c16350, k3, ← c16351, k3, ← c16352] at e_250_9
  have e_250_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_10
  simp only [← c16353, ← c16354, k3, ← c16355] at e_250_10
  have e_250_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_11
  simp only [← c16362, ← c16363, k3, ← c16364] at e_250_11
  have e_251_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_8
  simp only [← c16356, ← c16357, ← c16358] at e_251_8
  have e_251_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_9
  simp only [← c16359, ← c16360, ← c16361] at e_251_9
  refine ⟨?_, ?_⟩
  · have hc := e_251_8
    simp only [e_250_10] at hc
    linear_combination c16365.trans k2 - hc
  · have hc := e_250_11
    simp only [e_250_9, e_251_9, e_250_10] at hc
    linear_combination c16366.trans k2 - hc

theorem publicBatchWrapper89_f2310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 19) = bor (a (.wire 41 23)) (a (.virt 19013)) := by
  have c16367 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16368 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16369 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16370 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16371 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16372 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_12
  simp only [← c16367, ← c16368, ← c16369] at e_243_12
  have e_249_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_4
  simp only [← c16370, ← c16371, k3, ← c16372] at e_249_4
  have hr := e_249_4
  simp only [e_243_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 19) = a (.virt 201) := by
  have c16373 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16373

theorem publicBatchWrapper89_f2312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8205)) (a (.wire 169 63)) (a (.virt 19015)) (a (.virt 19016)) := by
  have c16374 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16375 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16376 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16377 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16378 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16379 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16380 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16381 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16382 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16383 := (publicBatchWrapper89_copies511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16384 := (publicBatchWrapper89_copies512 a h).1
  have c16385 := (publicBatchWrapper89_copies512 a h).2.1
  have c16386 := (publicBatchWrapper89_copies512 a h).2.2.1
  have c16387 := (publicBatchWrapper89_copies512 a h).2.2.2.1
  have c16388 := (publicBatchWrapper89_copies512 a h).2.2.2.2.1
  have c16389 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.1
  have c16390 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_250_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_12
  simp only [← c16374, k3, ← c16375, k3, ← c16376] at e_250_12
  have e_250_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_13
  simp only [← c16377, ← c16378, k3, ← c16379] at e_250_13
  have e_250_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_14
  simp only [← c16386, ← c16387, k3, ← c16388] at e_250_14
  have e_251_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_10
  simp only [← c16380, ← c16381, ← c16382] at e_251_10
  have e_251_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_11
  simp only [← c16383, ← c16384, ← c16385] at e_251_11
  refine ⟨?_, ?_⟩
  · have hc := e_251_10
    simp only [e_250_13] at hc
    linear_combination c16389.trans k2 - hc
  · have hc := e_250_14
    simp only [e_250_12, e_251_11, e_250_13] at hc
    linear_combination c16390.trans k2 - hc

theorem publicBatchWrapper89_f2313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8206)) (a (.wire 169 71)) (a (.virt 19017)) (a (.virt 19018)) := by
  have c16391 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.1
  have c16392 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.1
  have c16393 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.1
  have c16394 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16395 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16396 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16397 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16398 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16399 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16400 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16401 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16402 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16403 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16404 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16405 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16406 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16407 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_250_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_15
  simp only [← c16391, k3, ← c16392, k3, ← c16393] at e_250_15
  have e_250_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_16
  simp only [← c16394, ← c16395, k3, ← c16396] at e_250_16
  have e_250_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_17
  simp only [← c16403, ← c16404, k3, ← c16405] at e_250_17
  have e_251_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_12
  simp only [← c16397, ← c16398, ← c16399] at e_251_12
  have e_251_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_13
  simp only [← c16400, ← c16401, ← c16402] at e_251_13
  refine ⟨?_, ?_⟩
  · have hc := e_251_12
    simp only [e_250_16] at hc
    linear_combination c16406.trans k2 - hc
  · have hc := e_250_17
    simp only [e_250_15, e_251_13, e_250_16] at hc
    linear_combination c16407.trans k2 - hc

theorem publicBatchWrapper89_f2314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8207)) (a (.wire 169 79)) (a (.virt 19019)) (a (.virt 19020)) := by
  have c16408 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16409 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16410 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16411 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16412 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16413 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16414 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16415 := (publicBatchWrapper89_copies512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16416 := (publicBatchWrapper89_copies513 a h).1
  have c16417 := (publicBatchWrapper89_copies513 a h).2.1
  have c16418 := (publicBatchWrapper89_copies513 a h).2.2.1
  have c16419 := (publicBatchWrapper89_copies513 a h).2.2.2.1
  have c16420 := (publicBatchWrapper89_copies513 a h).2.2.2.2.1
  have c16421 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.1
  have c16422 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.1
  have c16423 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.1
  have c16424 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_250_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_18
  simp only [← c16408, k3, ← c16409, k3, ← c16410] at e_250_18
  have e_250_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_250_19
  simp only [← c16411, ← c16412, k3, ← c16413] at e_250_19
  have e_251_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_14
  simp only [← c16414, ← c16415, ← c16416] at e_251_14
  have e_251_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_15
  simp only [← c16417, ← c16418, ← c16419] at e_251_15
  have e_252_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_0
  simp only [← c16420, ← c16421, k3, ← c16422] at e_252_0
  refine ⟨?_, ?_⟩
  · have hc := e_251_14
    simp only [e_250_19] at hc
    linear_combination c16423.trans k2 - hc
  · have hc := e_252_0
    simp only [e_250_18, e_251_15, e_250_19] at hc
    linear_combination c16424.trans k2 - hc

theorem publicBatchWrapper89_f2315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8208)) (a (.wire 170 7)) (a (.virt 19021)) (a (.virt 19022)) := by
  have c16425 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.1
  have c16426 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16427 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16428 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16429 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16430 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16431 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16432 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16433 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16434 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16435 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16436 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16437 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16438 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16439 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16440 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16441 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_251_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_16
  simp only [← c16431, ← c16432, ← c16433] at e_251_16
  have e_251_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_17
  simp only [← c16434, ← c16435, ← c16436] at e_251_17
  have e_252_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_1
  simp only [← c16425, k3, ← c16426, k3, ← c16427] at e_252_1
  have e_252_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_2
  simp only [← c16428, ← c16429, k3, ← c16430] at e_252_2
  have e_252_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_3
  simp only [← c16437, ← c16438, k3, ← c16439] at e_252_3
  refine ⟨?_, ?_⟩
  · have hc := e_251_16
    simp only [e_252_2] at hc
    linear_combination c16440.trans k2 - hc
  · have hc := e_252_3
    simp only [e_252_1, e_251_17, e_252_2] at hc
    linear_combination c16441.trans k2 - hc

theorem publicBatchWrapper89_f2316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 251 75) = band (a (.virt 19015)) (a (.virt 19017)) := by
  have c16442 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16443 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16444 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_251_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_18
  simp only [← c16442, ← c16443, ← c16444] at e_251_18
  have hr := e_251_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 251 79) = band (a (.virt 19019)) (a (.virt 19021)) := by
  have c16445 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16446 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16447 := (publicBatchWrapper89_copies513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_251_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_251_19
  simp only [← c16445, ← c16446, ← c16447] at e_251_19
  have hr := e_251_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 253 3) = band (a (.wire 251 75)) (a (.wire 251 79)) := by
  have c16448 := (publicBatchWrapper89_copies514 a h).1
  have c16449 := (publicBatchWrapper89_copies514 a h).2.1
  have c16450 := (publicBatchWrapper89_copies514 a h).2.2.1
  have e_253_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_0
  simp only [← c16448, ← c16449, ← c16450] at e_253_0
  have hr := e_253_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 23) = bor (a (.wire 41 23)) (a (.wire 253 3)) := by
  have c16451 := (publicBatchWrapper89_copies514 a h).2.2.2.1
  have c16452 := (publicBatchWrapper89_copies514 a h).2.2.2.2.1
  have c16453 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.1
  have c16454 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.1
  have c16455 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.1
  have c16456 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_13
  simp only [← c16451, ← c16452, ← c16453] at e_243_13
  have e_249_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_5
  simp only [← c16454, ← c16455, k3, ← c16456] at e_249_5
  have hr := e_249_5
  simp only [e_243_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 23) = a (.virt 201) := by
  have c16457 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.1
  exact c16457

theorem publicBatchWrapper89_f2321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8403)) (a (.wire 170 23)) (a (.virt 19023)) (a (.virt 19024)) := by
  have c16458 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16459 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16460 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16461 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16462 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16463 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16464 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16465 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16466 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16467 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16468 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16469 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16470 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16471 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16472 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16473 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16474 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_252_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_4
  simp only [← c16458, k3, ← c16459, k3, ← c16460] at e_252_4
  have e_252_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_5
  simp only [← c16461, ← c16462, k3, ← c16463] at e_252_5
  have e_252_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_6
  simp only [← c16470, ← c16471, k3, ← c16472] at e_252_6
  have e_253_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_1
  simp only [← c16464, ← c16465, ← c16466] at e_253_1
  have e_253_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_2
  simp only [← c16467, ← c16468, ← c16469] at e_253_2
  refine ⟨?_, ?_⟩
  · have hc := e_253_1
    simp only [e_252_5] at hc
    linear_combination c16473.trans k2 - hc
  · have hc := e_252_6
    simp only [e_252_4, e_253_2, e_252_5] at hc
    linear_combination c16474.trans k2 - hc

theorem publicBatchWrapper89_f2322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 27) = bor (a (.wire 41 67)) (a (.virt 19023)) := by
  have c16475 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16476 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16477 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16478 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16479 := (publicBatchWrapper89_copies514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16480 := (publicBatchWrapper89_copies515 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_14
  simp only [← c16475, ← c16476, ← c16477] at e_243_14
  have e_249_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_6
  simp only [← c16478, ← c16479, k3, ← c16480] at e_249_6
  have hr := e_249_6
  simp only [e_243_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 27) = a (.virt 201) := by
  have c16481 := (publicBatchWrapper89_copies515 a h).2.1
  exact c16481

theorem publicBatchWrapper89_f2324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8404)) (a (.wire 170 31)) (a (.virt 19025)) (a (.virt 19026)) := by
  have c16482 := (publicBatchWrapper89_copies515 a h).2.2.1
  have c16483 := (publicBatchWrapper89_copies515 a h).2.2.2.1
  have c16484 := (publicBatchWrapper89_copies515 a h).2.2.2.2.1
  have c16485 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.1
  have c16486 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.1
  have c16487 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.1
  have c16488 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.1
  have c16489 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.1
  have c16490 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16491 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16492 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16493 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16494 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16495 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16496 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16497 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16498 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_252_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_7
  simp only [← c16482, k3, ← c16483, k3, ← c16484] at e_252_7
  have e_252_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_8
  simp only [← c16485, ← c16486, k3, ← c16487] at e_252_8
  have e_252_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_9
  simp only [← c16494, ← c16495, k3, ← c16496] at e_252_9
  have e_253_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_3
  simp only [← c16488, ← c16489, ← c16490] at e_253_3
  have e_253_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_4
  simp only [← c16491, ← c16492, ← c16493] at e_253_4
  refine ⟨?_, ?_⟩
  · have hc := e_253_3
    simp only [e_252_8] at hc
    linear_combination c16497.trans k2 - hc
  · have hc := e_252_9
    simp only [e_252_7, e_253_4, e_252_8] at hc
    linear_combination c16498.trans k2 - hc

theorem publicBatchWrapper89_f2325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 31) = bor (a (.wire 41 67)) (a (.virt 19025)) := by
  have c16499 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16500 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16501 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16502 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16503 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16504 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_15
  simp only [← c16499, ← c16500, ← c16501] at e_243_15
  have e_249_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_7
  simp only [← c16502, ← c16503, k3, ← c16504] at e_249_7
  have hr := e_249_7
  simp only [e_243_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 31) = a (.virt 201) := by
  have c16505 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16505

theorem publicBatchWrapper89_f2327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8405)) (a (.wire 169 63)) (a (.virt 19027)) (a (.virt 19028)) := by
  have c16506 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16507 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16508 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16509 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16510 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16511 := (publicBatchWrapper89_copies515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16512 := (publicBatchWrapper89_copies516 a h).1
  have c16513 := (publicBatchWrapper89_copies516 a h).2.1
  have c16514 := (publicBatchWrapper89_copies516 a h).2.2.1
  have c16515 := (publicBatchWrapper89_copies516 a h).2.2.2.1
  have c16516 := (publicBatchWrapper89_copies516 a h).2.2.2.2.1
  have c16517 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.1
  have c16518 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.1
  have c16519 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.1
  have c16520 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.1
  have c16521 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.1
  have c16522 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_252_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_10
  simp only [← c16506, k3, ← c16507, k3, ← c16508] at e_252_10
  have e_252_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_11
  simp only [← c16509, ← c16510, k3, ← c16511] at e_252_11
  have e_252_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_12
  simp only [← c16518, ← c16519, k3, ← c16520] at e_252_12
  have e_253_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_5
  simp only [← c16512, ← c16513, ← c16514] at e_253_5
  have e_253_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_6
  simp only [← c16515, ← c16516, ← c16517] at e_253_6
  refine ⟨?_, ?_⟩
  · have hc := e_253_5
    simp only [e_252_11] at hc
    linear_combination c16521.trans k2 - hc
  · have hc := e_252_12
    simp only [e_252_10, e_253_6, e_252_11] at hc
    linear_combination c16522.trans k2 - hc

theorem publicBatchWrapper89_f2328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8406)) (a (.wire 169 71)) (a (.virt 19029)) (a (.virt 19030)) := by
  have c16523 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16524 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16525 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16526 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16527 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16528 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16529 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16530 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16531 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16532 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16533 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16534 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16535 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16536 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16537 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16538 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16539 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_252_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_13
  simp only [← c16523, k3, ← c16524, k3, ← c16525] at e_252_13
  have e_252_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_14
  simp only [← c16526, ← c16527, k3, ← c16528] at e_252_14
  have e_252_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_15
  simp only [← c16535, ← c16536, k3, ← c16537] at e_252_15
  have e_253_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_7
  simp only [← c16529, ← c16530, ← c16531] at e_253_7
  have e_253_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_8
  simp only [← c16532, ← c16533, ← c16534] at e_253_8
  refine ⟨?_, ?_⟩
  · have hc := e_253_7
    simp only [e_252_14] at hc
    linear_combination c16538.trans k2 - hc
  · have hc := e_252_15
    simp only [e_252_13, e_253_8, e_252_14] at hc
    linear_combination c16539.trans k2 - hc

theorem publicBatchWrapper89_f2329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8407)) (a (.wire 169 79)) (a (.virt 19031)) (a (.virt 19032)) := by
  have c16540 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16541 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16542 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16543 := (publicBatchWrapper89_copies516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16544 := (publicBatchWrapper89_copies517 a h).1
  have c16545 := (publicBatchWrapper89_copies517 a h).2.1
  have c16546 := (publicBatchWrapper89_copies517 a h).2.2.1
  have c16547 := (publicBatchWrapper89_copies517 a h).2.2.2.1
  have c16548 := (publicBatchWrapper89_copies517 a h).2.2.2.2.1
  have c16549 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.1
  have c16550 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.1
  have c16551 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.1
  have c16552 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.1
  have c16553 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.1
  have c16554 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16555 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16556 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_252_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_16
  simp only [← c16540, k3, ← c16541, k3, ← c16542] at e_252_16
  have e_252_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_17
  simp only [← c16543, ← c16544, k3, ← c16545] at e_252_17
  have e_252_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_18
  simp only [← c16552, ← c16553, k3, ← c16554] at e_252_18
  have e_253_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_9
  simp only [← c16546, ← c16547, ← c16548] at e_253_9
  have e_253_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_10
  simp only [← c16549, ← c16550, ← c16551] at e_253_10
  refine ⟨?_, ?_⟩
  · have hc := e_253_9
    simp only [e_252_17] at hc
    linear_combination c16555.trans k2 - hc
  · have hc := e_252_18
    simp only [e_252_16, e_253_10, e_252_17] at hc
    linear_combination c16556.trans k2 - hc

theorem publicBatchWrapper89_f2330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8408)) (a (.wire 170 7)) (a (.virt 19033)) (a (.virt 19034)) := by
  have c16557 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16558 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16559 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16560 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16561 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16562 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16563 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16564 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16565 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16566 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16567 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16568 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16569 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16570 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16571 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16572 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16573 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_252_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row252 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_252_19
  simp only [← c16557, k3, ← c16558, k3, ← c16559] at e_252_19
  have e_253_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_11
  simp only [← c16563, ← c16564, ← c16565] at e_253_11
  have e_253_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_12
  simp only [← c16566, ← c16567, ← c16568] at e_253_12
  have e_254_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_0
  simp only [← c16560, ← c16561, k3, ← c16562] at e_254_0
  have e_254_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_1
  simp only [← c16569, ← c16570, k3, ← c16571] at e_254_1
  refine ⟨?_, ?_⟩
  · have hc := e_253_11
    simp only [e_254_0] at hc
    linear_combination c16572.trans k2 - hc
  · have hc := e_254_1
    simp only [e_252_19, e_253_12, e_254_0] at hc
    linear_combination c16573.trans k2 - hc

theorem publicBatchWrapper89_f2331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 253 55) = band (a (.virt 19027)) (a (.virt 19029)) := by
  have c16574 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16575 := (publicBatchWrapper89_copies517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16576 := (publicBatchWrapper89_copies518 a h).1
  have e_253_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_13
  simp only [← c16574, ← c16575, ← c16576] at e_253_13
  have hr := e_253_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 253 59) = band (a (.virt 19031)) (a (.virt 19033)) := by
  have c16577 := (publicBatchWrapper89_copies518 a h).2.1
  have c16578 := (publicBatchWrapper89_copies518 a h).2.2.1
  have c16579 := (publicBatchWrapper89_copies518 a h).2.2.2.1
  have e_253_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_14
  simp only [← c16577, ← c16578, ← c16579] at e_253_14
  have hr := e_253_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 253 63) = band (a (.wire 253 55)) (a (.wire 253 59)) := by
  have c16580 := (publicBatchWrapper89_copies518 a h).2.2.2.2.1
  have c16581 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.1
  have c16582 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.1
  have e_253_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_15
  simp only [← c16580, ← c16581, ← c16582] at e_253_15
  have hr := e_253_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 35) = bor (a (.wire 41 67)) (a (.wire 253 63)) := by
  have c16583 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.1
  have c16584 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.1
  have c16585 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.1
  have c16586 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16587 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16588 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_16
  simp only [← c16583, ← c16584, ← c16585] at e_243_16
  have e_249_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_8
  simp only [← c16586, ← c16587, k3, ← c16588] at e_249_8
  have hr := e_249_8
  simp only [e_243_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 35) = a (.virt 201) := by
  have c16589 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16589

theorem publicBatchWrapper89_f2336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8603)) (a (.wire 170 23)) (a (.virt 19035)) (a (.virt 19036)) := by
  have c16590 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16591 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16592 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16593 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16594 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16595 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16596 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16597 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16598 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16599 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16600 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16601 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16602 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16603 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16604 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16605 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16606 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_253_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_16
  simp only [← c16596, ← c16597, ← c16598] at e_253_16
  have e_253_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_17
  simp only [← c16599, ← c16600, ← c16601] at e_253_17
  have e_254_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_2
  simp only [← c16590, k3, ← c16591, k3, ← c16592] at e_254_2
  have e_254_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_3
  simp only [← c16593, ← c16594, k3, ← c16595] at e_254_3
  have e_254_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_4
  simp only [← c16602, ← c16603, k3, ← c16604] at e_254_4
  refine ⟨?_, ?_⟩
  · have hc := e_253_16
    simp only [e_254_3] at hc
    linear_combination c16605.trans k2 - hc
  · have hc := e_254_4
    simp only [e_254_2, e_253_17, e_254_3] at hc
    linear_combination c16606.trans k2 - hc

theorem publicBatchWrapper89_f2337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 39) = bor (a (.wire 43 31)) (a (.virt 19035)) := by
  have c16607 := (publicBatchWrapper89_copies518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16608 := (publicBatchWrapper89_copies519 a h).1
  have c16609 := (publicBatchWrapper89_copies519 a h).2.1
  have c16610 := (publicBatchWrapper89_copies519 a h).2.2.1
  have c16611 := (publicBatchWrapper89_copies519 a h).2.2.2.1
  have c16612 := (publicBatchWrapper89_copies519 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_17
  simp only [← c16607, ← c16608, ← c16609] at e_243_17
  have e_249_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_9
  simp only [← c16610, ← c16611, k3, ← c16612] at e_249_9
  have hr := e_249_9
  simp only [e_243_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 39) = a (.virt 201) := by
  have c16613 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.1
  exact c16613

theorem publicBatchWrapper89_f2339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8604)) (a (.wire 170 31)) (a (.virt 19037)) (a (.virt 19038)) := by
  have c16614 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.1
  have c16615 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.1
  have c16616 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.1
  have c16617 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.1
  have c16618 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16619 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16620 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16621 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16622 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16623 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16624 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16625 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16626 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16627 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16628 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16629 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16630 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_253_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_18
  simp only [← c16620, ← c16621, ← c16622] at e_253_18
  have e_253_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row253 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_253_19
  simp only [← c16623, ← c16624, ← c16625] at e_253_19
  have e_254_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_5
  simp only [← c16614, k3, ← c16615, k3, ← c16616] at e_254_5
  have e_254_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_6
  simp only [← c16617, ← c16618, k3, ← c16619] at e_254_6
  have e_254_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_7
  simp only [← c16626, ← c16627, k3, ← c16628] at e_254_7
  refine ⟨?_, ?_⟩
  · have hc := e_253_18
    simp only [e_254_6] at hc
    linear_combination c16629.trans k2 - hc
  · have hc := e_254_7
    simp only [e_254_5, e_253_19, e_254_6] at hc
    linear_combination c16630.trans k2 - hc

theorem publicBatchWrapper89_f2340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 43) = bor (a (.wire 43 31)) (a (.virt 19037)) := by
  have c16631 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16632 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16633 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16634 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16635 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16636 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_18
  simp only [← c16631, ← c16632, ← c16633] at e_243_18
  have e_249_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_10
  simp only [← c16634, ← c16635, k3, ← c16636] at e_249_10
  have hr := e_249_10
  simp only [e_243_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 43) = a (.virt 201) := by
  have c16637 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16637

theorem publicBatchWrapper89_f2342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8605)) (a (.wire 169 63)) (a (.virt 19039)) (a (.virt 19040)) := by
  have c16638 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16639 := (publicBatchWrapper89_copies519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16640 := (publicBatchWrapper89_copies520 a h).1
  have c16641 := (publicBatchWrapper89_copies520 a h).2.1
  have c16642 := (publicBatchWrapper89_copies520 a h).2.2.1
  have c16643 := (publicBatchWrapper89_copies520 a h).2.2.2.1
  have c16644 := (publicBatchWrapper89_copies520 a h).2.2.2.2.1
  have c16645 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.1
  have c16646 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.1
  have c16647 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.1
  have c16648 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.1
  have c16649 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.1
  have c16650 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16651 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16652 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16653 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16654 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_254_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_8
  simp only [← c16638, k3, ← c16639, k3, ← c16640] at e_254_8
  have e_254_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_9
  simp only [← c16641, ← c16642, k3, ← c16643] at e_254_9
  have e_254_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_10
  simp only [← c16650, ← c16651, k3, ← c16652] at e_254_10
  have e_255_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_0
  simp only [← c16644, ← c16645, ← c16646] at e_255_0
  have e_255_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_1
  simp only [← c16647, ← c16648, ← c16649] at e_255_1
  refine ⟨?_, ?_⟩
  · have hc := e_255_0
    simp only [e_254_9] at hc
    linear_combination c16653.trans k2 - hc
  · have hc := e_254_10
    simp only [e_254_8, e_255_1, e_254_9] at hc
    linear_combination c16654.trans k2 - hc

theorem publicBatchWrapper89_f2343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8606)) (a (.wire 169 71)) (a (.virt 19041)) (a (.virt 19042)) := by
  have c16655 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16656 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16657 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16658 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16659 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16660 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16661 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16662 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16663 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16664 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16665 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16666 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16667 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16668 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16669 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16670 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16671 := (publicBatchWrapper89_copies520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_254_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_11
  simp only [← c16655, k3, ← c16656, k3, ← c16657] at e_254_11
  have e_254_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_12
  simp only [← c16658, ← c16659, k3, ← c16660] at e_254_12
  have e_254_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_13
  simp only [← c16667, ← c16668, k3, ← c16669] at e_254_13
  have e_255_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_2
  simp only [← c16661, ← c16662, ← c16663] at e_255_2
  have e_255_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_3
  simp only [← c16664, ← c16665, ← c16666] at e_255_3
  refine ⟨?_, ?_⟩
  · have hc := e_255_2
    simp only [e_254_12] at hc
    linear_combination c16670.trans k2 - hc
  · have hc := e_254_13
    simp only [e_254_11, e_255_3, e_254_12] at hc
    linear_combination c16671.trans k2 - hc

theorem publicBatchWrapper89_f2344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8607)) (a (.wire 169 79)) (a (.virt 19043)) (a (.virt 19044)) := by
  have c16672 := (publicBatchWrapper89_copies521 a h).1
  have c16673 := (publicBatchWrapper89_copies521 a h).2.1
  have c16674 := (publicBatchWrapper89_copies521 a h).2.2.1
  have c16675 := (publicBatchWrapper89_copies521 a h).2.2.2.1
  have c16676 := (publicBatchWrapper89_copies521 a h).2.2.2.2.1
  have c16677 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.1
  have c16678 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.1
  have c16679 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.1
  have c16680 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.1
  have c16681 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.1
  have c16682 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16683 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16684 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16685 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16686 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16687 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16688 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_254_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_14
  simp only [← c16672, k3, ← c16673, k3, ← c16674] at e_254_14
  have e_254_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_15
  simp only [← c16675, ← c16676, k3, ← c16677] at e_254_15
  have e_254_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_16
  simp only [← c16684, ← c16685, k3, ← c16686] at e_254_16
  have e_255_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_4
  simp only [← c16678, ← c16679, ← c16680] at e_255_4
  have e_255_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_5
  simp only [← c16681, ← c16682, ← c16683] at e_255_5
  refine ⟨?_, ?_⟩
  · have hc := e_255_4
    simp only [e_254_15] at hc
    linear_combination c16687.trans k2 - hc
  · have hc := e_254_16
    simp only [e_254_14, e_255_5, e_254_15] at hc
    linear_combination c16688.trans k2 - hc

theorem publicBatchWrapper89_f2345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8608)) (a (.wire 170 7)) (a (.virt 19045)) (a (.virt 19046)) := by
  have c16689 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16690 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16691 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16692 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16693 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16694 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16695 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16696 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16697 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16698 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16699 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16700 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16701 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16702 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16703 := (publicBatchWrapper89_copies521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16704 := (publicBatchWrapper89_copies522 a h).1
  have c16705 := (publicBatchWrapper89_copies522 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_254_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_17
  simp only [← c16689, k3, ← c16690, k3, ← c16691] at e_254_17
  have e_254_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_18
  simp only [← c16692, ← c16693, k3, ← c16694] at e_254_18
  have e_254_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row254 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_254_19
  simp only [← c16701, ← c16702, k3, ← c16703] at e_254_19
  have e_255_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_6
  simp only [← c16695, ← c16696, ← c16697] at e_255_6
  have e_255_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_7
  simp only [← c16698, ← c16699, ← c16700] at e_255_7
  refine ⟨?_, ?_⟩
  · have hc := e_255_6
    simp only [e_254_18] at hc
    linear_combination c16704.trans k2 - hc
  · have hc := e_254_19
    simp only [e_254_17, e_255_7, e_254_18] at hc
    linear_combination c16705.trans k2 - hc

theorem publicBatchWrapper89_f2346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 255 35) = band (a (.virt 19039)) (a (.virt 19041)) := by
  have c16706 := (publicBatchWrapper89_copies522 a h).2.2.1
  have c16707 := (publicBatchWrapper89_copies522 a h).2.2.2.1
  have c16708 := (publicBatchWrapper89_copies522 a h).2.2.2.2.1
  have e_255_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_8
  simp only [← c16706, ← c16707, ← c16708] at e_255_8
  have hr := e_255_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 255 39) = band (a (.virt 19043)) (a (.virt 19045)) := by
  have c16709 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.1
  have c16710 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.1
  have c16711 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.1
  have e_255_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_9
  simp only [← c16709, ← c16710, ← c16711] at e_255_9
  have hr := e_255_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 255 43) = band (a (.wire 255 35)) (a (.wire 255 39)) := by
  have c16712 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.1
  have c16713 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.1
  have c16714 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_255_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_10
  simp only [← c16712, ← c16713, ← c16714] at e_255_10
  have hr := e_255_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 47) = bor (a (.wire 43 31)) (a (.wire 255 43)) := by
  have c16715 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16716 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16717 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16718 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16719 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16720 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_243_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_243_19
  simp only [← c16715, ← c16716, ← c16717] at e_243_19
  have e_249_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_11
  simp only [← c16718, ← c16719, k3, ← c16720] at e_249_11
  have hr := e_249_11
  simp only [e_243_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 47) = a (.virt 201) := by
  have c16721 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16721

theorem publicBatchWrapper89_f2351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8803)) (a (.wire 170 23)) (a (.virt 19047)) (a (.virt 19048)) := by
  have c16722 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16723 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16724 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16725 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16726 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16727 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16728 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16729 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16730 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16731 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16732 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16733 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16734 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16735 := (publicBatchWrapper89_copies522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16736 := (publicBatchWrapper89_copies523 a h).1
  have c16737 := (publicBatchWrapper89_copies523 a h).2.1
  have c16738 := (publicBatchWrapper89_copies523 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_255_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_11
  simp only [← c16728, ← c16729, ← c16730] at e_255_11
  have e_255_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_12
  simp only [← c16731, ← c16732, ← c16733] at e_255_12
  have e_256_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_0
  simp only [← c16722, k3, ← c16723, k3, ← c16724] at e_256_0
  have e_256_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_1
  simp only [← c16725, ← c16726, k3, ← c16727] at e_256_1
  have e_256_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_2
  simp only [← c16734, ← c16735, k3, ← c16736] at e_256_2
  refine ⟨?_, ?_⟩
  · have hc := e_255_11
    simp only [e_256_1] at hc
    linear_combination c16737.trans k2 - hc
  · have hc := e_256_2
    simp only [e_256_0, e_255_12, e_256_1] at hc
    linear_combination c16738.trans k2 - hc

theorem publicBatchWrapper89_f2352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 51) = bor (a (.wire 43 75)) (a (.virt 19047)) := by
  have c16739 := (publicBatchWrapper89_copies523 a h).2.2.2.1
  have c16740 := (publicBatchWrapper89_copies523 a h).2.2.2.2.1
  have c16741 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.1
  have c16742 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.1
  have c16743 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.1
  have c16744 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_249_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_12
  simp only [← c16742, ← c16743, k3, ← c16744] at e_249_12
  have e_257_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_0
  simp only [← c16739, ← c16740, ← c16741] at e_257_0
  have hr := e_249_12
  simp only [e_257_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 51) = a (.virt 201) := by
  have c16745 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.1
  exact c16745

theorem publicBatchWrapper89_f2354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8804)) (a (.wire 170 31)) (a (.virt 19049)) (a (.virt 19050)) := by
  have c16746 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16747 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16748 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16749 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16750 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16751 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16752 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16753 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16754 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16755 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16756 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16757 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16758 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16759 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16760 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16761 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16762 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_255_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_13
  simp only [← c16752, ← c16753, ← c16754] at e_255_13
  have e_255_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_14
  simp only [← c16755, ← c16756, ← c16757] at e_255_14
  have e_256_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_3
  simp only [← c16746, k3, ← c16747, k3, ← c16748] at e_256_3
  have e_256_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_4
  simp only [← c16749, ← c16750, k3, ← c16751] at e_256_4
  have e_256_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_5
  simp only [← c16758, ← c16759, k3, ← c16760] at e_256_5
  refine ⟨?_, ?_⟩
  · have hc := e_255_13
    simp only [e_256_4] at hc
    linear_combination c16761.trans k2 - hc
  · have hc := e_256_5
    simp only [e_256_3, e_255_14, e_256_4] at hc
    linear_combination c16762.trans k2 - hc

theorem publicBatchWrapper89_f2355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 55) = bor (a (.wire 43 75)) (a (.virt 19049)) := by
  have c16763 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16764 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16765 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16766 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16767 := (publicBatchWrapper89_copies523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16768 := (publicBatchWrapper89_copies524 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_249_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_13
  simp only [← c16766, ← c16767, k3, ← c16768] at e_249_13
  have e_257_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_1
  simp only [← c16763, ← c16764, ← c16765] at e_257_1
  have hr := e_249_13
  simp only [e_257_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 55) = a (.virt 201) := by
  have c16769 := (publicBatchWrapper89_copies524 a h).2.1
  exact c16769

theorem publicBatchWrapper89_f2357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8805)) (a (.wire 169 63)) (a (.virt 19051)) (a (.virt 19052)) := by
  have c16770 := (publicBatchWrapper89_copies524 a h).2.2.1
  have c16771 := (publicBatchWrapper89_copies524 a h).2.2.2.1
  have c16772 := (publicBatchWrapper89_copies524 a h).2.2.2.2.1
  have c16773 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.1
  have c16774 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.1
  have c16775 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.1
  have c16776 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.1
  have c16777 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.1
  have c16778 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16779 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16780 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16781 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16782 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16783 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16784 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16785 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16786 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_255_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_15
  simp only [← c16776, ← c16777, ← c16778] at e_255_15
  have e_255_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_16
  simp only [← c16779, ← c16780, ← c16781] at e_255_16
  have e_256_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_6
  simp only [← c16770, k3, ← c16771, k3, ← c16772] at e_256_6
  have e_256_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_7
  simp only [← c16773, ← c16774, k3, ← c16775] at e_256_7
  have e_256_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_8
  simp only [← c16782, ← c16783, k3, ← c16784] at e_256_8
  refine ⟨?_, ?_⟩
  · have hc := e_255_15
    simp only [e_256_7] at hc
    linear_combination c16785.trans k2 - hc
  · have hc := e_256_8
    simp only [e_256_6, e_255_16, e_256_7] at hc
    linear_combination c16786.trans k2 - hc

theorem publicBatchWrapper89_f2358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8806)) (a (.wire 169 71)) (a (.virt 19053)) (a (.virt 19054)) := by
  have c16787 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16788 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16789 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16790 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16791 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16792 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16793 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16794 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16795 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16796 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16797 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16798 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16799 := (publicBatchWrapper89_copies524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16800 := (publicBatchWrapper89_copies525 a h).1
  have c16801 := (publicBatchWrapper89_copies525 a h).2.1
  have c16802 := (publicBatchWrapper89_copies525 a h).2.2.1
  have c16803 := (publicBatchWrapper89_copies525 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_255_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_17
  simp only [← c16793, ← c16794, ← c16795] at e_255_17
  have e_255_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_18
  simp only [← c16796, ← c16797, ← c16798] at e_255_18
  have e_256_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_9
  simp only [← c16787, k3, ← c16788, k3, ← c16789] at e_256_9
  have e_256_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_10
  simp only [← c16790, ← c16791, k3, ← c16792] at e_256_10
  have e_256_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_11
  simp only [← c16799, ← c16800, k3, ← c16801] at e_256_11
  refine ⟨?_, ?_⟩
  · have hc := e_255_17
    simp only [e_256_10] at hc
    linear_combination c16802.trans k2 - hc
  · have hc := e_256_11
    simp only [e_256_9, e_255_18, e_256_10] at hc
    linear_combination c16803.trans k2 - hc

theorem publicBatchWrapper89_f2359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8807)) (a (.wire 169 79)) (a (.virt 19055)) (a (.virt 19056)) := by
  have c16804 := (publicBatchWrapper89_copies525 a h).2.2.2.2.1
  have c16805 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.1
  have c16806 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.1
  have c16807 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.1
  have c16808 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.1
  have c16809 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.1
  have c16810 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16811 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16812 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16813 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16814 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16815 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16816 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16817 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16818 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16819 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16820 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_255_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row255 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_255_19
  simp only [← c16810, ← c16811, ← c16812] at e_255_19
  have e_256_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_12
  simp only [← c16804, k3, ← c16805, k3, ← c16806] at e_256_12
  have e_256_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_13
  simp only [← c16807, ← c16808, k3, ← c16809] at e_256_13
  have e_256_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_14
  simp only [← c16816, ← c16817, k3, ← c16818] at e_256_14
  have e_258_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_0
  simp only [← c16813, ← c16814, ← c16815] at e_258_0
  refine ⟨?_, ?_⟩
  · have hc := e_255_19
    simp only [e_256_13] at hc
    linear_combination c16819.trans k2 - hc
  · have hc := e_256_14
    simp only [e_256_12, e_258_0, e_256_13] at hc
    linear_combination c16820.trans k2 - hc

theorem publicBatchWrapper89_f2360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8808)) (a (.wire 170 7)) (a (.virt 19057)) (a (.virt 19058)) := by
  have c16821 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16822 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16823 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16824 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16825 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16826 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16827 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16828 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16829 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16830 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16831 := (publicBatchWrapper89_copies525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16832 := (publicBatchWrapper89_copies526 a h).1
  have c16833 := (publicBatchWrapper89_copies526 a h).2.1
  have c16834 := (publicBatchWrapper89_copies526 a h).2.2.1
  have c16835 := (publicBatchWrapper89_copies526 a h).2.2.2.1
  have c16836 := (publicBatchWrapper89_copies526 a h).2.2.2.2.1
  have c16837 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_256_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_15
  simp only [← c16821, k3, ← c16822, k3, ← c16823] at e_256_15
  have e_256_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_16
  simp only [← c16824, ← c16825, k3, ← c16826] at e_256_16
  have e_256_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_17
  simp only [← c16833, ← c16834, k3, ← c16835] at e_256_17
  have e_258_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_1
  simp only [← c16827, ← c16828, ← c16829] at e_258_1
  have e_258_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_2
  simp only [← c16830, ← c16831, ← c16832] at e_258_2
  refine ⟨?_, ?_⟩
  · have hc := e_258_1
    simp only [e_256_16] at hc
    linear_combination c16836.trans k2 - hc
  · have hc := e_256_17
    simp only [e_256_15, e_258_2, e_256_16] at hc
    linear_combination c16837.trans k2 - hc

theorem publicBatchWrapper89_f2361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 258 15) = band (a (.virt 19051)) (a (.virt 19053)) := by
  have c16838 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.1
  have c16839 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.1
  have c16840 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.1
  have e_258_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_3
  simp only [← c16838, ← c16839, ← c16840] at e_258_3
  have hr := e_258_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 258 19) = band (a (.virt 19055)) (a (.virt 19057)) := by
  have c16841 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.1
  have c16842 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16843 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_258_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_4
  simp only [← c16841, ← c16842, ← c16843] at e_258_4
  have hr := e_258_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 258 23) = band (a (.wire 258 15)) (a (.wire 258 19)) := by
  have c16844 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16845 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16846 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_258_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_5
  simp only [← c16844, ← c16845, ← c16846] at e_258_5
  have hr := e_258_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 59) = bor (a (.wire 43 75)) (a (.wire 258 23)) := by
  have c16847 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16848 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16849 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16850 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16851 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16852 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_249_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_14
  simp only [← c16850, ← c16851, k3, ← c16852] at e_249_14
  have e_257_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_2
  simp only [← c16847, ← c16848, ← c16849] at e_257_2
  have hr := e_249_14
  simp only [e_257_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 59) = a (.virt 201) := by
  have c16853 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16853

theorem publicBatchWrapper89_f2366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9003)) (a (.wire 170 23)) (a (.virt 19059)) (a (.virt 19060)) := by
  have c16854 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16855 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16856 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16857 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16858 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16859 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16860 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16861 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16862 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16863 := (publicBatchWrapper89_copies526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16864 := (publicBatchWrapper89_copies527 a h).1
  have c16865 := (publicBatchWrapper89_copies527 a h).2.1
  have c16866 := (publicBatchWrapper89_copies527 a h).2.2.1
  have c16867 := (publicBatchWrapper89_copies527 a h).2.2.2.1
  have c16868 := (publicBatchWrapper89_copies527 a h).2.2.2.2.1
  have c16869 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.1
  have c16870 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_256_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_18
  simp only [← c16854, k3, ← c16855, k3, ← c16856] at e_256_18
  have e_256_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row256 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_256_19
  simp only [← c16857, ← c16858, k3, ← c16859] at e_256_19
  have e_258_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_6
  simp only [← c16860, ← c16861, ← c16862] at e_258_6
  have e_258_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_7
  simp only [← c16863, ← c16864, ← c16865] at e_258_7
  have e_259_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_0
  simp only [← c16866, ← c16867, k3, ← c16868] at e_259_0
  refine ⟨?_, ?_⟩
  · have hc := e_258_6
    simp only [e_256_19] at hc
    linear_combination c16869.trans k2 - hc
  · have hc := e_259_0
    simp only [e_256_18, e_258_7, e_256_19] at hc
    linear_combination c16870.trans k2 - hc

theorem publicBatchWrapper89_f2367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 63) = bor (a (.wire 45 39)) (a (.virt 19059)) := by
  have c16871 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.1
  have c16872 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.1
  have c16873 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.1
  have c16874 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16875 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16876 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_249_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_15
  simp only [← c16874, ← c16875, k3, ← c16876] at e_249_15
  have e_257_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_3
  simp only [← c16871, ← c16872, ← c16873] at e_257_3
  have hr := e_249_15
  simp only [e_257_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 63) = a (.virt 201) := by
  have c16877 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16877

theorem publicBatchWrapper89_f2369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9004)) (a (.wire 170 31)) (a (.virt 19061)) (a (.virt 19062)) := by
  have c16878 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16879 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16880 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16881 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16882 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16883 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16884 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16885 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16886 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16887 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16888 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16889 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16890 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16891 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16892 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16893 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16894 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_258_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_8
  simp only [← c16884, ← c16885, ← c16886] at e_258_8
  have e_258_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_9
  simp only [← c16887, ← c16888, ← c16889] at e_258_9
  have e_259_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_1
  simp only [← c16878, k3, ← c16879, k3, ← c16880] at e_259_1
  have e_259_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_2
  simp only [← c16881, ← c16882, k3, ← c16883] at e_259_2
  have e_259_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_3
  simp only [← c16890, ← c16891, k3, ← c16892] at e_259_3
  refine ⟨?_, ?_⟩
  · have hc := e_258_8
    simp only [e_259_2] at hc
    linear_combination c16893.trans k2 - hc
  · have hc := e_259_3
    simp only [e_259_1, e_258_9, e_259_2] at hc
    linear_combination c16894.trans k2 - hc

theorem publicBatchWrapper89_f2370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 67) = bor (a (.wire 45 39)) (a (.virt 19061)) := by
  have c16895 := (publicBatchWrapper89_copies527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16896 := (publicBatchWrapper89_copies528 a h).1
  have c16897 := (publicBatchWrapper89_copies528 a h).2.1
  have c16898 := (publicBatchWrapper89_copies528 a h).2.2.1
  have c16899 := (publicBatchWrapper89_copies528 a h).2.2.2.1
  have c16900 := (publicBatchWrapper89_copies528 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_249_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_16
  simp only [← c16898, ← c16899, k3, ← c16900] at e_249_16
  have e_257_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_4
  simp only [← c16895, ← c16896, ← c16897] at e_257_4
  have hr := e_249_16
  simp only [e_257_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 67) = a (.virt 201) := by
  have c16901 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.1
  exact c16901

theorem publicBatchWrapper89_f2372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9005)) (a (.wire 169 63)) (a (.virt 19063)) (a (.virt 19064)) := by
  have c16902 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.1
  have c16903 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.1
  have c16904 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.1
  have c16905 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.1
  have c16906 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16907 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16908 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16909 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16910 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16911 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16912 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16913 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16914 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16915 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16916 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16917 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16918 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_258_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_10
  simp only [← c16908, ← c16909, ← c16910] at e_258_10
  have e_258_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_11
  simp only [← c16911, ← c16912, ← c16913] at e_258_11
  have e_259_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_4
  simp only [← c16902, k3, ← c16903, k3, ← c16904] at e_259_4
  have e_259_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_5
  simp only [← c16905, ← c16906, k3, ← c16907] at e_259_5
  have e_259_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_6
  simp only [← c16914, ← c16915, k3, ← c16916] at e_259_6
  refine ⟨?_, ?_⟩
  · have hc := e_258_10
    simp only [e_259_5] at hc
    linear_combination c16917.trans k2 - hc
  · have hc := e_259_6
    simp only [e_259_4, e_258_11, e_259_5] at hc
    linear_combination c16918.trans k2 - hc

theorem publicBatchWrapper89_f2373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9006)) (a (.wire 169 71)) (a (.virt 19065)) (a (.virt 19066)) := by
  have c16919 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16920 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16921 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16922 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16923 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16924 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16925 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16926 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16927 := (publicBatchWrapper89_copies528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16928 := (publicBatchWrapper89_copies529 a h).1
  have c16929 := (publicBatchWrapper89_copies529 a h).2.1
  have c16930 := (publicBatchWrapper89_copies529 a h).2.2.1
  have c16931 := (publicBatchWrapper89_copies529 a h).2.2.2.1
  have c16932 := (publicBatchWrapper89_copies529 a h).2.2.2.2.1
  have c16933 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.1
  have c16934 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.1
  have c16935 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_258_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_12
  simp only [← c16925, ← c16926, ← c16927] at e_258_12
  have e_258_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_13
  simp only [← c16928, ← c16929, ← c16930] at e_258_13
  have e_259_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_7
  simp only [← c16919, k3, ← c16920, k3, ← c16921] at e_259_7
  have e_259_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_8
  simp only [← c16922, ← c16923, k3, ← c16924] at e_259_8
  have e_259_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_9
  simp only [← c16931, ← c16932, k3, ← c16933] at e_259_9
  refine ⟨?_, ?_⟩
  · have hc := e_258_12
    simp only [e_259_8] at hc
    linear_combination c16934.trans k2 - hc
  · have hc := e_259_9
    simp only [e_259_7, e_258_13, e_259_8] at hc
    linear_combination c16935.trans k2 - hc

theorem publicBatchWrapper89_f2374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9007)) (a (.wire 169 79)) (a (.virt 19067)) (a (.virt 19068)) := by
  have c16936 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.1
  have c16937 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.1
  have c16938 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16939 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16940 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16941 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16942 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16943 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16944 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16945 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16946 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16947 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16948 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16949 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16950 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16951 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16952 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_258_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_14
  simp only [← c16942, ← c16943, ← c16944] at e_258_14
  have e_258_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_15
  simp only [← c16945, ← c16946, ← c16947] at e_258_15
  have e_259_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_10
  simp only [← c16936, k3, ← c16937, k3, ← c16938] at e_259_10
  have e_259_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_11
  simp only [← c16939, ← c16940, k3, ← c16941] at e_259_11
  have e_259_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_12
  simp only [← c16948, ← c16949, k3, ← c16950] at e_259_12
  refine ⟨?_, ?_⟩
  · have hc := e_258_14
    simp only [e_259_11] at hc
    linear_combination c16951.trans k2 - hc
  · have hc := e_259_12
    simp only [e_259_10, e_258_15, e_259_11] at hc
    linear_combination c16952.trans k2 - hc

theorem publicBatchWrapper89_f2375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9008)) (a (.wire 170 7)) (a (.virt 19069)) (a (.virt 19070)) := by
  have c16953 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16954 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16955 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16956 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16957 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16958 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16959 := (publicBatchWrapper89_copies529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16960 := (publicBatchWrapper89_copies530 a h).1
  have c16961 := (publicBatchWrapper89_copies530 a h).2.1
  have c16962 := (publicBatchWrapper89_copies530 a h).2.2.1
  have c16963 := (publicBatchWrapper89_copies530 a h).2.2.2.1
  have c16964 := (publicBatchWrapper89_copies530 a h).2.2.2.2.1
  have c16965 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.1
  have c16966 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.1
  have c16967 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.1
  have c16968 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.1
  have c16969 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_258_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_16
  simp only [← c16959, ← c16960, ← c16961] at e_258_16
  have e_258_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_17
  simp only [← c16962, ← c16963, ← c16964] at e_258_17
  have e_259_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_13
  simp only [← c16953, k3, ← c16954, k3, ← c16955] at e_259_13
  have e_259_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_14
  simp only [← c16956, ← c16957, k3, ← c16958] at e_259_14
  have e_259_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_15
  simp only [← c16965, ← c16966, k3, ← c16967] at e_259_15
  refine ⟨?_, ?_⟩
  · have hc := e_258_16
    simp only [e_259_14] at hc
    linear_combination c16968.trans k2 - hc
  · have hc := e_259_15
    simp only [e_259_13, e_258_17, e_259_14] at hc
    linear_combination c16969.trans k2 - hc

theorem publicBatchWrapper89_f2376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 258 75) = band (a (.virt 19063)) (a (.virt 19065)) := by
  have c16970 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.1
  have c16971 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c16972 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_258_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_18
  simp only [← c16970, ← c16971, ← c16972] at e_258_18
  have hr := e_258_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 258 79) = band (a (.virt 19067)) (a (.virt 19069)) := by
  have c16973 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16974 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16975 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_258_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row258 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_258_19
  simp only [← c16973, ← c16974, ← c16975] at e_258_19
  have hr := e_258_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 260 3) = band (a (.wire 258 75)) (a (.wire 258 79)) := by
  have c16976 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16977 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16978 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_260_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_0
  simp only [← c16976, ← c16977, ← c16978] at e_260_0
  have hr := e_260_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 71) = bor (a (.wire 45 39)) (a (.wire 260 3)) := by
  have c16979 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16980 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16981 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16982 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16983 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16984 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_249_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_17
  simp only [← c16982, ← c16983, k3, ← c16984] at e_249_17
  have e_257_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_5
  simp only [← c16979, ← c16980, ← c16981] at e_257_5
  have hr := e_249_17
  simp only [e_257_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 71) = a (.virt 201) := by
  have c16985 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16985

theorem publicBatchWrapper89_f2381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9203)) (a (.wire 170 23)) (a (.virt 19071)) (a (.virt 19072)) := by
  have c16986 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16987 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16988 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16989 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16990 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16991 := (publicBatchWrapper89_copies530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c16992 := (publicBatchWrapper89_copies531 a h).1
  have c16993 := (publicBatchWrapper89_copies531 a h).2.1
  have c16994 := (publicBatchWrapper89_copies531 a h).2.2.1
  have c16995 := (publicBatchWrapper89_copies531 a h).2.2.2.1
  have c16996 := (publicBatchWrapper89_copies531 a h).2.2.2.2.1
  have c16997 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.1
  have c16998 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.1
  have c16999 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.1
  have c17000 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.1
  have c17001 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.1
  have c17002 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_259_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_16
  simp only [← c16986, k3, ← c16987, k3, ← c16988] at e_259_16
  have e_259_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_17
  simp only [← c16989, ← c16990, k3, ← c16991] at e_259_17
  have e_259_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_18
  simp only [← c16998, ← c16999, k3, ← c17000] at e_259_18
  have e_260_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_1
  simp only [← c16992, ← c16993, ← c16994] at e_260_1
  have e_260_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_2
  simp only [← c16995, ← c16996, ← c16997] at e_260_2
  refine ⟨?_, ?_⟩
  · have hc := e_260_1
    simp only [e_259_17] at hc
    linear_combination c17001.trans k2 - hc
  · have hc := e_259_18
    simp only [e_259_16, e_260_2, e_259_17] at hc
    linear_combination c17002.trans k2 - hc

theorem publicBatchWrapper89_f2382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 75) = bor (a (.wire 46 3)) (a (.virt 19071)) := by
  have c17003 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17004 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17005 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17006 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17007 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17008 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_249_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_18
  simp only [← c17006, ← c17007, k3, ← c17008] at e_249_18
  have e_257_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_6
  simp only [← c17003, ← c17004, ← c17005] at e_257_6
  have hr := e_249_18
  simp only [e_257_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 75) = a (.virt 201) := by
  have c17009 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17009

theorem publicBatchWrapper89_f2384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9204)) (a (.wire 170 31)) (a (.virt 19073)) (a (.virt 19074)) := by
  have c17010 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17011 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17012 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17013 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17014 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17015 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17016 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17017 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17018 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17019 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17020 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17021 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17022 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17023 := (publicBatchWrapper89_copies531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17024 := (publicBatchWrapper89_copies532 a h).1
  have c17025 := (publicBatchWrapper89_copies532 a h).2.1
  have c17026 := (publicBatchWrapper89_copies532 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_259_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row259 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_259_19
  simp only [← c17010, k3, ← c17011, k3, ← c17012] at e_259_19
  have e_260_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_3
  simp only [← c17016, ← c17017, ← c17018] at e_260_3
  have e_260_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_4
  simp only [← c17019, ← c17020, ← c17021] at e_260_4
  have e_261_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_0
  simp only [← c17013, ← c17014, k3, ← c17015] at e_261_0
  have e_261_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_1
  simp only [← c17022, ← c17023, k3, ← c17024] at e_261_1
  refine ⟨?_, ?_⟩
  · have hc := e_260_3
    simp only [e_261_0] at hc
    linear_combination c17025.trans k2 - hc
  · have hc := e_261_1
    simp only [e_259_19, e_260_4, e_261_0] at hc
    linear_combination c17026.trans k2 - hc

theorem publicBatchWrapper89_f2385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 79) = bor (a (.wire 46 3)) (a (.virt 19073)) := by
  have c17027 := (publicBatchWrapper89_copies532 a h).2.2.2.1
  have c17028 := (publicBatchWrapper89_copies532 a h).2.2.2.2.1
  have c17029 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.1
  have c17030 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.1
  have c17031 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.1
  have c17032 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_249_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_249_19
  simp only [← c17030, ← c17031, k3, ← c17032] at e_249_19
  have e_257_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_7
  simp only [← c17027, ← c17028, ← c17029] at e_257_7
  have hr := e_249_19
  simp only [e_257_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 249 79) = a (.virt 201) := by
  have c17033 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.1
  exact c17033

theorem publicBatchWrapper89_f2387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9205)) (a (.wire 169 63)) (a (.virt 19075)) (a (.virt 19076)) := by
  have c17034 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17035 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17036 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17037 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17038 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17039 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17040 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17041 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17042 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17043 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17044 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17045 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17046 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17047 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17048 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17049 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17050 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_260_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_5
  simp only [← c17040, ← c17041, ← c17042] at e_260_5
  have e_260_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_6
  simp only [← c17043, ← c17044, ← c17045] at e_260_6
  have e_261_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_2
  simp only [← c17034, k3, ← c17035, k3, ← c17036] at e_261_2
  have e_261_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_3
  simp only [← c17037, ← c17038, k3, ← c17039] at e_261_3
  have e_261_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_4
  simp only [← c17046, ← c17047, k3, ← c17048] at e_261_4
  refine ⟨?_, ?_⟩
  · have hc := e_260_5
    simp only [e_261_3] at hc
    linear_combination c17049.trans k2 - hc
  · have hc := e_261_4
    simp only [e_261_2, e_260_6, e_261_3] at hc
    linear_combination c17050.trans k2 - hc

theorem publicBatchWrapper89_f2388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9206)) (a (.wire 169 71)) (a (.virt 19077)) (a (.virt 19078)) := by
  have c17051 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17052 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17053 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17054 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17055 := (publicBatchWrapper89_copies532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17056 := (publicBatchWrapper89_copies533 a h).1
  have c17057 := (publicBatchWrapper89_copies533 a h).2.1
  have c17058 := (publicBatchWrapper89_copies533 a h).2.2.1
  have c17059 := (publicBatchWrapper89_copies533 a h).2.2.2.1
  have c17060 := (publicBatchWrapper89_copies533 a h).2.2.2.2.1
  have c17061 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.1
  have c17062 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.1
  have c17063 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.1
  have c17064 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.1
  have c17065 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.1
  have c17066 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17067 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_260_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_7
  simp only [← c17057, ← c17058, ← c17059] at e_260_7
  have e_260_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_8
  simp only [← c17060, ← c17061, ← c17062] at e_260_8
  have e_261_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_5
  simp only [← c17051, k3, ← c17052, k3, ← c17053] at e_261_5
  have e_261_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_6
  simp only [← c17054, ← c17055, k3, ← c17056] at e_261_6
  have e_261_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_7
  simp only [← c17063, ← c17064, k3, ← c17065] at e_261_7
  refine ⟨?_, ?_⟩
  · have hc := e_260_7
    simp only [e_261_6] at hc
    linear_combination c17066.trans k2 - hc
  · have hc := e_261_7
    simp only [e_261_5, e_260_8, e_261_6] at hc
    linear_combination c17067.trans k2 - hc

theorem publicBatchWrapper89_f2389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9207)) (a (.wire 169 79)) (a (.virt 19079)) (a (.virt 19080)) := by
  have c17068 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17069 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17070 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17071 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17072 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17073 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17074 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17075 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17076 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17077 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17078 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17079 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17080 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17081 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17082 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17083 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17084 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_260_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_9
  simp only [← c17074, ← c17075, ← c17076] at e_260_9
  have e_260_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_10
  simp only [← c17077, ← c17078, ← c17079] at e_260_10
  have e_261_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_8
  simp only [← c17068, k3, ← c17069, k3, ← c17070] at e_261_8
  have e_261_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_9
  simp only [← c17071, ← c17072, k3, ← c17073] at e_261_9
  have e_261_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_10
  simp only [← c17080, ← c17081, k3, ← c17082] at e_261_10
  refine ⟨?_, ?_⟩
  · have hc := e_260_9
    simp only [e_261_9] at hc
    linear_combination c17083.trans k2 - hc
  · have hc := e_261_10
    simp only [e_261_8, e_260_10, e_261_9] at hc
    linear_combination c17084.trans k2 - hc

theorem publicBatchWrapper89_f2390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9208)) (a (.wire 170 7)) (a (.virt 19081)) (a (.virt 19082)) := by
  have c17085 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17086 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17087 := (publicBatchWrapper89_copies533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17088 := (publicBatchWrapper89_copies534 a h).1
  have c17089 := (publicBatchWrapper89_copies534 a h).2.1
  have c17090 := (publicBatchWrapper89_copies534 a h).2.2.1
  have c17091 := (publicBatchWrapper89_copies534 a h).2.2.2.1
  have c17092 := (publicBatchWrapper89_copies534 a h).2.2.2.2.1
  have c17093 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.1
  have c17094 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.1
  have c17095 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.1
  have c17096 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.1
  have c17097 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.1
  have c17098 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17099 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17100 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17101 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_260_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_11
  simp only [← c17091, ← c17092, ← c17093] at e_260_11
  have e_260_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_12
  simp only [← c17094, ← c17095, ← c17096] at e_260_12
  have e_261_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_11
  simp only [← c17085, k3, ← c17086, k3, ← c17087] at e_261_11
  have e_261_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_12
  simp only [← c17088, ← c17089, k3, ← c17090] at e_261_12
  have e_261_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_13
  simp only [← c17097, ← c17098, k3, ← c17099] at e_261_13
  refine ⟨?_, ?_⟩
  · have hc := e_260_11
    simp only [e_261_12] at hc
    linear_combination c17100.trans k2 - hc
  · have hc := e_261_13
    simp only [e_261_11, e_260_12, e_261_12] at hc
    linear_combination c17101.trans k2 - hc

theorem publicBatchWrapper89_f2391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 260 55) = band (a (.virt 19075)) (a (.virt 19077)) := by
  have c17102 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17103 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17104 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_260_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_13
  simp only [← c17102, ← c17103, ← c17104] at e_260_13
  have hr := e_260_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 260 59) = band (a (.virt 19079)) (a (.virt 19081)) := by
  have c17105 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17106 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17107 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_260_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_14
  simp only [← c17105, ← c17106, ← c17107] at e_260_14
  have hr := e_260_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 260 63) = band (a (.wire 260 55)) (a (.wire 260 59)) := by
  have c17108 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17109 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17110 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_260_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_15
  simp only [← c17108, ← c17109, ← c17110] at e_260_15
  have hr := e_260_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 3) = bor (a (.wire 46 3)) (a (.wire 260 63)) := by
  have c17111 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17112 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17113 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17114 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17115 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17116 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_8
  simp only [← c17111, ← c17112, ← c17113] at e_257_8
  have e_262_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_0
  simp only [← c17114, ← c17115, k3, ← c17116] at e_262_0
  have hr := e_262_0
  simp only [e_257_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 3) = a (.virt 201) := by
  have c17117 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17117

theorem publicBatchWrapper89_f2396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9403)) (a (.wire 170 23)) (a (.virt 19083)) (a (.virt 19084)) := by
  have c17118 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17119 := (publicBatchWrapper89_copies534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17120 := (publicBatchWrapper89_copies535 a h).1
  have c17121 := (publicBatchWrapper89_copies535 a h).2.1
  have c17122 := (publicBatchWrapper89_copies535 a h).2.2.1
  have c17123 := (publicBatchWrapper89_copies535 a h).2.2.2.1
  have c17124 := (publicBatchWrapper89_copies535 a h).2.2.2.2.1
  have c17125 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.1
  have c17126 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.1
  have c17127 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.1
  have c17128 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.1
  have c17129 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.1
  have c17130 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17131 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17132 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17133 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17134 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_260_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_16
  simp only [← c17124, ← c17125, ← c17126] at e_260_16
  have e_260_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_17
  simp only [← c17127, ← c17128, ← c17129] at e_260_17
  have e_261_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_14
  simp only [← c17118, k3, ← c17119, k3, ← c17120] at e_261_14
  have e_261_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_15
  simp only [← c17121, ← c17122, k3, ← c17123] at e_261_15
  have e_261_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_16
  simp only [← c17130, ← c17131, k3, ← c17132] at e_261_16
  refine ⟨?_, ?_⟩
  · have hc := e_260_16
    simp only [e_261_15] at hc
    linear_combination c17133.trans k2 - hc
  · have hc := e_261_16
    simp only [e_261_14, e_260_17, e_261_15] at hc
    linear_combination c17134.trans k2 - hc

theorem publicBatchWrapper89_f2397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 7) = bor (a (.wire 46 47)) (a (.virt 19083)) := by
  have c17135 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17136 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17137 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17138 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17139 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17140 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_9
  simp only [← c17135, ← c17136, ← c17137] at e_257_9
  have e_262_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_1
  simp only [← c17138, ← c17139, k3, ← c17140] at e_262_1
  have hr := e_262_1
  simp only [e_257_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 7) = a (.virt 201) := by
  have c17141 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17141

theorem publicBatchWrapper89_f2399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9404)) (a (.wire 170 31)) (a (.virt 19085)) (a (.virt 19086)) := by
  have c17142 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17143 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17144 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17145 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17146 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17147 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17148 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17149 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17150 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17151 := (publicBatchWrapper89_copies535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17152 := (publicBatchWrapper89_copies536 a h).1
  have c17153 := (publicBatchWrapper89_copies536 a h).2.1
  have c17154 := (publicBatchWrapper89_copies536 a h).2.2.1
  have c17155 := (publicBatchWrapper89_copies536 a h).2.2.2.1
  have c17156 := (publicBatchWrapper89_copies536 a h).2.2.2.2.1
  have c17157 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.1
  have c17158 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_260_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_18
  simp only [← c17148, ← c17149, ← c17150] at e_260_18
  have e_260_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row260 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_260_19
  simp only [← c17151, ← c17152, ← c17153] at e_260_19
  have e_261_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_17
  simp only [← c17142, k3, ← c17143, k3, ← c17144] at e_261_17
  have e_261_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_18
  simp only [← c17145, ← c17146, k3, ← c17147] at e_261_18
  have e_261_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row261 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_261_19
  simp only [← c17154, ← c17155, k3, ← c17156] at e_261_19
  refine ⟨?_, ?_⟩
  · have hc := e_260_18
    simp only [e_261_18] at hc
    linear_combination c17157.trans k2 - hc
  · have hc := e_261_19
    simp only [e_261_17, e_260_19, e_261_18] at hc
    linear_combination c17158.trans k2 - hc

theorem publicBatchWrapper89_f2400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 11) = bor (a (.wire 46 47)) (a (.virt 19085)) := by
  have c17159 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.1
  have c17160 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.1
  have c17161 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.1
  have c17162 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17163 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17164 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_10
  simp only [← c17159, ← c17160, ← c17161] at e_257_10
  have e_262_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_2
  simp only [← c17162, ← c17163, k3, ← c17164] at e_262_2
  have hr := e_262_2
  simp only [e_257_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 11) = a (.virt 201) := by
  have c17165 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17165

theorem publicBatchWrapper89_f2402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9405)) (a (.wire 169 63)) (a (.virt 19087)) (a (.virt 19088)) := by
  have c17166 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17167 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17168 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17169 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17170 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17171 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17172 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17173 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17174 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17175 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17176 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17177 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17178 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17179 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17180 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17181 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17182 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_263_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_0
  simp only [← c17166, k3, ← c17167, k3, ← c17168] at e_263_0
  have e_263_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_1
  simp only [← c17169, ← c17170, k3, ← c17171] at e_263_1
  have e_263_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_2
  simp only [← c17178, ← c17179, k3, ← c17180] at e_263_2
  have e_264_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_0
  simp only [← c17172, ← c17173, ← c17174] at e_264_0
  have e_264_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_1
  simp only [← c17175, ← c17176, ← c17177] at e_264_1
  refine ⟨?_, ?_⟩
  · have hc := e_264_0
    simp only [e_263_1] at hc
    linear_combination c17181.trans k2 - hc
  · have hc := e_263_2
    simp only [e_263_0, e_264_1, e_263_1] at hc
    linear_combination c17182.trans k2 - hc

theorem publicBatchWrapper89_f2403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9406)) (a (.wire 169 71)) (a (.virt 19089)) (a (.virt 19090)) := by
  have c17183 := (publicBatchWrapper89_copies536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17184 := (publicBatchWrapper89_copies537 a h).1
  have c17185 := (publicBatchWrapper89_copies537 a h).2.1
  have c17186 := (publicBatchWrapper89_copies537 a h).2.2.1
  have c17187 := (publicBatchWrapper89_copies537 a h).2.2.2.1
  have c17188 := (publicBatchWrapper89_copies537 a h).2.2.2.2.1
  have c17189 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.1
  have c17190 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.1
  have c17191 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.1
  have c17192 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.1
  have c17193 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.1
  have c17194 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17195 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17196 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17197 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17198 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17199 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_263_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_3
  simp only [← c17183, k3, ← c17184, k3, ← c17185] at e_263_3
  have e_263_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_4
  simp only [← c17186, ← c17187, k3, ← c17188] at e_263_4
  have e_263_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_5
  simp only [← c17195, ← c17196, k3, ← c17197] at e_263_5
  have e_264_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_2
  simp only [← c17189, ← c17190, ← c17191] at e_264_2
  have e_264_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_3
  simp only [← c17192, ← c17193, ← c17194] at e_264_3
  refine ⟨?_, ?_⟩
  · have hc := e_264_2
    simp only [e_263_4] at hc
    linear_combination c17198.trans k2 - hc
  · have hc := e_263_5
    simp only [e_263_3, e_264_3, e_263_4] at hc
    linear_combination c17199.trans k2 - hc

theorem publicBatchWrapper89_f2404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9407)) (a (.wire 169 79)) (a (.virt 19091)) (a (.virt 19092)) := by
  have c17200 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17201 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17202 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17203 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17204 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17205 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17206 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17207 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17208 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17209 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17210 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17211 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17212 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17213 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17214 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17215 := (publicBatchWrapper89_copies537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17216 := (publicBatchWrapper89_copies538 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_263_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_6
  simp only [← c17200, k3, ← c17201, k3, ← c17202] at e_263_6
  have e_263_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_7
  simp only [← c17203, ← c17204, k3, ← c17205] at e_263_7
  have e_263_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_8
  simp only [← c17212, ← c17213, k3, ← c17214] at e_263_8
  have e_264_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_4
  simp only [← c17206, ← c17207, ← c17208] at e_264_4
  have e_264_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_5
  simp only [← c17209, ← c17210, ← c17211] at e_264_5
  refine ⟨?_, ?_⟩
  · have hc := e_264_4
    simp only [e_263_7] at hc
    linear_combination c17215.trans k2 - hc
  · have hc := e_263_8
    simp only [e_263_6, e_264_5, e_263_7] at hc
    linear_combination c17216.trans k2 - hc

theorem publicBatchWrapper89_f2405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9408)) (a (.wire 170 7)) (a (.virt 19093)) (a (.virt 19094)) := by
  have c17217 := (publicBatchWrapper89_copies538 a h).2.1
  have c17218 := (publicBatchWrapper89_copies538 a h).2.2.1
  have c17219 := (publicBatchWrapper89_copies538 a h).2.2.2.1
  have c17220 := (publicBatchWrapper89_copies538 a h).2.2.2.2.1
  have c17221 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.1
  have c17222 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.1
  have c17223 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.1
  have c17224 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.1
  have c17225 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.1
  have c17226 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17227 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17228 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17229 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17230 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17231 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17232 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17233 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_263_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_9
  simp only [← c17217, k3, ← c17218, k3, ← c17219] at e_263_9
  have e_263_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_10
  simp only [← c17220, ← c17221, k3, ← c17222] at e_263_10
  have e_263_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_11
  simp only [← c17229, ← c17230, k3, ← c17231] at e_263_11
  have e_264_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_6
  simp only [← c17223, ← c17224, ← c17225] at e_264_6
  have e_264_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_7
  simp only [← c17226, ← c17227, ← c17228] at e_264_7
  refine ⟨?_, ?_⟩
  · have hc := e_264_6
    simp only [e_263_10] at hc
    linear_combination c17232.trans k2 - hc
  · have hc := e_263_11
    simp only [e_263_9, e_264_7, e_263_10] at hc
    linear_combination c17233.trans k2 - hc

theorem publicBatchWrapper89_f2406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 264 35) = band (a (.virt 19087)) (a (.virt 19089)) := by
  have c17234 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17235 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17236 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_264_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_8
  simp only [← c17234, ← c17235, ← c17236] at e_264_8
  have hr := e_264_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 264 39) = band (a (.virt 19091)) (a (.virt 19093)) := by
  have c17237 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17238 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17239 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_264_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_9
  simp only [← c17237, ← c17238, ← c17239] at e_264_9
  have hr := e_264_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 264 43) = band (a (.wire 264 35)) (a (.wire 264 39)) := by
  have c17240 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17241 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17242 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_264_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_10
  simp only [← c17240, ← c17241, ← c17242] at e_264_10
  have hr := e_264_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 15) = bor (a (.wire 46 47)) (a (.wire 264 43)) := by
  have c17243 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17244 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17245 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17246 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17247 := (publicBatchWrapper89_copies538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17248 := (publicBatchWrapper89_copies539 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_11
  simp only [← c17243, ← c17244, ← c17245] at e_257_11
  have e_262_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_3
  simp only [← c17246, ← c17247, k3, ← c17248] at e_262_3
  have hr := e_262_3
  simp only [e_257_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 15) = a (.virt 201) := by
  have c17249 := (publicBatchWrapper89_copies539 a h).2.1
  exact c17249

theorem publicBatchWrapper89_f2411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9603)) (a (.wire 170 23)) (a (.virt 19095)) (a (.virt 19096)) := by
  have c17250 := (publicBatchWrapper89_copies539 a h).2.2.1
  have c17251 := (publicBatchWrapper89_copies539 a h).2.2.2.1
  have c17252 := (publicBatchWrapper89_copies539 a h).2.2.2.2.1
  have c17253 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.1
  have c17254 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.1
  have c17255 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.1
  have c17256 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.1
  have c17257 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.1
  have c17258 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17259 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17260 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17261 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17262 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17263 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17264 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17265 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17266 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_263_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_12
  simp only [← c17250, k3, ← c17251, k3, ← c17252] at e_263_12
  have e_263_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_13
  simp only [← c17253, ← c17254, k3, ← c17255] at e_263_13
  have e_263_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_14
  simp only [← c17262, ← c17263, k3, ← c17264] at e_263_14
  have e_264_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_11
  simp only [← c17256, ← c17257, ← c17258] at e_264_11
  have e_264_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_12
  simp only [← c17259, ← c17260, ← c17261] at e_264_12
  refine ⟨?_, ?_⟩
  · have hc := e_264_11
    simp only [e_263_13] at hc
    linear_combination c17265.trans k2 - hc
  · have hc := e_263_14
    simp only [e_263_12, e_264_12, e_263_13] at hc
    linear_combination c17266.trans k2 - hc

theorem publicBatchWrapper89_f2412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 19) = bor (a (.wire 48 11)) (a (.virt 19095)) := by
  have c17267 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17268 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17269 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17270 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17271 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17272 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_12
  simp only [← c17267, ← c17268, ← c17269] at e_257_12
  have e_262_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_4
  simp only [← c17270, ← c17271, k3, ← c17272] at e_262_4
  have hr := e_262_4
  simp only [e_257_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 19) = a (.virt 201) := by
  have c17273 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17273

theorem publicBatchWrapper89_f2414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9604)) (a (.wire 170 31)) (a (.virt 19097)) (a (.virt 19098)) := by
  have c17274 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17275 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17276 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17277 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17278 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17279 := (publicBatchWrapper89_copies539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17280 := (publicBatchWrapper89_copies540 a h).1
  have c17281 := (publicBatchWrapper89_copies540 a h).2.1
  have c17282 := (publicBatchWrapper89_copies540 a h).2.2.1
  have c17283 := (publicBatchWrapper89_copies540 a h).2.2.2.1
  have c17284 := (publicBatchWrapper89_copies540 a h).2.2.2.2.1
  have c17285 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.1
  have c17286 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.1
  have c17287 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.1
  have c17288 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.1
  have c17289 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.1
  have c17290 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_263_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_15
  simp only [← c17274, k3, ← c17275, k3, ← c17276] at e_263_15
  have e_263_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_16
  simp only [← c17277, ← c17278, k3, ← c17279] at e_263_16
  have e_263_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_17
  simp only [← c17286, ← c17287, k3, ← c17288] at e_263_17
  have e_264_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_13
  simp only [← c17280, ← c17281, ← c17282] at e_264_13
  have e_264_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_14
  simp only [← c17283, ← c17284, ← c17285] at e_264_14
  refine ⟨?_, ?_⟩
  · have hc := e_264_13
    simp only [e_263_16] at hc
    linear_combination c17289.trans k2 - hc
  · have hc := e_263_17
    simp only [e_263_15, e_264_14, e_263_16] at hc
    linear_combination c17290.trans k2 - hc

theorem publicBatchWrapper89_f2415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 23) = bor (a (.wire 48 11)) (a (.virt 19097)) := by
  have c17291 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17292 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17293 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17294 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17295 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17296 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_13
  simp only [← c17291, ← c17292, ← c17293] at e_257_13
  have e_262_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_5
  simp only [← c17294, ← c17295, k3, ← c17296] at e_262_5
  have hr := e_262_5
  simp only [e_257_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 23) = a (.virt 201) := by
  have c17297 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17297

theorem publicBatchWrapper89_f2417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9605)) (a (.wire 169 63)) (a (.virt 19099)) (a (.virt 19100)) := by
  have c17298 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17299 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17300 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17301 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17302 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17303 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17304 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17305 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17306 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17307 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17308 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17309 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17310 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17311 := (publicBatchWrapper89_copies540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17312 := (publicBatchWrapper89_copies541 a h).1
  have c17313 := (publicBatchWrapper89_copies541 a h).2.1
  have c17314 := (publicBatchWrapper89_copies541 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_263_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_18
  simp only [← c17298, k3, ← c17299, k3, ← c17300] at e_263_18
  have e_263_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row263 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_263_19
  simp only [← c17301, ← c17302, k3, ← c17303] at e_263_19
  have e_264_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_15
  simp only [← c17304, ← c17305, ← c17306] at e_264_15
  have e_264_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_16
  simp only [← c17307, ← c17308, ← c17309] at e_264_16
  have e_265_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_0
  simp only [← c17310, ← c17311, k3, ← c17312] at e_265_0
  refine ⟨?_, ?_⟩
  · have hc := e_264_15
    simp only [e_263_19] at hc
    linear_combination c17313.trans k2 - hc
  · have hc := e_265_0
    simp only [e_263_18, e_264_16, e_263_19] at hc
    linear_combination c17314.trans k2 - hc

theorem publicBatchWrapper89_f2418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9606)) (a (.wire 169 71)) (a (.virt 19101)) (a (.virt 19102)) := by
  have c17315 := (publicBatchWrapper89_copies541 a h).2.2.2.1
  have c17316 := (publicBatchWrapper89_copies541 a h).2.2.2.2.1
  have c17317 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.1
  have c17318 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.1
  have c17319 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.1
  have c17320 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.1
  have c17321 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.1
  have c17322 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17323 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17324 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17325 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17326 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17327 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17328 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17329 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17330 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17331 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_264_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_17
  simp only [← c17321, ← c17322, ← c17323] at e_264_17
  have e_264_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_18
  simp only [← c17324, ← c17325, ← c17326] at e_264_18
  have e_265_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_1
  simp only [← c17315, k3, ← c17316, k3, ← c17317] at e_265_1
  have e_265_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_2
  simp only [← c17318, ← c17319, k3, ← c17320] at e_265_2
  have e_265_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_3
  simp only [← c17327, ← c17328, k3, ← c17329] at e_265_3
  refine ⟨?_, ?_⟩
  · have hc := e_264_17
    simp only [e_265_2] at hc
    linear_combination c17330.trans k2 - hc
  · have hc := e_265_3
    simp only [e_265_1, e_264_18, e_265_2] at hc
    linear_combination c17331.trans k2 - hc

theorem publicBatchWrapper89_f2419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9607)) (a (.wire 169 79)) (a (.virt 19103)) (a (.virt 19104)) := by
  have c17332 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17333 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17334 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17335 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17336 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17337 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17338 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17339 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17340 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17341 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17342 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17343 := (publicBatchWrapper89_copies541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17344 := (publicBatchWrapper89_copies542 a h).1
  have c17345 := (publicBatchWrapper89_copies542 a h).2.1
  have c17346 := (publicBatchWrapper89_copies542 a h).2.2.1
  have c17347 := (publicBatchWrapper89_copies542 a h).2.2.2.1
  have c17348 := (publicBatchWrapper89_copies542 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_264_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row264 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_264_19
  simp only [← c17338, ← c17339, ← c17340] at e_264_19
  have e_265_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_4
  simp only [← c17332, k3, ← c17333, k3, ← c17334] at e_265_4
  have e_265_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_5
  simp only [← c17335, ← c17336, k3, ← c17337] at e_265_5
  have e_265_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_6
  simp only [← c17344, ← c17345, k3, ← c17346] at e_265_6
  have e_266_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_0
  simp only [← c17341, ← c17342, ← c17343] at e_266_0
  refine ⟨?_, ?_⟩
  · have hc := e_264_19
    simp only [e_265_5] at hc
    linear_combination c17347.trans k2 - hc
  · have hc := e_265_6
    simp only [e_265_4, e_266_0, e_265_5] at hc
    linear_combination c17348.trans k2 - hc

theorem publicBatchWrapper89_f2420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9608)) (a (.wire 170 7)) (a (.virt 19105)) (a (.virt 19106)) := by
  have c17349 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.1
  have c17350 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.1
  have c17351 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.1
  have c17352 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.1
  have c17353 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.1
  have c17354 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17355 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17356 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17357 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17358 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17359 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17360 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17361 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17362 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17363 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17364 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17365 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_265_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_7
  simp only [← c17349, k3, ← c17350, k3, ← c17351] at e_265_7
  have e_265_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_8
  simp only [← c17352, ← c17353, k3, ← c17354] at e_265_8
  have e_265_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_9
  simp only [← c17361, ← c17362, k3, ← c17363] at e_265_9
  have e_266_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_1
  simp only [← c17355, ← c17356, ← c17357] at e_266_1
  have e_266_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_2
  simp only [← c17358, ← c17359, ← c17360] at e_266_2
  refine ⟨?_, ?_⟩
  · have hc := e_266_1
    simp only [e_265_8] at hc
    linear_combination c17364.trans k2 - hc
  · have hc := e_265_9
    simp only [e_265_7, e_266_2, e_265_8] at hc
    linear_combination c17365.trans k2 - hc

theorem publicBatchWrapper89_f2421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 266 15) = band (a (.virt 19099)) (a (.virt 19101)) := by
  have c17366 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17367 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17368 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_266_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_3
  simp only [← c17366, ← c17367, ← c17368] at e_266_3
  have hr := e_266_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 266 19) = band (a (.virt 19103)) (a (.virt 19105)) := by
  have c17369 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17370 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17371 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_266_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_4
  simp only [← c17369, ← c17370, ← c17371] at e_266_4
  have hr := e_266_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 266 23) = band (a (.wire 266 15)) (a (.wire 266 19)) := by
  have c17372 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17373 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17374 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_266_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_5
  simp only [← c17372, ← c17373, ← c17374] at e_266_5
  have hr := e_266_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 27) = bor (a (.wire 48 11)) (a (.wire 266 23)) := by
  have c17375 := (publicBatchWrapper89_copies542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17376 := (publicBatchWrapper89_copies543 a h).1
  have c17377 := (publicBatchWrapper89_copies543 a h).2.1
  have c17378 := (publicBatchWrapper89_copies543 a h).2.2.1
  have c17379 := (publicBatchWrapper89_copies543 a h).2.2.2.1
  have c17380 := (publicBatchWrapper89_copies543 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_14
  simp only [← c17375, ← c17376, ← c17377] at e_257_14
  have e_262_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_6
  simp only [← c17378, ← c17379, k3, ← c17380] at e_262_6
  have hr := e_262_6
  simp only [e_257_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 27) = a (.virt 201) := by
  have c17381 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.1
  exact c17381

theorem publicBatchWrapper89_f2426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9803)) (a (.wire 170 23)) (a (.virt 19107)) (a (.virt 19108)) := by
  have c17382 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.1
  have c17383 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.1
  have c17384 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.1
  have c17385 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.1
  have c17386 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17387 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17388 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17389 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17390 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17391 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17392 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17393 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17394 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17395 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17396 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17397 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17398 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_265_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_10
  simp only [← c17382, k3, ← c17383, k3, ← c17384] at e_265_10
  have e_265_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_11
  simp only [← c17385, ← c17386, k3, ← c17387] at e_265_11
  have e_265_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_12
  simp only [← c17394, ← c17395, k3, ← c17396] at e_265_12
  have e_266_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_6
  simp only [← c17388, ← c17389, ← c17390] at e_266_6
  have e_266_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_7
  simp only [← c17391, ← c17392, ← c17393] at e_266_7
  refine ⟨?_, ?_⟩
  · have hc := e_266_6
    simp only [e_265_11] at hc
    linear_combination c17397.trans k2 - hc
  · have hc := e_265_12
    simp only [e_265_10, e_266_7, e_265_11] at hc
    linear_combination c17398.trans k2 - hc

theorem publicBatchWrapper89_f2427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 31) = bor (a (.wire 48 55)) (a (.virt 19107)) := by
  have c17399 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17400 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17401 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17402 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17403 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17404 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_15
  simp only [← c17399, ← c17400, ← c17401] at e_257_15
  have e_262_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_7
  simp only [← c17402, ← c17403, k3, ← c17404] at e_262_7
  have hr := e_262_7
  simp only [e_257_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 31) = a (.virt 201) := by
  have c17405 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17405

theorem publicBatchWrapper89_f2429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9804)) (a (.wire 170 31)) (a (.virt 19109)) (a (.virt 19110)) := by
  have c17406 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17407 := (publicBatchWrapper89_copies543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17408 := (publicBatchWrapper89_copies544 a h).1
  have c17409 := (publicBatchWrapper89_copies544 a h).2.1
  have c17410 := (publicBatchWrapper89_copies544 a h).2.2.1
  have c17411 := (publicBatchWrapper89_copies544 a h).2.2.2.1
  have c17412 := (publicBatchWrapper89_copies544 a h).2.2.2.2.1
  have c17413 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.1
  have c17414 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.1
  have c17415 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.1
  have c17416 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.1
  have c17417 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.1
  have c17418 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17419 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17420 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17421 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17422 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_265_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_13
  simp only [← c17406, k3, ← c17407, k3, ← c17408] at e_265_13
  have e_265_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_14
  simp only [← c17409, ← c17410, k3, ← c17411] at e_265_14
  have e_265_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_15
  simp only [← c17418, ← c17419, k3, ← c17420] at e_265_15
  have e_266_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_8
  simp only [← c17412, ← c17413, ← c17414] at e_266_8
  have e_266_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_9
  simp only [← c17415, ← c17416, ← c17417] at e_266_9
  refine ⟨?_, ?_⟩
  · have hc := e_266_8
    simp only [e_265_14] at hc
    linear_combination c17421.trans k2 - hc
  · have hc := e_265_15
    simp only [e_265_13, e_266_9, e_265_14] at hc
    linear_combination c17422.trans k2 - hc

theorem publicBatchWrapper89_f2430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 35) = bor (a (.wire 48 55)) (a (.virt 19109)) := by
  have c17423 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17424 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17425 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17426 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17427 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17428 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_16
  simp only [← c17423, ← c17424, ← c17425] at e_257_16
  have e_262_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_8
  simp only [← c17426, ← c17427, k3, ← c17428] at e_262_8
  have hr := e_262_8
  simp only [e_257_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 35) = a (.virt 201) := by
  have c17429 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17429

theorem publicBatchWrapper89_f2432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9805)) (a (.wire 169 63)) (a (.virt 19111)) (a (.virt 19112)) := by
  have c17430 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17431 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17432 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17433 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17434 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17435 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17436 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17437 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17438 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17439 := (publicBatchWrapper89_copies544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17440 := (publicBatchWrapper89_copies545 a h).1
  have c17441 := (publicBatchWrapper89_copies545 a h).2.1
  have c17442 := (publicBatchWrapper89_copies545 a h).2.2.1
  have c17443 := (publicBatchWrapper89_copies545 a h).2.2.2.1
  have c17444 := (publicBatchWrapper89_copies545 a h).2.2.2.2.1
  have c17445 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.1
  have c17446 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_265_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_16
  simp only [← c17430, k3, ← c17431, k3, ← c17432] at e_265_16
  have e_265_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_17
  simp only [← c17433, ← c17434, k3, ← c17435] at e_265_17
  have e_265_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_18
  simp only [← c17442, ← c17443, k3, ← c17444] at e_265_18
  have e_266_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_10
  simp only [← c17436, ← c17437, ← c17438] at e_266_10
  have e_266_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_11
  simp only [← c17439, ← c17440, ← c17441] at e_266_11
  refine ⟨?_, ?_⟩
  · have hc := e_266_10
    simp only [e_265_17] at hc
    linear_combination c17445.trans k2 - hc
  · have hc := e_265_18
    simp only [e_265_16, e_266_11, e_265_17] at hc
    linear_combination c17446.trans k2 - hc

theorem publicBatchWrapper89_f2433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9806)) (a (.wire 169 71)) (a (.virt 19113)) (a (.virt 19114)) := by
  have c17447 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.1
  have c17448 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.1
  have c17449 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.1
  have c17450 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17451 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17452 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17453 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17454 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17455 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17456 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17457 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17458 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17459 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17460 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17461 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17462 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17463 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_265_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row265 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_265_19
  simp only [← c17447, k3, ← c17448, k3, ← c17449] at e_265_19
  have e_266_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_12
  simp only [← c17453, ← c17454, ← c17455] at e_266_12
  have e_266_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_13
  simp only [← c17456, ← c17457, ← c17458] at e_266_13
  have e_267_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_0
  simp only [← c17450, ← c17451, k3, ← c17452] at e_267_0
  have e_267_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_1
  simp only [← c17459, ← c17460, k3, ← c17461] at e_267_1
  refine ⟨?_, ?_⟩
  · have hc := e_266_12
    simp only [e_267_0] at hc
    linear_combination c17462.trans k2 - hc
  · have hc := e_267_1
    simp only [e_265_19, e_266_13, e_267_0] at hc
    linear_combination c17463.trans k2 - hc

theorem publicBatchWrapper89_f2434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9807)) (a (.wire 169 79)) (a (.virt 19115)) (a (.virt 19116)) := by
  have c17464 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17465 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17466 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17467 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17468 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17469 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17470 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17471 := (publicBatchWrapper89_copies545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17472 := (publicBatchWrapper89_copies546 a h).1
  have c17473 := (publicBatchWrapper89_copies546 a h).2.1
  have c17474 := (publicBatchWrapper89_copies546 a h).2.2.1
  have c17475 := (publicBatchWrapper89_copies546 a h).2.2.2.1
  have c17476 := (publicBatchWrapper89_copies546 a h).2.2.2.2.1
  have c17477 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.1
  have c17478 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.1
  have c17479 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.1
  have c17480 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_266_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_14
  simp only [← c17470, ← c17471, ← c17472] at e_266_14
  have e_266_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_15
  simp only [← c17473, ← c17474, ← c17475] at e_266_15
  have e_267_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_2
  simp only [← c17464, k3, ← c17465, k3, ← c17466] at e_267_2
  have e_267_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_3
  simp only [← c17467, ← c17468, k3, ← c17469] at e_267_3
  have e_267_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_4
  simp only [← c17476, ← c17477, k3, ← c17478] at e_267_4
  refine ⟨?_, ?_⟩
  · have hc := e_266_14
    simp only [e_267_3] at hc
    linear_combination c17479.trans k2 - hc
  · have hc := e_267_4
    simp only [e_267_2, e_266_15, e_267_3] at hc
    linear_combination c17480.trans k2 - hc

theorem publicBatchWrapper89_f2435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9808)) (a (.wire 170 7)) (a (.virt 19117)) (a (.virt 19118)) := by
  have c17481 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.1
  have c17482 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17483 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17484 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17485 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17486 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17487 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17488 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17489 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17490 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17491 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17492 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17493 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17494 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17495 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17496 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17497 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_266_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_16
  simp only [← c17487, ← c17488, ← c17489] at e_266_16
  have e_266_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_17
  simp only [← c17490, ← c17491, ← c17492] at e_266_17
  have e_267_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_5
  simp only [← c17481, k3, ← c17482, k3, ← c17483] at e_267_5
  have e_267_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_6
  simp only [← c17484, ← c17485, k3, ← c17486] at e_267_6
  have e_267_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_7
  simp only [← c17493, ← c17494, k3, ← c17495] at e_267_7
  refine ⟨?_, ?_⟩
  · have hc := e_266_16
    simp only [e_267_6] at hc
    linear_combination c17496.trans k2 - hc
  · have hc := e_267_7
    simp only [e_267_5, e_266_17, e_267_6] at hc
    linear_combination c17497.trans k2 - hc

theorem publicBatchWrapper89_f2436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 266 75) = band (a (.virt 19111)) (a (.virt 19113)) := by
  have c17498 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17499 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17500 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_266_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_18
  simp only [← c17498, ← c17499, ← c17500] at e_266_18
  have hr := e_266_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 266 79) = band (a (.virt 19115)) (a (.virt 19117)) := by
  have c17501 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17502 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17503 := (publicBatchWrapper89_copies546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_266_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row266 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_266_19
  simp only [← c17501, ← c17502, ← c17503] at e_266_19
  have hr := e_266_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 268 3) = band (a (.wire 266 75)) (a (.wire 266 79)) := by
  have c17504 := (publicBatchWrapper89_copies547 a h).1
  have c17505 := (publicBatchWrapper89_copies547 a h).2.1
  have c17506 := (publicBatchWrapper89_copies547 a h).2.2.1
  have e_268_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_0
  simp only [← c17504, ← c17505, ← c17506] at e_268_0
  have hr := e_268_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 39) = bor (a (.wire 48 55)) (a (.wire 268 3)) := by
  have c17507 := (publicBatchWrapper89_copies547 a h).2.2.2.1
  have c17508 := (publicBatchWrapper89_copies547 a h).2.2.2.2.1
  have c17509 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.1
  have c17510 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.1
  have c17511 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.1
  have c17512 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_17
  simp only [← c17507, ← c17508, ← c17509] at e_257_17
  have e_262_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_9
  simp only [← c17510, ← c17511, k3, ← c17512] at e_262_9
  have hr := e_262_9
  simp only [e_257_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 39) = a (.virt 201) := by
  have c17513 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.1
  exact c17513

theorem publicBatchWrapper89_f2441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10003)) (a (.wire 170 23)) (a (.virt 19119)) (a (.virt 19120)) := by
  have c17514 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17515 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17516 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17517 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17518 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17519 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17520 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17521 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17522 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17523 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17524 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17525 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17526 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17527 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17528 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17529 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17530 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_267_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_8
  simp only [← c17514, k3, ← c17515, k3, ← c17516] at e_267_8
  have e_267_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_9
  simp only [← c17517, ← c17518, k3, ← c17519] at e_267_9
  have e_267_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_10
  simp only [← c17526, ← c17527, k3, ← c17528] at e_267_10
  have e_268_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_1
  simp only [← c17520, ← c17521, ← c17522] at e_268_1
  have e_268_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_2
  simp only [← c17523, ← c17524, ← c17525] at e_268_2
  refine ⟨?_, ?_⟩
  · have hc := e_268_1
    simp only [e_267_9] at hc
    linear_combination c17529.trans k2 - hc
  · have hc := e_267_10
    simp only [e_267_8, e_268_2, e_267_9] at hc
    linear_combination c17530.trans k2 - hc

theorem publicBatchWrapper89_f2442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 43) = bor (a (.wire 50 19)) (a (.virt 19119)) := by
  have c17531 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17532 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17533 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17534 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17535 := (publicBatchWrapper89_copies547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17536 := (publicBatchWrapper89_copies548 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_18
  simp only [← c17531, ← c17532, ← c17533] at e_257_18
  have e_262_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_10
  simp only [← c17534, ← c17535, k3, ← c17536] at e_262_10
  have hr := e_262_10
  simp only [e_257_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 43) = a (.virt 201) := by
  have c17537 := (publicBatchWrapper89_copies548 a h).2.1
  exact c17537

theorem publicBatchWrapper89_f2444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10004)) (a (.wire 170 31)) (a (.virt 19121)) (a (.virt 19122)) := by
  have c17538 := (publicBatchWrapper89_copies548 a h).2.2.1
  have c17539 := (publicBatchWrapper89_copies548 a h).2.2.2.1
  have c17540 := (publicBatchWrapper89_copies548 a h).2.2.2.2.1
  have c17541 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.1
  have c17542 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.1
  have c17543 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.1
  have c17544 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.1
  have c17545 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.1
  have c17546 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17547 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17548 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17549 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17550 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17551 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17552 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17553 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17554 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_267_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_11
  simp only [← c17538, k3, ← c17539, k3, ← c17540] at e_267_11
  have e_267_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_12
  simp only [← c17541, ← c17542, k3, ← c17543] at e_267_12
  have e_267_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_13
  simp only [← c17550, ← c17551, k3, ← c17552] at e_267_13
  have e_268_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_3
  simp only [← c17544, ← c17545, ← c17546] at e_268_3
  have e_268_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_4
  simp only [← c17547, ← c17548, ← c17549] at e_268_4
  refine ⟨?_, ?_⟩
  · have hc := e_268_3
    simp only [e_267_12] at hc
    linear_combination c17553.trans k2 - hc
  · have hc := e_267_13
    simp only [e_267_11, e_268_4, e_267_12] at hc
    linear_combination c17554.trans k2 - hc

theorem publicBatchWrapper89_f2445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 47) = bor (a (.wire 50 19)) (a (.virt 19121)) := by
  have c17555 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17556 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17557 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17558 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17559 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17560 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_257_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row257 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_257_19
  simp only [← c17555, ← c17556, ← c17557] at e_257_19
  have e_262_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_11
  simp only [← c17558, ← c17559, k3, ← c17560] at e_262_11
  have hr := e_262_11
  simp only [e_257_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 47) = a (.virt 201) := by
  have c17561 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17561

theorem publicBatchWrapper89_f2447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10005)) (a (.wire 169 63)) (a (.virt 19123)) (a (.virt 19124)) := by
  have c17562 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17563 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17564 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17565 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17566 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17567 := (publicBatchWrapper89_copies548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17568 := (publicBatchWrapper89_copies549 a h).1
  have c17569 := (publicBatchWrapper89_copies549 a h).2.1
  have c17570 := (publicBatchWrapper89_copies549 a h).2.2.1
  have c17571 := (publicBatchWrapper89_copies549 a h).2.2.2.1
  have c17572 := (publicBatchWrapper89_copies549 a h).2.2.2.2.1
  have c17573 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.1
  have c17574 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.1
  have c17575 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.1
  have c17576 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.1
  have c17577 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.1
  have c17578 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_267_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_14
  simp only [← c17562, k3, ← c17563, k3, ← c17564] at e_267_14
  have e_267_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_15
  simp only [← c17565, ← c17566, k3, ← c17567] at e_267_15
  have e_267_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_16
  simp only [← c17574, ← c17575, k3, ← c17576] at e_267_16
  have e_268_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_5
  simp only [← c17568, ← c17569, ← c17570] at e_268_5
  have e_268_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_6
  simp only [← c17571, ← c17572, ← c17573] at e_268_6
  refine ⟨?_, ?_⟩
  · have hc := e_268_5
    simp only [e_267_15] at hc
    linear_combination c17577.trans k2 - hc
  · have hc := e_267_16
    simp only [e_267_14, e_268_6, e_267_15] at hc
    linear_combination c17578.trans k2 - hc

theorem publicBatchWrapper89_f2448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10006)) (a (.wire 169 71)) (a (.virt 19125)) (a (.virt 19126)) := by
  have c17579 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17580 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17581 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17582 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17583 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17584 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17585 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17586 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17587 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17588 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17589 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17590 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17591 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17592 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17593 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17594 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17595 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_267_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_17
  simp only [← c17579, k3, ← c17580, k3, ← c17581] at e_267_17
  have e_267_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_18
  simp only [← c17582, ← c17583, k3, ← c17584] at e_267_18
  have e_267_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row267 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_267_19
  simp only [← c17591, ← c17592, k3, ← c17593] at e_267_19
  have e_268_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_7
  simp only [← c17585, ← c17586, ← c17587] at e_268_7
  have e_268_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_8
  simp only [← c17588, ← c17589, ← c17590] at e_268_8
  refine ⟨?_, ?_⟩
  · have hc := e_268_7
    simp only [e_267_18] at hc
    linear_combination c17594.trans k2 - hc
  · have hc := e_267_19
    simp only [e_267_17, e_268_8, e_267_18] at hc
    linear_combination c17595.trans k2 - hc

theorem publicBatchWrapper89_f2449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10007)) (a (.wire 169 79)) (a (.virt 19127)) (a (.virt 19128)) := by
  have c17596 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17597 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17598 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17599 := (publicBatchWrapper89_copies549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17600 := (publicBatchWrapper89_copies550 a h).1
  have c17601 := (publicBatchWrapper89_copies550 a h).2.1
  have c17602 := (publicBatchWrapper89_copies550 a h).2.2.1
  have c17603 := (publicBatchWrapper89_copies550 a h).2.2.2.1
  have c17604 := (publicBatchWrapper89_copies550 a h).2.2.2.2.1
  have c17605 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.1
  have c17606 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.1
  have c17607 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.1
  have c17608 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.1
  have c17609 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.1
  have c17610 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17611 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17612 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_268_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_9
  simp only [← c17602, ← c17603, ← c17604] at e_268_9
  have e_268_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_10
  simp only [← c17605, ← c17606, ← c17607] at e_268_10
  have e_269_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_0
  simp only [← c17596, k3, ← c17597, k3, ← c17598] at e_269_0
  have e_269_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_1
  simp only [← c17599, ← c17600, k3, ← c17601] at e_269_1
  have e_269_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_2
  simp only [← c17608, ← c17609, k3, ← c17610] at e_269_2
  refine ⟨?_, ?_⟩
  · have hc := e_268_9
    simp only [e_269_1] at hc
    linear_combination c17611.trans k2 - hc
  · have hc := e_269_2
    simp only [e_269_0, e_268_10, e_269_1] at hc
    linear_combination c17612.trans k2 - hc

theorem publicBatchWrapper89_f2450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10008)) (a (.wire 170 7)) (a (.virt 19129)) (a (.virt 19130)) := by
  have c17613 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17614 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17615 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17616 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17617 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17618 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17619 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17620 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17621 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17622 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17623 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17624 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17625 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17626 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17627 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17628 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17629 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_268_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_11
  simp only [← c17619, ← c17620, ← c17621] at e_268_11
  have e_268_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_12
  simp only [← c17622, ← c17623, ← c17624] at e_268_12
  have e_269_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_3
  simp only [← c17613, k3, ← c17614, k3, ← c17615] at e_269_3
  have e_269_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_4
  simp only [← c17616, ← c17617, k3, ← c17618] at e_269_4
  have e_269_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_5
  simp only [← c17625, ← c17626, k3, ← c17627] at e_269_5
  refine ⟨?_, ?_⟩
  · have hc := e_268_11
    simp only [e_269_4] at hc
    linear_combination c17628.trans k2 - hc
  · have hc := e_269_5
    simp only [e_269_3, e_268_12, e_269_4] at hc
    linear_combination c17629.trans k2 - hc

theorem publicBatchWrapper89_f2451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 268 55) = band (a (.virt 19123)) (a (.virt 19125)) := by
  have c17630 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17631 := (publicBatchWrapper89_copies550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17632 := (publicBatchWrapper89_copies551 a h).1
  have e_268_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_13
  simp only [← c17630, ← c17631, ← c17632] at e_268_13
  have hr := e_268_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 268 59) = band (a (.virt 19127)) (a (.virt 19129)) := by
  have c17633 := (publicBatchWrapper89_copies551 a h).2.1
  have c17634 := (publicBatchWrapper89_copies551 a h).2.2.1
  have c17635 := (publicBatchWrapper89_copies551 a h).2.2.2.1
  have e_268_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_14
  simp only [← c17633, ← c17634, ← c17635] at e_268_14
  have hr := e_268_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 268 63) = band (a (.wire 268 55)) (a (.wire 268 59)) := by
  have c17636 := (publicBatchWrapper89_copies551 a h).2.2.2.2.1
  have c17637 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.1
  have c17638 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.1
  have e_268_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_15
  simp only [← c17636, ← c17637, ← c17638] at e_268_15
  have hr := e_268_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 51) = bor (a (.wire 50 19)) (a (.wire 268 63)) := by
  have c17639 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.1
  have c17640 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.1
  have c17641 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.1
  have c17642 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17643 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17644 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_262_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_12
  simp only [← c17642, ← c17643, k3, ← c17644] at e_262_12
  have e_270_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_0
  simp only [← c17639, ← c17640, ← c17641] at e_270_0
  have hr := e_262_12
  simp only [e_270_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 51) = a (.virt 201) := by
  have c17645 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17645

theorem publicBatchWrapper89_f2456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10203)) (a (.wire 170 23)) (a (.virt 19131)) (a (.virt 19132)) := by
  have c17646 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17647 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17648 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17649 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17650 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17651 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17652 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17653 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17654 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17655 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17656 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17657 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17658 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17659 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17660 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17661 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17662 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_268_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_16
  simp only [← c17652, ← c17653, ← c17654] at e_268_16
  have e_268_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_17
  simp only [← c17655, ← c17656, ← c17657] at e_268_17
  have e_269_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_6
  simp only [← c17646, k3, ← c17647, k3, ← c17648] at e_269_6
  have e_269_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_7
  simp only [← c17649, ← c17650, k3, ← c17651] at e_269_7
  have e_269_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_8
  simp only [← c17658, ← c17659, k3, ← c17660] at e_269_8
  refine ⟨?_, ?_⟩
  · have hc := e_268_16
    simp only [e_269_7] at hc
    linear_combination c17661.trans k2 - hc
  · have hc := e_269_8
    simp only [e_269_6, e_268_17, e_269_7] at hc
    linear_combination c17662.trans k2 - hc

theorem publicBatchWrapper89_f2457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 55) = bor (a (.wire 50 63)) (a (.virt 19131)) := by
  have c17663 := (publicBatchWrapper89_copies551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17664 := (publicBatchWrapper89_copies552 a h).1
  have c17665 := (publicBatchWrapper89_copies552 a h).2.1
  have c17666 := (publicBatchWrapper89_copies552 a h).2.2.1
  have c17667 := (publicBatchWrapper89_copies552 a h).2.2.2.1
  have c17668 := (publicBatchWrapper89_copies552 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_262_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_13
  simp only [← c17666, ← c17667, k3, ← c17668] at e_262_13
  have e_270_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_1
  simp only [← c17663, ← c17664, ← c17665] at e_270_1
  have hr := e_262_13
  simp only [e_270_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 55) = a (.virt 201) := by
  have c17669 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.1
  exact c17669

theorem publicBatchWrapper89_f2459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10204)) (a (.wire 170 31)) (a (.virt 19133)) (a (.virt 19134)) := by
  have c17670 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.1
  have c17671 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.1
  have c17672 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.1
  have c17673 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.1
  have c17674 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17675 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17676 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17677 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17678 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17679 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17680 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17681 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17682 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17683 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17684 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17685 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17686 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_268_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_18
  simp only [← c17676, ← c17677, ← c17678] at e_268_18
  have e_268_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row268 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_268_19
  simp only [← c17679, ← c17680, ← c17681] at e_268_19
  have e_269_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_9
  simp only [← c17670, k3, ← c17671, k3, ← c17672] at e_269_9
  have e_269_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_10
  simp only [← c17673, ← c17674, k3, ← c17675] at e_269_10
  have e_269_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_11
  simp only [← c17682, ← c17683, k3, ← c17684] at e_269_11
  refine ⟨?_, ?_⟩
  · have hc := e_268_18
    simp only [e_269_10] at hc
    linear_combination c17685.trans k2 - hc
  · have hc := e_269_11
    simp only [e_269_9, e_268_19, e_269_10] at hc
    linear_combination c17686.trans k2 - hc

theorem publicBatchWrapper89_f2460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 59) = bor (a (.wire 50 63)) (a (.virt 19133)) := by
  have c17687 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17688 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17689 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17690 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17691 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17692 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_262_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_14
  simp only [← c17690, ← c17691, k3, ← c17692] at e_262_14
  have e_270_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_2
  simp only [← c17687, ← c17688, ← c17689] at e_270_2
  have hr := e_262_14
  simp only [e_270_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 59) = a (.virt 201) := by
  have c17693 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17693

theorem publicBatchWrapper89_f2462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10205)) (a (.wire 169 63)) (a (.virt 19135)) (a (.virt 19136)) := by
  have c17694 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17695 := (publicBatchWrapper89_copies552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17696 := (publicBatchWrapper89_copies553 a h).1
  have c17697 := (publicBatchWrapper89_copies553 a h).2.1
  have c17698 := (publicBatchWrapper89_copies553 a h).2.2.1
  have c17699 := (publicBatchWrapper89_copies553 a h).2.2.2.1
  have c17700 := (publicBatchWrapper89_copies553 a h).2.2.2.2.1
  have c17701 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.1
  have c17702 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.1
  have c17703 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.1
  have c17704 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.1
  have c17705 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.1
  have c17706 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17707 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17708 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17709 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17710 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_269_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_12
  simp only [← c17694, k3, ← c17695, k3, ← c17696] at e_269_12
  have e_269_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_13
  simp only [← c17697, ← c17698, k3, ← c17699] at e_269_13
  have e_269_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_14
  simp only [← c17706, ← c17707, k3, ← c17708] at e_269_14
  have e_271_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_0
  simp only [← c17700, ← c17701, ← c17702] at e_271_0
  have e_271_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_1
  simp only [← c17703, ← c17704, ← c17705] at e_271_1
  refine ⟨?_, ?_⟩
  · have hc := e_271_0
    simp only [e_269_13] at hc
    linear_combination c17709.trans k2 - hc
  · have hc := e_269_14
    simp only [e_269_12, e_271_1, e_269_13] at hc
    linear_combination c17710.trans k2 - hc

theorem publicBatchWrapper89_f2463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10206)) (a (.wire 169 71)) (a (.virt 19137)) (a (.virt 19138)) := by
  have c17711 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17712 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17713 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17714 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17715 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17716 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17717 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17718 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17719 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17720 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17721 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17722 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17723 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17724 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17725 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17726 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17727 := (publicBatchWrapper89_copies553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_269_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_15
  simp only [← c17711, k3, ← c17712, k3, ← c17713] at e_269_15
  have e_269_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_16
  simp only [← c17714, ← c17715, k3, ← c17716] at e_269_16
  have e_269_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_17
  simp only [← c17723, ← c17724, k3, ← c17725] at e_269_17
  have e_271_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_2
  simp only [← c17717, ← c17718, ← c17719] at e_271_2
  have e_271_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_3
  simp only [← c17720, ← c17721, ← c17722] at e_271_3
  refine ⟨?_, ?_⟩
  · have hc := e_271_2
    simp only [e_269_16] at hc
    linear_combination c17726.trans k2 - hc
  · have hc := e_269_17
    simp only [e_269_15, e_271_3, e_269_16] at hc
    linear_combination c17727.trans k2 - hc

theorem publicBatchWrapper89_f2464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10207)) (a (.wire 169 79)) (a (.virt 19139)) (a (.virt 19140)) := by
  have c17728 := (publicBatchWrapper89_copies554 a h).1
  have c17729 := (publicBatchWrapper89_copies554 a h).2.1
  have c17730 := (publicBatchWrapper89_copies554 a h).2.2.1
  have c17731 := (publicBatchWrapper89_copies554 a h).2.2.2.1
  have c17732 := (publicBatchWrapper89_copies554 a h).2.2.2.2.1
  have c17733 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.1
  have c17734 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.1
  have c17735 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.1
  have c17736 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.1
  have c17737 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.1
  have c17738 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17739 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17740 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17741 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17742 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17743 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17744 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_269_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_18
  simp only [← c17728, k3, ← c17729, k3, ← c17730] at e_269_18
  have e_269_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row269 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_269_19
  simp only [← c17731, ← c17732, k3, ← c17733] at e_269_19
  have e_271_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_4
  simp only [← c17734, ← c17735, ← c17736] at e_271_4
  have e_271_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_5
  simp only [← c17737, ← c17738, ← c17739] at e_271_5
  have e_272_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_0
  simp only [← c17740, ← c17741, k3, ← c17742] at e_272_0
  refine ⟨?_, ?_⟩
  · have hc := e_271_4
    simp only [e_269_19] at hc
    linear_combination c17743.trans k2 - hc
  · have hc := e_272_0
    simp only [e_269_18, e_271_5, e_269_19] at hc
    linear_combination c17744.trans k2 - hc

theorem publicBatchWrapper89_f2465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10208)) (a (.wire 170 7)) (a (.virt 19141)) (a (.virt 19142)) := by
  have c17745 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17746 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17747 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17748 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17749 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17750 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17751 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17752 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17753 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17754 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17755 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17756 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17757 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17758 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17759 := (publicBatchWrapper89_copies554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17760 := (publicBatchWrapper89_copies555 a h).1
  have c17761 := (publicBatchWrapper89_copies555 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_271_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_6
  simp only [← c17751, ← c17752, ← c17753] at e_271_6
  have e_271_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_7
  simp only [← c17754, ← c17755, ← c17756] at e_271_7
  have e_272_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_1
  simp only [← c17745, k3, ← c17746, k3, ← c17747] at e_272_1
  have e_272_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_2
  simp only [← c17748, ← c17749, k3, ← c17750] at e_272_2
  have e_272_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_3
  simp only [← c17757, ← c17758, k3, ← c17759] at e_272_3
  refine ⟨?_, ?_⟩
  · have hc := e_271_6
    simp only [e_272_2] at hc
    linear_combination c17760.trans k2 - hc
  · have hc := e_272_3
    simp only [e_272_1, e_271_7, e_272_2] at hc
    linear_combination c17761.trans k2 - hc

theorem publicBatchWrapper89_f2466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 271 35) = band (a (.virt 19135)) (a (.virt 19137)) := by
  have c17762 := (publicBatchWrapper89_copies555 a h).2.2.1
  have c17763 := (publicBatchWrapper89_copies555 a h).2.2.2.1
  have c17764 := (publicBatchWrapper89_copies555 a h).2.2.2.2.1
  have e_271_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_8
  simp only [← c17762, ← c17763, ← c17764] at e_271_8
  have hr := e_271_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 271 39) = band (a (.virt 19139)) (a (.virt 19141)) := by
  have c17765 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.1
  have c17766 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.1
  have c17767 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.1
  have e_271_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_9
  simp only [← c17765, ← c17766, ← c17767] at e_271_9
  have hr := e_271_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 271 43) = band (a (.wire 271 35)) (a (.wire 271 39)) := by
  have c17768 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.1
  have c17769 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.1
  have c17770 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_271_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_10
  simp only [← c17768, ← c17769, ← c17770] at e_271_10
  have hr := e_271_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 63) = bor (a (.wire 50 63)) (a (.wire 271 43)) := by
  have c17771 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17772 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17773 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17774 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17775 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17776 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_262_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_15
  simp only [← c17774, ← c17775, k3, ← c17776] at e_262_15
  have e_270_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_3
  simp only [← c17771, ← c17772, ← c17773] at e_270_3
  have hr := e_262_15
  simp only [e_270_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 63) = a (.virt 201) := by
  have c17777 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17777

theorem publicBatchWrapper89_f2471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10403)) (a (.wire 170 23)) (a (.virt 19143)) (a (.virt 19144)) := by
  have c17778 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17779 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17780 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17781 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17782 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17783 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17784 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17785 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17786 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17787 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17788 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17789 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17790 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17791 := (publicBatchWrapper89_copies555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17792 := (publicBatchWrapper89_copies556 a h).1
  have c17793 := (publicBatchWrapper89_copies556 a h).2.1
  have c17794 := (publicBatchWrapper89_copies556 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_271_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_11
  simp only [← c17784, ← c17785, ← c17786] at e_271_11
  have e_271_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_12
  simp only [← c17787, ← c17788, ← c17789] at e_271_12
  have e_272_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_4
  simp only [← c17778, k3, ← c17779, k3, ← c17780] at e_272_4
  have e_272_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_5
  simp only [← c17781, ← c17782, k3, ← c17783] at e_272_5
  have e_272_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_6
  simp only [← c17790, ← c17791, k3, ← c17792] at e_272_6
  refine ⟨?_, ?_⟩
  · have hc := e_271_11
    simp only [e_272_5] at hc
    linear_combination c17793.trans k2 - hc
  · have hc := e_272_6
    simp only [e_272_4, e_271_12, e_272_5] at hc
    linear_combination c17794.trans k2 - hc

theorem publicBatchWrapper89_f2472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 67) = bor (a (.wire 52 27)) (a (.virt 19143)) := by
  have c17795 := (publicBatchWrapper89_copies556 a h).2.2.2.1
  have c17796 := (publicBatchWrapper89_copies556 a h).2.2.2.2.1
  have c17797 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.1
  have c17798 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.1
  have c17799 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.1
  have c17800 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_262_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_16
  simp only [← c17798, ← c17799, k3, ← c17800] at e_262_16
  have e_270_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_4
  simp only [← c17795, ← c17796, ← c17797] at e_270_4
  have hr := e_262_16
  simp only [e_270_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 67) = a (.virt 201) := by
  have c17801 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.1
  exact c17801

theorem publicBatchWrapper89_f2474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10404)) (a (.wire 170 31)) (a (.virt 19145)) (a (.virt 19146)) := by
  have c17802 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17803 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17804 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17805 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17806 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17807 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17808 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17809 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17810 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17811 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17812 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17813 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17814 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17815 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17816 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17817 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17818 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_271_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_13
  simp only [← c17808, ← c17809, ← c17810] at e_271_13
  have e_271_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_14
  simp only [← c17811, ← c17812, ← c17813] at e_271_14
  have e_272_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_7
  simp only [← c17802, k3, ← c17803, k3, ← c17804] at e_272_7
  have e_272_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_8
  simp only [← c17805, ← c17806, k3, ← c17807] at e_272_8
  have e_272_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_9
  simp only [← c17814, ← c17815, k3, ← c17816] at e_272_9
  refine ⟨?_, ?_⟩
  · have hc := e_271_13
    simp only [e_272_8] at hc
    linear_combination c17817.trans k2 - hc
  · have hc := e_272_9
    simp only [e_272_7, e_271_14, e_272_8] at hc
    linear_combination c17818.trans k2 - hc

theorem publicBatchWrapper89_f2475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 71) = bor (a (.wire 52 27)) (a (.virt 19145)) := by
  have c17819 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17820 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17821 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17822 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17823 := (publicBatchWrapper89_copies556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17824 := (publicBatchWrapper89_copies557 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_262_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_17
  simp only [← c17822, ← c17823, k3, ← c17824] at e_262_17
  have e_270_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_5
  simp only [← c17819, ← c17820, ← c17821] at e_270_5
  have hr := e_262_17
  simp only [e_270_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 71) = a (.virt 201) := by
  have c17825 := (publicBatchWrapper89_copies557 a h).2.1
  exact c17825

theorem publicBatchWrapper89_f2477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10405)) (a (.wire 169 63)) (a (.virt 19147)) (a (.virt 19148)) := by
  have c17826 := (publicBatchWrapper89_copies557 a h).2.2.1
  have c17827 := (publicBatchWrapper89_copies557 a h).2.2.2.1
  have c17828 := (publicBatchWrapper89_copies557 a h).2.2.2.2.1
  have c17829 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.1
  have c17830 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.1
  have c17831 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.1
  have c17832 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.1
  have c17833 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.1
  have c17834 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17835 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17836 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17837 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17838 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17839 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17840 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17841 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17842 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_271_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_15
  simp only [← c17832, ← c17833, ← c17834] at e_271_15
  have e_271_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_16
  simp only [← c17835, ← c17836, ← c17837] at e_271_16
  have e_272_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_10
  simp only [← c17826, k3, ← c17827, k3, ← c17828] at e_272_10
  have e_272_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_11
  simp only [← c17829, ← c17830, k3, ← c17831] at e_272_11
  have e_272_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_12
  simp only [← c17838, ← c17839, k3, ← c17840] at e_272_12
  refine ⟨?_, ?_⟩
  · have hc := e_271_15
    simp only [e_272_11] at hc
    linear_combination c17841.trans k2 - hc
  · have hc := e_272_12
    simp only [e_272_10, e_271_16, e_272_11] at hc
    linear_combination c17842.trans k2 - hc

theorem publicBatchWrapper89_f2478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10406)) (a (.wire 169 71)) (a (.virt 19149)) (a (.virt 19150)) := by
  have c17843 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17844 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17845 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17846 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17847 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17848 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17849 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17850 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17851 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17852 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17853 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17854 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17855 := (publicBatchWrapper89_copies557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17856 := (publicBatchWrapper89_copies558 a h).1
  have c17857 := (publicBatchWrapper89_copies558 a h).2.1
  have c17858 := (publicBatchWrapper89_copies558 a h).2.2.1
  have c17859 := (publicBatchWrapper89_copies558 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_271_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_17
  simp only [← c17849, ← c17850, ← c17851] at e_271_17
  have e_271_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_18
  simp only [← c17852, ← c17853, ← c17854] at e_271_18
  have e_272_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_13
  simp only [← c17843, k3, ← c17844, k3, ← c17845] at e_272_13
  have e_272_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_14
  simp only [← c17846, ← c17847, k3, ← c17848] at e_272_14
  have e_272_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_15
  simp only [← c17855, ← c17856, k3, ← c17857] at e_272_15
  refine ⟨?_, ?_⟩
  · have hc := e_271_17
    simp only [e_272_14] at hc
    linear_combination c17858.trans k2 - hc
  · have hc := e_272_15
    simp only [e_272_13, e_271_18, e_272_14] at hc
    linear_combination c17859.trans k2 - hc

theorem publicBatchWrapper89_f2479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10407)) (a (.wire 169 79)) (a (.virt 19151)) (a (.virt 19152)) := by
  have c17860 := (publicBatchWrapper89_copies558 a h).2.2.2.2.1
  have c17861 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.1
  have c17862 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.1
  have c17863 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.1
  have c17864 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.1
  have c17865 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.1
  have c17866 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17867 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17868 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17869 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17870 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17871 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17872 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17873 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17874 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17875 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17876 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_271_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row271 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_271_19
  simp only [← c17866, ← c17867, ← c17868] at e_271_19
  have e_272_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_16
  simp only [← c17860, k3, ← c17861, k3, ← c17862] at e_272_16
  have e_272_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_17
  simp only [← c17863, ← c17864, k3, ← c17865] at e_272_17
  have e_272_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_18
  simp only [← c17872, ← c17873, k3, ← c17874] at e_272_18
  have e_273_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_0
  simp only [← c17869, ← c17870, ← c17871] at e_273_0
  refine ⟨?_, ?_⟩
  · have hc := e_271_19
    simp only [e_272_17] at hc
    linear_combination c17875.trans k2 - hc
  · have hc := e_272_18
    simp only [e_272_16, e_273_0, e_272_17] at hc
    linear_combination c17876.trans k2 - hc

theorem publicBatchWrapper89_f2480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10408)) (a (.wire 170 7)) (a (.virt 19153)) (a (.virt 19154)) := by
  have c17877 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17878 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17879 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17880 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17881 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17882 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17883 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17884 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17885 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17886 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17887 := (publicBatchWrapper89_copies558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17888 := (publicBatchWrapper89_copies559 a h).1
  have c17889 := (publicBatchWrapper89_copies559 a h).2.1
  have c17890 := (publicBatchWrapper89_copies559 a h).2.2.1
  have c17891 := (publicBatchWrapper89_copies559 a h).2.2.2.1
  have c17892 := (publicBatchWrapper89_copies559 a h).2.2.2.2.1
  have c17893 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_272_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row272 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_272_19
  simp only [← c17877, k3, ← c17878, k3, ← c17879] at e_272_19
  have e_273_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_1
  simp only [← c17883, ← c17884, ← c17885] at e_273_1
  have e_273_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_2
  simp only [← c17886, ← c17887, ← c17888] at e_273_2
  have e_274_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_0
  simp only [← c17880, ← c17881, k3, ← c17882] at e_274_0
  have e_274_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_1
  simp only [← c17889, ← c17890, k3, ← c17891] at e_274_1
  refine ⟨?_, ?_⟩
  · have hc := e_273_1
    simp only [e_274_0] at hc
    linear_combination c17892.trans k2 - hc
  · have hc := e_274_1
    simp only [e_272_19, e_273_2, e_274_0] at hc
    linear_combination c17893.trans k2 - hc

theorem publicBatchWrapper89_f2481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 273 15) = band (a (.virt 19147)) (a (.virt 19149)) := by
  have c17894 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.1
  have c17895 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.1
  have c17896 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.1
  have e_273_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_3
  simp only [← c17894, ← c17895, ← c17896] at e_273_3
  have hr := e_273_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 273 19) = band (a (.virt 19151)) (a (.virt 19153)) := by
  have c17897 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.1
  have c17898 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17899 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_273_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_4
  simp only [← c17897, ← c17898, ← c17899] at e_273_4
  have hr := e_273_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 273 23) = band (a (.wire 273 15)) (a (.wire 273 19)) := by
  have c17900 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17901 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17902 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_273_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_5
  simp only [← c17900, ← c17901, ← c17902] at e_273_5
  have hr := e_273_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 75) = bor (a (.wire 52 27)) (a (.wire 273 23)) := by
  have c17903 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17904 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17905 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17906 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17907 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17908 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_262_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_18
  simp only [← c17906, ← c17907, k3, ← c17908] at e_262_18
  have e_270_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_6
  simp only [← c17903, ← c17904, ← c17905] at e_270_6
  have hr := e_262_18
  simp only [e_270_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 75) = a (.virt 201) := by
  have c17909 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17909

theorem publicBatchWrapper89_f2486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10603)) (a (.wire 170 23)) (a (.virt 19155)) (a (.virt 19156)) := by
  have c17910 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17911 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17912 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17913 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17914 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17915 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17916 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17917 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17918 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17919 := (publicBatchWrapper89_copies559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17920 := (publicBatchWrapper89_copies560 a h).1
  have c17921 := (publicBatchWrapper89_copies560 a h).2.1
  have c17922 := (publicBatchWrapper89_copies560 a h).2.2.1
  have c17923 := (publicBatchWrapper89_copies560 a h).2.2.2.1
  have c17924 := (publicBatchWrapper89_copies560 a h).2.2.2.2.1
  have c17925 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.1
  have c17926 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_273_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_6
  simp only [← c17916, ← c17917, ← c17918] at e_273_6
  have e_273_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_7
  simp only [← c17919, ← c17920, ← c17921] at e_273_7
  have e_274_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_2
  simp only [← c17910, k3, ← c17911, k3, ← c17912] at e_274_2
  have e_274_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_3
  simp only [← c17913, ← c17914, k3, ← c17915] at e_274_3
  have e_274_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_4
  simp only [← c17922, ← c17923, k3, ← c17924] at e_274_4
  refine ⟨?_, ?_⟩
  · have hc := e_273_6
    simp only [e_274_3] at hc
    linear_combination c17925.trans k2 - hc
  · have hc := e_274_4
    simp only [e_274_2, e_273_7, e_274_3] at hc
    linear_combination c17926.trans k2 - hc

theorem publicBatchWrapper89_f2487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 79) = bor (a (.wire 52 71)) (a (.virt 19155)) := by
  have c17927 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.1
  have c17928 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.1
  have c17929 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.1
  have c17930 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17931 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17932 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_262_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row262 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_262_19
  simp only [← c17930, ← c17931, k3, ← c17932] at e_262_19
  have e_270_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_7
  simp only [← c17927, ← c17928, ← c17929] at e_270_7
  have hr := e_262_19
  simp only [e_270_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 262 79) = a (.virt 201) := by
  have c17933 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17933

theorem publicBatchWrapper89_f2489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10604)) (a (.wire 170 31)) (a (.virt 19157)) (a (.virt 19158)) := by
  have c17934 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17935 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17936 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17937 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17938 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17939 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17940 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17941 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17942 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17943 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17944 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17945 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17946 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17947 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17948 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17949 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17950 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_273_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_8
  simp only [← c17940, ← c17941, ← c17942] at e_273_8
  have e_273_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_9
  simp only [← c17943, ← c17944, ← c17945] at e_273_9
  have e_274_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_5
  simp only [← c17934, k3, ← c17935, k3, ← c17936] at e_274_5
  have e_274_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_6
  simp only [← c17937, ← c17938, k3, ← c17939] at e_274_6
  have e_274_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_7
  simp only [← c17946, ← c17947, k3, ← c17948] at e_274_7
  refine ⟨?_, ?_⟩
  · have hc := e_273_8
    simp only [e_274_6] at hc
    linear_combination c17949.trans k2 - hc
  · have hc := e_274_7
    simp only [e_274_5, e_273_9, e_274_6] at hc
    linear_combination c17950.trans k2 - hc

theorem publicBatchWrapper89_f2490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 3) = bor (a (.wire 52 71)) (a (.virt 19157)) := by
  have c17951 := (publicBatchWrapper89_copies560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17952 := (publicBatchWrapper89_copies561 a h).1
  have c17953 := (publicBatchWrapper89_copies561 a h).2.1
  have c17954 := (publicBatchWrapper89_copies561 a h).2.2.1
  have c17955 := (publicBatchWrapper89_copies561 a h).2.2.2.1
  have c17956 := (publicBatchWrapper89_copies561 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_8
  simp only [← c17951, ← c17952, ← c17953] at e_270_8
  have e_275_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_0
  simp only [← c17954, ← c17955, k3, ← c17956] at e_275_0
  have hr := e_275_0
  simp only [e_270_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 3) = a (.virt 201) := by
  have c17957 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.1
  exact c17957

theorem publicBatchWrapper89_f2492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10605)) (a (.wire 169 63)) (a (.virt 19159)) (a (.virt 19160)) := by
  have c17958 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.1
  have c17959 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.1
  have c17960 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.1
  have c17961 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.1
  have c17962 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17963 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17964 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17965 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17966 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17967 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17968 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17969 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17970 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17971 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17972 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17973 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17974 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_273_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_10
  simp only [← c17964, ← c17965, ← c17966] at e_273_10
  have e_273_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_11
  simp only [← c17967, ← c17968, ← c17969] at e_273_11
  have e_274_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_8
  simp only [← c17958, k3, ← c17959, k3, ← c17960] at e_274_8
  have e_274_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_9
  simp only [← c17961, ← c17962, k3, ← c17963] at e_274_9
  have e_274_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_10
  simp only [← c17970, ← c17971, k3, ← c17972] at e_274_10
  refine ⟨?_, ?_⟩
  · have hc := e_273_10
    simp only [e_274_9] at hc
    linear_combination c17973.trans k2 - hc
  · have hc := e_274_10
    simp only [e_274_8, e_273_11, e_274_9] at hc
    linear_combination c17974.trans k2 - hc

theorem publicBatchWrapper89_f2493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10606)) (a (.wire 169 71)) (a (.virt 19161)) (a (.virt 19162)) := by
  have c17975 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17976 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17977 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17978 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17979 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17980 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17981 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17982 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17983 := (publicBatchWrapper89_copies561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c17984 := (publicBatchWrapper89_copies562 a h).1
  have c17985 := (publicBatchWrapper89_copies562 a h).2.1
  have c17986 := (publicBatchWrapper89_copies562 a h).2.2.1
  have c17987 := (publicBatchWrapper89_copies562 a h).2.2.2.1
  have c17988 := (publicBatchWrapper89_copies562 a h).2.2.2.2.1
  have c17989 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.1
  have c17990 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.1
  have c17991 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_273_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_12
  simp only [← c17981, ← c17982, ← c17983] at e_273_12
  have e_273_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_13
  simp only [← c17984, ← c17985, ← c17986] at e_273_13
  have e_274_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_11
  simp only [← c17975, k3, ← c17976, k3, ← c17977] at e_274_11
  have e_274_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_12
  simp only [← c17978, ← c17979, k3, ← c17980] at e_274_12
  have e_274_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_13
  simp only [← c17987, ← c17988, k3, ← c17989] at e_274_13
  refine ⟨?_, ?_⟩
  · have hc := e_273_12
    simp only [e_274_12] at hc
    linear_combination c17990.trans k2 - hc
  · have hc := e_274_13
    simp only [e_274_11, e_273_13, e_274_12] at hc
    linear_combination c17991.trans k2 - hc

theorem publicBatchWrapper89_f2494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10607)) (a (.wire 169 79)) (a (.virt 19163)) (a (.virt 19164)) := by
  have c17992 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.1
  have c17993 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.1
  have c17994 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.1
  have c17995 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c17996 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17997 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17998 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17999 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18000 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18001 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18002 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18003 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18004 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18005 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18006 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18007 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18008 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_273_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_14
  simp only [← c17998, ← c17999, ← c18000] at e_273_14
  have e_273_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_15
  simp only [← c18001, ← c18002, ← c18003] at e_273_15
  have e_274_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_14
  simp only [← c17992, k3, ← c17993, k3, ← c17994] at e_274_14
  have e_274_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_15
  simp only [← c17995, ← c17996, k3, ← c17997] at e_274_15
  have e_274_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_16
  simp only [← c18004, ← c18005, k3, ← c18006] at e_274_16
  refine ⟨?_, ?_⟩
  · have hc := e_273_14
    simp only [e_274_15] at hc
    linear_combination c18007.trans k2 - hc
  · have hc := e_274_16
    simp only [e_274_14, e_273_15, e_274_15] at hc
    linear_combination c18008.trans k2 - hc

theorem publicBatchWrapper89_f2495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10608)) (a (.wire 170 7)) (a (.virt 19165)) (a (.virt 19166)) := by
  have c18009 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18010 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18011 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18012 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18013 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18014 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18015 := (publicBatchWrapper89_copies562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18016 := (publicBatchWrapper89_copies563 a h).1
  have c18017 := (publicBatchWrapper89_copies563 a h).2.1
  have c18018 := (publicBatchWrapper89_copies563 a h).2.2.1
  have c18019 := (publicBatchWrapper89_copies563 a h).2.2.2.1
  have c18020 := (publicBatchWrapper89_copies563 a h).2.2.2.2.1
  have c18021 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.1
  have c18022 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.1
  have c18023 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.1
  have c18024 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.1
  have c18025 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_273_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_16
  simp only [← c18015, ← c18016, ← c18017] at e_273_16
  have e_273_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_17
  simp only [← c18018, ← c18019, ← c18020] at e_273_17
  have e_274_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_17
  simp only [← c18009, k3, ← c18010, k3, ← c18011] at e_274_17
  have e_274_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_18
  simp only [← c18012, ← c18013, k3, ← c18014] at e_274_18
  have e_274_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row274 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_274_19
  simp only [← c18021, ← c18022, k3, ← c18023] at e_274_19
  refine ⟨?_, ?_⟩
  · have hc := e_273_16
    simp only [e_274_18] at hc
    linear_combination c18024.trans k2 - hc
  · have hc := e_274_19
    simp only [e_274_17, e_273_17, e_274_18] at hc
    linear_combination c18025.trans k2 - hc

theorem publicBatchWrapper89_f2496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 273 75) = band (a (.virt 19159)) (a (.virt 19161)) := by
  have c18026 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18027 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18028 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_273_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_18
  simp only [← c18026, ← c18027, ← c18028] at e_273_18
  have hr := e_273_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 273 79) = band (a (.virt 19163)) (a (.virt 19165)) := by
  have c18029 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18030 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18031 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_273_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row273 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_273_19
  simp only [← c18029, ← c18030, ← c18031] at e_273_19
  have hr := e_273_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 276 3) = band (a (.wire 273 75)) (a (.wire 273 79)) := by
  have c18032 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18033 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18034 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_276_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_0
  simp only [← c18032, ← c18033, ← c18034] at e_276_0
  have hr := e_276_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 7) = bor (a (.wire 52 71)) (a (.wire 276 3)) := by
  have c18035 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18036 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18037 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18038 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18039 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18040 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_9
  simp only [← c18035, ← c18036, ← c18037] at e_270_9
  have e_275_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_1
  simp only [← c18038, ← c18039, k3, ← c18040] at e_275_1
  have hr := e_275_1
  simp only [e_270_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 7) = a (.virt 201) := by
  have c18041 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18041

theorem publicBatchWrapper89_f2501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10803)) (a (.wire 170 23)) (a (.virt 19167)) (a (.virt 19168)) := by
  have c18042 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18043 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18044 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18045 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18046 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18047 := (publicBatchWrapper89_copies563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18048 := (publicBatchWrapper89_copies564 a h).1
  have c18049 := (publicBatchWrapper89_copies564 a h).2.1
  have c18050 := (publicBatchWrapper89_copies564 a h).2.2.1
  have c18051 := (publicBatchWrapper89_copies564 a h).2.2.2.1
  have c18052 := (publicBatchWrapper89_copies564 a h).2.2.2.2.1
  have c18053 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.1
  have c18054 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.1
  have c18055 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.1
  have c18056 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.1
  have c18057 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.1
  have c18058 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_276_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_1
  simp only [← c18048, ← c18049, ← c18050] at e_276_1
  have e_276_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_2
  simp only [← c18051, ← c18052, ← c18053] at e_276_2
  have e_277_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_0
  simp only [← c18042, k3, ← c18043, k3, ← c18044] at e_277_0
  have e_277_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_1
  simp only [← c18045, ← c18046, k3, ← c18047] at e_277_1
  have e_277_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_2
  simp only [← c18054, ← c18055, k3, ← c18056] at e_277_2
  refine ⟨?_, ?_⟩
  · have hc := e_276_1
    simp only [e_277_1] at hc
    linear_combination c18057.trans k2 - hc
  · have hc := e_277_2
    simp only [e_277_0, e_276_2, e_277_1] at hc
    linear_combination c18058.trans k2 - hc

theorem publicBatchWrapper89_f2502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 11) = bor (a (.wire 53 35)) (a (.virt 19167)) := by
  have c18059 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18060 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18061 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18062 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18063 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18064 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_10
  simp only [← c18059, ← c18060, ← c18061] at e_270_10
  have e_275_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_2
  simp only [← c18062, ← c18063, k3, ← c18064] at e_275_2
  have hr := e_275_2
  simp only [e_270_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 11) = a (.virt 201) := by
  have c18065 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18065

theorem publicBatchWrapper89_f2504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10804)) (a (.wire 170 31)) (a (.virt 19169)) (a (.virt 19170)) := by
  have c18066 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18067 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18068 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18069 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18070 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18071 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18072 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18073 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18074 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18075 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18076 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18077 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18078 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18079 := (publicBatchWrapper89_copies564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18080 := (publicBatchWrapper89_copies565 a h).1
  have c18081 := (publicBatchWrapper89_copies565 a h).2.1
  have c18082 := (publicBatchWrapper89_copies565 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_276_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_3
  simp only [← c18072, ← c18073, ← c18074] at e_276_3
  have e_276_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_4
  simp only [← c18075, ← c18076, ← c18077] at e_276_4
  have e_277_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_3
  simp only [← c18066, k3, ← c18067, k3, ← c18068] at e_277_3
  have e_277_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_4
  simp only [← c18069, ← c18070, k3, ← c18071] at e_277_4
  have e_277_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_5
  simp only [← c18078, ← c18079, k3, ← c18080] at e_277_5
  refine ⟨?_, ?_⟩
  · have hc := e_276_3
    simp only [e_277_4] at hc
    linear_combination c18081.trans k2 - hc
  · have hc := e_277_5
    simp only [e_277_3, e_276_4, e_277_4] at hc
    linear_combination c18082.trans k2 - hc

theorem publicBatchWrapper89_f2505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 15) = bor (a (.wire 53 35)) (a (.virt 19169)) := by
  have c18083 := (publicBatchWrapper89_copies565 a h).2.2.2.1
  have c18084 := (publicBatchWrapper89_copies565 a h).2.2.2.2.1
  have c18085 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.1
  have c18086 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.1
  have c18087 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.1
  have c18088 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_11
  simp only [← c18083, ← c18084, ← c18085] at e_270_11
  have e_275_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_3
  simp only [← c18086, ← c18087, k3, ← c18088] at e_275_3
  have hr := e_275_3
  simp only [e_270_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 15) = a (.virt 201) := by
  have c18089 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.1
  exact c18089

theorem publicBatchWrapper89_f2507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10805)) (a (.wire 169 63)) (a (.virt 19171)) (a (.virt 19172)) := by
  have c18090 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18091 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18092 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18093 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18094 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18095 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18096 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18097 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18098 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18099 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18100 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18101 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18102 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18103 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18104 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18105 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18106 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_276_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_5
  simp only [← c18096, ← c18097, ← c18098] at e_276_5
  have e_276_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_6
  simp only [← c18099, ← c18100, ← c18101] at e_276_6
  have e_277_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_6
  simp only [← c18090, k3, ← c18091, k3, ← c18092] at e_277_6
  have e_277_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_7
  simp only [← c18093, ← c18094, k3, ← c18095] at e_277_7
  have e_277_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_8
  simp only [← c18102, ← c18103, k3, ← c18104] at e_277_8
  refine ⟨?_, ?_⟩
  · have hc := e_276_5
    simp only [e_277_7] at hc
    linear_combination c18105.trans k2 - hc
  · have hc := e_277_8
    simp only [e_277_6, e_276_6, e_277_7] at hc
    linear_combination c18106.trans k2 - hc

theorem publicBatchWrapper89_f2508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10806)) (a (.wire 169 71)) (a (.virt 19173)) (a (.virt 19174)) := by
  have c18107 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18108 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18109 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18110 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18111 := (publicBatchWrapper89_copies565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18112 := (publicBatchWrapper89_copies566 a h).1
  have c18113 := (publicBatchWrapper89_copies566 a h).2.1
  have c18114 := (publicBatchWrapper89_copies566 a h).2.2.1
  have c18115 := (publicBatchWrapper89_copies566 a h).2.2.2.1
  have c18116 := (publicBatchWrapper89_copies566 a h).2.2.2.2.1
  have c18117 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.1
  have c18118 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.1
  have c18119 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.1
  have c18120 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.1
  have c18121 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.1
  have c18122 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18123 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_276_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_7
  simp only [← c18113, ← c18114, ← c18115] at e_276_7
  have e_276_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_8
  simp only [← c18116, ← c18117, ← c18118] at e_276_8
  have e_277_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_9
  simp only [← c18107, k3, ← c18108, k3, ← c18109] at e_277_9
  have e_277_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_10
  simp only [← c18110, ← c18111, k3, ← c18112] at e_277_10
  have e_277_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_11
  simp only [← c18119, ← c18120, k3, ← c18121] at e_277_11
  refine ⟨?_, ?_⟩
  · have hc := e_276_7
    simp only [e_277_10] at hc
    linear_combination c18122.trans k2 - hc
  · have hc := e_277_11
    simp only [e_277_9, e_276_8, e_277_10] at hc
    linear_combination c18123.trans k2 - hc

theorem publicBatchWrapper89_f2509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10807)) (a (.wire 169 79)) (a (.virt 19175)) (a (.virt 19176)) := by
  have c18124 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18125 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18126 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18127 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18128 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18129 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18130 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18131 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18132 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18133 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18134 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18135 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18136 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18137 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18138 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18139 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18140 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_276_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_9
  simp only [← c18130, ← c18131, ← c18132] at e_276_9
  have e_276_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_10
  simp only [← c18133, ← c18134, ← c18135] at e_276_10
  have e_277_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_12
  simp only [← c18124, k3, ← c18125, k3, ← c18126] at e_277_12
  have e_277_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_13
  simp only [← c18127, ← c18128, k3, ← c18129] at e_277_13
  have e_277_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_14
  simp only [← c18136, ← c18137, k3, ← c18138] at e_277_14
  refine ⟨?_, ?_⟩
  · have hc := e_276_9
    simp only [e_277_13] at hc
    linear_combination c18139.trans k2 - hc
  · have hc := e_277_14
    simp only [e_277_12, e_276_10, e_277_13] at hc
    linear_combination c18140.trans k2 - hc

theorem publicBatchWrapper89_f2510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10808)) (a (.wire 170 7)) (a (.virt 19177)) (a (.virt 19178)) := by
  have c18141 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18142 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18143 := (publicBatchWrapper89_copies566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18144 := (publicBatchWrapper89_copies567 a h).1
  have c18145 := (publicBatchWrapper89_copies567 a h).2.1
  have c18146 := (publicBatchWrapper89_copies567 a h).2.2.1
  have c18147 := (publicBatchWrapper89_copies567 a h).2.2.2.1
  have c18148 := (publicBatchWrapper89_copies567 a h).2.2.2.2.1
  have c18149 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.1
  have c18150 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.1
  have c18151 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.1
  have c18152 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.1
  have c18153 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.1
  have c18154 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18155 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18156 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18157 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_276_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_11
  simp only [← c18147, ← c18148, ← c18149] at e_276_11
  have e_276_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_12
  simp only [← c18150, ← c18151, ← c18152] at e_276_12
  have e_277_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_15
  simp only [← c18141, k3, ← c18142, k3, ← c18143] at e_277_15
  have e_277_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_16
  simp only [← c18144, ← c18145, k3, ← c18146] at e_277_16
  have e_277_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_17
  simp only [← c18153, ← c18154, k3, ← c18155] at e_277_17
  refine ⟨?_, ?_⟩
  · have hc := e_276_11
    simp only [e_277_16] at hc
    linear_combination c18156.trans k2 - hc
  · have hc := e_277_17
    simp only [e_277_15, e_276_12, e_277_16] at hc
    linear_combination c18157.trans k2 - hc

theorem publicBatchWrapper89_f2511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 276 55) = band (a (.virt 19171)) (a (.virt 19173)) := by
  have c18158 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18159 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18160 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_276_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_13
  simp only [← c18158, ← c18159, ← c18160] at e_276_13
  have hr := e_276_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 276 59) = band (a (.virt 19175)) (a (.virt 19177)) := by
  have c18161 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18162 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18163 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_276_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_14
  simp only [← c18161, ← c18162, ← c18163] at e_276_14
  have hr := e_276_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 276 63) = band (a (.wire 276 55)) (a (.wire 276 59)) := by
  have c18164 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18165 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18166 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_276_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_15
  simp only [← c18164, ← c18165, ← c18166] at e_276_15
  have hr := e_276_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 19) = bor (a (.wire 53 35)) (a (.wire 276 63)) := by
  have c18167 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18168 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18169 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18170 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18171 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18172 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_12
  simp only [← c18167, ← c18168, ← c18169] at e_270_12
  have e_275_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_4
  simp only [← c18170, ← c18171, k3, ← c18172] at e_275_4
  have hr := e_275_4
  simp only [e_270_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 19) = a (.virt 201) := by
  have c18173 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18173

theorem publicBatchWrapper89_f2516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11003)) (a (.wire 170 23)) (a (.virt 19179)) (a (.virt 19180)) := by
  have c18174 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18175 := (publicBatchWrapper89_copies567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18176 := (publicBatchWrapper89_copies568 a h).1
  have c18177 := (publicBatchWrapper89_copies568 a h).2.1
  have c18178 := (publicBatchWrapper89_copies568 a h).2.2.1
  have c18179 := (publicBatchWrapper89_copies568 a h).2.2.2.1
  have c18180 := (publicBatchWrapper89_copies568 a h).2.2.2.2.1
  have c18181 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.1
  have c18182 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.1
  have c18183 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.1
  have c18184 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.1
  have c18185 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.1
  have c18186 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18187 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18188 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18189 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18190 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_276_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_16
  simp only [← c18180, ← c18181, ← c18182] at e_276_16
  have e_276_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_17
  simp only [← c18183, ← c18184, ← c18185] at e_276_17
  have e_277_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_18
  simp only [← c18174, k3, ← c18175, k3, ← c18176] at e_277_18
  have e_277_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row277 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_277_19
  simp only [← c18177, ← c18178, k3, ← c18179] at e_277_19
  have e_278_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_0
  simp only [← c18186, ← c18187, k3, ← c18188] at e_278_0
  refine ⟨?_, ?_⟩
  · have hc := e_276_16
    simp only [e_277_19] at hc
    linear_combination c18189.trans k2 - hc
  · have hc := e_278_0
    simp only [e_277_18, e_276_17, e_277_19] at hc
    linear_combination c18190.trans k2 - hc

theorem publicBatchWrapper89_f2517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 23) = bor (a (.wire 53 79)) (a (.virt 19179)) := by
  have c18191 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18192 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18193 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18194 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18195 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18196 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_13
  simp only [← c18191, ← c18192, ← c18193] at e_270_13
  have e_275_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_5
  simp only [← c18194, ← c18195, k3, ← c18196] at e_275_5
  have hr := e_275_5
  simp only [e_270_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 23) = a (.virt 201) := by
  have c18197 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18197

theorem publicBatchWrapper89_f2519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11004)) (a (.wire 170 31)) (a (.virt 19181)) (a (.virt 19182)) := by
  have c18198 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18199 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18200 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18201 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18202 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18203 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18204 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18205 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18206 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18207 := (publicBatchWrapper89_copies568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18208 := (publicBatchWrapper89_copies569 a h).1
  have c18209 := (publicBatchWrapper89_copies569 a h).2.1
  have c18210 := (publicBatchWrapper89_copies569 a h).2.2.1
  have c18211 := (publicBatchWrapper89_copies569 a h).2.2.2.1
  have c18212 := (publicBatchWrapper89_copies569 a h).2.2.2.2.1
  have c18213 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.1
  have c18214 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_276_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_18
  simp only [← c18204, ← c18205, ← c18206] at e_276_18
  have e_276_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row276 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_276_19
  simp only [← c18207, ← c18208, ← c18209] at e_276_19
  have e_278_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_1
  simp only [← c18198, k3, ← c18199, k3, ← c18200] at e_278_1
  have e_278_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_2
  simp only [← c18201, ← c18202, k3, ← c18203] at e_278_2
  have e_278_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_3
  simp only [← c18210, ← c18211, k3, ← c18212] at e_278_3
  refine ⟨?_, ?_⟩
  · have hc := e_276_18
    simp only [e_278_2] at hc
    linear_combination c18213.trans k2 - hc
  · have hc := e_278_3
    simp only [e_278_1, e_276_19, e_278_2] at hc
    linear_combination c18214.trans k2 - hc

theorem publicBatchWrapper89_f2520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 27) = bor (a (.wire 53 79)) (a (.virt 19181)) := by
  have c18215 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.1
  have c18216 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.1
  have c18217 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.1
  have c18218 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18219 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18220 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_14
  simp only [← c18215, ← c18216, ← c18217] at e_270_14
  have e_275_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_6
  simp only [← c18218, ← c18219, k3, ← c18220] at e_275_6
  have hr := e_275_6
  simp only [e_270_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 27) = a (.virt 201) := by
  have c18221 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18221

theorem publicBatchWrapper89_f2522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11005)) (a (.wire 169 63)) (a (.virt 19183)) (a (.virt 19184)) := by
  have c18222 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18223 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18224 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18225 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18226 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18227 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18228 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18229 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18230 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18231 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18232 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18233 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18234 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18235 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18236 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18237 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18238 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_278_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_4
  simp only [← c18222, k3, ← c18223, k3, ← c18224] at e_278_4
  have e_278_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_5
  simp only [← c18225, ← c18226, k3, ← c18227] at e_278_5
  have e_278_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_6
  simp only [← c18234, ← c18235, k3, ← c18236] at e_278_6
  have e_279_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_0
  simp only [← c18228, ← c18229, ← c18230] at e_279_0
  have e_279_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_1
  simp only [← c18231, ← c18232, ← c18233] at e_279_1
  refine ⟨?_, ?_⟩
  · have hc := e_279_0
    simp only [e_278_5] at hc
    linear_combination c18237.trans k2 - hc
  · have hc := e_278_6
    simp only [e_278_4, e_279_1, e_278_5] at hc
    linear_combination c18238.trans k2 - hc

theorem publicBatchWrapper89_f2523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11006)) (a (.wire 169 71)) (a (.virt 19185)) (a (.virt 19186)) := by
  have c18239 := (publicBatchWrapper89_copies569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18240 := (publicBatchWrapper89_copies570 a h).1
  have c18241 := (publicBatchWrapper89_copies570 a h).2.1
  have c18242 := (publicBatchWrapper89_copies570 a h).2.2.1
  have c18243 := (publicBatchWrapper89_copies570 a h).2.2.2.1
  have c18244 := (publicBatchWrapper89_copies570 a h).2.2.2.2.1
  have c18245 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.1
  have c18246 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.1
  have c18247 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.1
  have c18248 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.1
  have c18249 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.1
  have c18250 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18251 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18252 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18253 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18254 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18255 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_278_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_7
  simp only [← c18239, k3, ← c18240, k3, ← c18241] at e_278_7
  have e_278_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_8
  simp only [← c18242, ← c18243, k3, ← c18244] at e_278_8
  have e_278_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_9
  simp only [← c18251, ← c18252, k3, ← c18253] at e_278_9
  have e_279_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_2
  simp only [← c18245, ← c18246, ← c18247] at e_279_2
  have e_279_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_3
  simp only [← c18248, ← c18249, ← c18250] at e_279_3
  refine ⟨?_, ?_⟩
  · have hc := e_279_2
    simp only [e_278_8] at hc
    linear_combination c18254.trans k2 - hc
  · have hc := e_278_9
    simp only [e_278_7, e_279_3, e_278_8] at hc
    linear_combination c18255.trans k2 - hc

theorem publicBatchWrapper89_f2524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11007)) (a (.wire 169 79)) (a (.virt 19187)) (a (.virt 19188)) := by
  have c18256 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18257 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18258 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18259 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18260 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18261 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18262 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18263 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18264 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18265 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18266 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18267 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18268 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18269 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18270 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18271 := (publicBatchWrapper89_copies570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18272 := (publicBatchWrapper89_copies571 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_278_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_10
  simp only [← c18256, k3, ← c18257, k3, ← c18258] at e_278_10
  have e_278_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_11
  simp only [← c18259, ← c18260, k3, ← c18261] at e_278_11
  have e_278_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_12
  simp only [← c18268, ← c18269, k3, ← c18270] at e_278_12
  have e_279_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_4
  simp only [← c18262, ← c18263, ← c18264] at e_279_4
  have e_279_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_5
  simp only [← c18265, ← c18266, ← c18267] at e_279_5
  refine ⟨?_, ?_⟩
  · have hc := e_279_4
    simp only [e_278_11] at hc
    linear_combination c18271.trans k2 - hc
  · have hc := e_278_12
    simp only [e_278_10, e_279_5, e_278_11] at hc
    linear_combination c18272.trans k2 - hc

theorem publicBatchWrapper89_f2525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11008)) (a (.wire 170 7)) (a (.virt 19189)) (a (.virt 19190)) := by
  have c18273 := (publicBatchWrapper89_copies571 a h).2.1
  have c18274 := (publicBatchWrapper89_copies571 a h).2.2.1
  have c18275 := (publicBatchWrapper89_copies571 a h).2.2.2.1
  have c18276 := (publicBatchWrapper89_copies571 a h).2.2.2.2.1
  have c18277 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.1
  have c18278 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.1
  have c18279 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.1
  have c18280 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.1
  have c18281 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.1
  have c18282 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18283 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18284 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18285 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18286 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18287 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18288 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18289 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_278_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_13
  simp only [← c18273, k3, ← c18274, k3, ← c18275] at e_278_13
  have e_278_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_14
  simp only [← c18276, ← c18277, k3, ← c18278] at e_278_14
  have e_278_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_15
  simp only [← c18285, ← c18286, k3, ← c18287] at e_278_15
  have e_279_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_6
  simp only [← c18279, ← c18280, ← c18281] at e_279_6
  have e_279_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_7
  simp only [← c18282, ← c18283, ← c18284] at e_279_7
  refine ⟨?_, ?_⟩
  · have hc := e_279_6
    simp only [e_278_14] at hc
    linear_combination c18288.trans k2 - hc
  · have hc := e_278_15
    simp only [e_278_13, e_279_7, e_278_14] at hc
    linear_combination c18289.trans k2 - hc

theorem publicBatchWrapper89_f2526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 279 35) = band (a (.virt 19183)) (a (.virt 19185)) := by
  have c18290 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18291 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18292 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_279_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_8
  simp only [← c18290, ← c18291, ← c18292] at e_279_8
  have hr := e_279_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 279 39) = band (a (.virt 19187)) (a (.virt 19189)) := by
  have c18293 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18294 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18295 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_279_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_9
  simp only [← c18293, ← c18294, ← c18295] at e_279_9
  have hr := e_279_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 279 43) = band (a (.wire 279 35)) (a (.wire 279 39)) := by
  have c18296 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18297 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18298 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_279_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_10
  simp only [← c18296, ← c18297, ← c18298] at e_279_10
  have hr := e_279_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 31) = bor (a (.wire 53 79)) (a (.wire 279 43)) := by
  have c18299 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18300 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18301 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18302 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18303 := (publicBatchWrapper89_copies571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18304 := (publicBatchWrapper89_copies572 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_15
  simp only [← c18299, ← c18300, ← c18301] at e_270_15
  have e_275_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_7
  simp only [← c18302, ← c18303, k3, ← c18304] at e_275_7
  have hr := e_275_7
  simp only [e_270_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 31) = a (.virt 201) := by
  have c18305 := (publicBatchWrapper89_copies572 a h).2.1
  exact c18305

theorem publicBatchWrapper89_f2531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11203)) (a (.wire 170 23)) (a (.virt 19191)) (a (.virt 19192)) := by
  have c18306 := (publicBatchWrapper89_copies572 a h).2.2.1
  have c18307 := (publicBatchWrapper89_copies572 a h).2.2.2.1
  have c18308 := (publicBatchWrapper89_copies572 a h).2.2.2.2.1
  have c18309 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.1
  have c18310 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.1
  have c18311 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.1
  have c18312 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.1
  have c18313 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.1
  have c18314 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18315 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18316 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18317 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18318 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18319 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18320 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18321 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18322 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_278_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_16
  simp only [← c18306, k3, ← c18307, k3, ← c18308] at e_278_16
  have e_278_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_17
  simp only [← c18309, ← c18310, k3, ← c18311] at e_278_17
  have e_278_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_18
  simp only [← c18318, ← c18319, k3, ← c18320] at e_278_18
  have e_279_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_11
  simp only [← c18312, ← c18313, ← c18314] at e_279_11
  have e_279_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_12
  simp only [← c18315, ← c18316, ← c18317] at e_279_12
  refine ⟨?_, ?_⟩
  · have hc := e_279_11
    simp only [e_278_17] at hc
    linear_combination c18321.trans k2 - hc
  · have hc := e_278_18
    simp only [e_278_16, e_279_12, e_278_17] at hc
    linear_combination c18322.trans k2 - hc

theorem publicBatchWrapper89_f2532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 35) = bor (a (.wire 55 43)) (a (.virt 19191)) := by
  have c18323 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18324 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18325 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18326 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18327 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18328 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_16
  simp only [← c18323, ← c18324, ← c18325] at e_270_16
  have e_275_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_8
  simp only [← c18326, ← c18327, k3, ← c18328] at e_275_8
  have hr := e_275_8
  simp only [e_270_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 35) = a (.virt 201) := by
  have c18329 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18329

theorem publicBatchWrapper89_f2534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11204)) (a (.wire 170 31)) (a (.virt 19193)) (a (.virt 19194)) := by
  have c18330 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18331 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18332 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18333 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18334 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18335 := (publicBatchWrapper89_copies572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18336 := (publicBatchWrapper89_copies573 a h).1
  have c18337 := (publicBatchWrapper89_copies573 a h).2.1
  have c18338 := (publicBatchWrapper89_copies573 a h).2.2.1
  have c18339 := (publicBatchWrapper89_copies573 a h).2.2.2.1
  have c18340 := (publicBatchWrapper89_copies573 a h).2.2.2.2.1
  have c18341 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.1
  have c18342 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.1
  have c18343 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.1
  have c18344 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.1
  have c18345 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.1
  have c18346 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_278_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row278 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_278_19
  simp only [← c18330, k3, ← c18331, k3, ← c18332] at e_278_19
  have e_279_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_13
  simp only [← c18336, ← c18337, ← c18338] at e_279_13
  have e_279_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_14
  simp only [← c18339, ← c18340, ← c18341] at e_279_14
  have e_280_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_0
  simp only [← c18333, ← c18334, k3, ← c18335] at e_280_0
  have e_280_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_1
  simp only [← c18342, ← c18343, k3, ← c18344] at e_280_1
  refine ⟨?_, ?_⟩
  · have hc := e_279_13
    simp only [e_280_0] at hc
    linear_combination c18345.trans k2 - hc
  · have hc := e_280_1
    simp only [e_278_19, e_279_14, e_280_0] at hc
    linear_combination c18346.trans k2 - hc

theorem publicBatchWrapper89_f2535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 39) = bor (a (.wire 55 43)) (a (.virt 19193)) := by
  have c18347 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18348 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18349 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18350 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18351 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18352 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_17
  simp only [← c18347, ← c18348, ← c18349] at e_270_17
  have e_275_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_9
  simp only [← c18350, ← c18351, k3, ← c18352] at e_275_9
  have hr := e_275_9
  simp only [e_270_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 39) = a (.virt 201) := by
  have c18353 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18353

theorem publicBatchWrapper89_f2537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11205)) (a (.wire 169 63)) (a (.virt 19195)) (a (.virt 19196)) := by
  have c18354 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18355 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18356 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18357 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18358 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18359 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18360 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18361 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18362 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18363 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18364 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18365 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18366 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18367 := (publicBatchWrapper89_copies573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18368 := (publicBatchWrapper89_copies574 a h).1
  have c18369 := (publicBatchWrapper89_copies574 a h).2.1
  have c18370 := (publicBatchWrapper89_copies574 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_279_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_15
  simp only [← c18360, ← c18361, ← c18362] at e_279_15
  have e_279_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_16
  simp only [← c18363, ← c18364, ← c18365] at e_279_16
  have e_280_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_2
  simp only [← c18354, k3, ← c18355, k3, ← c18356] at e_280_2
  have e_280_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_3
  simp only [← c18357, ← c18358, k3, ← c18359] at e_280_3
  have e_280_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_4
  simp only [← c18366, ← c18367, k3, ← c18368] at e_280_4
  refine ⟨?_, ?_⟩
  · have hc := e_279_15
    simp only [e_280_3] at hc
    linear_combination c18369.trans k2 - hc
  · have hc := e_280_4
    simp only [e_280_2, e_279_16, e_280_3] at hc
    linear_combination c18370.trans k2 - hc

theorem publicBatchWrapper89_f2538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11206)) (a (.wire 169 71)) (a (.virt 19197)) (a (.virt 19198)) := by
  have c18371 := (publicBatchWrapper89_copies574 a h).2.2.2.1
  have c18372 := (publicBatchWrapper89_copies574 a h).2.2.2.2.1
  have c18373 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.1
  have c18374 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.1
  have c18375 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.1
  have c18376 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.1
  have c18377 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.1
  have c18378 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18379 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18380 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18381 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18382 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18383 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18384 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18385 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18386 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18387 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_279_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_17
  simp only [← c18377, ← c18378, ← c18379] at e_279_17
  have e_279_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_18
  simp only [← c18380, ← c18381, ← c18382] at e_279_18
  have e_280_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_5
  simp only [← c18371, k3, ← c18372, k3, ← c18373] at e_280_5
  have e_280_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_6
  simp only [← c18374, ← c18375, k3, ← c18376] at e_280_6
  have e_280_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_7
  simp only [← c18383, ← c18384, k3, ← c18385] at e_280_7
  refine ⟨?_, ?_⟩
  · have hc := e_279_17
    simp only [e_280_6] at hc
    linear_combination c18386.trans k2 - hc
  · have hc := e_280_7
    simp only [e_280_5, e_279_18, e_280_6] at hc
    linear_combination c18387.trans k2 - hc

theorem publicBatchWrapper89_f2539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11207)) (a (.wire 169 79)) (a (.virt 19199)) (a (.virt 19200)) := by
  have c18388 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18389 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18390 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18391 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18392 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18393 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18394 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18395 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18396 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18397 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18398 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18399 := (publicBatchWrapper89_copies574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18400 := (publicBatchWrapper89_copies575 a h).1
  have c18401 := (publicBatchWrapper89_copies575 a h).2.1
  have c18402 := (publicBatchWrapper89_copies575 a h).2.2.1
  have c18403 := (publicBatchWrapper89_copies575 a h).2.2.2.1
  have c18404 := (publicBatchWrapper89_copies575 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_279_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row279 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_279_19
  simp only [← c18394, ← c18395, ← c18396] at e_279_19
  have e_280_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_8
  simp only [← c18388, k3, ← c18389, k3, ← c18390] at e_280_8
  have e_280_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_9
  simp only [← c18391, ← c18392, k3, ← c18393] at e_280_9
  have e_280_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_10
  simp only [← c18400, ← c18401, k3, ← c18402] at e_280_10
  have e_281_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_0
  simp only [← c18397, ← c18398, ← c18399] at e_281_0
  refine ⟨?_, ?_⟩
  · have hc := e_279_19
    simp only [e_280_9] at hc
    linear_combination c18403.trans k2 - hc
  · have hc := e_280_10
    simp only [e_280_8, e_281_0, e_280_9] at hc
    linear_combination c18404.trans k2 - hc

theorem publicBatchWrapper89_f2540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11208)) (a (.wire 170 7)) (a (.virt 19201)) (a (.virt 19202)) := by
  have c18405 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.1
  have c18406 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.1
  have c18407 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.1
  have c18408 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.1
  have c18409 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.1
  have c18410 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18411 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18412 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18413 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18414 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18415 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18416 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18417 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18418 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18419 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18420 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18421 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_280_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_11
  simp only [← c18405, k3, ← c18406, k3, ← c18407] at e_280_11
  have e_280_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_12
  simp only [← c18408, ← c18409, k3, ← c18410] at e_280_12
  have e_280_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_13
  simp only [← c18417, ← c18418, k3, ← c18419] at e_280_13
  have e_281_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_1
  simp only [← c18411, ← c18412, ← c18413] at e_281_1
  have e_281_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_2
  simp only [← c18414, ← c18415, ← c18416] at e_281_2
  refine ⟨?_, ?_⟩
  · have hc := e_281_1
    simp only [e_280_12] at hc
    linear_combination c18420.trans k2 - hc
  · have hc := e_280_13
    simp only [e_280_11, e_281_2, e_280_12] at hc
    linear_combination c18421.trans k2 - hc

theorem publicBatchWrapper89_f2541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 281 15) = band (a (.virt 19195)) (a (.virt 19197)) := by
  have c18422 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18423 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18424 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_281_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_3
  simp only [← c18422, ← c18423, ← c18424] at e_281_3
  have hr := e_281_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 281 19) = band (a (.virt 19199)) (a (.virt 19201)) := by
  have c18425 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18426 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18427 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_281_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_4
  simp only [← c18425, ← c18426, ← c18427] at e_281_4
  have hr := e_281_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 281 23) = band (a (.wire 281 15)) (a (.wire 281 19)) := by
  have c18428 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18429 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18430 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_281_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_5
  simp only [← c18428, ← c18429, ← c18430] at e_281_5
  have hr := e_281_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 43) = bor (a (.wire 55 43)) (a (.wire 281 23)) := by
  have c18431 := (publicBatchWrapper89_copies575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18432 := (publicBatchWrapper89_copies576 a h).1
  have c18433 := (publicBatchWrapper89_copies576 a h).2.1
  have c18434 := (publicBatchWrapper89_copies576 a h).2.2.1
  have c18435 := (publicBatchWrapper89_copies576 a h).2.2.2.1
  have c18436 := (publicBatchWrapper89_copies576 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_18
  simp only [← c18431, ← c18432, ← c18433] at e_270_18
  have e_275_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_10
  simp only [← c18434, ← c18435, k3, ← c18436] at e_275_10
  have hr := e_275_10
  simp only [e_270_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 43) = a (.virt 201) := by
  have c18437 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.1
  exact c18437

theorem publicBatchWrapper89_f2546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11403)) (a (.wire 170 23)) (a (.virt 19203)) (a (.virt 19204)) := by
  have c18438 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.1
  have c18439 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.1
  have c18440 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.1
  have c18441 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.1
  have c18442 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18443 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18444 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18445 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18446 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18447 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18448 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18449 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18450 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18451 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18452 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18453 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18454 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_280_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_14
  simp only [← c18438, k3, ← c18439, k3, ← c18440] at e_280_14
  have e_280_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_15
  simp only [← c18441, ← c18442, k3, ← c18443] at e_280_15
  have e_280_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_16
  simp only [← c18450, ← c18451, k3, ← c18452] at e_280_16
  have e_281_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_6
  simp only [← c18444, ← c18445, ← c18446] at e_281_6
  have e_281_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_7
  simp only [← c18447, ← c18448, ← c18449] at e_281_7
  refine ⟨?_, ?_⟩
  · have hc := e_281_6
    simp only [e_280_15] at hc
    linear_combination c18453.trans k2 - hc
  · have hc := e_280_16
    simp only [e_280_14, e_281_7, e_280_15] at hc
    linear_combination c18454.trans k2 - hc

theorem publicBatchWrapper89_f2547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 47) = bor (a (.wire 57 7)) (a (.virt 19203)) := by
  have c18455 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18456 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18457 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18458 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18459 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18460 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_270_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row270 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_270_19
  simp only [← c18455, ← c18456, ← c18457] at e_270_19
  have e_275_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_11
  simp only [← c18458, ← c18459, k3, ← c18460] at e_275_11
  have hr := e_275_11
  simp only [e_270_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 47) = a (.virt 201) := by
  have c18461 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18461

theorem publicBatchWrapper89_f2549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11404)) (a (.wire 170 31)) (a (.virt 19205)) (a (.virt 19206)) := by
  have c18462 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18463 := (publicBatchWrapper89_copies576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18464 := (publicBatchWrapper89_copies577 a h).1
  have c18465 := (publicBatchWrapper89_copies577 a h).2.1
  have c18466 := (publicBatchWrapper89_copies577 a h).2.2.1
  have c18467 := (publicBatchWrapper89_copies577 a h).2.2.2.1
  have c18468 := (publicBatchWrapper89_copies577 a h).2.2.2.2.1
  have c18469 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.1
  have c18470 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.1
  have c18471 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.1
  have c18472 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.1
  have c18473 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.1
  have c18474 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18475 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18476 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18477 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18478 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_280_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_17
  simp only [← c18462, k3, ← c18463, k3, ← c18464] at e_280_17
  have e_280_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_18
  simp only [← c18465, ← c18466, k3, ← c18467] at e_280_18
  have e_280_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row280 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_280_19
  simp only [← c18474, ← c18475, k3, ← c18476] at e_280_19
  have e_281_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_8
  simp only [← c18468, ← c18469, ← c18470] at e_281_8
  have e_281_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_9
  simp only [← c18471, ← c18472, ← c18473] at e_281_9
  refine ⟨?_, ?_⟩
  · have hc := e_281_8
    simp only [e_280_18] at hc
    linear_combination c18477.trans k2 - hc
  · have hc := e_280_19
    simp only [e_280_17, e_281_9, e_280_18] at hc
    linear_combination c18478.trans k2 - hc

theorem publicBatchWrapper89_f2550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 51) = bor (a (.wire 57 7)) (a (.virt 19205)) := by
  have c18479 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18480 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18481 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18482 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18483 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18484 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_275_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_12
  simp only [← c18482, ← c18483, k3, ← c18484] at e_275_12
  have e_282_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_0
  simp only [← c18479, ← c18480, ← c18481] at e_282_0
  have hr := e_275_12
  simp only [e_282_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 51) = a (.virt 201) := by
  have c18485 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18485

theorem publicBatchWrapper89_f2552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11405)) (a (.wire 169 63)) (a (.virt 19207)) (a (.virt 19208)) := by
  have c18486 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18487 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18488 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18489 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18490 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18491 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18492 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18493 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18494 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18495 := (publicBatchWrapper89_copies577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18496 := (publicBatchWrapper89_copies578 a h).1
  have c18497 := (publicBatchWrapper89_copies578 a h).2.1
  have c18498 := (publicBatchWrapper89_copies578 a h).2.2.1
  have c18499 := (publicBatchWrapper89_copies578 a h).2.2.2.1
  have c18500 := (publicBatchWrapper89_copies578 a h).2.2.2.2.1
  have c18501 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.1
  have c18502 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_281_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_10
  simp only [← c18492, ← c18493, ← c18494] at e_281_10
  have e_281_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_11
  simp only [← c18495, ← c18496, ← c18497] at e_281_11
  have e_283_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_0
  simp only [← c18486, k3, ← c18487, k3, ← c18488] at e_283_0
  have e_283_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_1
  simp only [← c18489, ← c18490, k3, ← c18491] at e_283_1
  have e_283_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_2
  simp only [← c18498, ← c18499, k3, ← c18500] at e_283_2
  refine ⟨?_, ?_⟩
  · have hc := e_281_10
    simp only [e_283_1] at hc
    linear_combination c18501.trans k2 - hc
  · have hc := e_283_2
    simp only [e_283_0, e_281_11, e_283_1] at hc
    linear_combination c18502.trans k2 - hc

theorem publicBatchWrapper89_f2553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11406)) (a (.wire 169 71)) (a (.virt 19209)) (a (.virt 19210)) := by
  have c18503 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.1
  have c18504 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.1
  have c18505 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.1
  have c18506 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18507 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18508 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18509 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18510 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18511 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18512 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18513 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18514 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18515 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18516 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18517 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18518 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18519 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_281_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_12
  simp only [← c18509, ← c18510, ← c18511] at e_281_12
  have e_281_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_13
  simp only [← c18512, ← c18513, ← c18514] at e_281_13
  have e_283_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_3
  simp only [← c18503, k3, ← c18504, k3, ← c18505] at e_283_3
  have e_283_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_4
  simp only [← c18506, ← c18507, k3, ← c18508] at e_283_4
  have e_283_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_5
  simp only [← c18515, ← c18516, k3, ← c18517] at e_283_5
  refine ⟨?_, ?_⟩
  · have hc := e_281_12
    simp only [e_283_4] at hc
    linear_combination c18518.trans k2 - hc
  · have hc := e_283_5
    simp only [e_283_3, e_281_13, e_283_4] at hc
    linear_combination c18519.trans k2 - hc

theorem publicBatchWrapper89_f2554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11407)) (a (.wire 169 79)) (a (.virt 19211)) (a (.virt 19212)) := by
  have c18520 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18521 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18522 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18523 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18524 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18525 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18526 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18527 := (publicBatchWrapper89_copies578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18528 := (publicBatchWrapper89_copies579 a h).1
  have c18529 := (publicBatchWrapper89_copies579 a h).2.1
  have c18530 := (publicBatchWrapper89_copies579 a h).2.2.1
  have c18531 := (publicBatchWrapper89_copies579 a h).2.2.2.1
  have c18532 := (publicBatchWrapper89_copies579 a h).2.2.2.2.1
  have c18533 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.1
  have c18534 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.1
  have c18535 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.1
  have c18536 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_281_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_14
  simp only [← c18526, ← c18527, ← c18528] at e_281_14
  have e_281_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_15
  simp only [← c18529, ← c18530, ← c18531] at e_281_15
  have e_283_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_6
  simp only [← c18520, k3, ← c18521, k3, ← c18522] at e_283_6
  have e_283_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_7
  simp only [← c18523, ← c18524, k3, ← c18525] at e_283_7
  have e_283_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_8
  simp only [← c18532, ← c18533, k3, ← c18534] at e_283_8
  refine ⟨?_, ?_⟩
  · have hc := e_281_14
    simp only [e_283_7] at hc
    linear_combination c18535.trans k2 - hc
  · have hc := e_283_8
    simp only [e_283_6, e_281_15, e_283_7] at hc
    linear_combination c18536.trans k2 - hc

theorem publicBatchWrapper89_f2555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11408)) (a (.wire 170 7)) (a (.virt 19213)) (a (.virt 19214)) := by
  have c18537 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.1
  have c18538 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18539 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18540 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18541 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18542 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18543 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18544 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18545 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18546 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18547 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18548 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18549 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18550 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18551 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18552 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18553 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_281_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_16
  simp only [← c18543, ← c18544, ← c18545] at e_281_16
  have e_281_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_17
  simp only [← c18546, ← c18547, ← c18548] at e_281_17
  have e_283_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_9
  simp only [← c18537, k3, ← c18538, k3, ← c18539] at e_283_9
  have e_283_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_10
  simp only [← c18540, ← c18541, k3, ← c18542] at e_283_10
  have e_283_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_11
  simp only [← c18549, ← c18550, k3, ← c18551] at e_283_11
  refine ⟨?_, ?_⟩
  · have hc := e_281_16
    simp only [e_283_10] at hc
    linear_combination c18552.trans k2 - hc
  · have hc := e_283_11
    simp only [e_283_9, e_281_17, e_283_10] at hc
    linear_combination c18553.trans k2 - hc

theorem publicBatchWrapper89_f2556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 281 75) = band (a (.virt 19207)) (a (.virt 19209)) := by
  have c18554 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18555 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18556 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_281_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_18
  simp only [← c18554, ← c18555, ← c18556] at e_281_18
  have hr := e_281_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 281 79) = band (a (.virt 19211)) (a (.virt 19213)) := by
  have c18557 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18558 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18559 := (publicBatchWrapper89_copies579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_281_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row281 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_281_19
  simp only [← c18557, ← c18558, ← c18559] at e_281_19
  have hr := e_281_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 284 3) = band (a (.wire 281 75)) (a (.wire 281 79)) := by
  have c18560 := (publicBatchWrapper89_copies580 a h).1
  have c18561 := (publicBatchWrapper89_copies580 a h).2.1
  have c18562 := (publicBatchWrapper89_copies580 a h).2.2.1
  have e_284_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_0
  simp only [← c18560, ← c18561, ← c18562] at e_284_0
  have hr := e_284_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 55) = bor (a (.wire 57 7)) (a (.wire 284 3)) := by
  have c18563 := (publicBatchWrapper89_copies580 a h).2.2.2.1
  have c18564 := (publicBatchWrapper89_copies580 a h).2.2.2.2.1
  have c18565 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.1
  have c18566 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.1
  have c18567 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.1
  have c18568 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_275_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_13
  simp only [← c18566, ← c18567, k3, ← c18568] at e_275_13
  have e_282_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_1
  simp only [← c18563, ← c18564, ← c18565] at e_282_1
  have hr := e_275_13
  simp only [e_282_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 55) = a (.virt 201) := by
  have c18569 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.1
  exact c18569

theorem publicBatchWrapper89_f2561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11603)) (a (.wire 170 23)) (a (.virt 19215)) (a (.virt 19216)) := by
  have c18570 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18571 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18572 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18573 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18574 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18575 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18576 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18577 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18578 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18579 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18580 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18581 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18582 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18583 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18584 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18585 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18586 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_283_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_12
  simp only [← c18570, k3, ← c18571, k3, ← c18572] at e_283_12
  have e_283_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_13
  simp only [← c18573, ← c18574, k3, ← c18575] at e_283_13
  have e_283_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_14
  simp only [← c18582, ← c18583, k3, ← c18584] at e_283_14
  have e_284_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_1
  simp only [← c18576, ← c18577, ← c18578] at e_284_1
  have e_284_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_2
  simp only [← c18579, ← c18580, ← c18581] at e_284_2
  refine ⟨?_, ?_⟩
  · have hc := e_284_1
    simp only [e_283_13] at hc
    linear_combination c18585.trans k2 - hc
  · have hc := e_283_14
    simp only [e_283_12, e_284_2, e_283_13] at hc
    linear_combination c18586.trans k2 - hc

theorem publicBatchWrapper89_f2562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 59) = bor (a (.wire 57 51)) (a (.virt 19215)) := by
  have c18587 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18588 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18589 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18590 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18591 := (publicBatchWrapper89_copies580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18592 := (publicBatchWrapper89_copies581 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_275_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_14
  simp only [← c18590, ← c18591, k3, ← c18592] at e_275_14
  have e_282_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_2
  simp only [← c18587, ← c18588, ← c18589] at e_282_2
  have hr := e_275_14
  simp only [e_282_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 59) = a (.virt 201) := by
  have c18593 := (publicBatchWrapper89_copies581 a h).2.1
  exact c18593

theorem publicBatchWrapper89_f2564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11604)) (a (.wire 170 31)) (a (.virt 19217)) (a (.virt 19218)) := by
  have c18594 := (publicBatchWrapper89_copies581 a h).2.2.1
  have c18595 := (publicBatchWrapper89_copies581 a h).2.2.2.1
  have c18596 := (publicBatchWrapper89_copies581 a h).2.2.2.2.1
  have c18597 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.1
  have c18598 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.1
  have c18599 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.1
  have c18600 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.1
  have c18601 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.1
  have c18602 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18603 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18604 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18605 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18606 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18607 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18608 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18609 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18610 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_283_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_15
  simp only [← c18594, k3, ← c18595, k3, ← c18596] at e_283_15
  have e_283_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_16
  simp only [← c18597, ← c18598, k3, ← c18599] at e_283_16
  have e_283_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_17
  simp only [← c18606, ← c18607, k3, ← c18608] at e_283_17
  have e_284_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_3
  simp only [← c18600, ← c18601, ← c18602] at e_284_3
  have e_284_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_4
  simp only [← c18603, ← c18604, ← c18605] at e_284_4
  refine ⟨?_, ?_⟩
  · have hc := e_284_3
    simp only [e_283_16] at hc
    linear_combination c18609.trans k2 - hc
  · have hc := e_283_17
    simp only [e_283_15, e_284_4, e_283_16] at hc
    linear_combination c18610.trans k2 - hc

theorem publicBatchWrapper89_f2565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 63) = bor (a (.wire 57 51)) (a (.virt 19217)) := by
  have c18611 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18612 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18613 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18614 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18615 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18616 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_275_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_15
  simp only [← c18614, ← c18615, k3, ← c18616] at e_275_15
  have e_282_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_3
  simp only [← c18611, ← c18612, ← c18613] at e_282_3
  have hr := e_275_15
  simp only [e_282_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 63) = a (.virt 201) := by
  have c18617 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18617

theorem publicBatchWrapper89_f2567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11605)) (a (.wire 169 63)) (a (.virt 19219)) (a (.virt 19220)) := by
  have c18618 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18619 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18620 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18621 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18622 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18623 := (publicBatchWrapper89_copies581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18624 := (publicBatchWrapper89_copies582 a h).1
  have c18625 := (publicBatchWrapper89_copies582 a h).2.1
  have c18626 := (publicBatchWrapper89_copies582 a h).2.2.1
  have c18627 := (publicBatchWrapper89_copies582 a h).2.2.2.1
  have c18628 := (publicBatchWrapper89_copies582 a h).2.2.2.2.1
  have c18629 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.1
  have c18630 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.1
  have c18631 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.1
  have c18632 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.1
  have c18633 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.1
  have c18634 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_283_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_18
  simp only [← c18618, k3, ← c18619, k3, ← c18620] at e_283_18
  have e_283_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row283 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_283_19
  simp only [← c18621, ← c18622, k3, ← c18623] at e_283_19
  have e_284_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_5
  simp only [← c18624, ← c18625, ← c18626] at e_284_5
  have e_284_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_6
  simp only [← c18627, ← c18628, ← c18629] at e_284_6
  have e_285_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_0
  simp only [← c18630, ← c18631, k3, ← c18632] at e_285_0
  refine ⟨?_, ?_⟩
  · have hc := e_284_5
    simp only [e_283_19] at hc
    linear_combination c18633.trans k2 - hc
  · have hc := e_285_0
    simp only [e_283_18, e_284_6, e_283_19] at hc
    linear_combination c18634.trans k2 - hc

theorem publicBatchWrapper89_f2568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11606)) (a (.wire 169 71)) (a (.virt 19221)) (a (.virt 19222)) := by
  have c18635 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18636 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18637 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18638 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18639 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18640 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18641 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18642 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18643 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18644 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18645 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18646 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18647 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18648 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18649 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18650 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18651 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_284_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_7
  simp only [← c18641, ← c18642, ← c18643] at e_284_7
  have e_284_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_8
  simp only [← c18644, ← c18645, ← c18646] at e_284_8
  have e_285_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_1
  simp only [← c18635, k3, ← c18636, k3, ← c18637] at e_285_1
  have e_285_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_2
  simp only [← c18638, ← c18639, k3, ← c18640] at e_285_2
  have e_285_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_3
  simp only [← c18647, ← c18648, k3, ← c18649] at e_285_3
  refine ⟨?_, ?_⟩
  · have hc := e_284_7
    simp only [e_285_2] at hc
    linear_combination c18650.trans k2 - hc
  · have hc := e_285_3
    simp only [e_285_1, e_284_8, e_285_2] at hc
    linear_combination c18651.trans k2 - hc

theorem publicBatchWrapper89_f2569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11607)) (a (.wire 169 79)) (a (.virt 19223)) (a (.virt 19224)) := by
  have c18652 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18653 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18654 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18655 := (publicBatchWrapper89_copies582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18656 := (publicBatchWrapper89_copies583 a h).1
  have c18657 := (publicBatchWrapper89_copies583 a h).2.1
  have c18658 := (publicBatchWrapper89_copies583 a h).2.2.1
  have c18659 := (publicBatchWrapper89_copies583 a h).2.2.2.1
  have c18660 := (publicBatchWrapper89_copies583 a h).2.2.2.2.1
  have c18661 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.1
  have c18662 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.1
  have c18663 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.1
  have c18664 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.1
  have c18665 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.1
  have c18666 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18667 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18668 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_284_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_9
  simp only [← c18658, ← c18659, ← c18660] at e_284_9
  have e_284_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_10
  simp only [← c18661, ← c18662, ← c18663] at e_284_10
  have e_285_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_4
  simp only [← c18652, k3, ← c18653, k3, ← c18654] at e_285_4
  have e_285_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_5
  simp only [← c18655, ← c18656, k3, ← c18657] at e_285_5
  have e_285_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_6
  simp only [← c18664, ← c18665, k3, ← c18666] at e_285_6
  refine ⟨?_, ?_⟩
  · have hc := e_284_9
    simp only [e_285_5] at hc
    linear_combination c18667.trans k2 - hc
  · have hc := e_285_6
    simp only [e_285_4, e_284_10, e_285_5] at hc
    linear_combination c18668.trans k2 - hc

theorem publicBatchWrapper89_f2570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11608)) (a (.wire 170 7)) (a (.virt 19225)) (a (.virt 19226)) := by
  have c18669 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18670 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18671 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18672 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18673 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18674 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18675 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18676 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18677 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18678 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18679 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18680 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18681 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18682 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18683 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18684 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18685 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_284_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_11
  simp only [← c18675, ← c18676, ← c18677] at e_284_11
  have e_284_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_12
  simp only [← c18678, ← c18679, ← c18680] at e_284_12
  have e_285_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_7
  simp only [← c18669, k3, ← c18670, k3, ← c18671] at e_285_7
  have e_285_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_8
  simp only [← c18672, ← c18673, k3, ← c18674] at e_285_8
  have e_285_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_9
  simp only [← c18681, ← c18682, k3, ← c18683] at e_285_9
  refine ⟨?_, ?_⟩
  · have hc := e_284_11
    simp only [e_285_8] at hc
    linear_combination c18684.trans k2 - hc
  · have hc := e_285_9
    simp only [e_285_7, e_284_12, e_285_8] at hc
    linear_combination c18685.trans k2 - hc

theorem publicBatchWrapper89_f2571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 284 55) = band (a (.virt 19219)) (a (.virt 19221)) := by
  have c18686 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18687 := (publicBatchWrapper89_copies583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18688 := (publicBatchWrapper89_copies584 a h).1
  have e_284_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_13
  simp only [← c18686, ← c18687, ← c18688] at e_284_13
  have hr := e_284_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 284 59) = band (a (.virt 19223)) (a (.virt 19225)) := by
  have c18689 := (publicBatchWrapper89_copies584 a h).2.1
  have c18690 := (publicBatchWrapper89_copies584 a h).2.2.1
  have c18691 := (publicBatchWrapper89_copies584 a h).2.2.2.1
  have e_284_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_14
  simp only [← c18689, ← c18690, ← c18691] at e_284_14
  have hr := e_284_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 284 63) = band (a (.wire 284 55)) (a (.wire 284 59)) := by
  have c18692 := (publicBatchWrapper89_copies584 a h).2.2.2.2.1
  have c18693 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.1
  have c18694 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.1
  have e_284_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_15
  simp only [← c18692, ← c18693, ← c18694] at e_284_15
  have hr := e_284_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 67) = bor (a (.wire 57 51)) (a (.wire 284 63)) := by
  have c18695 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.1
  have c18696 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.1
  have c18697 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.1
  have c18698 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18699 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18700 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_275_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_16
  simp only [← c18698, ← c18699, k3, ← c18700] at e_275_16
  have e_282_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_4
  simp only [← c18695, ← c18696, ← c18697] at e_282_4
  have hr := e_275_16
  simp only [e_282_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 67) = a (.virt 201) := by
  have c18701 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18701

theorem publicBatchWrapper89_f2576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11803)) (a (.wire 170 23)) (a (.virt 19227)) (a (.virt 19228)) := by
  have c18702 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18703 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18704 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18705 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18706 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18707 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18708 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18709 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18710 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18711 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18712 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18713 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18714 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18715 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18716 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18717 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18718 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_284_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_16
  simp only [← c18708, ← c18709, ← c18710] at e_284_16
  have e_284_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_17
  simp only [← c18711, ← c18712, ← c18713] at e_284_17
  have e_285_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_10
  simp only [← c18702, k3, ← c18703, k3, ← c18704] at e_285_10
  have e_285_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_11
  simp only [← c18705, ← c18706, k3, ← c18707] at e_285_11
  have e_285_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_12
  simp only [← c18714, ← c18715, k3, ← c18716] at e_285_12
  refine ⟨?_, ?_⟩
  · have hc := e_284_16
    simp only [e_285_11] at hc
    linear_combination c18717.trans k2 - hc
  · have hc := e_285_12
    simp only [e_285_10, e_284_17, e_285_11] at hc
    linear_combination c18718.trans k2 - hc

theorem publicBatchWrapper89_f2577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 71) = bor (a (.wire 58 15)) (a (.virt 19227)) := by
  have c18719 := (publicBatchWrapper89_copies584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18720 := (publicBatchWrapper89_copies585 a h).1
  have c18721 := (publicBatchWrapper89_copies585 a h).2.1
  have c18722 := (publicBatchWrapper89_copies585 a h).2.2.1
  have c18723 := (publicBatchWrapper89_copies585 a h).2.2.2.1
  have c18724 := (publicBatchWrapper89_copies585 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_275_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_17
  simp only [← c18722, ← c18723, k3, ← c18724] at e_275_17
  have e_282_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_5
  simp only [← c18719, ← c18720, ← c18721] at e_282_5
  have hr := e_275_17
  simp only [e_282_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 71) = a (.virt 201) := by
  have c18725 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.1
  exact c18725

theorem publicBatchWrapper89_f2579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11804)) (a (.wire 170 31)) (a (.virt 19229)) (a (.virt 19230)) := by
  have c18726 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.1
  have c18727 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.1
  have c18728 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.1
  have c18729 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.1
  have c18730 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18731 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18732 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18733 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18734 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18735 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18736 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18737 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18738 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18739 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18740 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18741 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18742 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_284_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_18
  simp only [← c18732, ← c18733, ← c18734] at e_284_18
  have e_284_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row284 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_284_19
  simp only [← c18735, ← c18736, ← c18737] at e_284_19
  have e_285_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_13
  simp only [← c18726, k3, ← c18727, k3, ← c18728] at e_285_13
  have e_285_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_14
  simp only [← c18729, ← c18730, k3, ← c18731] at e_285_14
  have e_285_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_15
  simp only [← c18738, ← c18739, k3, ← c18740] at e_285_15
  refine ⟨?_, ?_⟩
  · have hc := e_284_18
    simp only [e_285_14] at hc
    linear_combination c18741.trans k2 - hc
  · have hc := e_285_15
    simp only [e_285_13, e_284_19, e_285_14] at hc
    linear_combination c18742.trans k2 - hc

theorem publicBatchWrapper89_f2580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 75) = bor (a (.wire 58 15)) (a (.virt 19229)) := by
  have c18743 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18744 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18745 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18746 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18747 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18748 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_275_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_18
  simp only [← c18746, ← c18747, k3, ← c18748] at e_275_18
  have e_282_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_6
  simp only [← c18743, ← c18744, ← c18745] at e_282_6
  have hr := e_275_18
  simp only [e_282_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 75) = a (.virt 201) := by
  have c18749 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18749

theorem publicBatchWrapper89_f2582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11805)) (a (.wire 169 63)) (a (.virt 19231)) (a (.virt 19232)) := by
  have c18750 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18751 := (publicBatchWrapper89_copies585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18752 := (publicBatchWrapper89_copies586 a h).1
  have c18753 := (publicBatchWrapper89_copies586 a h).2.1
  have c18754 := (publicBatchWrapper89_copies586 a h).2.2.1
  have c18755 := (publicBatchWrapper89_copies586 a h).2.2.2.1
  have c18756 := (publicBatchWrapper89_copies586 a h).2.2.2.2.1
  have c18757 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.1
  have c18758 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.1
  have c18759 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.1
  have c18760 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.1
  have c18761 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.1
  have c18762 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18763 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18764 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18765 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18766 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_285_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_16
  simp only [← c18750, k3, ← c18751, k3, ← c18752] at e_285_16
  have e_285_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_17
  simp only [← c18753, ← c18754, k3, ← c18755] at e_285_17
  have e_285_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_18
  simp only [← c18762, ← c18763, k3, ← c18764] at e_285_18
  have e_286_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_0
  simp only [← c18756, ← c18757, ← c18758] at e_286_0
  have e_286_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_1
  simp only [← c18759, ← c18760, ← c18761] at e_286_1
  refine ⟨?_, ?_⟩
  · have hc := e_286_0
    simp only [e_285_17] at hc
    linear_combination c18765.trans k2 - hc
  · have hc := e_285_18
    simp only [e_285_16, e_286_1, e_285_17] at hc
    linear_combination c18766.trans k2 - hc

theorem publicBatchWrapper89_f2583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11806)) (a (.wire 169 71)) (a (.virt 19233)) (a (.virt 19234)) := by
  have c18767 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18768 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18769 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18770 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18771 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18772 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18773 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18774 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18775 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18776 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18777 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18778 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18779 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18780 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18781 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18782 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18783 := (publicBatchWrapper89_copies586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_285_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row285 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_285_19
  simp only [← c18767, k3, ← c18768, k3, ← c18769] at e_285_19
  have e_286_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_2
  simp only [← c18773, ← c18774, ← c18775] at e_286_2
  have e_286_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_3
  simp only [← c18776, ← c18777, ← c18778] at e_286_3
  have e_287_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_0
  simp only [← c18770, ← c18771, k3, ← c18772] at e_287_0
  have e_287_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_1
  simp only [← c18779, ← c18780, k3, ← c18781] at e_287_1
  refine ⟨?_, ?_⟩
  · have hc := e_286_2
    simp only [e_287_0] at hc
    linear_combination c18782.trans k2 - hc
  · have hc := e_287_1
    simp only [e_285_19, e_286_3, e_287_0] at hc
    linear_combination c18783.trans k2 - hc

theorem publicBatchWrapper89_f2584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11807)) (a (.wire 169 79)) (a (.virt 19235)) (a (.virt 19236)) := by
  have c18784 := (publicBatchWrapper89_copies587 a h).1
  have c18785 := (publicBatchWrapper89_copies587 a h).2.1
  have c18786 := (publicBatchWrapper89_copies587 a h).2.2.1
  have c18787 := (publicBatchWrapper89_copies587 a h).2.2.2.1
  have c18788 := (publicBatchWrapper89_copies587 a h).2.2.2.2.1
  have c18789 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.1
  have c18790 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.1
  have c18791 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.1
  have c18792 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.1
  have c18793 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.1
  have c18794 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18795 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18796 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18797 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18798 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18799 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18800 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_286_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_4
  simp only [← c18790, ← c18791, ← c18792] at e_286_4
  have e_286_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_5
  simp only [← c18793, ← c18794, ← c18795] at e_286_5
  have e_287_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_2
  simp only [← c18784, k3, ← c18785, k3, ← c18786] at e_287_2
  have e_287_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_3
  simp only [← c18787, ← c18788, k3, ← c18789] at e_287_3
  have e_287_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_4
  simp only [← c18796, ← c18797, k3, ← c18798] at e_287_4
  refine ⟨?_, ?_⟩
  · have hc := e_286_4
    simp only [e_287_3] at hc
    linear_combination c18799.trans k2 - hc
  · have hc := e_287_4
    simp only [e_287_2, e_286_5, e_287_3] at hc
    linear_combination c18800.trans k2 - hc

theorem publicBatchWrapper89_f2585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11808)) (a (.wire 170 7)) (a (.virt 19237)) (a (.virt 19238)) := by
  have c18801 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18802 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18803 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18804 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18805 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18806 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18807 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18808 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18809 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18810 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18811 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18812 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18813 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18814 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18815 := (publicBatchWrapper89_copies587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18816 := (publicBatchWrapper89_copies588 a h).1
  have c18817 := (publicBatchWrapper89_copies588 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_286_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_6
  simp only [← c18807, ← c18808, ← c18809] at e_286_6
  have e_286_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_7
  simp only [← c18810, ← c18811, ← c18812] at e_286_7
  have e_287_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_5
  simp only [← c18801, k3, ← c18802, k3, ← c18803] at e_287_5
  have e_287_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_6
  simp only [← c18804, ← c18805, k3, ← c18806] at e_287_6
  have e_287_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_7
  simp only [← c18813, ← c18814, k3, ← c18815] at e_287_7
  refine ⟨?_, ?_⟩
  · have hc := e_286_6
    simp only [e_287_6] at hc
    linear_combination c18816.trans k2 - hc
  · have hc := e_287_7
    simp only [e_287_5, e_286_7, e_287_6] at hc
    linear_combination c18817.trans k2 - hc

theorem publicBatchWrapper89_f2586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 286 35) = band (a (.virt 19231)) (a (.virt 19233)) := by
  have c18818 := (publicBatchWrapper89_copies588 a h).2.2.1
  have c18819 := (publicBatchWrapper89_copies588 a h).2.2.2.1
  have c18820 := (publicBatchWrapper89_copies588 a h).2.2.2.2.1
  have e_286_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_8
  simp only [← c18818, ← c18819, ← c18820] at e_286_8
  have hr := e_286_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 286 39) = band (a (.virt 19235)) (a (.virt 19237)) := by
  have c18821 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.1
  have c18822 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.1
  have c18823 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.1
  have e_286_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_9
  simp only [← c18821, ← c18822, ← c18823] at e_286_9
  have hr := e_286_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 286 43) = band (a (.wire 286 35)) (a (.wire 286 39)) := by
  have c18824 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.1
  have c18825 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.1
  have c18826 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_286_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_10
  simp only [← c18824, ← c18825, ← c18826] at e_286_10
  have hr := e_286_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 79) = bor (a (.wire 58 15)) (a (.wire 286 43)) := by
  have c18827 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18828 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18829 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18830 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18831 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18832 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_275_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row275 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_275_19
  simp only [← c18830, ← c18831, k3, ← c18832] at e_275_19
  have e_282_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_7
  simp only [← c18827, ← c18828, ← c18829] at e_282_7
  have hr := e_275_19
  simp only [e_282_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 275 79) = a (.virt 201) := by
  have c18833 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18833

theorem publicBatchWrapper89_f2591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12003)) (a (.wire 170 23)) (a (.virt 19239)) (a (.virt 19240)) := by
  have c18834 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18835 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18836 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18837 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18838 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18839 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18840 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18841 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18842 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18843 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18844 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18845 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18846 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18847 := (publicBatchWrapper89_copies588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18848 := (publicBatchWrapper89_copies589 a h).1
  have c18849 := (publicBatchWrapper89_copies589 a h).2.1
  have c18850 := (publicBatchWrapper89_copies589 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_286_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_11
  simp only [← c18840, ← c18841, ← c18842] at e_286_11
  have e_286_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_12
  simp only [← c18843, ← c18844, ← c18845] at e_286_12
  have e_287_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_8
  simp only [← c18834, k3, ← c18835, k3, ← c18836] at e_287_8
  have e_287_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_9
  simp only [← c18837, ← c18838, k3, ← c18839] at e_287_9
  have e_287_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_10
  simp only [← c18846, ← c18847, k3, ← c18848] at e_287_10
  refine ⟨?_, ?_⟩
  · have hc := e_286_11
    simp only [e_287_9] at hc
    linear_combination c18849.trans k2 - hc
  · have hc := e_287_10
    simp only [e_287_8, e_286_12, e_287_9] at hc
    linear_combination c18850.trans k2 - hc

theorem publicBatchWrapper89_f2592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 3) = bor (a (.wire 58 59)) (a (.virt 19239)) := by
  have c18851 := (publicBatchWrapper89_copies589 a h).2.2.2.1
  have c18852 := (publicBatchWrapper89_copies589 a h).2.2.2.2.1
  have c18853 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.1
  have c18854 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.1
  have c18855 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.1
  have c18856 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_8
  simp only [← c18851, ← c18852, ← c18853] at e_282_8
  have e_288_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_0
  simp only [← c18854, ← c18855, k3, ← c18856] at e_288_0
  have hr := e_288_0
  simp only [e_282_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 3) = a (.virt 201) := by
  have c18857 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.1
  exact c18857

theorem publicBatchWrapper89_f2594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12004)) (a (.wire 170 31)) (a (.virt 19241)) (a (.virt 19242)) := by
  have c18858 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18859 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18860 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18861 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18862 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18863 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18864 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18865 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18866 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18867 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18868 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18869 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18870 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18871 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18872 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18873 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18874 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_286_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_13
  simp only [← c18864, ← c18865, ← c18866] at e_286_13
  have e_286_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_14
  simp only [← c18867, ← c18868, ← c18869] at e_286_14
  have e_287_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_11
  simp only [← c18858, k3, ← c18859, k3, ← c18860] at e_287_11
  have e_287_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_12
  simp only [← c18861, ← c18862, k3, ← c18863] at e_287_12
  have e_287_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_13
  simp only [← c18870, ← c18871, k3, ← c18872] at e_287_13
  refine ⟨?_, ?_⟩
  · have hc := e_286_13
    simp only [e_287_12] at hc
    linear_combination c18873.trans k2 - hc
  · have hc := e_287_13
    simp only [e_287_11, e_286_14, e_287_12] at hc
    linear_combination c18874.trans k2 - hc

theorem publicBatchWrapper89_f2595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 7) = bor (a (.wire 58 59)) (a (.virt 19241)) := by
  have c18875 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18876 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18877 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18878 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18879 := (publicBatchWrapper89_copies589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18880 := (publicBatchWrapper89_copies590 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_9
  simp only [← c18875, ← c18876, ← c18877] at e_282_9
  have e_288_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_1
  simp only [← c18878, ← c18879, k3, ← c18880] at e_288_1
  have hr := e_288_1
  simp only [e_282_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 7) = a (.virt 201) := by
  have c18881 := (publicBatchWrapper89_copies590 a h).2.1
  exact c18881

theorem publicBatchWrapper89_f2597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12005)) (a (.wire 169 63)) (a (.virt 19243)) (a (.virt 19244)) := by
  have c18882 := (publicBatchWrapper89_copies590 a h).2.2.1
  have c18883 := (publicBatchWrapper89_copies590 a h).2.2.2.1
  have c18884 := (publicBatchWrapper89_copies590 a h).2.2.2.2.1
  have c18885 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.1
  have c18886 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.1
  have c18887 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.1
  have c18888 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.1
  have c18889 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.1
  have c18890 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18891 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18892 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18893 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18894 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18895 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18896 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18897 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18898 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_286_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_15
  simp only [← c18888, ← c18889, ← c18890] at e_286_15
  have e_286_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_16
  simp only [← c18891, ← c18892, ← c18893] at e_286_16
  have e_287_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_14
  simp only [← c18882, k3, ← c18883, k3, ← c18884] at e_287_14
  have e_287_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_15
  simp only [← c18885, ← c18886, k3, ← c18887] at e_287_15
  have e_287_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_16
  simp only [← c18894, ← c18895, k3, ← c18896] at e_287_16
  refine ⟨?_, ?_⟩
  · have hc := e_286_15
    simp only [e_287_15] at hc
    linear_combination c18897.trans k2 - hc
  · have hc := e_287_16
    simp only [e_287_14, e_286_16, e_287_15] at hc
    linear_combination c18898.trans k2 - hc

theorem publicBatchWrapper89_f2598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12006)) (a (.wire 169 71)) (a (.virt 19245)) (a (.virt 19246)) := by
  have c18899 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18900 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18901 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18902 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18903 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18904 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18905 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18906 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18907 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18908 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18909 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18910 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18911 := (publicBatchWrapper89_copies590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18912 := (publicBatchWrapper89_copies591 a h).1
  have c18913 := (publicBatchWrapper89_copies591 a h).2.1
  have c18914 := (publicBatchWrapper89_copies591 a h).2.2.1
  have c18915 := (publicBatchWrapper89_copies591 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_286_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_17
  simp only [← c18905, ← c18906, ← c18907] at e_286_17
  have e_286_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_18
  simp only [← c18908, ← c18909, ← c18910] at e_286_18
  have e_287_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_17
  simp only [← c18899, k3, ← c18900, k3, ← c18901] at e_287_17
  have e_287_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_18
  simp only [← c18902, ← c18903, k3, ← c18904] at e_287_18
  have e_287_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row287 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_287_19
  simp only [← c18911, ← c18912, k3, ← c18913] at e_287_19
  refine ⟨?_, ?_⟩
  · have hc := e_286_17
    simp only [e_287_18] at hc
    linear_combination c18914.trans k2 - hc
  · have hc := e_287_19
    simp only [e_287_17, e_286_18, e_287_18] at hc
    linear_combination c18915.trans k2 - hc

theorem publicBatchWrapper89_f2599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12007)) (a (.wire 169 79)) (a (.virt 19247)) (a (.virt 19248)) := by
  have c18916 := (publicBatchWrapper89_copies591 a h).2.2.2.2.1
  have c18917 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.1
  have c18918 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.1
  have c18919 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.1
  have c18920 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.1
  have c18921 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.1
  have c18922 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18923 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18924 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18925 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18926 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18927 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18928 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18929 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18930 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18931 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18932 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_286_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row286 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_286_19
  simp only [← c18922, ← c18923, ← c18924] at e_286_19
  have e_289_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_0
  simp only [← c18916, k3, ← c18917, k3, ← c18918] at e_289_0
  have e_289_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_1
  simp only [← c18919, ← c18920, k3, ← c18921] at e_289_1
  have e_289_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_2
  simp only [← c18928, ← c18929, k3, ← c18930] at e_289_2
  have e_290_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_0
  simp only [← c18925, ← c18926, ← c18927] at e_290_0
  refine ⟨?_, ?_⟩
  · have hc := e_286_19
    simp only [e_289_1] at hc
    linear_combination c18931.trans k2 - hc
  · have hc := e_289_2
    simp only [e_289_0, e_290_0, e_289_1] at hc
    linear_combination c18932.trans k2 - hc

theorem publicBatchWrapper89_f2600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12008)) (a (.wire 170 7)) (a (.virt 19249)) (a (.virt 19250)) := by
  have c18933 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18934 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18935 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18936 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18937 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18938 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18939 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18940 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18941 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18942 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18943 := (publicBatchWrapper89_copies591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18944 := (publicBatchWrapper89_copies592 a h).1
  have c18945 := (publicBatchWrapper89_copies592 a h).2.1
  have c18946 := (publicBatchWrapper89_copies592 a h).2.2.1
  have c18947 := (publicBatchWrapper89_copies592 a h).2.2.2.1
  have c18948 := (publicBatchWrapper89_copies592 a h).2.2.2.2.1
  have c18949 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_289_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_3
  simp only [← c18933, k3, ← c18934, k3, ← c18935] at e_289_3
  have e_289_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_4
  simp only [← c18936, ← c18937, k3, ← c18938] at e_289_4
  have e_289_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_5
  simp only [← c18945, ← c18946, k3, ← c18947] at e_289_5
  have e_290_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_1
  simp only [← c18939, ← c18940, ← c18941] at e_290_1
  have e_290_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_2
  simp only [← c18942, ← c18943, ← c18944] at e_290_2
  refine ⟨?_, ?_⟩
  · have hc := e_290_1
    simp only [e_289_4] at hc
    linear_combination c18948.trans k2 - hc
  · have hc := e_289_5
    simp only [e_289_3, e_290_2, e_289_4] at hc
    linear_combination c18949.trans k2 - hc

theorem publicBatchWrapper89_f2601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 290 15) = band (a (.virt 19243)) (a (.virt 19245)) := by
  have c18950 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.1
  have c18951 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.1
  have c18952 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.1
  have e_290_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_3
  simp only [← c18950, ← c18951, ← c18952] at e_290_3
  have hr := e_290_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 290 19) = band (a (.virt 19247)) (a (.virt 19249)) := by
  have c18953 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.1
  have c18954 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18955 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_290_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_4
  simp only [← c18953, ← c18954, ← c18955] at e_290_4
  have hr := e_290_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 290 23) = band (a (.wire 290 15)) (a (.wire 290 19)) := by
  have c18956 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18957 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18958 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_290_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_5
  simp only [← c18956, ← c18957, ← c18958] at e_290_5
  have hr := e_290_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 11) = bor (a (.wire 58 59)) (a (.wire 290 23)) := by
  have c18959 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18960 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18961 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18962 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18963 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18964 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_10
  simp only [← c18959, ← c18960, ← c18961] at e_282_10
  have e_288_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_2
  simp only [← c18962, ← c18963, k3, ← c18964] at e_288_2
  have hr := e_288_2
  simp only [e_282_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 11) = a (.virt 201) := by
  have c18965 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18965

theorem publicBatchWrapper89_f2606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12203)) (a (.wire 170 23)) (a (.virt 19251)) (a (.virt 19252)) := by
  have c18966 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18967 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18968 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18969 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18970 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18971 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18972 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18973 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18974 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18975 := (publicBatchWrapper89_copies592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c18976 := (publicBatchWrapper89_copies593 a h).1
  have c18977 := (publicBatchWrapper89_copies593 a h).2.1
  have c18978 := (publicBatchWrapper89_copies593 a h).2.2.1
  have c18979 := (publicBatchWrapper89_copies593 a h).2.2.2.1
  have c18980 := (publicBatchWrapper89_copies593 a h).2.2.2.2.1
  have c18981 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.1
  have c18982 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_289_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_6
  simp only [← c18966, k3, ← c18967, k3, ← c18968] at e_289_6
  have e_289_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_7
  simp only [← c18969, ← c18970, k3, ← c18971] at e_289_7
  have e_289_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_8
  simp only [← c18978, ← c18979, k3, ← c18980] at e_289_8
  have e_290_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_6
  simp only [← c18972, ← c18973, ← c18974] at e_290_6
  have e_290_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_7
  simp only [← c18975, ← c18976, ← c18977] at e_290_7
  refine ⟨?_, ?_⟩
  · have hc := e_290_6
    simp only [e_289_7] at hc
    linear_combination c18981.trans k2 - hc
  · have hc := e_289_8
    simp only [e_289_6, e_290_7, e_289_7] at hc
    linear_combination c18982.trans k2 - hc

theorem publicBatchWrapper89_f2607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 15) = bor (a (.wire 60 23)) (a (.virt 19251)) := by
  have c18983 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.1
  have c18984 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.1
  have c18985 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.1
  have c18986 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.1
  have c18987 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c18988 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_11
  simp only [← c18983, ← c18984, ← c18985] at e_282_11
  have e_288_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_3
  simp only [← c18986, ← c18987, k3, ← c18988] at e_288_3
  have hr := e_288_3
  simp only [e_282_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 15) = a (.virt 201) := by
  have c18989 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18989

theorem publicBatchWrapper89_f2609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12204)) (a (.wire 170 31)) (a (.virt 19253)) (a (.virt 19254)) := by
  have c18990 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18991 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18992 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18993 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18994 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18995 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18996 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18997 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18998 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18999 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19000 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19001 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19002 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19003 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19004 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19005 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19006 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_289_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_9
  simp only [← c18990, k3, ← c18991, k3, ← c18992] at e_289_9
  have e_289_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_10
  simp only [← c18993, ← c18994, k3, ← c18995] at e_289_10
  have e_289_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_11
  simp only [← c19002, ← c19003, k3, ← c19004] at e_289_11
  have e_290_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_8
  simp only [← c18996, ← c18997, ← c18998] at e_290_8
  have e_290_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_9
  simp only [← c18999, ← c19000, ← c19001] at e_290_9
  refine ⟨?_, ?_⟩
  · have hc := e_290_8
    simp only [e_289_10] at hc
    linear_combination c19005.trans k2 - hc
  · have hc := e_289_11
    simp only [e_289_9, e_290_9, e_289_10] at hc
    linear_combination c19006.trans k2 - hc

theorem publicBatchWrapper89_f2610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 19) = bor (a (.wire 60 23)) (a (.virt 19253)) := by
  have c19007 := (publicBatchWrapper89_copies593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19008 := (publicBatchWrapper89_copies594 a h).1
  have c19009 := (publicBatchWrapper89_copies594 a h).2.1
  have c19010 := (publicBatchWrapper89_copies594 a h).2.2.1
  have c19011 := (publicBatchWrapper89_copies594 a h).2.2.2.1
  have c19012 := (publicBatchWrapper89_copies594 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_12
  simp only [← c19007, ← c19008, ← c19009] at e_282_12
  have e_288_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_4
  simp only [← c19010, ← c19011, k3, ← c19012] at e_288_4
  have hr := e_288_4
  simp only [e_282_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 19) = a (.virt 201) := by
  have c19013 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.1
  exact c19013

theorem publicBatchWrapper89_f2612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12205)) (a (.wire 169 63)) (a (.virt 19255)) (a (.virt 19256)) := by
  have c19014 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.1
  have c19015 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.1
  have c19016 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.1
  have c19017 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.1
  have c19018 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19019 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19020 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19021 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19022 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19023 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19024 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19025 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19026 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19027 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19028 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19029 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19030 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_289_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_12
  simp only [← c19014, k3, ← c19015, k3, ← c19016] at e_289_12
  have e_289_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_13
  simp only [← c19017, ← c19018, k3, ← c19019] at e_289_13
  have e_289_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_14
  simp only [← c19026, ← c19027, k3, ← c19028] at e_289_14
  have e_290_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_10
  simp only [← c19020, ← c19021, ← c19022] at e_290_10
  have e_290_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_11
  simp only [← c19023, ← c19024, ← c19025] at e_290_11
  refine ⟨?_, ?_⟩
  · have hc := e_290_10
    simp only [e_289_13] at hc
    linear_combination c19029.trans k2 - hc
  · have hc := e_289_14
    simp only [e_289_12, e_290_11, e_289_13] at hc
    linear_combination c19030.trans k2 - hc

theorem publicBatchWrapper89_f2613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12206)) (a (.wire 169 71)) (a (.virt 19257)) (a (.virt 19258)) := by
  have c19031 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19032 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19033 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19034 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19035 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19036 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19037 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19038 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19039 := (publicBatchWrapper89_copies594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19040 := (publicBatchWrapper89_copies595 a h).1
  have c19041 := (publicBatchWrapper89_copies595 a h).2.1
  have c19042 := (publicBatchWrapper89_copies595 a h).2.2.1
  have c19043 := (publicBatchWrapper89_copies595 a h).2.2.2.1
  have c19044 := (publicBatchWrapper89_copies595 a h).2.2.2.2.1
  have c19045 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.1
  have c19046 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.1
  have c19047 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_289_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_15
  simp only [← c19031, k3, ← c19032, k3, ← c19033] at e_289_15
  have e_289_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_16
  simp only [← c19034, ← c19035, k3, ← c19036] at e_289_16
  have e_289_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_17
  simp only [← c19043, ← c19044, k3, ← c19045] at e_289_17
  have e_290_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_12
  simp only [← c19037, ← c19038, ← c19039] at e_290_12
  have e_290_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_13
  simp only [← c19040, ← c19041, ← c19042] at e_290_13
  refine ⟨?_, ?_⟩
  · have hc := e_290_12
    simp only [e_289_16] at hc
    linear_combination c19046.trans k2 - hc
  · have hc := e_289_17
    simp only [e_289_15, e_290_13, e_289_16] at hc
    linear_combination c19047.trans k2 - hc

theorem publicBatchWrapper89_f2614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12207)) (a (.wire 169 79)) (a (.virt 19259)) (a (.virt 19260)) := by
  have c19048 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.1
  have c19049 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.1
  have c19050 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19051 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19052 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19053 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19054 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19055 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19056 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19057 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19058 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19059 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19060 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19061 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19062 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19063 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19064 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_289_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_18
  simp only [← c19048, k3, ← c19049, k3, ← c19050] at e_289_18
  have e_289_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row289 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_289_19
  simp only [← c19051, ← c19052, k3, ← c19053] at e_289_19
  have e_290_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_14
  simp only [← c19054, ← c19055, ← c19056] at e_290_14
  have e_290_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_15
  simp only [← c19057, ← c19058, ← c19059] at e_290_15
  have e_291_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_0
  simp only [← c19060, ← c19061, k3, ← c19062] at e_291_0
  refine ⟨?_, ?_⟩
  · have hc := e_290_14
    simp only [e_289_19] at hc
    linear_combination c19063.trans k2 - hc
  · have hc := e_291_0
    simp only [e_289_18, e_290_15, e_289_19] at hc
    linear_combination c19064.trans k2 - hc

theorem publicBatchWrapper89_f2615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12208)) (a (.wire 170 7)) (a (.virt 19261)) (a (.virt 19262)) := by
  have c19065 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19066 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19067 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19068 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19069 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19070 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19071 := (publicBatchWrapper89_copies595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19072 := (publicBatchWrapper89_copies596 a h).1
  have c19073 := (publicBatchWrapper89_copies596 a h).2.1
  have c19074 := (publicBatchWrapper89_copies596 a h).2.2.1
  have c19075 := (publicBatchWrapper89_copies596 a h).2.2.2.1
  have c19076 := (publicBatchWrapper89_copies596 a h).2.2.2.2.1
  have c19077 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.1
  have c19078 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.1
  have c19079 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.1
  have c19080 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.1
  have c19081 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_290_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_16
  simp only [← c19071, ← c19072, ← c19073] at e_290_16
  have e_290_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_17
  simp only [← c19074, ← c19075, ← c19076] at e_290_17
  have e_291_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_1
  simp only [← c19065, k3, ← c19066, k3, ← c19067] at e_291_1
  have e_291_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_2
  simp only [← c19068, ← c19069, k3, ← c19070] at e_291_2
  have e_291_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_3
  simp only [← c19077, ← c19078, k3, ← c19079] at e_291_3
  refine ⟨?_, ?_⟩
  · have hc := e_290_16
    simp only [e_291_2] at hc
    linear_combination c19080.trans k2 - hc
  · have hc := e_291_3
    simp only [e_291_1, e_290_17, e_291_2] at hc
    linear_combination c19081.trans k2 - hc

theorem publicBatchWrapper89_f2616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 290 75) = band (a (.virt 19255)) (a (.virt 19257)) := by
  have c19082 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19083 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19084 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_290_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_18
  simp only [← c19082, ← c19083, ← c19084] at e_290_18
  have hr := e_290_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 290 79) = band (a (.virt 19259)) (a (.virt 19261)) := by
  have c19085 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19086 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19087 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_290_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row290 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_290_19
  simp only [← c19085, ← c19086, ← c19087] at e_290_19
  have hr := e_290_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 292 3) = band (a (.wire 290 75)) (a (.wire 290 79)) := by
  have c19088 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19089 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19090 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_292_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_0
  simp only [← c19088, ← c19089, ← c19090] at e_292_0
  have hr := e_292_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 23) = bor (a (.wire 60 23)) (a (.wire 292 3)) := by
  have c19091 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19092 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19093 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19094 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19095 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19096 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_13
  simp only [← c19091, ← c19092, ← c19093] at e_282_13
  have e_288_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_5
  simp only [← c19094, ← c19095, k3, ← c19096] at e_288_5
  have hr := e_288_5
  simp only [e_282_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 23) = a (.virt 201) := by
  have c19097 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19097

theorem publicBatchWrapper89_f2621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12403)) (a (.wire 170 23)) (a (.virt 19263)) (a (.virt 19264)) := by
  have c19098 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19099 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19100 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19101 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19102 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19103 := (publicBatchWrapper89_copies596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19104 := (publicBatchWrapper89_copies597 a h).1
  have c19105 := (publicBatchWrapper89_copies597 a h).2.1
  have c19106 := (publicBatchWrapper89_copies597 a h).2.2.1
  have c19107 := (publicBatchWrapper89_copies597 a h).2.2.2.1
  have c19108 := (publicBatchWrapper89_copies597 a h).2.2.2.2.1
  have c19109 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.1
  have c19110 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.1
  have c19111 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.1
  have c19112 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.1
  have c19113 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.1
  have c19114 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_291_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_4
  simp only [← c19098, k3, ← c19099, k3, ← c19100] at e_291_4
  have e_291_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_5
  simp only [← c19101, ← c19102, k3, ← c19103] at e_291_5
  have e_291_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_6
  simp only [← c19110, ← c19111, k3, ← c19112] at e_291_6
  have e_292_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_1
  simp only [← c19104, ← c19105, ← c19106] at e_292_1
  have e_292_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_2
  simp only [← c19107, ← c19108, ← c19109] at e_292_2
  refine ⟨?_, ?_⟩
  · have hc := e_292_1
    simp only [e_291_5] at hc
    linear_combination c19113.trans k2 - hc
  · have hc := e_291_6
    simp only [e_291_4, e_292_2, e_291_5] at hc
    linear_combination c19114.trans k2 - hc

theorem publicBatchWrapper89_f2622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 27) = bor (a (.wire 60 67)) (a (.virt 19263)) := by
  have c19115 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19116 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19117 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19118 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19119 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19120 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_14
  simp only [← c19115, ← c19116, ← c19117] at e_282_14
  have e_288_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_6
  simp only [← c19118, ← c19119, k3, ← c19120] at e_288_6
  have hr := e_288_6
  simp only [e_282_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 27) = a (.virt 201) := by
  have c19121 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19121

theorem publicBatchWrapper89_f2624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12404)) (a (.wire 170 31)) (a (.virt 19265)) (a (.virt 19266)) := by
  have c19122 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19123 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19124 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19125 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19126 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19127 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19128 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19129 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19130 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19131 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19132 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19133 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19134 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19135 := (publicBatchWrapper89_copies597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19136 := (publicBatchWrapper89_copies598 a h).1
  have c19137 := (publicBatchWrapper89_copies598 a h).2.1
  have c19138 := (publicBatchWrapper89_copies598 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_291_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_7
  simp only [← c19122, k3, ← c19123, k3, ← c19124] at e_291_7
  have e_291_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_8
  simp only [← c19125, ← c19126, k3, ← c19127] at e_291_8
  have e_291_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_9
  simp only [← c19134, ← c19135, k3, ← c19136] at e_291_9
  have e_292_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_3
  simp only [← c19128, ← c19129, ← c19130] at e_292_3
  have e_292_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_4
  simp only [← c19131, ← c19132, ← c19133] at e_292_4
  refine ⟨?_, ?_⟩
  · have hc := e_292_3
    simp only [e_291_8] at hc
    linear_combination c19137.trans k2 - hc
  · have hc := e_291_9
    simp only [e_291_7, e_292_4, e_291_8] at hc
    linear_combination c19138.trans k2 - hc

theorem publicBatchWrapper89_f2625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 31) = bor (a (.wire 60 67)) (a (.virt 19265)) := by
  have c19139 := (publicBatchWrapper89_copies598 a h).2.2.2.1
  have c19140 := (publicBatchWrapper89_copies598 a h).2.2.2.2.1
  have c19141 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.1
  have c19142 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.1
  have c19143 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.1
  have c19144 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_15
  simp only [← c19139, ← c19140, ← c19141] at e_282_15
  have e_288_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_7
  simp only [← c19142, ← c19143, k3, ← c19144] at e_288_7
  have hr := e_288_7
  simp only [e_282_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 31) = a (.virt 201) := by
  have c19145 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.1
  exact c19145

theorem publicBatchWrapper89_f2627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12405)) (a (.wire 169 63)) (a (.virt 19267)) (a (.virt 19268)) := by
  have c19146 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19147 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19148 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19149 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19150 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19151 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19152 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19153 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19154 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19155 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19156 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19157 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19158 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19159 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19160 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19161 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19162 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_291_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_10
  simp only [← c19146, k3, ← c19147, k3, ← c19148] at e_291_10
  have e_291_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_11
  simp only [← c19149, ← c19150, k3, ← c19151] at e_291_11
  have e_291_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_12
  simp only [← c19158, ← c19159, k3, ← c19160] at e_291_12
  have e_292_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_5
  simp only [← c19152, ← c19153, ← c19154] at e_292_5
  have e_292_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_6
  simp only [← c19155, ← c19156, ← c19157] at e_292_6
  refine ⟨?_, ?_⟩
  · have hc := e_292_5
    simp only [e_291_11] at hc
    linear_combination c19161.trans k2 - hc
  · have hc := e_291_12
    simp only [e_291_10, e_292_6, e_291_11] at hc
    linear_combination c19162.trans k2 - hc

theorem publicBatchWrapper89_f2628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12406)) (a (.wire 169 71)) (a (.virt 19269)) (a (.virt 19270)) := by
  have c19163 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19164 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19165 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19166 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19167 := (publicBatchWrapper89_copies598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19168 := (publicBatchWrapper89_copies599 a h).1
  have c19169 := (publicBatchWrapper89_copies599 a h).2.1
  have c19170 := (publicBatchWrapper89_copies599 a h).2.2.1
  have c19171 := (publicBatchWrapper89_copies599 a h).2.2.2.1
  have c19172 := (publicBatchWrapper89_copies599 a h).2.2.2.2.1
  have c19173 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.1
  have c19174 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.1
  have c19175 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.1
  have c19176 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.1
  have c19177 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.1
  have c19178 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19179 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_291_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_13
  simp only [← c19163, k3, ← c19164, k3, ← c19165] at e_291_13
  have e_291_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_14
  simp only [← c19166, ← c19167, k3, ← c19168] at e_291_14
  have e_291_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_15
  simp only [← c19175, ← c19176, k3, ← c19177] at e_291_15
  have e_292_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_7
  simp only [← c19169, ← c19170, ← c19171] at e_292_7
  have e_292_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_8
  simp only [← c19172, ← c19173, ← c19174] at e_292_8
  refine ⟨?_, ?_⟩
  · have hc := e_292_7
    simp only [e_291_14] at hc
    linear_combination c19178.trans k2 - hc
  · have hc := e_291_15
    simp only [e_291_13, e_292_8, e_291_14] at hc
    linear_combination c19179.trans k2 - hc

theorem publicBatchWrapper89_f2629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12407)) (a (.wire 169 79)) (a (.virt 19271)) (a (.virt 19272)) := by
  have c19180 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19181 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19182 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19183 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19184 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19185 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19186 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19187 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19188 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19189 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19190 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19191 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19192 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19193 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19194 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19195 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19196 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_291_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_16
  simp only [← c19180, k3, ← c19181, k3, ← c19182] at e_291_16
  have e_291_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_17
  simp only [← c19183, ← c19184, k3, ← c19185] at e_291_17
  have e_291_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_18
  simp only [← c19192, ← c19193, k3, ← c19194] at e_291_18
  have e_292_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_9
  simp only [← c19186, ← c19187, ← c19188] at e_292_9
  have e_292_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_10
  simp only [← c19189, ← c19190, ← c19191] at e_292_10
  refine ⟨?_, ?_⟩
  · have hc := e_292_9
    simp only [e_291_17] at hc
    linear_combination c19195.trans k2 - hc
  · have hc := e_291_18
    simp only [e_291_16, e_292_10, e_291_17] at hc
    linear_combination c19196.trans k2 - hc

theorem publicBatchWrapper89_f2630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12408)) (a (.wire 170 7)) (a (.virt 19273)) (a (.virt 19274)) := by
  have c19197 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19198 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19199 := (publicBatchWrapper89_copies599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19200 := (publicBatchWrapper89_copies600 a h).1
  have c19201 := (publicBatchWrapper89_copies600 a h).2.1
  have c19202 := (publicBatchWrapper89_copies600 a h).2.2.1
  have c19203 := (publicBatchWrapper89_copies600 a h).2.2.2.1
  have c19204 := (publicBatchWrapper89_copies600 a h).2.2.2.2.1
  have c19205 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.1
  have c19206 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.1
  have c19207 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.1
  have c19208 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.1
  have c19209 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.1
  have c19210 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19211 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19212 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19213 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_291_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row291 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_291_19
  simp only [← c19197, k3, ← c19198, k3, ← c19199] at e_291_19
  have e_292_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_11
  simp only [← c19203, ← c19204, ← c19205] at e_292_11
  have e_292_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_12
  simp only [← c19206, ← c19207, ← c19208] at e_292_12
  have e_293_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_0
  simp only [← c19200, ← c19201, k3, ← c19202] at e_293_0
  have e_293_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_1
  simp only [← c19209, ← c19210, k3, ← c19211] at e_293_1
  refine ⟨?_, ?_⟩
  · have hc := e_292_11
    simp only [e_293_0] at hc
    linear_combination c19212.trans k2 - hc
  · have hc := e_293_1
    simp only [e_291_19, e_292_12, e_293_0] at hc
    linear_combination c19213.trans k2 - hc

theorem publicBatchWrapper89_f2631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 292 55) = band (a (.virt 19267)) (a (.virt 19269)) := by
  have c19214 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19215 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19216 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_292_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_13
  simp only [← c19214, ← c19215, ← c19216] at e_292_13
  have hr := e_292_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 292 59) = band (a (.virt 19271)) (a (.virt 19273)) := by
  have c19217 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19218 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19219 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_292_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_14
  simp only [← c19217, ← c19218, ← c19219] at e_292_14
  have hr := e_292_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 292 63) = band (a (.wire 292 55)) (a (.wire 292 59)) := by
  have c19220 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19221 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19222 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_292_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_15
  simp only [← c19220, ← c19221, ← c19222] at e_292_15
  have hr := e_292_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 35) = bor (a (.wire 60 67)) (a (.wire 292 63)) := by
  have c19223 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19224 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19225 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19226 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19227 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19228 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_16
  simp only [← c19223, ← c19224, ← c19225] at e_282_16
  have e_288_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_8
  simp only [← c19226, ← c19227, k3, ← c19228] at e_288_8
  have hr := e_288_8
  simp only [e_282_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 35) = a (.virt 201) := by
  have c19229 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19229

theorem publicBatchWrapper89_f2636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12603)) (a (.wire 170 23)) (a (.virt 19275)) (a (.virt 19276)) := by
  have c19230 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19231 := (publicBatchWrapper89_copies600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19232 := (publicBatchWrapper89_copies601 a h).1
  have c19233 := (publicBatchWrapper89_copies601 a h).2.1
  have c19234 := (publicBatchWrapper89_copies601 a h).2.2.1
  have c19235 := (publicBatchWrapper89_copies601 a h).2.2.2.1
  have c19236 := (publicBatchWrapper89_copies601 a h).2.2.2.2.1
  have c19237 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.1
  have c19238 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.1
  have c19239 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.1
  have c19240 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.1
  have c19241 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.1
  have c19242 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19243 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19244 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19245 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19246 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_292_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_16
  simp only [← c19236, ← c19237, ← c19238] at e_292_16
  have e_292_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_17
  simp only [← c19239, ← c19240, ← c19241] at e_292_17
  have e_293_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_2
  simp only [← c19230, k3, ← c19231, k3, ← c19232] at e_293_2
  have e_293_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_3
  simp only [← c19233, ← c19234, k3, ← c19235] at e_293_3
  have e_293_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_4
  simp only [← c19242, ← c19243, k3, ← c19244] at e_293_4
  refine ⟨?_, ?_⟩
  · have hc := e_292_16
    simp only [e_293_3] at hc
    linear_combination c19245.trans k2 - hc
  · have hc := e_293_4
    simp only [e_293_2, e_292_17, e_293_3] at hc
    linear_combination c19246.trans k2 - hc

theorem publicBatchWrapper89_f2637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 39) = bor (a (.wire 62 31)) (a (.virt 19275)) := by
  have c19247 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19248 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19249 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19250 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19251 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19252 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_17
  simp only [← c19247, ← c19248, ← c19249] at e_282_17
  have e_288_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_9
  simp only [← c19250, ← c19251, k3, ← c19252] at e_288_9
  have hr := e_288_9
  simp only [e_282_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 39) = a (.virt 201) := by
  have c19253 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19253

theorem publicBatchWrapper89_f2639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12604)) (a (.wire 170 31)) (a (.virt 19277)) (a (.virt 19278)) := by
  have c19254 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19255 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19256 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19257 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19258 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19259 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19260 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19261 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19262 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19263 := (publicBatchWrapper89_copies601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19264 := (publicBatchWrapper89_copies602 a h).1
  have c19265 := (publicBatchWrapper89_copies602 a h).2.1
  have c19266 := (publicBatchWrapper89_copies602 a h).2.2.1
  have c19267 := (publicBatchWrapper89_copies602 a h).2.2.2.1
  have c19268 := (publicBatchWrapper89_copies602 a h).2.2.2.2.1
  have c19269 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.1
  have c19270 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_292_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_18
  simp only [← c19260, ← c19261, ← c19262] at e_292_18
  have e_292_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row292 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_292_19
  simp only [← c19263, ← c19264, ← c19265] at e_292_19
  have e_293_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_5
  simp only [← c19254, k3, ← c19255, k3, ← c19256] at e_293_5
  have e_293_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_6
  simp only [← c19257, ← c19258, k3, ← c19259] at e_293_6
  have e_293_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_7
  simp only [← c19266, ← c19267, k3, ← c19268] at e_293_7
  refine ⟨?_, ?_⟩
  · have hc := e_292_18
    simp only [e_293_6] at hc
    linear_combination c19269.trans k2 - hc
  · have hc := e_293_7
    simp only [e_293_5, e_292_19, e_293_6] at hc
    linear_combination c19270.trans k2 - hc

theorem publicBatchWrapper89_f2640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 43) = bor (a (.wire 62 31)) (a (.virt 19277)) := by
  have c19271 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.1
  have c19272 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.1
  have c19273 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.1
  have c19274 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19275 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19276 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_18
  simp only [← c19271, ← c19272, ← c19273] at e_282_18
  have e_288_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_10
  simp only [← c19274, ← c19275, k3, ← c19276] at e_288_10
  have hr := e_288_10
  simp only [e_282_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 43) = a (.virt 201) := by
  have c19277 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19277

theorem publicBatchWrapper89_f2642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12605)) (a (.wire 169 63)) (a (.virt 19279)) (a (.virt 19280)) := by
  have c19278 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19279 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19280 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19281 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19282 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19283 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19284 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19285 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19286 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19287 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19288 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19289 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19290 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19291 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19292 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19293 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19294 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_293_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_8
  simp only [← c19278, k3, ← c19279, k3, ← c19280] at e_293_8
  have e_293_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_9
  simp only [← c19281, ← c19282, k3, ← c19283] at e_293_9
  have e_293_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_10
  simp only [← c19290, ← c19291, k3, ← c19292] at e_293_10
  have e_294_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_0
  simp only [← c19284, ← c19285, ← c19286] at e_294_0
  have e_294_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_1
  simp only [← c19287, ← c19288, ← c19289] at e_294_1
  refine ⟨?_, ?_⟩
  · have hc := e_294_0
    simp only [e_293_9] at hc
    linear_combination c19293.trans k2 - hc
  · have hc := e_293_10
    simp only [e_293_8, e_294_1, e_293_9] at hc
    linear_combination c19294.trans k2 - hc

theorem publicBatchWrapper89_f2643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12606)) (a (.wire 169 71)) (a (.virt 19281)) (a (.virt 19282)) := by
  have c19295 := (publicBatchWrapper89_copies602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19296 := (publicBatchWrapper89_copies603 a h).1
  have c19297 := (publicBatchWrapper89_copies603 a h).2.1
  have c19298 := (publicBatchWrapper89_copies603 a h).2.2.1
  have c19299 := (publicBatchWrapper89_copies603 a h).2.2.2.1
  have c19300 := (publicBatchWrapper89_copies603 a h).2.2.2.2.1
  have c19301 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.1
  have c19302 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.1
  have c19303 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.1
  have c19304 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.1
  have c19305 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.1
  have c19306 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19307 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19308 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19309 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19310 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19311 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_293_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_11
  simp only [← c19295, k3, ← c19296, k3, ← c19297] at e_293_11
  have e_293_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_12
  simp only [← c19298, ← c19299, k3, ← c19300] at e_293_12
  have e_293_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_13
  simp only [← c19307, ← c19308, k3, ← c19309] at e_293_13
  have e_294_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_2
  simp only [← c19301, ← c19302, ← c19303] at e_294_2
  have e_294_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_3
  simp only [← c19304, ← c19305, ← c19306] at e_294_3
  refine ⟨?_, ?_⟩
  · have hc := e_294_2
    simp only [e_293_12] at hc
    linear_combination c19310.trans k2 - hc
  · have hc := e_293_13
    simp only [e_293_11, e_294_3, e_293_12] at hc
    linear_combination c19311.trans k2 - hc

theorem publicBatchWrapper89_f2644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12607)) (a (.wire 169 79)) (a (.virt 19283)) (a (.virt 19284)) := by
  have c19312 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19313 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19314 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19315 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19316 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19317 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19318 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19319 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19320 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19321 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19322 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19323 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19324 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19325 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19326 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19327 := (publicBatchWrapper89_copies603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19328 := (publicBatchWrapper89_copies604 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_293_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_14
  simp only [← c19312, k3, ← c19313, k3, ← c19314] at e_293_14
  have e_293_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_15
  simp only [← c19315, ← c19316, k3, ← c19317] at e_293_15
  have e_293_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_16
  simp only [← c19324, ← c19325, k3, ← c19326] at e_293_16
  have e_294_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_4
  simp only [← c19318, ← c19319, ← c19320] at e_294_4
  have e_294_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_5
  simp only [← c19321, ← c19322, ← c19323] at e_294_5
  refine ⟨?_, ?_⟩
  · have hc := e_294_4
    simp only [e_293_15] at hc
    linear_combination c19327.trans k2 - hc
  · have hc := e_293_16
    simp only [e_293_14, e_294_5, e_293_15] at hc
    linear_combination c19328.trans k2 - hc

theorem publicBatchWrapper89_f2645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12608)) (a (.wire 170 7)) (a (.virt 19285)) (a (.virt 19286)) := by
  have c19329 := (publicBatchWrapper89_copies604 a h).2.1
  have c19330 := (publicBatchWrapper89_copies604 a h).2.2.1
  have c19331 := (publicBatchWrapper89_copies604 a h).2.2.2.1
  have c19332 := (publicBatchWrapper89_copies604 a h).2.2.2.2.1
  have c19333 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.1
  have c19334 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.1
  have c19335 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.1
  have c19336 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.1
  have c19337 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.1
  have c19338 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19339 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19340 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19341 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19342 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19343 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19344 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19345 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_293_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_17
  simp only [← c19329, k3, ← c19330, k3, ← c19331] at e_293_17
  have e_293_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_18
  simp only [← c19332, ← c19333, k3, ← c19334] at e_293_18
  have e_293_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row293 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_293_19
  simp only [← c19341, ← c19342, k3, ← c19343] at e_293_19
  have e_294_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_6
  simp only [← c19335, ← c19336, ← c19337] at e_294_6
  have e_294_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_7
  simp only [← c19338, ← c19339, ← c19340] at e_294_7
  refine ⟨?_, ?_⟩
  · have hc := e_294_6
    simp only [e_293_18] at hc
    linear_combination c19344.trans k2 - hc
  · have hc := e_293_19
    simp only [e_293_17, e_294_7, e_293_18] at hc
    linear_combination c19345.trans k2 - hc

theorem publicBatchWrapper89_f2646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 294 35) = band (a (.virt 19279)) (a (.virt 19281)) := by
  have c19346 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19347 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19348 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_294_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_8
  simp only [← c19346, ← c19347, ← c19348] at e_294_8
  have hr := e_294_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 294 39) = band (a (.virt 19283)) (a (.virt 19285)) := by
  have c19349 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19350 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19351 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_294_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_9
  simp only [← c19349, ← c19350, ← c19351] at e_294_9
  have hr := e_294_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 294 43) = band (a (.wire 294 35)) (a (.wire 294 39)) := by
  have c19352 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19353 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19354 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_294_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_10
  simp only [← c19352, ← c19353, ← c19354] at e_294_10
  have hr := e_294_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 47) = bor (a (.wire 62 31)) (a (.wire 294 43)) := by
  have c19355 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19356 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19357 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19358 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19359 := (publicBatchWrapper89_copies604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19360 := (publicBatchWrapper89_copies605 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_282_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row282 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_282_19
  simp only [← c19355, ← c19356, ← c19357] at e_282_19
  have e_288_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_11
  simp only [← c19358, ← c19359, k3, ← c19360] at e_288_11
  have hr := e_288_11
  simp only [e_282_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 47) = a (.virt 201) := by
  have c19361 := (publicBatchWrapper89_copies605 a h).2.1
  exact c19361

theorem publicBatchWrapper89_f2651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12803)) (a (.wire 170 23)) (a (.virt 19287)) (a (.virt 19288)) := by
  have c19362 := (publicBatchWrapper89_copies605 a h).2.2.1
  have c19363 := (publicBatchWrapper89_copies605 a h).2.2.2.1
  have c19364 := (publicBatchWrapper89_copies605 a h).2.2.2.2.1
  have c19365 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.1
  have c19366 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.1
  have c19367 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.1
  have c19368 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.1
  have c19369 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.1
  have c19370 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19371 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19372 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19373 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19374 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19375 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19376 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19377 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19378 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_294_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_11
  simp only [← c19368, ← c19369, ← c19370] at e_294_11
  have e_294_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_12
  simp only [← c19371, ← c19372, ← c19373] at e_294_12
  have e_295_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_0
  simp only [← c19362, k3, ← c19363, k3, ← c19364] at e_295_0
  have e_295_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_1
  simp only [← c19365, ← c19366, k3, ← c19367] at e_295_1
  have e_295_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_2
  simp only [← c19374, ← c19375, k3, ← c19376] at e_295_2
  refine ⟨?_, ?_⟩
  · have hc := e_294_11
    simp only [e_295_1] at hc
    linear_combination c19377.trans k2 - hc
  · have hc := e_295_2
    simp only [e_295_0, e_294_12, e_295_1] at hc
    linear_combination c19378.trans k2 - hc

theorem publicBatchWrapper89_f2652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 51) = bor (a (.wire 62 75)) (a (.virt 19287)) := by
  have c19379 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19380 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19381 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19382 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19383 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19384 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_288_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_12
  simp only [← c19382, ← c19383, k3, ← c19384] at e_288_12
  have e_296_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_0
  simp only [← c19379, ← c19380, ← c19381] at e_296_0
  have hr := e_288_12
  simp only [e_296_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 51) = a (.virt 201) := by
  have c19385 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19385

theorem publicBatchWrapper89_f2654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12804)) (a (.wire 170 31)) (a (.virt 19289)) (a (.virt 19290)) := by
  have c19386 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19387 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19388 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19389 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19390 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19391 := (publicBatchWrapper89_copies605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19392 := (publicBatchWrapper89_copies606 a h).1
  have c19393 := (publicBatchWrapper89_copies606 a h).2.1
  have c19394 := (publicBatchWrapper89_copies606 a h).2.2.1
  have c19395 := (publicBatchWrapper89_copies606 a h).2.2.2.1
  have c19396 := (publicBatchWrapper89_copies606 a h).2.2.2.2.1
  have c19397 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.1
  have c19398 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.1
  have c19399 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.1
  have c19400 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.1
  have c19401 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.1
  have c19402 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_294_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_13
  simp only [← c19392, ← c19393, ← c19394] at e_294_13
  have e_294_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_14
  simp only [← c19395, ← c19396, ← c19397] at e_294_14
  have e_295_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_3
  simp only [← c19386, k3, ← c19387, k3, ← c19388] at e_295_3
  have e_295_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_4
  simp only [← c19389, ← c19390, k3, ← c19391] at e_295_4
  have e_295_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_5
  simp only [← c19398, ← c19399, k3, ← c19400] at e_295_5
  refine ⟨?_, ?_⟩
  · have hc := e_294_13
    simp only [e_295_4] at hc
    linear_combination c19401.trans k2 - hc
  · have hc := e_295_5
    simp only [e_295_3, e_294_14, e_295_4] at hc
    linear_combination c19402.trans k2 - hc

theorem publicBatchWrapper89_f2655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 55) = bor (a (.wire 62 75)) (a (.virt 19289)) := by
  have c19403 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19404 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19405 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19406 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19407 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19408 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_288_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_13
  simp only [← c19406, ← c19407, k3, ← c19408] at e_288_13
  have e_296_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_1
  simp only [← c19403, ← c19404, ← c19405] at e_296_1
  have hr := e_288_13
  simp only [e_296_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 55) = a (.virt 201) := by
  have c19409 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19409

theorem publicBatchWrapper89_f2657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12805)) (a (.wire 169 63)) (a (.virt 19291)) (a (.virt 19292)) := by
  have c19410 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19411 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19412 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19413 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19414 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19415 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19416 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19417 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19418 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19419 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19420 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19421 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19422 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19423 := (publicBatchWrapper89_copies606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19424 := (publicBatchWrapper89_copies607 a h).1
  have c19425 := (publicBatchWrapper89_copies607 a h).2.1
  have c19426 := (publicBatchWrapper89_copies607 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_294_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_15
  simp only [← c19416, ← c19417, ← c19418] at e_294_15
  have e_294_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_16
  simp only [← c19419, ← c19420, ← c19421] at e_294_16
  have e_295_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_6
  simp only [← c19410, k3, ← c19411, k3, ← c19412] at e_295_6
  have e_295_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_7
  simp only [← c19413, ← c19414, k3, ← c19415] at e_295_7
  have e_295_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_8
  simp only [← c19422, ← c19423, k3, ← c19424] at e_295_8
  refine ⟨?_, ?_⟩
  · have hc := e_294_15
    simp only [e_295_7] at hc
    linear_combination c19425.trans k2 - hc
  · have hc := e_295_8
    simp only [e_295_6, e_294_16, e_295_7] at hc
    linear_combination c19426.trans k2 - hc

theorem publicBatchWrapper89_f2658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12806)) (a (.wire 169 71)) (a (.virt 19293)) (a (.virt 19294)) := by
  have c19427 := (publicBatchWrapper89_copies607 a h).2.2.2.1
  have c19428 := (publicBatchWrapper89_copies607 a h).2.2.2.2.1
  have c19429 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.1
  have c19430 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.1
  have c19431 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.1
  have c19432 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.1
  have c19433 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.1
  have c19434 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19435 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19436 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19437 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19438 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19439 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19440 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19441 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19442 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19443 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_294_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_17
  simp only [← c19433, ← c19434, ← c19435] at e_294_17
  have e_294_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_18
  simp only [← c19436, ← c19437, ← c19438] at e_294_18
  have e_295_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_9
  simp only [← c19427, k3, ← c19428, k3, ← c19429] at e_295_9
  have e_295_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_10
  simp only [← c19430, ← c19431, k3, ← c19432] at e_295_10
  have e_295_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_11
  simp only [← c19439, ← c19440, k3, ← c19441] at e_295_11
  refine ⟨?_, ?_⟩
  · have hc := e_294_17
    simp only [e_295_10] at hc
    linear_combination c19442.trans k2 - hc
  · have hc := e_295_11
    simp only [e_295_9, e_294_18, e_295_10] at hc
    linear_combination c19443.trans k2 - hc

theorem publicBatchWrapper89_f2659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12807)) (a (.wire 169 79)) (a (.virt 19295)) (a (.virt 19296)) := by
  have c19444 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19445 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19446 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19447 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19448 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19449 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19450 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19451 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19452 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19453 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19454 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19455 := (publicBatchWrapper89_copies607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19456 := (publicBatchWrapper89_copies608 a h).1
  have c19457 := (publicBatchWrapper89_copies608 a h).2.1
  have c19458 := (publicBatchWrapper89_copies608 a h).2.2.1
  have c19459 := (publicBatchWrapper89_copies608 a h).2.2.2.1
  have c19460 := (publicBatchWrapper89_copies608 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_294_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row294 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_294_19
  simp only [← c19450, ← c19451, ← c19452] at e_294_19
  have e_295_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_12
  simp only [← c19444, k3, ← c19445, k3, ← c19446] at e_295_12
  have e_295_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_13
  simp only [← c19447, ← c19448, k3, ← c19449] at e_295_13
  have e_295_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_14
  simp only [← c19456, ← c19457, k3, ← c19458] at e_295_14
  have e_297_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_0
  simp only [← c19453, ← c19454, ← c19455] at e_297_0
  refine ⟨?_, ?_⟩
  · have hc := e_294_19
    simp only [e_295_13] at hc
    linear_combination c19459.trans k2 - hc
  · have hc := e_295_14
    simp only [e_295_12, e_297_0, e_295_13] at hc
    linear_combination c19460.trans k2 - hc

theorem publicBatchWrapper89_f2660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12808)) (a (.wire 170 7)) (a (.virt 19297)) (a (.virt 19298)) := by
  have c19461 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.1
  have c19462 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.1
  have c19463 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.1
  have c19464 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.1
  have c19465 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.1
  have c19466 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19467 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19468 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19469 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19470 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19471 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19472 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19473 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19474 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19475 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19476 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19477 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_295_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_15
  simp only [← c19461, k3, ← c19462, k3, ← c19463] at e_295_15
  have e_295_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_16
  simp only [← c19464, ← c19465, k3, ← c19466] at e_295_16
  have e_295_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_17
  simp only [← c19473, ← c19474, k3, ← c19475] at e_295_17
  have e_297_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_1
  simp only [← c19467, ← c19468, ← c19469] at e_297_1
  have e_297_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_2
  simp only [← c19470, ← c19471, ← c19472] at e_297_2
  refine ⟨?_, ?_⟩
  · have hc := e_297_1
    simp only [e_295_16] at hc
    linear_combination c19476.trans k2 - hc
  · have hc := e_295_17
    simp only [e_295_15, e_297_2, e_295_16] at hc
    linear_combination c19477.trans k2 - hc

theorem publicBatchWrapper89_f2661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 297 15) = band (a (.virt 19291)) (a (.virt 19293)) := by
  have c19478 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19479 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19480 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_297_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_3
  simp only [← c19478, ← c19479, ← c19480] at e_297_3
  have hr := e_297_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 297 19) = band (a (.virt 19295)) (a (.virt 19297)) := by
  have c19481 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19482 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19483 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_297_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_4
  simp only [← c19481, ← c19482, ← c19483] at e_297_4
  have hr := e_297_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 297 23) = band (a (.wire 297 15)) (a (.wire 297 19)) := by
  have c19484 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19485 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19486 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_297_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_5
  simp only [← c19484, ← c19485, ← c19486] at e_297_5
  have hr := e_297_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 59) = bor (a (.wire 62 75)) (a (.wire 297 23)) := by
  have c19487 := (publicBatchWrapper89_copies608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19488 := (publicBatchWrapper89_copies609 a h).1
  have c19489 := (publicBatchWrapper89_copies609 a h).2.1
  have c19490 := (publicBatchWrapper89_copies609 a h).2.2.1
  have c19491 := (publicBatchWrapper89_copies609 a h).2.2.2.1
  have c19492 := (publicBatchWrapper89_copies609 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_288_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_14
  simp only [← c19490, ← c19491, k3, ← c19492] at e_288_14
  have e_296_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_2
  simp only [← c19487, ← c19488, ← c19489] at e_296_2
  have hr := e_288_14
  simp only [e_296_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 59) = a (.virt 201) := by
  have c19493 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.1
  exact c19493

theorem publicBatchWrapper89_f2666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13003)) (a (.wire 170 23)) (a (.virt 19299)) (a (.virt 19300)) := by
  have c19494 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.1
  have c19495 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.1
  have c19496 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.1
  have c19497 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.1
  have c19498 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19499 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19500 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19501 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19502 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19503 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19504 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19505 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19506 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19507 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19508 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19509 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19510 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_295_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_18
  simp only [← c19494, k3, ← c19495, k3, ← c19496] at e_295_18
  have e_295_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row295 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_295_19
  simp only [← c19497, ← c19498, k3, ← c19499] at e_295_19
  have e_297_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_6
  simp only [← c19500, ← c19501, ← c19502] at e_297_6
  have e_297_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_7
  simp only [← c19503, ← c19504, ← c19505] at e_297_7
  have e_298_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_0
  simp only [← c19506, ← c19507, k3, ← c19508] at e_298_0
  refine ⟨?_, ?_⟩
  · have hc := e_297_6
    simp only [e_295_19] at hc
    linear_combination c19509.trans k2 - hc
  · have hc := e_298_0
    simp only [e_295_18, e_297_7, e_295_19] at hc
    linear_combination c19510.trans k2 - hc

theorem publicBatchWrapper89_f2667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 63) = bor (a (.wire 64 39)) (a (.virt 19299)) := by
  have c19511 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19512 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19513 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19514 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19515 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19516 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_288_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_15
  simp only [← c19514, ← c19515, k3, ← c19516] at e_288_15
  have e_296_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_3
  simp only [← c19511, ← c19512, ← c19513] at e_296_3
  have hr := e_288_15
  simp only [e_296_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 63) = a (.virt 201) := by
  have c19517 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19517

theorem publicBatchWrapper89_f2669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13004)) (a (.wire 170 31)) (a (.virt 19301)) (a (.virt 19302)) := by
  have c19518 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19519 := (publicBatchWrapper89_copies609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19520 := (publicBatchWrapper89_copies610 a h).1
  have c19521 := (publicBatchWrapper89_copies610 a h).2.1
  have c19522 := (publicBatchWrapper89_copies610 a h).2.2.1
  have c19523 := (publicBatchWrapper89_copies610 a h).2.2.2.1
  have c19524 := (publicBatchWrapper89_copies610 a h).2.2.2.2.1
  have c19525 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.1
  have c19526 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.1
  have c19527 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.1
  have c19528 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.1
  have c19529 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.1
  have c19530 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19531 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19532 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19533 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19534 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_297_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_8
  simp only [← c19524, ← c19525, ← c19526] at e_297_8
  have e_297_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_9
  simp only [← c19527, ← c19528, ← c19529] at e_297_9
  have e_298_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_1
  simp only [← c19518, k3, ← c19519, k3, ← c19520] at e_298_1
  have e_298_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_2
  simp only [← c19521, ← c19522, k3, ← c19523] at e_298_2
  have e_298_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_3
  simp only [← c19530, ← c19531, k3, ← c19532] at e_298_3
  refine ⟨?_, ?_⟩
  · have hc := e_297_8
    simp only [e_298_2] at hc
    linear_combination c19533.trans k2 - hc
  · have hc := e_298_3
    simp only [e_298_1, e_297_9, e_298_2] at hc
    linear_combination c19534.trans k2 - hc

theorem publicBatchWrapper89_f2670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 67) = bor (a (.wire 64 39)) (a (.virt 19301)) := by
  have c19535 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19536 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19537 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19538 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19539 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19540 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_288_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_16
  simp only [← c19538, ← c19539, k3, ← c19540] at e_288_16
  have e_296_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_4
  simp only [← c19535, ← c19536, ← c19537] at e_296_4
  have hr := e_288_16
  simp only [e_296_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 67) = a (.virt 201) := by
  have c19541 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19541

theorem publicBatchWrapper89_f2672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13005)) (a (.wire 169 63)) (a (.virt 19303)) (a (.virt 19304)) := by
  have c19542 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19543 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19544 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19545 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19546 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19547 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19548 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19549 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19550 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19551 := (publicBatchWrapper89_copies610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19552 := (publicBatchWrapper89_copies611 a h).1
  have c19553 := (publicBatchWrapper89_copies611 a h).2.1
  have c19554 := (publicBatchWrapper89_copies611 a h).2.2.1
  have c19555 := (publicBatchWrapper89_copies611 a h).2.2.2.1
  have c19556 := (publicBatchWrapper89_copies611 a h).2.2.2.2.1
  have c19557 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.1
  have c19558 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_297_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_10
  simp only [← c19548, ← c19549, ← c19550] at e_297_10
  have e_297_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_11
  simp only [← c19551, ← c19552, ← c19553] at e_297_11
  have e_298_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_4
  simp only [← c19542, k3, ← c19543, k3, ← c19544] at e_298_4
  have e_298_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_5
  simp only [← c19545, ← c19546, k3, ← c19547] at e_298_5
  have e_298_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_6
  simp only [← c19554, ← c19555, k3, ← c19556] at e_298_6
  refine ⟨?_, ?_⟩
  · have hc := e_297_10
    simp only [e_298_5] at hc
    linear_combination c19557.trans k2 - hc
  · have hc := e_298_6
    simp only [e_298_4, e_297_11, e_298_5] at hc
    linear_combination c19558.trans k2 - hc

theorem publicBatchWrapper89_f2673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13006)) (a (.wire 169 71)) (a (.virt 19305)) (a (.virt 19306)) := by
  have c19559 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.1
  have c19560 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.1
  have c19561 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.1
  have c19562 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19563 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19564 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19565 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19566 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19567 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19568 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19569 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19570 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19571 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19572 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19573 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19574 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19575 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_297_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_12
  simp only [← c19565, ← c19566, ← c19567] at e_297_12
  have e_297_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_13
  simp only [← c19568, ← c19569, ← c19570] at e_297_13
  have e_298_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_7
  simp only [← c19559, k3, ← c19560, k3, ← c19561] at e_298_7
  have e_298_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_8
  simp only [← c19562, ← c19563, k3, ← c19564] at e_298_8
  have e_298_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_9
  simp only [← c19571, ← c19572, k3, ← c19573] at e_298_9
  refine ⟨?_, ?_⟩
  · have hc := e_297_12
    simp only [e_298_8] at hc
    linear_combination c19574.trans k2 - hc
  · have hc := e_298_9
    simp only [e_298_7, e_297_13, e_298_8] at hc
    linear_combination c19575.trans k2 - hc

theorem publicBatchWrapper89_f2674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13007)) (a (.wire 169 79)) (a (.virt 19307)) (a (.virt 19308)) := by
  have c19576 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19577 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19578 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19579 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19580 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19581 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19582 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19583 := (publicBatchWrapper89_copies611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19584 := (publicBatchWrapper89_copies612 a h).1
  have c19585 := (publicBatchWrapper89_copies612 a h).2.1
  have c19586 := (publicBatchWrapper89_copies612 a h).2.2.1
  have c19587 := (publicBatchWrapper89_copies612 a h).2.2.2.1
  have c19588 := (publicBatchWrapper89_copies612 a h).2.2.2.2.1
  have c19589 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.1
  have c19590 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.1
  have c19591 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.1
  have c19592 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_297_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_14
  simp only [← c19582, ← c19583, ← c19584] at e_297_14
  have e_297_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_15
  simp only [← c19585, ← c19586, ← c19587] at e_297_15
  have e_298_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_10
  simp only [← c19576, k3, ← c19577, k3, ← c19578] at e_298_10
  have e_298_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_11
  simp only [← c19579, ← c19580, k3, ← c19581] at e_298_11
  have e_298_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_12
  simp only [← c19588, ← c19589, k3, ← c19590] at e_298_12
  refine ⟨?_, ?_⟩
  · have hc := e_297_14
    simp only [e_298_11] at hc
    linear_combination c19591.trans k2 - hc
  · have hc := e_298_12
    simp only [e_298_10, e_297_15, e_298_11] at hc
    linear_combination c19592.trans k2 - hc

theorem publicBatchWrapper89_f2675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13008)) (a (.wire 170 7)) (a (.virt 19309)) (a (.virt 19310)) := by
  have c19593 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.1
  have c19594 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19595 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19596 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19597 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19598 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19599 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19600 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19601 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19602 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19603 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19604 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19605 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19606 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19607 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19608 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19609 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_297_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_16
  simp only [← c19599, ← c19600, ← c19601] at e_297_16
  have e_297_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_17
  simp only [← c19602, ← c19603, ← c19604] at e_297_17
  have e_298_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_13
  simp only [← c19593, k3, ← c19594, k3, ← c19595] at e_298_13
  have e_298_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_14
  simp only [← c19596, ← c19597, k3, ← c19598] at e_298_14
  have e_298_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_15
  simp only [← c19605, ← c19606, k3, ← c19607] at e_298_15
  refine ⟨?_, ?_⟩
  · have hc := e_297_16
    simp only [e_298_14] at hc
    linear_combination c19608.trans k2 - hc
  · have hc := e_298_15
    simp only [e_298_13, e_297_17, e_298_14] at hc
    linear_combination c19609.trans k2 - hc

theorem publicBatchWrapper89_f2676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 297 75) = band (a (.virt 19303)) (a (.virt 19305)) := by
  have c19610 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19611 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19612 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_297_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_18
  simp only [← c19610, ← c19611, ← c19612] at e_297_18
  have hr := e_297_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 297 79) = band (a (.virt 19307)) (a (.virt 19309)) := by
  have c19613 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19614 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19615 := (publicBatchWrapper89_copies612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_297_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row297 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_297_19
  simp only [← c19613, ← c19614, ← c19615] at e_297_19
  have hr := e_297_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 299 3) = band (a (.wire 297 75)) (a (.wire 297 79)) := by
  have c19616 := (publicBatchWrapper89_copies613 a h).1
  have c19617 := (publicBatchWrapper89_copies613 a h).2.1
  have c19618 := (publicBatchWrapper89_copies613 a h).2.2.1
  have e_299_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_0
  simp only [← c19616, ← c19617, ← c19618] at e_299_0
  have hr := e_299_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 71) = bor (a (.wire 64 39)) (a (.wire 299 3)) := by
  have c19619 := (publicBatchWrapper89_copies613 a h).2.2.2.1
  have c19620 := (publicBatchWrapper89_copies613 a h).2.2.2.2.1
  have c19621 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.1
  have c19622 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.1
  have c19623 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.1
  have c19624 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_288_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_17
  simp only [← c19622, ← c19623, k3, ← c19624] at e_288_17
  have e_296_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_5
  simp only [← c19619, ← c19620, ← c19621] at e_296_5
  have hr := e_288_17
  simp only [e_296_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 71) = a (.virt 201) := by
  have c19625 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.1
  exact c19625

theorem publicBatchWrapper89_f2681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13203)) (a (.wire 170 23)) (a (.virt 19311)) (a (.virt 19312)) := by
  have c19626 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19627 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19628 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19629 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19630 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19631 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19632 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19633 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19634 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19635 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19636 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19637 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19638 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19639 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19640 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19641 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19642 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_298_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_16
  simp only [← c19626, k3, ← c19627, k3, ← c19628] at e_298_16
  have e_298_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_17
  simp only [← c19629, ← c19630, k3, ← c19631] at e_298_17
  have e_298_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_18
  simp only [← c19638, ← c19639, k3, ← c19640] at e_298_18
  have e_299_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_1
  simp only [← c19632, ← c19633, ← c19634] at e_299_1
  have e_299_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_2
  simp only [← c19635, ← c19636, ← c19637] at e_299_2
  refine ⟨?_, ?_⟩
  · have hc := e_299_1
    simp only [e_298_17] at hc
    linear_combination c19641.trans k2 - hc
  · have hc := e_298_18
    simp only [e_298_16, e_299_2, e_298_17] at hc
    linear_combination c19642.trans k2 - hc

theorem publicBatchWrapper89_f2682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 75) = bor (a (.wire 65 3)) (a (.virt 19311)) := by
  have c19643 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19644 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19645 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19646 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19647 := (publicBatchWrapper89_copies613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19648 := (publicBatchWrapper89_copies614 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_288_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_18
  simp only [← c19646, ← c19647, k3, ← c19648] at e_288_18
  have e_296_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_6
  simp only [← c19643, ← c19644, ← c19645] at e_296_6
  have hr := e_288_18
  simp only [e_296_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 75) = a (.virt 201) := by
  have c19649 := (publicBatchWrapper89_copies614 a h).2.1
  exact c19649

theorem publicBatchWrapper89_f2684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13204)) (a (.wire 170 31)) (a (.virt 19313)) (a (.virt 19314)) := by
  have c19650 := (publicBatchWrapper89_copies614 a h).2.2.1
  have c19651 := (publicBatchWrapper89_copies614 a h).2.2.2.1
  have c19652 := (publicBatchWrapper89_copies614 a h).2.2.2.2.1
  have c19653 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.1
  have c19654 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.1
  have c19655 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.1
  have c19656 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.1
  have c19657 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.1
  have c19658 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19659 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19660 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19661 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19662 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19663 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19664 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19665 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19666 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_298_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row298 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_298_19
  simp only [← c19650, k3, ← c19651, k3, ← c19652] at e_298_19
  have e_299_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_3
  simp only [← c19656, ← c19657, ← c19658] at e_299_3
  have e_299_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_4
  simp only [← c19659, ← c19660, ← c19661] at e_299_4
  have e_300_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_0
  simp only [← c19653, ← c19654, k3, ← c19655] at e_300_0
  have e_300_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_1
  simp only [← c19662, ← c19663, k3, ← c19664] at e_300_1
  refine ⟨?_, ?_⟩
  · have hc := e_299_3
    simp only [e_300_0] at hc
    linear_combination c19665.trans k2 - hc
  · have hc := e_300_1
    simp only [e_298_19, e_299_4, e_300_0] at hc
    linear_combination c19666.trans k2 - hc

theorem publicBatchWrapper89_f2685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 79) = bor (a (.wire 65 3)) (a (.virt 19313)) := by
  have c19667 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19668 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19669 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19670 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19671 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19672 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_288_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row288 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_288_19
  simp only [← c19670, ← c19671, k3, ← c19672] at e_288_19
  have e_296_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_7
  simp only [← c19667, ← c19668, ← c19669] at e_296_7
  have hr := e_288_19
  simp only [e_296_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 288 79) = a (.virt 201) := by
  have c19673 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19673

theorem publicBatchWrapper89_f2687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13205)) (a (.wire 169 63)) (a (.virt 19315)) (a (.virt 19316)) := by
  have c19674 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19675 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19676 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19677 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19678 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19679 := (publicBatchWrapper89_copies614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19680 := (publicBatchWrapper89_copies615 a h).1
  have c19681 := (publicBatchWrapper89_copies615 a h).2.1
  have c19682 := (publicBatchWrapper89_copies615 a h).2.2.1
  have c19683 := (publicBatchWrapper89_copies615 a h).2.2.2.1
  have c19684 := (publicBatchWrapper89_copies615 a h).2.2.2.2.1
  have c19685 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.1
  have c19686 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.1
  have c19687 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.1
  have c19688 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.1
  have c19689 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.1
  have c19690 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_299_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_5
  simp only [← c19680, ← c19681, ← c19682] at e_299_5
  have e_299_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_6
  simp only [← c19683, ← c19684, ← c19685] at e_299_6
  have e_300_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_2
  simp only [← c19674, k3, ← c19675, k3, ← c19676] at e_300_2
  have e_300_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_3
  simp only [← c19677, ← c19678, k3, ← c19679] at e_300_3
  have e_300_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_4
  simp only [← c19686, ← c19687, k3, ← c19688] at e_300_4
  refine ⟨?_, ?_⟩
  · have hc := e_299_5
    simp only [e_300_3] at hc
    linear_combination c19689.trans k2 - hc
  · have hc := e_300_4
    simp only [e_300_2, e_299_6, e_300_3] at hc
    linear_combination c19690.trans k2 - hc

theorem publicBatchWrapper89_f2688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13206)) (a (.wire 169 71)) (a (.virt 19317)) (a (.virt 19318)) := by
  have c19691 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19692 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19693 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19694 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19695 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19696 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19697 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19698 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19699 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19700 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19701 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19702 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19703 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19704 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19705 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19706 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19707 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_299_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_7
  simp only [← c19697, ← c19698, ← c19699] at e_299_7
  have e_299_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_8
  simp only [← c19700, ← c19701, ← c19702] at e_299_8
  have e_300_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_5
  simp only [← c19691, k3, ← c19692, k3, ← c19693] at e_300_5
  have e_300_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_6
  simp only [← c19694, ← c19695, k3, ← c19696] at e_300_6
  have e_300_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_7
  simp only [← c19703, ← c19704, k3, ← c19705] at e_300_7
  refine ⟨?_, ?_⟩
  · have hc := e_299_7
    simp only [e_300_6] at hc
    linear_combination c19706.trans k2 - hc
  · have hc := e_300_7
    simp only [e_300_5, e_299_8, e_300_6] at hc
    linear_combination c19707.trans k2 - hc

theorem publicBatchWrapper89_f2689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13207)) (a (.wire 169 79)) (a (.virt 19319)) (a (.virt 19320)) := by
  have c19708 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19709 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19710 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19711 := (publicBatchWrapper89_copies615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19712 := (publicBatchWrapper89_copies616 a h).1
  have c19713 := (publicBatchWrapper89_copies616 a h).2.1
  have c19714 := (publicBatchWrapper89_copies616 a h).2.2.1
  have c19715 := (publicBatchWrapper89_copies616 a h).2.2.2.1
  have c19716 := (publicBatchWrapper89_copies616 a h).2.2.2.2.1
  have c19717 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.1
  have c19718 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.1
  have c19719 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.1
  have c19720 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.1
  have c19721 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.1
  have c19722 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19723 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19724 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_299_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_9
  simp only [← c19714, ← c19715, ← c19716] at e_299_9
  have e_299_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_10
  simp only [← c19717, ← c19718, ← c19719] at e_299_10
  have e_300_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_8
  simp only [← c19708, k3, ← c19709, k3, ← c19710] at e_300_8
  have e_300_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_9
  simp only [← c19711, ← c19712, k3, ← c19713] at e_300_9
  have e_300_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_10
  simp only [← c19720, ← c19721, k3, ← c19722] at e_300_10
  refine ⟨?_, ?_⟩
  · have hc := e_299_9
    simp only [e_300_9] at hc
    linear_combination c19723.trans k2 - hc
  · have hc := e_300_10
    simp only [e_300_8, e_299_10, e_300_9] at hc
    linear_combination c19724.trans k2 - hc

theorem publicBatchWrapper89_f2690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13208)) (a (.wire 170 7)) (a (.virt 19321)) (a (.virt 19322)) := by
  have c19725 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19726 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19727 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19728 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19729 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19730 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19731 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19732 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19733 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19734 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19735 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19736 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19737 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19738 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19739 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19740 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19741 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_299_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_11
  simp only [← c19731, ← c19732, ← c19733] at e_299_11
  have e_299_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_12
  simp only [← c19734, ← c19735, ← c19736] at e_299_12
  have e_300_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_11
  simp only [← c19725, k3, ← c19726, k3, ← c19727] at e_300_11
  have e_300_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_12
  simp only [← c19728, ← c19729, k3, ← c19730] at e_300_12
  have e_300_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_13
  simp only [← c19737, ← c19738, k3, ← c19739] at e_300_13
  refine ⟨?_, ?_⟩
  · have hc := e_299_11
    simp only [e_300_12] at hc
    linear_combination c19740.trans k2 - hc
  · have hc := e_300_13
    simp only [e_300_11, e_299_12, e_300_12] at hc
    linear_combination c19741.trans k2 - hc

theorem publicBatchWrapper89_f2691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 299 55) = band (a (.virt 19315)) (a (.virt 19317)) := by
  have c19742 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19743 := (publicBatchWrapper89_copies616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19744 := (publicBatchWrapper89_copies617 a h).1
  have e_299_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_13
  simp only [← c19742, ← c19743, ← c19744] at e_299_13
  have hr := e_299_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 299 59) = band (a (.virt 19319)) (a (.virt 19321)) := by
  have c19745 := (publicBatchWrapper89_copies617 a h).2.1
  have c19746 := (publicBatchWrapper89_copies617 a h).2.2.1
  have c19747 := (publicBatchWrapper89_copies617 a h).2.2.2.1
  have e_299_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_14
  simp only [← c19745, ← c19746, ← c19747] at e_299_14
  have hr := e_299_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 299 63) = band (a (.wire 299 55)) (a (.wire 299 59)) := by
  have c19748 := (publicBatchWrapper89_copies617 a h).2.2.2.2.1
  have c19749 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.1
  have c19750 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.1
  have e_299_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_15
  simp only [← c19748, ← c19749, ← c19750] at e_299_15
  have hr := e_299_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 3) = bor (a (.wire 65 3)) (a (.wire 299 63)) := by
  have c19751 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.1
  have c19752 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.1
  have c19753 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.1
  have c19754 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19755 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19756 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_8
  simp only [← c19751, ← c19752, ← c19753] at e_296_8
  have e_301_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_0
  simp only [← c19754, ← c19755, k3, ← c19756] at e_301_0
  have hr := e_301_0
  simp only [e_296_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 3) = a (.virt 201) := by
  have c19757 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19757

theorem publicBatchWrapper89_f2696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13403)) (a (.wire 170 23)) (a (.virt 19323)) (a (.virt 19324)) := by
  have c19758 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19759 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19760 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19761 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19762 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19763 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19764 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19765 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19766 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19767 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19768 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19769 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19770 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19771 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19772 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19773 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19774 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_299_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_16
  simp only [← c19764, ← c19765, ← c19766] at e_299_16
  have e_299_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_17
  simp only [← c19767, ← c19768, ← c19769] at e_299_17
  have e_300_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_14
  simp only [← c19758, k3, ← c19759, k3, ← c19760] at e_300_14
  have e_300_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_15
  simp only [← c19761, ← c19762, k3, ← c19763] at e_300_15
  have e_300_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_16
  simp only [← c19770, ← c19771, k3, ← c19772] at e_300_16
  refine ⟨?_, ?_⟩
  · have hc := e_299_16
    simp only [e_300_15] at hc
    linear_combination c19773.trans k2 - hc
  · have hc := e_300_16
    simp only [e_300_14, e_299_17, e_300_15] at hc
    linear_combination c19774.trans k2 - hc

theorem publicBatchWrapper89_f2697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 7) = bor (a (.wire 65 47)) (a (.virt 19323)) := by
  have c19775 := (publicBatchWrapper89_copies617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19776 := (publicBatchWrapper89_copies618 a h).1
  have c19777 := (publicBatchWrapper89_copies618 a h).2.1
  have c19778 := (publicBatchWrapper89_copies618 a h).2.2.1
  have c19779 := (publicBatchWrapper89_copies618 a h).2.2.2.1
  have c19780 := (publicBatchWrapper89_copies618 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_9
  simp only [← c19775, ← c19776, ← c19777] at e_296_9
  have e_301_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_1
  simp only [← c19778, ← c19779, k3, ← c19780] at e_301_1
  have hr := e_301_1
  simp only [e_296_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 7) = a (.virt 201) := by
  have c19781 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.1
  exact c19781

theorem publicBatchWrapper89_f2699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13404)) (a (.wire 170 31)) (a (.virt 19325)) (a (.virt 19326)) := by
  have c19782 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.1
  have c19783 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.1
  have c19784 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.1
  have c19785 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.1
  have c19786 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19787 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19788 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19789 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19790 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19791 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19792 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19793 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19794 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19795 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19796 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19797 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19798 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_299_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_18
  simp only [← c19788, ← c19789, ← c19790] at e_299_18
  have e_299_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row299 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_299_19
  simp only [← c19791, ← c19792, ← c19793] at e_299_19
  have e_300_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_17
  simp only [← c19782, k3, ← c19783, k3, ← c19784] at e_300_17
  have e_300_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_18
  simp only [← c19785, ← c19786, k3, ← c19787] at e_300_18
  have e_300_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row300 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_300_19
  simp only [← c19794, ← c19795, k3, ← c19796] at e_300_19
  refine ⟨?_, ?_⟩
  · have hc := e_299_18
    simp only [e_300_18] at hc
    linear_combination c19797.trans k2 - hc
  · have hc := e_300_19
    simp only [e_300_17, e_299_19, e_300_18] at hc
    linear_combination c19798.trans k2 - hc

theorem publicBatchWrapper89_f2700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 11) = bor (a (.wire 65 47)) (a (.virt 19325)) := by
  have c19799 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19800 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19801 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19802 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19803 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19804 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_10
  simp only [← c19799, ← c19800, ← c19801] at e_296_10
  have e_301_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_2
  simp only [← c19802, ← c19803, k3, ← c19804] at e_301_2
  have hr := e_301_2
  simp only [e_296_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 11) = a (.virt 201) := by
  have c19805 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19805

theorem publicBatchWrapper89_f2702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13405)) (a (.wire 169 63)) (a (.virt 19327)) (a (.virt 19328)) := by
  have c19806 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19807 := (publicBatchWrapper89_copies618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19808 := (publicBatchWrapper89_copies619 a h).1
  have c19809 := (publicBatchWrapper89_copies619 a h).2.1
  have c19810 := (publicBatchWrapper89_copies619 a h).2.2.1
  have c19811 := (publicBatchWrapper89_copies619 a h).2.2.2.1
  have c19812 := (publicBatchWrapper89_copies619 a h).2.2.2.2.1
  have c19813 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.1
  have c19814 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.1
  have c19815 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.1
  have c19816 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.1
  have c19817 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.1
  have c19818 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19819 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19820 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19821 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19822 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_302_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_0
  simp only [← c19806, k3, ← c19807, k3, ← c19808] at e_302_0
  have e_302_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_1
  simp only [← c19809, ← c19810, k3, ← c19811] at e_302_1
  have e_302_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_2
  simp only [← c19818, ← c19819, k3, ← c19820] at e_302_2
  have e_303_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_0
  simp only [← c19812, ← c19813, ← c19814] at e_303_0
  have e_303_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_1
  simp only [← c19815, ← c19816, ← c19817] at e_303_1
  refine ⟨?_, ?_⟩
  · have hc := e_303_0
    simp only [e_302_1] at hc
    linear_combination c19821.trans k2 - hc
  · have hc := e_302_2
    simp only [e_302_0, e_303_1, e_302_1] at hc
    linear_combination c19822.trans k2 - hc

theorem publicBatchWrapper89_f2703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13406)) (a (.wire 169 71)) (a (.virt 19329)) (a (.virt 19330)) := by
  have c19823 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19824 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19825 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19826 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19827 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19828 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19829 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19830 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19831 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19832 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19833 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19834 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19835 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19836 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19837 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19838 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19839 := (publicBatchWrapper89_copies619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_302_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_3
  simp only [← c19823, k3, ← c19824, k3, ← c19825] at e_302_3
  have e_302_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_4
  simp only [← c19826, ← c19827, k3, ← c19828] at e_302_4
  have e_302_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_5
  simp only [← c19835, ← c19836, k3, ← c19837] at e_302_5
  have e_303_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_2
  simp only [← c19829, ← c19830, ← c19831] at e_303_2
  have e_303_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_3
  simp only [← c19832, ← c19833, ← c19834] at e_303_3
  refine ⟨?_, ?_⟩
  · have hc := e_303_2
    simp only [e_302_4] at hc
    linear_combination c19838.trans k2 - hc
  · have hc := e_302_5
    simp only [e_302_3, e_303_3, e_302_4] at hc
    linear_combination c19839.trans k2 - hc

theorem publicBatchWrapper89_f2704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13407)) (a (.wire 169 79)) (a (.virt 19331)) (a (.virt 19332)) := by
  have c19840 := (publicBatchWrapper89_copies620 a h).1
  have c19841 := (publicBatchWrapper89_copies620 a h).2.1
  have c19842 := (publicBatchWrapper89_copies620 a h).2.2.1
  have c19843 := (publicBatchWrapper89_copies620 a h).2.2.2.1
  have c19844 := (publicBatchWrapper89_copies620 a h).2.2.2.2.1
  have c19845 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.1
  have c19846 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.1
  have c19847 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.1
  have c19848 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.1
  have c19849 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.1
  have c19850 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19851 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19852 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19853 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19854 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19855 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19856 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_302_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_6
  simp only [← c19840, k3, ← c19841, k3, ← c19842] at e_302_6
  have e_302_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_7
  simp only [← c19843, ← c19844, k3, ← c19845] at e_302_7
  have e_302_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_8
  simp only [← c19852, ← c19853, k3, ← c19854] at e_302_8
  have e_303_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_4
  simp only [← c19846, ← c19847, ← c19848] at e_303_4
  have e_303_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_5
  simp only [← c19849, ← c19850, ← c19851] at e_303_5
  refine ⟨?_, ?_⟩
  · have hc := e_303_4
    simp only [e_302_7] at hc
    linear_combination c19855.trans k2 - hc
  · have hc := e_302_8
    simp only [e_302_6, e_303_5, e_302_7] at hc
    linear_combination c19856.trans k2 - hc

theorem publicBatchWrapper89_f2705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13408)) (a (.wire 170 7)) (a (.virt 19333)) (a (.virt 19334)) := by
  have c19857 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19858 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19859 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19860 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19861 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19862 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19863 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19864 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19865 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19866 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19867 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19868 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19869 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19870 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19871 := (publicBatchWrapper89_copies620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19872 := (publicBatchWrapper89_copies621 a h).1
  have c19873 := (publicBatchWrapper89_copies621 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_302_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_9
  simp only [← c19857, k3, ← c19858, k3, ← c19859] at e_302_9
  have e_302_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_10
  simp only [← c19860, ← c19861, k3, ← c19862] at e_302_10
  have e_302_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_11
  simp only [← c19869, ← c19870, k3, ← c19871] at e_302_11
  have e_303_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_6
  simp only [← c19863, ← c19864, ← c19865] at e_303_6
  have e_303_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_7
  simp only [← c19866, ← c19867, ← c19868] at e_303_7
  refine ⟨?_, ?_⟩
  · have hc := e_303_6
    simp only [e_302_10] at hc
    linear_combination c19872.trans k2 - hc
  · have hc := e_302_11
    simp only [e_302_9, e_303_7, e_302_10] at hc
    linear_combination c19873.trans k2 - hc

theorem publicBatchWrapper89_f2706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 303 35) = band (a (.virt 19327)) (a (.virt 19329)) := by
  have c19874 := (publicBatchWrapper89_copies621 a h).2.2.1
  have c19875 := (publicBatchWrapper89_copies621 a h).2.2.2.1
  have c19876 := (publicBatchWrapper89_copies621 a h).2.2.2.2.1
  have e_303_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_8
  simp only [← c19874, ← c19875, ← c19876] at e_303_8
  have hr := e_303_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 303 39) = band (a (.virt 19331)) (a (.virt 19333)) := by
  have c19877 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.1
  have c19878 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.1
  have c19879 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.1
  have e_303_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_9
  simp only [← c19877, ← c19878, ← c19879] at e_303_9
  have hr := e_303_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 303 43) = band (a (.wire 303 35)) (a (.wire 303 39)) := by
  have c19880 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.1
  have c19881 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.1
  have c19882 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_303_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_10
  simp only [← c19880, ← c19881, ← c19882] at e_303_10
  have hr := e_303_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 15) = bor (a (.wire 65 47)) (a (.wire 303 43)) := by
  have c19883 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19884 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19885 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19886 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19887 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19888 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_11
  simp only [← c19883, ← c19884, ← c19885] at e_296_11
  have e_301_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_3
  simp only [← c19886, ← c19887, k3, ← c19888] at e_301_3
  have hr := e_301_3
  simp only [e_296_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 15) = a (.virt 201) := by
  have c19889 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19889

theorem publicBatchWrapper89_f2711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13603)) (a (.wire 170 23)) (a (.virt 19335)) (a (.virt 19336)) := by
  have c19890 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19891 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19892 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19893 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19894 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19895 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19896 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19897 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19898 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19899 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19900 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19901 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19902 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19903 := (publicBatchWrapper89_copies621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19904 := (publicBatchWrapper89_copies622 a h).1
  have c19905 := (publicBatchWrapper89_copies622 a h).2.1
  have c19906 := (publicBatchWrapper89_copies622 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_302_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_12
  simp only [← c19890, k3, ← c19891, k3, ← c19892] at e_302_12
  have e_302_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_13
  simp only [← c19893, ← c19894, k3, ← c19895] at e_302_13
  have e_302_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_14
  simp only [← c19902, ← c19903, k3, ← c19904] at e_302_14
  have e_303_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_11
  simp only [← c19896, ← c19897, ← c19898] at e_303_11
  have e_303_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_12
  simp only [← c19899, ← c19900, ← c19901] at e_303_12
  refine ⟨?_, ?_⟩
  · have hc := e_303_11
    simp only [e_302_13] at hc
    linear_combination c19905.trans k2 - hc
  · have hc := e_302_14
    simp only [e_302_12, e_303_12, e_302_13] at hc
    linear_combination c19906.trans k2 - hc

theorem publicBatchWrapper89_f2712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 19) = bor (a (.wire 67 11)) (a (.virt 19335)) := by
  have c19907 := (publicBatchWrapper89_copies622 a h).2.2.2.1
  have c19908 := (publicBatchWrapper89_copies622 a h).2.2.2.2.1
  have c19909 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.1
  have c19910 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.1
  have c19911 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.1
  have c19912 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_12
  simp only [← c19907, ← c19908, ← c19909] at e_296_12
  have e_301_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_4
  simp only [← c19910, ← c19911, k3, ← c19912] at e_301_4
  have hr := e_301_4
  simp only [e_296_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 19) = a (.virt 201) := by
  have c19913 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.1
  exact c19913

theorem publicBatchWrapper89_f2714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13604)) (a (.wire 170 31)) (a (.virt 19337)) (a (.virt 19338)) := by
  have c19914 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19915 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19916 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19917 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19918 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19919 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19920 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19921 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19922 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19923 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19924 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19925 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19926 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19927 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19928 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19929 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19930 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_302_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_15
  simp only [← c19914, k3, ← c19915, k3, ← c19916] at e_302_15
  have e_302_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_16
  simp only [← c19917, ← c19918, k3, ← c19919] at e_302_16
  have e_302_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_17
  simp only [← c19926, ← c19927, k3, ← c19928] at e_302_17
  have e_303_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_13
  simp only [← c19920, ← c19921, ← c19922] at e_303_13
  have e_303_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_14
  simp only [← c19923, ← c19924, ← c19925] at e_303_14
  refine ⟨?_, ?_⟩
  · have hc := e_303_13
    simp only [e_302_16] at hc
    linear_combination c19929.trans k2 - hc
  · have hc := e_302_17
    simp only [e_302_15, e_303_14, e_302_16] at hc
    linear_combination c19930.trans k2 - hc

theorem publicBatchWrapper89_f2715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 23) = bor (a (.wire 67 11)) (a (.virt 19337)) := by
  have c19931 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19932 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19933 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19934 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19935 := (publicBatchWrapper89_copies622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19936 := (publicBatchWrapper89_copies623 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_13
  simp only [← c19931, ← c19932, ← c19933] at e_296_13
  have e_301_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_5
  simp only [← c19934, ← c19935, k3, ← c19936] at e_301_5
  have hr := e_301_5
  simp only [e_296_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 23) = a (.virt 201) := by
  have c19937 := (publicBatchWrapper89_copies623 a h).2.1
  exact c19937

theorem publicBatchWrapper89_f2717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13605)) (a (.wire 169 63)) (a (.virt 19339)) (a (.virt 19340)) := by
  have c19938 := (publicBatchWrapper89_copies623 a h).2.2.1
  have c19939 := (publicBatchWrapper89_copies623 a h).2.2.2.1
  have c19940 := (publicBatchWrapper89_copies623 a h).2.2.2.2.1
  have c19941 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.1
  have c19942 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.1
  have c19943 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.1
  have c19944 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.1
  have c19945 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.1
  have c19946 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19947 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19948 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19949 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19950 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19951 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19952 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19953 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19954 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_302_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_18
  simp only [← c19938, k3, ← c19939, k3, ← c19940] at e_302_18
  have e_302_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row302 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_302_19
  simp only [← c19941, ← c19942, k3, ← c19943] at e_302_19
  have e_303_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_15
  simp only [← c19944, ← c19945, ← c19946] at e_303_15
  have e_303_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_16
  simp only [← c19947, ← c19948, ← c19949] at e_303_16
  have e_304_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_0
  simp only [← c19950, ← c19951, k3, ← c19952] at e_304_0
  refine ⟨?_, ?_⟩
  · have hc := e_303_15
    simp only [e_302_19] at hc
    linear_combination c19953.trans k2 - hc
  · have hc := e_304_0
    simp only [e_302_18, e_303_16, e_302_19] at hc
    linear_combination c19954.trans k2 - hc

theorem publicBatchWrapper89_f2718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13606)) (a (.wire 169 71)) (a (.virt 19341)) (a (.virt 19342)) := by
  have c19955 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19956 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19957 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19958 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19959 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19960 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19961 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19962 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19963 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19964 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19965 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19966 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19967 := (publicBatchWrapper89_copies623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c19968 := (publicBatchWrapper89_copies624 a h).1
  have c19969 := (publicBatchWrapper89_copies624 a h).2.1
  have c19970 := (publicBatchWrapper89_copies624 a h).2.2.1
  have c19971 := (publicBatchWrapper89_copies624 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_303_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_17
  simp only [← c19961, ← c19962, ← c19963] at e_303_17
  have e_303_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_18
  simp only [← c19964, ← c19965, ← c19966] at e_303_18
  have e_304_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_1
  simp only [← c19955, k3, ← c19956, k3, ← c19957] at e_304_1
  have e_304_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_2
  simp only [← c19958, ← c19959, k3, ← c19960] at e_304_2
  have e_304_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_3
  simp only [← c19967, ← c19968, k3, ← c19969] at e_304_3
  refine ⟨?_, ?_⟩
  · have hc := e_303_17
    simp only [e_304_2] at hc
    linear_combination c19970.trans k2 - hc
  · have hc := e_304_3
    simp only [e_304_1, e_303_18, e_304_2] at hc
    linear_combination c19971.trans k2 - hc

theorem publicBatchWrapper89_f2719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13607)) (a (.wire 169 79)) (a (.virt 19343)) (a (.virt 19344)) := by
  have c19972 := (publicBatchWrapper89_copies624 a h).2.2.2.2.1
  have c19973 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.1
  have c19974 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.1
  have c19975 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.1
  have c19976 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.1
  have c19977 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.1
  have c19978 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.1
  have c19979 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c19980 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19981 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19982 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19983 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19984 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19985 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19986 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19987 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19988 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_303_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row303 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_303_19
  simp only [← c19978, ← c19979, ← c19980] at e_303_19
  have e_304_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_4
  simp only [← c19972, k3, ← c19973, k3, ← c19974] at e_304_4
  have e_304_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_5
  simp only [← c19975, ← c19976, k3, ← c19977] at e_304_5
  have e_304_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_6
  simp only [← c19984, ← c19985, k3, ← c19986] at e_304_6
  have e_305_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_0
  simp only [← c19981, ← c19982, ← c19983] at e_305_0
  refine ⟨?_, ?_⟩
  · have hc := e_303_19
    simp only [e_304_5] at hc
    linear_combination c19987.trans k2 - hc
  · have hc := e_304_6
    simp only [e_304_4, e_305_0, e_304_5] at hc
    linear_combination c19988.trans k2 - hc

theorem publicBatchWrapper89_f2720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13608)) (a (.wire 170 7)) (a (.virt 19345)) (a (.virt 19346)) := by
  have c19989 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19990 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19991 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19992 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19993 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19994 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19995 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19996 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19997 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19998 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19999 := (publicBatchWrapper89_copies624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20000 := (publicBatchWrapper89_copies625 a h).1
  have c20001 := (publicBatchWrapper89_copies625 a h).2.1
  have c20002 := (publicBatchWrapper89_copies625 a h).2.2.1
  have c20003 := (publicBatchWrapper89_copies625 a h).2.2.2.1
  have c20004 := (publicBatchWrapper89_copies625 a h).2.2.2.2.1
  have c20005 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_304_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_7
  simp only [← c19989, k3, ← c19990, k3, ← c19991] at e_304_7
  have e_304_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_8
  simp only [← c19992, ← c19993, k3, ← c19994] at e_304_8
  have e_304_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_9
  simp only [← c20001, ← c20002, k3, ← c20003] at e_304_9
  have e_305_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_1
  simp only [← c19995, ← c19996, ← c19997] at e_305_1
  have e_305_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_2
  simp only [← c19998, ← c19999, ← c20000] at e_305_2
  refine ⟨?_, ?_⟩
  · have hc := e_305_1
    simp only [e_304_8] at hc
    linear_combination c20004.trans k2 - hc
  · have hc := e_304_9
    simp only [e_304_7, e_305_2, e_304_8] at hc
    linear_combination c20005.trans k2 - hc

theorem publicBatchWrapper89_f2721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 305 15) = band (a (.virt 19339)) (a (.virt 19341)) := by
  have c20006 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.1
  have c20007 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.1
  have c20008 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.1
  have e_305_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_3
  simp only [← c20006, ← c20007, ← c20008] at e_305_3
  have hr := e_305_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 305 19) = band (a (.virt 19343)) (a (.virt 19345)) := by
  have c20009 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.1
  have c20010 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20011 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_305_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_4
  simp only [← c20009, ← c20010, ← c20011] at e_305_4
  have hr := e_305_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 305 23) = band (a (.wire 305 15)) (a (.wire 305 19)) := by
  have c20012 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20013 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20014 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_305_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_5
  simp only [← c20012, ← c20013, ← c20014] at e_305_5
  have hr := e_305_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 27) = bor (a (.wire 67 11)) (a (.wire 305 23)) := by
  have c20015 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20016 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20017 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20018 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20019 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20020 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_14
  simp only [← c20015, ← c20016, ← c20017] at e_296_14
  have e_301_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_6
  simp only [← c20018, ← c20019, k3, ← c20020] at e_301_6
  have hr := e_301_6
  simp only [e_296_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 27) = a (.virt 201) := by
  have c20021 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20021

theorem publicBatchWrapper89_f2726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13803)) (a (.wire 170 23)) (a (.virt 19347)) (a (.virt 19348)) := by
  have c20022 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20023 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20024 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20025 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20026 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20027 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20028 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20029 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20030 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20031 := (publicBatchWrapper89_copies625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20032 := (publicBatchWrapper89_copies626 a h).1
  have c20033 := (publicBatchWrapper89_copies626 a h).2.1
  have c20034 := (publicBatchWrapper89_copies626 a h).2.2.1
  have c20035 := (publicBatchWrapper89_copies626 a h).2.2.2.1
  have c20036 := (publicBatchWrapper89_copies626 a h).2.2.2.2.1
  have c20037 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.1
  have c20038 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_304_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_10
  simp only [← c20022, k3, ← c20023, k3, ← c20024] at e_304_10
  have e_304_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_11
  simp only [← c20025, ← c20026, k3, ← c20027] at e_304_11
  have e_304_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_12
  simp only [← c20034, ← c20035, k3, ← c20036] at e_304_12
  have e_305_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_6
  simp only [← c20028, ← c20029, ← c20030] at e_305_6
  have e_305_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_7
  simp only [← c20031, ← c20032, ← c20033] at e_305_7
  refine ⟨?_, ?_⟩
  · have hc := e_305_6
    simp only [e_304_11] at hc
    linear_combination c20037.trans k2 - hc
  · have hc := e_304_12
    simp only [e_304_10, e_305_7, e_304_11] at hc
    linear_combination c20038.trans k2 - hc

theorem publicBatchWrapper89_f2727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 31) = bor (a (.wire 67 55)) (a (.virt 19347)) := by
  have c20039 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.1
  have c20040 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.1
  have c20041 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.1
  have c20042 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20043 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20044 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_15
  simp only [← c20039, ← c20040, ← c20041] at e_296_15
  have e_301_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_7
  simp only [← c20042, ← c20043, k3, ← c20044] at e_301_7
  have hr := e_301_7
  simp only [e_296_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 31) = a (.virt 201) := by
  have c20045 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20045

theorem publicBatchWrapper89_f2729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13804)) (a (.wire 170 31)) (a (.virt 19349)) (a (.virt 19350)) := by
  have c20046 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20047 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20048 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20049 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20050 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20051 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20052 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20053 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20054 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20055 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20056 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20057 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20058 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20059 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20060 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20061 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20062 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_304_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_13
  simp only [← c20046, k3, ← c20047, k3, ← c20048] at e_304_13
  have e_304_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_14
  simp only [← c20049, ← c20050, k3, ← c20051] at e_304_14
  have e_304_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_15
  simp only [← c20058, ← c20059, k3, ← c20060] at e_304_15
  have e_305_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_8
  simp only [← c20052, ← c20053, ← c20054] at e_305_8
  have e_305_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_9
  simp only [← c20055, ← c20056, ← c20057] at e_305_9
  refine ⟨?_, ?_⟩
  · have hc := e_305_8
    simp only [e_304_14] at hc
    linear_combination c20061.trans k2 - hc
  · have hc := e_304_15
    simp only [e_304_13, e_305_9, e_304_14] at hc
    linear_combination c20062.trans k2 - hc

theorem publicBatchWrapper89_f2730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 35) = bor (a (.wire 67 55)) (a (.virt 19349)) := by
  have c20063 := (publicBatchWrapper89_copies626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20064 := (publicBatchWrapper89_copies627 a h).1
  have c20065 := (publicBatchWrapper89_copies627 a h).2.1
  have c20066 := (publicBatchWrapper89_copies627 a h).2.2.1
  have c20067 := (publicBatchWrapper89_copies627 a h).2.2.2.1
  have c20068 := (publicBatchWrapper89_copies627 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_16
  simp only [← c20063, ← c20064, ← c20065] at e_296_16
  have e_301_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_8
  simp only [← c20066, ← c20067, k3, ← c20068] at e_301_8
  have hr := e_301_8
  simp only [e_296_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 35) = a (.virt 201) := by
  have c20069 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.1
  exact c20069

theorem publicBatchWrapper89_f2732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13805)) (a (.wire 169 63)) (a (.virt 19351)) (a (.virt 19352)) := by
  have c20070 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.1
  have c20071 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.1
  have c20072 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.1
  have c20073 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.1
  have c20074 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20075 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20076 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20077 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20078 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20079 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20080 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20081 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20082 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20083 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20084 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20085 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20086 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_304_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_16
  simp only [← c20070, k3, ← c20071, k3, ← c20072] at e_304_16
  have e_304_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_17
  simp only [← c20073, ← c20074, k3, ← c20075] at e_304_17
  have e_304_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_18
  simp only [← c20082, ← c20083, k3, ← c20084] at e_304_18
  have e_305_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_10
  simp only [← c20076, ← c20077, ← c20078] at e_305_10
  have e_305_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_11
  simp only [← c20079, ← c20080, ← c20081] at e_305_11
  refine ⟨?_, ?_⟩
  · have hc := e_305_10
    simp only [e_304_17] at hc
    linear_combination c20085.trans k2 - hc
  · have hc := e_304_18
    simp only [e_304_16, e_305_11, e_304_17] at hc
    linear_combination c20086.trans k2 - hc

theorem publicBatchWrapper89_f2733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13806)) (a (.wire 169 71)) (a (.virt 19353)) (a (.virt 19354)) := by
  have c20087 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20088 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20089 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20090 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20091 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20092 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20093 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20094 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20095 := (publicBatchWrapper89_copies627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20096 := (publicBatchWrapper89_copies628 a h).1
  have c20097 := (publicBatchWrapper89_copies628 a h).2.1
  have c20098 := (publicBatchWrapper89_copies628 a h).2.2.1
  have c20099 := (publicBatchWrapper89_copies628 a h).2.2.2.1
  have c20100 := (publicBatchWrapper89_copies628 a h).2.2.2.2.1
  have c20101 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.1
  have c20102 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.1
  have c20103 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_304_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row304 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_304_19
  simp only [← c20087, k3, ← c20088, k3, ← c20089] at e_304_19
  have e_305_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_12
  simp only [← c20093, ← c20094, ← c20095] at e_305_12
  have e_305_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_13
  simp only [← c20096, ← c20097, ← c20098] at e_305_13
  have e_306_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_0
  simp only [← c20090, ← c20091, k3, ← c20092] at e_306_0
  have e_306_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_1
  simp only [← c20099, ← c20100, k3, ← c20101] at e_306_1
  refine ⟨?_, ?_⟩
  · have hc := e_305_12
    simp only [e_306_0] at hc
    linear_combination c20102.trans k2 - hc
  · have hc := e_306_1
    simp only [e_304_19, e_305_13, e_306_0] at hc
    linear_combination c20103.trans k2 - hc

theorem publicBatchWrapper89_f2734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13807)) (a (.wire 169 79)) (a (.virt 19355)) (a (.virt 19356)) := by
  have c20104 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.1
  have c20105 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.1
  have c20106 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20107 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20108 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20109 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20110 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20111 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20112 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20113 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20114 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20115 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20116 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20117 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20118 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20119 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20120 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_305_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_14
  simp only [← c20110, ← c20111, ← c20112] at e_305_14
  have e_305_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_15
  simp only [← c20113, ← c20114, ← c20115] at e_305_15
  have e_306_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_2
  simp only [← c20104, k3, ← c20105, k3, ← c20106] at e_306_2
  have e_306_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_3
  simp only [← c20107, ← c20108, k3, ← c20109] at e_306_3
  have e_306_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_4
  simp only [← c20116, ← c20117, k3, ← c20118] at e_306_4
  refine ⟨?_, ?_⟩
  · have hc := e_305_14
    simp only [e_306_3] at hc
    linear_combination c20119.trans k2 - hc
  · have hc := e_306_4
    simp only [e_306_2, e_305_15, e_306_3] at hc
    linear_combination c20120.trans k2 - hc

theorem publicBatchWrapper89_f2735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13808)) (a (.wire 170 7)) (a (.virt 19357)) (a (.virt 19358)) := by
  have c20121 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20122 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20123 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20124 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20125 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20126 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20127 := (publicBatchWrapper89_copies628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20128 := (publicBatchWrapper89_copies629 a h).1
  have c20129 := (publicBatchWrapper89_copies629 a h).2.1
  have c20130 := (publicBatchWrapper89_copies629 a h).2.2.1
  have c20131 := (publicBatchWrapper89_copies629 a h).2.2.2.1
  have c20132 := (publicBatchWrapper89_copies629 a h).2.2.2.2.1
  have c20133 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.1
  have c20134 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.1
  have c20135 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.1
  have c20136 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.1
  have c20137 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_305_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_16
  simp only [← c20127, ← c20128, ← c20129] at e_305_16
  have e_305_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_17
  simp only [← c20130, ← c20131, ← c20132] at e_305_17
  have e_306_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_5
  simp only [← c20121, k3, ← c20122, k3, ← c20123] at e_306_5
  have e_306_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_6
  simp only [← c20124, ← c20125, k3, ← c20126] at e_306_6
  have e_306_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_7
  simp only [← c20133, ← c20134, k3, ← c20135] at e_306_7
  refine ⟨?_, ?_⟩
  · have hc := e_305_16
    simp only [e_306_6] at hc
    linear_combination c20136.trans k2 - hc
  · have hc := e_306_7
    simp only [e_306_5, e_305_17, e_306_6] at hc
    linear_combination c20137.trans k2 - hc

theorem publicBatchWrapper89_f2736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 305 75) = band (a (.virt 19351)) (a (.virt 19353)) := by
  have c20138 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20139 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20140 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_305_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_18
  simp only [← c20138, ← c20139, ← c20140] at e_305_18
  have hr := e_305_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 305 79) = band (a (.virt 19355)) (a (.virt 19357)) := by
  have c20141 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20142 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20143 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_305_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row305 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_305_19
  simp only [← c20141, ← c20142, ← c20143] at e_305_19
  have hr := e_305_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 307 3) = band (a (.wire 305 75)) (a (.wire 305 79)) := by
  have c20144 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20145 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20146 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_307_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_0
  simp only [← c20144, ← c20145, ← c20146] at e_307_0
  have hr := e_307_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 39) = bor (a (.wire 67 55)) (a (.wire 307 3)) := by
  have c20147 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20148 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20149 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20150 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20151 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20152 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_17
  simp only [← c20147, ← c20148, ← c20149] at e_296_17
  have e_301_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_9
  simp only [← c20150, ← c20151, k3, ← c20152] at e_301_9
  have hr := e_301_9
  simp only [e_296_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 39) = a (.virt 201) := by
  have c20153 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20153

theorem publicBatchWrapper89_f2741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14003)) (a (.wire 170 23)) (a (.virt 19359)) (a (.virt 19360)) := by
  have c20154 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20155 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20156 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20157 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20158 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20159 := (publicBatchWrapper89_copies629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20160 := (publicBatchWrapper89_copies630 a h).1
  have c20161 := (publicBatchWrapper89_copies630 a h).2.1
  have c20162 := (publicBatchWrapper89_copies630 a h).2.2.1
  have c20163 := (publicBatchWrapper89_copies630 a h).2.2.2.1
  have c20164 := (publicBatchWrapper89_copies630 a h).2.2.2.2.1
  have c20165 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.1
  have c20166 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.1
  have c20167 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.1
  have c20168 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.1
  have c20169 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.1
  have c20170 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_306_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_8
  simp only [← c20154, k3, ← c20155, k3, ← c20156] at e_306_8
  have e_306_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_9
  simp only [← c20157, ← c20158, k3, ← c20159] at e_306_9
  have e_306_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_10
  simp only [← c20166, ← c20167, k3, ← c20168] at e_306_10
  have e_307_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_1
  simp only [← c20160, ← c20161, ← c20162] at e_307_1
  have e_307_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_2
  simp only [← c20163, ← c20164, ← c20165] at e_307_2
  refine ⟨?_, ?_⟩
  · have hc := e_307_1
    simp only [e_306_9] at hc
    linear_combination c20169.trans k2 - hc
  · have hc := e_306_10
    simp only [e_306_8, e_307_2, e_306_9] at hc
    linear_combination c20170.trans k2 - hc

theorem publicBatchWrapper89_f2742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 43) = bor (a (.wire 69 19)) (a (.virt 19359)) := by
  have c20171 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20172 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20173 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20174 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20175 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20176 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_18
  simp only [← c20171, ← c20172, ← c20173] at e_296_18
  have e_301_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_10
  simp only [← c20174, ← c20175, k3, ← c20176] at e_301_10
  have hr := e_301_10
  simp only [e_296_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 43) = a (.virt 201) := by
  have c20177 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20177

theorem publicBatchWrapper89_f2744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14004)) (a (.wire 170 31)) (a (.virt 19361)) (a (.virt 19362)) := by
  have c20178 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20179 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20180 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20181 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20182 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20183 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20184 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20185 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20186 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20187 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20188 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20189 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20190 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20191 := (publicBatchWrapper89_copies630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20192 := (publicBatchWrapper89_copies631 a h).1
  have c20193 := (publicBatchWrapper89_copies631 a h).2.1
  have c20194 := (publicBatchWrapper89_copies631 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_306_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_11
  simp only [← c20178, k3, ← c20179, k3, ← c20180] at e_306_11
  have e_306_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_12
  simp only [← c20181, ← c20182, k3, ← c20183] at e_306_12
  have e_306_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_13
  simp only [← c20190, ← c20191, k3, ← c20192] at e_306_13
  have e_307_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_3
  simp only [← c20184, ← c20185, ← c20186] at e_307_3
  have e_307_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_4
  simp only [← c20187, ← c20188, ← c20189] at e_307_4
  refine ⟨?_, ?_⟩
  · have hc := e_307_3
    simp only [e_306_12] at hc
    linear_combination c20193.trans k2 - hc
  · have hc := e_306_13
    simp only [e_306_11, e_307_4, e_306_12] at hc
    linear_combination c20194.trans k2 - hc

theorem publicBatchWrapper89_f2745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 47) = bor (a (.wire 69 19)) (a (.virt 19361)) := by
  have c20195 := (publicBatchWrapper89_copies631 a h).2.2.2.1
  have c20196 := (publicBatchWrapper89_copies631 a h).2.2.2.2.1
  have c20197 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.1
  have c20198 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.1
  have c20199 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.1
  have c20200 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_296_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row296 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_296_19
  simp only [← c20195, ← c20196, ← c20197] at e_296_19
  have e_301_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_11
  simp only [← c20198, ← c20199, k3, ← c20200] at e_301_11
  have hr := e_301_11
  simp only [e_296_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 47) = a (.virt 201) := by
  have c20201 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.1
  exact c20201

theorem publicBatchWrapper89_f2747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14005)) (a (.wire 169 63)) (a (.virt 19363)) (a (.virt 19364)) := by
  have c20202 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20203 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20204 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20205 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20206 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20207 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20208 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20209 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20210 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20211 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20212 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20213 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20214 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20215 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20216 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20217 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20218 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_306_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_14
  simp only [← c20202, k3, ← c20203, k3, ← c20204] at e_306_14
  have e_306_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_15
  simp only [← c20205, ← c20206, k3, ← c20207] at e_306_15
  have e_306_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_16
  simp only [← c20214, ← c20215, k3, ← c20216] at e_306_16
  have e_307_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_5
  simp only [← c20208, ← c20209, ← c20210] at e_307_5
  have e_307_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_6
  simp only [← c20211, ← c20212, ← c20213] at e_307_6
  refine ⟨?_, ?_⟩
  · have hc := e_307_5
    simp only [e_306_15] at hc
    linear_combination c20217.trans k2 - hc
  · have hc := e_306_16
    simp only [e_306_14, e_307_6, e_306_15] at hc
    linear_combination c20218.trans k2 - hc

theorem publicBatchWrapper89_f2748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14006)) (a (.wire 169 71)) (a (.virt 19365)) (a (.virt 19366)) := by
  have c20219 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20220 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20221 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20222 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20223 := (publicBatchWrapper89_copies631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20224 := (publicBatchWrapper89_copies632 a h).1
  have c20225 := (publicBatchWrapper89_copies632 a h).2.1
  have c20226 := (publicBatchWrapper89_copies632 a h).2.2.1
  have c20227 := (publicBatchWrapper89_copies632 a h).2.2.2.1
  have c20228 := (publicBatchWrapper89_copies632 a h).2.2.2.2.1
  have c20229 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.1
  have c20230 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.1
  have c20231 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.1
  have c20232 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.1
  have c20233 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.1
  have c20234 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20235 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_306_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_17
  simp only [← c20219, k3, ← c20220, k3, ← c20221] at e_306_17
  have e_306_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_18
  simp only [← c20222, ← c20223, k3, ← c20224] at e_306_18
  have e_306_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row306 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_306_19
  simp only [← c20231, ← c20232, k3, ← c20233] at e_306_19
  have e_307_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_7
  simp only [← c20225, ← c20226, ← c20227] at e_307_7
  have e_307_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_8
  simp only [← c20228, ← c20229, ← c20230] at e_307_8
  refine ⟨?_, ?_⟩
  · have hc := e_307_7
    simp only [e_306_18] at hc
    linear_combination c20234.trans k2 - hc
  · have hc := e_306_19
    simp only [e_306_17, e_307_8, e_306_18] at hc
    linear_combination c20235.trans k2 - hc

theorem publicBatchWrapper89_f2749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14007)) (a (.wire 169 79)) (a (.virt 19367)) (a (.virt 19368)) := by
  have c20236 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20237 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20238 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20239 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20240 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20241 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20242 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20243 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20244 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20245 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20246 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20247 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20248 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20249 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20250 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20251 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20252 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_307_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_9
  simp only [← c20242, ← c20243, ← c20244] at e_307_9
  have e_307_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_10
  simp only [← c20245, ← c20246, ← c20247] at e_307_10
  have e_308_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_0
  simp only [← c20236, k3, ← c20237, k3, ← c20238] at e_308_0
  have e_308_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_1
  simp only [← c20239, ← c20240, k3, ← c20241] at e_308_1
  have e_308_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_2
  simp only [← c20248, ← c20249, k3, ← c20250] at e_308_2
  refine ⟨?_, ?_⟩
  · have hc := e_307_9
    simp only [e_308_1] at hc
    linear_combination c20251.trans k2 - hc
  · have hc := e_308_2
    simp only [e_308_0, e_307_10, e_308_1] at hc
    linear_combination c20252.trans k2 - hc

theorem publicBatchWrapper89_f2750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14008)) (a (.wire 170 7)) (a (.virt 19369)) (a (.virt 19370)) := by
  have c20253 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20254 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20255 := (publicBatchWrapper89_copies632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20256 := (publicBatchWrapper89_copies633 a h).1
  have c20257 := (publicBatchWrapper89_copies633 a h).2.1
  have c20258 := (publicBatchWrapper89_copies633 a h).2.2.1
  have c20259 := (publicBatchWrapper89_copies633 a h).2.2.2.1
  have c20260 := (publicBatchWrapper89_copies633 a h).2.2.2.2.1
  have c20261 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.1
  have c20262 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.1
  have c20263 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.1
  have c20264 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.1
  have c20265 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.1
  have c20266 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20267 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20268 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20269 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_307_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_11
  simp only [← c20259, ← c20260, ← c20261] at e_307_11
  have e_307_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_12
  simp only [← c20262, ← c20263, ← c20264] at e_307_12
  have e_308_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_3
  simp only [← c20253, k3, ← c20254, k3, ← c20255] at e_308_3
  have e_308_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_4
  simp only [← c20256, ← c20257, k3, ← c20258] at e_308_4
  have e_308_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_5
  simp only [← c20265, ← c20266, k3, ← c20267] at e_308_5
  refine ⟨?_, ?_⟩
  · have hc := e_307_11
    simp only [e_308_4] at hc
    linear_combination c20268.trans k2 - hc
  · have hc := e_308_5
    simp only [e_308_3, e_307_12, e_308_4] at hc
    linear_combination c20269.trans k2 - hc

theorem publicBatchWrapper89_f2751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 307 55) = band (a (.virt 19363)) (a (.virt 19365)) := by
  have c20270 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20271 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20272 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_307_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_13
  simp only [← c20270, ← c20271, ← c20272] at e_307_13
  have hr := e_307_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 307 59) = band (a (.virt 19367)) (a (.virt 19369)) := by
  have c20273 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20274 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20275 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_307_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_14
  simp only [← c20273, ← c20274, ← c20275] at e_307_14
  have hr := e_307_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 307 63) = band (a (.wire 307 55)) (a (.wire 307 59)) := by
  have c20276 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20277 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20278 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_307_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_15
  simp only [← c20276, ← c20277, ← c20278] at e_307_15
  have hr := e_307_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 51) = bor (a (.wire 69 19)) (a (.wire 307 63)) := by
  have c20279 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20280 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20281 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20282 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20283 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20284 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_301_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_12
  simp only [← c20282, ← c20283, k3, ← c20284] at e_301_12
  have e_309_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_0
  simp only [← c20279, ← c20280, ← c20281] at e_309_0
  have hr := e_301_12
  simp only [e_309_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 51) = a (.virt 201) := by
  have c20285 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20285

theorem publicBatchWrapper89_f2756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14203)) (a (.wire 170 23)) (a (.virt 19371)) (a (.virt 19372)) := by
  have c20286 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20287 := (publicBatchWrapper89_copies633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20288 := (publicBatchWrapper89_copies634 a h).1
  have c20289 := (publicBatchWrapper89_copies634 a h).2.1
  have c20290 := (publicBatchWrapper89_copies634 a h).2.2.1
  have c20291 := (publicBatchWrapper89_copies634 a h).2.2.2.1
  have c20292 := (publicBatchWrapper89_copies634 a h).2.2.2.2.1
  have c20293 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.1
  have c20294 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.1
  have c20295 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.1
  have c20296 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.1
  have c20297 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.1
  have c20298 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20299 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20300 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20301 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20302 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_307_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_16
  simp only [← c20292, ← c20293, ← c20294] at e_307_16
  have e_307_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_17
  simp only [← c20295, ← c20296, ← c20297] at e_307_17
  have e_308_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_6
  simp only [← c20286, k3, ← c20287, k3, ← c20288] at e_308_6
  have e_308_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_7
  simp only [← c20289, ← c20290, k3, ← c20291] at e_308_7
  have e_308_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_8
  simp only [← c20298, ← c20299, k3, ← c20300] at e_308_8
  refine ⟨?_, ?_⟩
  · have hc := e_307_16
    simp only [e_308_7] at hc
    linear_combination c20301.trans k2 - hc
  · have hc := e_308_8
    simp only [e_308_6, e_307_17, e_308_7] at hc
    linear_combination c20302.trans k2 - hc

theorem publicBatchWrapper89_f2757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 55) = bor (a (.wire 69 63)) (a (.virt 19371)) := by
  have c20303 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20304 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20305 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20306 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20307 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20308 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_301_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_13
  simp only [← c20306, ← c20307, k3, ← c20308] at e_301_13
  have e_309_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_1
  simp only [← c20303, ← c20304, ← c20305] at e_309_1
  have hr := e_301_13
  simp only [e_309_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 55) = a (.virt 201) := by
  have c20309 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20309

theorem publicBatchWrapper89_f2759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14204)) (a (.wire 170 31)) (a (.virt 19373)) (a (.virt 19374)) := by
  have c20310 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20311 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20312 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20313 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20314 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20315 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20316 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20317 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20318 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20319 := (publicBatchWrapper89_copies634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20320 := (publicBatchWrapper89_copies635 a h).1
  have c20321 := (publicBatchWrapper89_copies635 a h).2.1
  have c20322 := (publicBatchWrapper89_copies635 a h).2.2.1
  have c20323 := (publicBatchWrapper89_copies635 a h).2.2.2.1
  have c20324 := (publicBatchWrapper89_copies635 a h).2.2.2.2.1
  have c20325 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.1
  have c20326 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_307_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_18
  simp only [← c20316, ← c20317, ← c20318] at e_307_18
  have e_307_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row307 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_307_19
  simp only [← c20319, ← c20320, ← c20321] at e_307_19
  have e_308_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_9
  simp only [← c20310, k3, ← c20311, k3, ← c20312] at e_308_9
  have e_308_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_10
  simp only [← c20313, ← c20314, k3, ← c20315] at e_308_10
  have e_308_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_11
  simp only [← c20322, ← c20323, k3, ← c20324] at e_308_11
  refine ⟨?_, ?_⟩
  · have hc := e_307_18
    simp only [e_308_10] at hc
    linear_combination c20325.trans k2 - hc
  · have hc := e_308_11
    simp only [e_308_9, e_307_19, e_308_10] at hc
    linear_combination c20326.trans k2 - hc

theorem publicBatchWrapper89_f2760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 59) = bor (a (.wire 69 63)) (a (.virt 19373)) := by
  have c20327 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.1
  have c20328 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.1
  have c20329 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.1
  have c20330 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20331 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20332 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_301_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_14
  simp only [← c20330, ← c20331, k3, ← c20332] at e_301_14
  have e_309_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_2
  simp only [← c20327, ← c20328, ← c20329] at e_309_2
  have hr := e_301_14
  simp only [e_309_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 59) = a (.virt 201) := by
  have c20333 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20333

theorem publicBatchWrapper89_f2762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14205)) (a (.wire 169 63)) (a (.virt 19375)) (a (.virt 19376)) := by
  have c20334 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20335 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20336 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20337 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20338 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20339 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20340 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20341 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20342 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20343 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20344 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20345 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20346 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20347 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20348 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20349 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20350 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_308_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_12
  simp only [← c20334, k3, ← c20335, k3, ← c20336] at e_308_12
  have e_308_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_13
  simp only [← c20337, ← c20338, k3, ← c20339] at e_308_13
  have e_308_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_14
  simp only [← c20346, ← c20347, k3, ← c20348] at e_308_14
  have e_310_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_0
  simp only [← c20340, ← c20341, ← c20342] at e_310_0
  have e_310_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_1
  simp only [← c20343, ← c20344, ← c20345] at e_310_1
  refine ⟨?_, ?_⟩
  · have hc := e_310_0
    simp only [e_308_13] at hc
    linear_combination c20349.trans k2 - hc
  · have hc := e_308_14
    simp only [e_308_12, e_310_1, e_308_13] at hc
    linear_combination c20350.trans k2 - hc

theorem publicBatchWrapper89_f2763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14206)) (a (.wire 169 71)) (a (.virt 19377)) (a (.virt 19378)) := by
  have c20351 := (publicBatchWrapper89_copies635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20352 := (publicBatchWrapper89_copies636 a h).1
  have c20353 := (publicBatchWrapper89_copies636 a h).2.1
  have c20354 := (publicBatchWrapper89_copies636 a h).2.2.1
  have c20355 := (publicBatchWrapper89_copies636 a h).2.2.2.1
  have c20356 := (publicBatchWrapper89_copies636 a h).2.2.2.2.1
  have c20357 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.1
  have c20358 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.1
  have c20359 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.1
  have c20360 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.1
  have c20361 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.1
  have c20362 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20363 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20364 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20365 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20366 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20367 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_308_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_15
  simp only [← c20351, k3, ← c20352, k3, ← c20353] at e_308_15
  have e_308_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_16
  simp only [← c20354, ← c20355, k3, ← c20356] at e_308_16
  have e_308_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_17
  simp only [← c20363, ← c20364, k3, ← c20365] at e_308_17
  have e_310_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_2
  simp only [← c20357, ← c20358, ← c20359] at e_310_2
  have e_310_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_3
  simp only [← c20360, ← c20361, ← c20362] at e_310_3
  refine ⟨?_, ?_⟩
  · have hc := e_310_2
    simp only [e_308_16] at hc
    linear_combination c20366.trans k2 - hc
  · have hc := e_308_17
    simp only [e_308_15, e_310_3, e_308_16] at hc
    linear_combination c20367.trans k2 - hc

theorem publicBatchWrapper89_f2764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14207)) (a (.wire 169 79)) (a (.virt 19379)) (a (.virt 19380)) := by
  have c20368 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20369 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20370 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20371 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20372 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20373 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20374 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20375 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20376 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20377 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20378 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20379 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20380 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20381 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20382 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20383 := (publicBatchWrapper89_copies636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20384 := (publicBatchWrapper89_copies637 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_308_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_18
  simp only [← c20368, k3, ← c20369, k3, ← c20370] at e_308_18
  have e_308_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row308 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_308_19
  simp only [← c20371, ← c20372, k3, ← c20373] at e_308_19
  have e_310_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_4
  simp only [← c20374, ← c20375, ← c20376] at e_310_4
  have e_310_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_5
  simp only [← c20377, ← c20378, ← c20379] at e_310_5
  have e_311_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_0
  simp only [← c20380, ← c20381, k3, ← c20382] at e_311_0
  refine ⟨?_, ?_⟩
  · have hc := e_310_4
    simp only [e_308_19] at hc
    linear_combination c20383.trans k2 - hc
  · have hc := e_311_0
    simp only [e_308_18, e_310_5, e_308_19] at hc
    linear_combination c20384.trans k2 - hc

theorem publicBatchWrapper89_f2765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14208)) (a (.wire 170 7)) (a (.virt 19381)) (a (.virt 19382)) := by
  have c20385 := (publicBatchWrapper89_copies637 a h).2.1
  have c20386 := (publicBatchWrapper89_copies637 a h).2.2.1
  have c20387 := (publicBatchWrapper89_copies637 a h).2.2.2.1
  have c20388 := (publicBatchWrapper89_copies637 a h).2.2.2.2.1
  have c20389 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.1
  have c20390 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.1
  have c20391 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.1
  have c20392 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.1
  have c20393 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.1
  have c20394 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20395 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20396 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20397 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20398 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20399 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20400 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20401 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_310_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_6
  simp only [← c20391, ← c20392, ← c20393] at e_310_6
  have e_310_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_7
  simp only [← c20394, ← c20395, ← c20396] at e_310_7
  have e_311_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_1
  simp only [← c20385, k3, ← c20386, k3, ← c20387] at e_311_1
  have e_311_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_2
  simp only [← c20388, ← c20389, k3, ← c20390] at e_311_2
  have e_311_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_3
  simp only [← c20397, ← c20398, k3, ← c20399] at e_311_3
  refine ⟨?_, ?_⟩
  · have hc := e_310_6
    simp only [e_311_2] at hc
    linear_combination c20400.trans k2 - hc
  · have hc := e_311_3
    simp only [e_311_1, e_310_7, e_311_2] at hc
    linear_combination c20401.trans k2 - hc

theorem publicBatchWrapper89_f2766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 310 35) = band (a (.virt 19375)) (a (.virt 19377)) := by
  have c20402 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20403 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20404 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_310_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_8
  simp only [← c20402, ← c20403, ← c20404] at e_310_8
  have hr := e_310_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 310 39) = band (a (.virt 19379)) (a (.virt 19381)) := by
  have c20405 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20406 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20407 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_310_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_9
  simp only [← c20405, ← c20406, ← c20407] at e_310_9
  have hr := e_310_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 310 43) = band (a (.wire 310 35)) (a (.wire 310 39)) := by
  have c20408 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20409 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20410 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_310_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_10
  simp only [← c20408, ← c20409, ← c20410] at e_310_10
  have hr := e_310_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 63) = bor (a (.wire 69 63)) (a (.wire 310 43)) := by
  have c20411 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20412 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20413 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20414 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20415 := (publicBatchWrapper89_copies637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20416 := (publicBatchWrapper89_copies638 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_301_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_15
  simp only [← c20414, ← c20415, k3, ← c20416] at e_301_15
  have e_309_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_3
  simp only [← c20411, ← c20412, ← c20413] at e_309_3
  have hr := e_301_15
  simp only [e_309_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 63) = a (.virt 201) := by
  have c20417 := (publicBatchWrapper89_copies638 a h).2.1
  exact c20417

theorem publicBatchWrapper89_f2771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14403)) (a (.wire 170 23)) (a (.virt 19383)) (a (.virt 19384)) := by
  have c20418 := (publicBatchWrapper89_copies638 a h).2.2.1
  have c20419 := (publicBatchWrapper89_copies638 a h).2.2.2.1
  have c20420 := (publicBatchWrapper89_copies638 a h).2.2.2.2.1
  have c20421 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.1
  have c20422 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.1
  have c20423 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.1
  have c20424 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.1
  have c20425 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.1
  have c20426 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20427 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20428 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20429 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20430 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20431 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20432 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20433 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20434 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_310_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_11
  simp only [← c20424, ← c20425, ← c20426] at e_310_11
  have e_310_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_12
  simp only [← c20427, ← c20428, ← c20429] at e_310_12
  have e_311_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_4
  simp only [← c20418, k3, ← c20419, k3, ← c20420] at e_311_4
  have e_311_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_5
  simp only [← c20421, ← c20422, k3, ← c20423] at e_311_5
  have e_311_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_6
  simp only [← c20430, ← c20431, k3, ← c20432] at e_311_6
  refine ⟨?_, ?_⟩
  · have hc := e_310_11
    simp only [e_311_5] at hc
    linear_combination c20433.trans k2 - hc
  · have hc := e_311_6
    simp only [e_311_4, e_310_12, e_311_5] at hc
    linear_combination c20434.trans k2 - hc

theorem publicBatchWrapper89_f2772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 67) = bor (a (.wire 71 27)) (a (.virt 19383)) := by
  have c20435 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20436 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20437 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20438 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20439 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20440 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_301_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_16
  simp only [← c20438, ← c20439, k3, ← c20440] at e_301_16
  have e_309_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_4
  simp only [← c20435, ← c20436, ← c20437] at e_309_4
  have hr := e_301_16
  simp only [e_309_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 67) = a (.virt 201) := by
  have c20441 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20441

theorem publicBatchWrapper89_f2774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14404)) (a (.wire 170 31)) (a (.virt 19385)) (a (.virt 19386)) := by
  have c20442 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20443 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20444 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20445 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20446 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20447 := (publicBatchWrapper89_copies638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20448 := (publicBatchWrapper89_copies639 a h).1
  have c20449 := (publicBatchWrapper89_copies639 a h).2.1
  have c20450 := (publicBatchWrapper89_copies639 a h).2.2.1
  have c20451 := (publicBatchWrapper89_copies639 a h).2.2.2.1
  have c20452 := (publicBatchWrapper89_copies639 a h).2.2.2.2.1
  have c20453 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.1
  have c20454 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.1
  have c20455 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.1
  have c20456 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.1
  have c20457 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.1
  have c20458 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_310_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_13
  simp only [← c20448, ← c20449, ← c20450] at e_310_13
  have e_310_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_14
  simp only [← c20451, ← c20452, ← c20453] at e_310_14
  have e_311_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_7
  simp only [← c20442, k3, ← c20443, k3, ← c20444] at e_311_7
  have e_311_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_8
  simp only [← c20445, ← c20446, k3, ← c20447] at e_311_8
  have e_311_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_9
  simp only [← c20454, ← c20455, k3, ← c20456] at e_311_9
  refine ⟨?_, ?_⟩
  · have hc := e_310_13
    simp only [e_311_8] at hc
    linear_combination c20457.trans k2 - hc
  · have hc := e_311_9
    simp only [e_311_7, e_310_14, e_311_8] at hc
    linear_combination c20458.trans k2 - hc

theorem publicBatchWrapper89_f2775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 71) = bor (a (.wire 71 27)) (a (.virt 19385)) := by
  have c20459 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20460 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20461 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20462 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20463 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20464 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_301_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_17
  simp only [← c20462, ← c20463, k3, ← c20464] at e_301_17
  have e_309_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_5
  simp only [← c20459, ← c20460, ← c20461] at e_309_5
  have hr := e_301_17
  simp only [e_309_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 71) = a (.virt 201) := by
  have c20465 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20465

theorem publicBatchWrapper89_f2777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14405)) (a (.wire 169 63)) (a (.virt 19387)) (a (.virt 19388)) := by
  have c20466 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20467 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20468 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20469 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20470 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20471 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20472 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20473 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20474 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20475 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20476 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20477 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20478 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20479 := (publicBatchWrapper89_copies639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20480 := (publicBatchWrapper89_copies640 a h).1
  have c20481 := (publicBatchWrapper89_copies640 a h).2.1
  have c20482 := (publicBatchWrapper89_copies640 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_310_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_15
  simp only [← c20472, ← c20473, ← c20474] at e_310_15
  have e_310_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_16
  simp only [← c20475, ← c20476, ← c20477] at e_310_16
  have e_311_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_10
  simp only [← c20466, k3, ← c20467, k3, ← c20468] at e_311_10
  have e_311_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_11
  simp only [← c20469, ← c20470, k3, ← c20471] at e_311_11
  have e_311_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_12
  simp only [← c20478, ← c20479, k3, ← c20480] at e_311_12
  refine ⟨?_, ?_⟩
  · have hc := e_310_15
    simp only [e_311_11] at hc
    linear_combination c20481.trans k2 - hc
  · have hc := e_311_12
    simp only [e_311_10, e_310_16, e_311_11] at hc
    linear_combination c20482.trans k2 - hc

theorem publicBatchWrapper89_f2778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14406)) (a (.wire 169 71)) (a (.virt 19389)) (a (.virt 19390)) := by
  have c20483 := (publicBatchWrapper89_copies640 a h).2.2.2.1
  have c20484 := (publicBatchWrapper89_copies640 a h).2.2.2.2.1
  have c20485 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.1
  have c20486 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.1
  have c20487 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.1
  have c20488 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.1
  have c20489 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.1
  have c20490 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20491 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20492 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20493 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20494 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20495 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20496 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20497 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20498 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20499 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_310_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_17
  simp only [← c20489, ← c20490, ← c20491] at e_310_17
  have e_310_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_18
  simp only [← c20492, ← c20493, ← c20494] at e_310_18
  have e_311_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_13
  simp only [← c20483, k3, ← c20484, k3, ← c20485] at e_311_13
  have e_311_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_14
  simp only [← c20486, ← c20487, k3, ← c20488] at e_311_14
  have e_311_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_15
  simp only [← c20495, ← c20496, k3, ← c20497] at e_311_15
  refine ⟨?_, ?_⟩
  · have hc := e_310_17
    simp only [e_311_14] at hc
    linear_combination c20498.trans k2 - hc
  · have hc := e_311_15
    simp only [e_311_13, e_310_18, e_311_14] at hc
    linear_combination c20499.trans k2 - hc

theorem publicBatchWrapper89_f2779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14407)) (a (.wire 169 79)) (a (.virt 19391)) (a (.virt 19392)) := by
  have c20500 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20501 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20502 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20503 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20504 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20505 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20506 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20507 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20508 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20509 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20510 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20511 := (publicBatchWrapper89_copies640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20512 := (publicBatchWrapper89_copies641 a h).1
  have c20513 := (publicBatchWrapper89_copies641 a h).2.1
  have c20514 := (publicBatchWrapper89_copies641 a h).2.2.1
  have c20515 := (publicBatchWrapper89_copies641 a h).2.2.2.1
  have c20516 := (publicBatchWrapper89_copies641 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_310_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row310 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_310_19
  simp only [← c20506, ← c20507, ← c20508] at e_310_19
  have e_311_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_16
  simp only [← c20500, k3, ← c20501, k3, ← c20502] at e_311_16
  have e_311_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_17
  simp only [← c20503, ← c20504, k3, ← c20505] at e_311_17
  have e_311_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_18
  simp only [← c20512, ← c20513, k3, ← c20514] at e_311_18
  have e_312_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_0
  simp only [← c20509, ← c20510, ← c20511] at e_312_0
  refine ⟨?_, ?_⟩
  · have hc := e_310_19
    simp only [e_311_17] at hc
    linear_combination c20515.trans k2 - hc
  · have hc := e_311_18
    simp only [e_311_16, e_312_0, e_311_17] at hc
    linear_combination c20516.trans k2 - hc

theorem publicBatchWrapper89_f2780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14408)) (a (.wire 170 7)) (a (.virt 19393)) (a (.virt 19394)) := by
  have c20517 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.1
  have c20518 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.1
  have c20519 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.1
  have c20520 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.1
  have c20521 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.1
  have c20522 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20523 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20524 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20525 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20526 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20527 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20528 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20529 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20530 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20531 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20532 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20533 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_311_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row311 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_311_19
  simp only [← c20517, k3, ← c20518, k3, ← c20519] at e_311_19
  have e_312_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_1
  simp only [← c20523, ← c20524, ← c20525] at e_312_1
  have e_312_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_2
  simp only [← c20526, ← c20527, ← c20528] at e_312_2
  have e_313_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_0
  simp only [← c20520, ← c20521, k3, ← c20522] at e_313_0
  have e_313_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_1
  simp only [← c20529, ← c20530, k3, ← c20531] at e_313_1
  refine ⟨?_, ?_⟩
  · have hc := e_312_1
    simp only [e_313_0] at hc
    linear_combination c20532.trans k2 - hc
  · have hc := e_313_1
    simp only [e_311_19, e_312_2, e_313_0] at hc
    linear_combination c20533.trans k2 - hc

theorem publicBatchWrapper89_f2781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 312 15) = band (a (.virt 19387)) (a (.virt 19389)) := by
  have c20534 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20535 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20536 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_312_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_3
  simp only [← c20534, ← c20535, ← c20536] at e_312_3
  have hr := e_312_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 312 19) = band (a (.virt 19391)) (a (.virt 19393)) := by
  have c20537 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20538 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20539 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_312_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_4
  simp only [← c20537, ← c20538, ← c20539] at e_312_4
  have hr := e_312_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 312 23) = band (a (.wire 312 15)) (a (.wire 312 19)) := by
  have c20540 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20541 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20542 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_312_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_5
  simp only [← c20540, ← c20541, ← c20542] at e_312_5
  have hr := e_312_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 75) = bor (a (.wire 71 27)) (a (.wire 312 23)) := by
  have c20543 := (publicBatchWrapper89_copies641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20544 := (publicBatchWrapper89_copies642 a h).1
  have c20545 := (publicBatchWrapper89_copies642 a h).2.1
  have c20546 := (publicBatchWrapper89_copies642 a h).2.2.1
  have c20547 := (publicBatchWrapper89_copies642 a h).2.2.2.1
  have c20548 := (publicBatchWrapper89_copies642 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_301_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_18
  simp only [← c20546, ← c20547, k3, ← c20548] at e_301_18
  have e_309_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_6
  simp only [← c20543, ← c20544, ← c20545] at e_309_6
  have hr := e_301_18
  simp only [e_309_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 75) = a (.virt 201) := by
  have c20549 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.1
  exact c20549

theorem publicBatchWrapper89_f2786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14603)) (a (.wire 170 23)) (a (.virt 19395)) (a (.virt 19396)) := by
  have c20550 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.1
  have c20551 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.1
  have c20552 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.1
  have c20553 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.1
  have c20554 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20555 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20556 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20557 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20558 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20559 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20560 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20561 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20562 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20563 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20564 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20565 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20566 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_312_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_6
  simp only [← c20556, ← c20557, ← c20558] at e_312_6
  have e_312_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_7
  simp only [← c20559, ← c20560, ← c20561] at e_312_7
  have e_313_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_2
  simp only [← c20550, k3, ← c20551, k3, ← c20552] at e_313_2
  have e_313_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_3
  simp only [← c20553, ← c20554, k3, ← c20555] at e_313_3
  have e_313_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_4
  simp only [← c20562, ← c20563, k3, ← c20564] at e_313_4
  refine ⟨?_, ?_⟩
  · have hc := e_312_6
    simp only [e_313_3] at hc
    linear_combination c20565.trans k2 - hc
  · have hc := e_313_4
    simp only [e_313_2, e_312_7, e_313_3] at hc
    linear_combination c20566.trans k2 - hc

theorem publicBatchWrapper89_f2787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 79) = bor (a (.wire 71 71)) (a (.virt 19395)) := by
  have c20567 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20568 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20569 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20570 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20571 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20572 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_301_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row301 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_301_19
  simp only [← c20570, ← c20571, k3, ← c20572] at e_301_19
  have e_309_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_7
  simp only [← c20567, ← c20568, ← c20569] at e_309_7
  have hr := e_301_19
  simp only [e_309_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 301 79) = a (.virt 201) := by
  have c20573 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20573

theorem publicBatchWrapper89_f2789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14604)) (a (.wire 170 31)) (a (.virt 19397)) (a (.virt 19398)) := by
  have c20574 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20575 := (publicBatchWrapper89_copies642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20576 := (publicBatchWrapper89_copies643 a h).1
  have c20577 := (publicBatchWrapper89_copies643 a h).2.1
  have c20578 := (publicBatchWrapper89_copies643 a h).2.2.1
  have c20579 := (publicBatchWrapper89_copies643 a h).2.2.2.1
  have c20580 := (publicBatchWrapper89_copies643 a h).2.2.2.2.1
  have c20581 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.1
  have c20582 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.1
  have c20583 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.1
  have c20584 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.1
  have c20585 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.1
  have c20586 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20587 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20588 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20589 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20590 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_312_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_8
  simp only [← c20580, ← c20581, ← c20582] at e_312_8
  have e_312_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_9
  simp only [← c20583, ← c20584, ← c20585] at e_312_9
  have e_313_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_5
  simp only [← c20574, k3, ← c20575, k3, ← c20576] at e_313_5
  have e_313_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_6
  simp only [← c20577, ← c20578, k3, ← c20579] at e_313_6
  have e_313_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_7
  simp only [← c20586, ← c20587, k3, ← c20588] at e_313_7
  refine ⟨?_, ?_⟩
  · have hc := e_312_8
    simp only [e_313_6] at hc
    linear_combination c20589.trans k2 - hc
  · have hc := e_313_7
    simp only [e_313_5, e_312_9, e_313_6] at hc
    linear_combination c20590.trans k2 - hc

theorem publicBatchWrapper89_f2790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 3) = bor (a (.wire 71 71)) (a (.virt 19397)) := by
  have c20591 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20592 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20593 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20594 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20595 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20596 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_8
  simp only [← c20591, ← c20592, ← c20593] at e_309_8
  have e_314_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_0
  simp only [← c20594, ← c20595, k3, ← c20596] at e_314_0
  have hr := e_314_0
  simp only [e_309_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 3) = a (.virt 201) := by
  have c20597 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20597

theorem publicBatchWrapper89_f2792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14605)) (a (.wire 169 63)) (a (.virt 19399)) (a (.virt 19400)) := by
  have c20598 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20599 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20600 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20601 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20602 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20603 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20604 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20605 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20606 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20607 := (publicBatchWrapper89_copies643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20608 := (publicBatchWrapper89_copies644 a h).1
  have c20609 := (publicBatchWrapper89_copies644 a h).2.1
  have c20610 := (publicBatchWrapper89_copies644 a h).2.2.1
  have c20611 := (publicBatchWrapper89_copies644 a h).2.2.2.1
  have c20612 := (publicBatchWrapper89_copies644 a h).2.2.2.2.1
  have c20613 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.1
  have c20614 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_312_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_10
  simp only [← c20604, ← c20605, ← c20606] at e_312_10
  have e_312_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_11
  simp only [← c20607, ← c20608, ← c20609] at e_312_11
  have e_313_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_8
  simp only [← c20598, k3, ← c20599, k3, ← c20600] at e_313_8
  have e_313_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_9
  simp only [← c20601, ← c20602, k3, ← c20603] at e_313_9
  have e_313_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_10
  simp only [← c20610, ← c20611, k3, ← c20612] at e_313_10
  refine ⟨?_, ?_⟩
  · have hc := e_312_10
    simp only [e_313_9] at hc
    linear_combination c20613.trans k2 - hc
  · have hc := e_313_10
    simp only [e_313_8, e_312_11, e_313_9] at hc
    linear_combination c20614.trans k2 - hc

theorem publicBatchWrapper89_f2793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14606)) (a (.wire 169 71)) (a (.virt 19401)) (a (.virt 19402)) := by
  have c20615 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.1
  have c20616 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.1
  have c20617 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.1
  have c20618 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20619 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20620 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20621 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20622 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20623 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20624 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20625 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20626 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20627 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20628 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20629 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20630 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20631 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_312_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_12
  simp only [← c20621, ← c20622, ← c20623] at e_312_12
  have e_312_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_13
  simp only [← c20624, ← c20625, ← c20626] at e_312_13
  have e_313_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_11
  simp only [← c20615, k3, ← c20616, k3, ← c20617] at e_313_11
  have e_313_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_12
  simp only [← c20618, ← c20619, k3, ← c20620] at e_313_12
  have e_313_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_13
  simp only [← c20627, ← c20628, k3, ← c20629] at e_313_13
  refine ⟨?_, ?_⟩
  · have hc := e_312_12
    simp only [e_313_12] at hc
    linear_combination c20630.trans k2 - hc
  · have hc := e_313_13
    simp only [e_313_11, e_312_13, e_313_12] at hc
    linear_combination c20631.trans k2 - hc

theorem publicBatchWrapper89_f2794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14607)) (a (.wire 169 79)) (a (.virt 19403)) (a (.virt 19404)) := by
  have c20632 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20633 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20634 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20635 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20636 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20637 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20638 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20639 := (publicBatchWrapper89_copies644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20640 := (publicBatchWrapper89_copies645 a h).1
  have c20641 := (publicBatchWrapper89_copies645 a h).2.1
  have c20642 := (publicBatchWrapper89_copies645 a h).2.2.1
  have c20643 := (publicBatchWrapper89_copies645 a h).2.2.2.1
  have c20644 := (publicBatchWrapper89_copies645 a h).2.2.2.2.1
  have c20645 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.1
  have c20646 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.1
  have c20647 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.1
  have c20648 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_312_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_14
  simp only [← c20638, ← c20639, ← c20640] at e_312_14
  have e_312_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_15
  simp only [← c20641, ← c20642, ← c20643] at e_312_15
  have e_313_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_14
  simp only [← c20632, k3, ← c20633, k3, ← c20634] at e_313_14
  have e_313_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_15
  simp only [← c20635, ← c20636, k3, ← c20637] at e_313_15
  have e_313_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_16
  simp only [← c20644, ← c20645, k3, ← c20646] at e_313_16
  refine ⟨?_, ?_⟩
  · have hc := e_312_14
    simp only [e_313_15] at hc
    linear_combination c20647.trans k2 - hc
  · have hc := e_313_16
    simp only [e_313_14, e_312_15, e_313_15] at hc
    linear_combination c20648.trans k2 - hc

theorem publicBatchWrapper89_f2795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14608)) (a (.wire 170 7)) (a (.virt 19405)) (a (.virt 19406)) := by
  have c20649 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.1
  have c20650 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20651 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20652 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20653 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20654 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20655 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20656 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20657 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20658 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20659 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20660 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20661 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20662 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20663 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20664 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20665 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_312_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_16
  simp only [← c20655, ← c20656, ← c20657] at e_312_16
  have e_312_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_17
  simp only [← c20658, ← c20659, ← c20660] at e_312_17
  have e_313_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_17
  simp only [← c20649, k3, ← c20650, k3, ← c20651] at e_313_17
  have e_313_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_18
  simp only [← c20652, ← c20653, k3, ← c20654] at e_313_18
  have e_313_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row313 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_313_19
  simp only [← c20661, ← c20662, k3, ← c20663] at e_313_19
  refine ⟨?_, ?_⟩
  · have hc := e_312_16
    simp only [e_313_18] at hc
    linear_combination c20664.trans k2 - hc
  · have hc := e_313_19
    simp only [e_313_17, e_312_17, e_313_18] at hc
    linear_combination c20665.trans k2 - hc

theorem publicBatchWrapper89_f2796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 312 75) = band (a (.virt 19399)) (a (.virt 19401)) := by
  have c20666 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20667 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20668 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_312_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_18
  simp only [← c20666, ← c20667, ← c20668] at e_312_18
  have hr := e_312_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 312 79) = band (a (.virt 19403)) (a (.virt 19405)) := by
  have c20669 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20670 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20671 := (publicBatchWrapper89_copies645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_312_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row312 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_312_19
  simp only [← c20669, ← c20670, ← c20671] at e_312_19
  have hr := e_312_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 315 3) = band (a (.wire 312 75)) (a (.wire 312 79)) := by
  have c20672 := (publicBatchWrapper89_copies646 a h).1
  have c20673 := (publicBatchWrapper89_copies646 a h).2.1
  have c20674 := (publicBatchWrapper89_copies646 a h).2.2.1
  have e_315_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_0
  simp only [← c20672, ← c20673, ← c20674] at e_315_0
  have hr := e_315_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 7) = bor (a (.wire 71 71)) (a (.wire 315 3)) := by
  have c20675 := (publicBatchWrapper89_copies646 a h).2.2.2.1
  have c20676 := (publicBatchWrapper89_copies646 a h).2.2.2.2.1
  have c20677 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.1
  have c20678 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.1
  have c20679 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.1
  have c20680 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_9
  simp only [← c20675, ← c20676, ← c20677] at e_309_9
  have e_314_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_1
  simp only [← c20678, ← c20679, k3, ← c20680] at e_314_1
  have hr := e_314_1
  simp only [e_309_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 7) = a (.virt 201) := by
  have c20681 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.1
  exact c20681

theorem publicBatchWrapper89_f2801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14803)) (a (.wire 170 23)) (a (.virt 19407)) (a (.virt 19408)) := by
  have c20682 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20683 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20684 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20685 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20686 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20687 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20688 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20689 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20690 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20691 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20692 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20693 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20694 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20695 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20696 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20697 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20698 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_315_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_1
  simp only [← c20688, ← c20689, ← c20690] at e_315_1
  have e_315_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_2
  simp only [← c20691, ← c20692, ← c20693] at e_315_2
  have e_316_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_0
  simp only [← c20682, k3, ← c20683, k3, ← c20684] at e_316_0
  have e_316_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_1
  simp only [← c20685, ← c20686, k3, ← c20687] at e_316_1
  have e_316_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_2
  simp only [← c20694, ← c20695, k3, ← c20696] at e_316_2
  refine ⟨?_, ?_⟩
  · have hc := e_315_1
    simp only [e_316_1] at hc
    linear_combination c20697.trans k2 - hc
  · have hc := e_316_2
    simp only [e_316_0, e_315_2, e_316_1] at hc
    linear_combination c20698.trans k2 - hc

theorem publicBatchWrapper89_f2802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 11) = bor (a (.wire 72 35)) (a (.virt 19407)) := by
  have c20699 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20700 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20701 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20702 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20703 := (publicBatchWrapper89_copies646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20704 := (publicBatchWrapper89_copies647 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_10
  simp only [← c20699, ← c20700, ← c20701] at e_309_10
  have e_314_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_2
  simp only [← c20702, ← c20703, k3, ← c20704] at e_314_2
  have hr := e_314_2
  simp only [e_309_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 11) = a (.virt 201) := by
  have c20705 := (publicBatchWrapper89_copies647 a h).2.1
  exact c20705

theorem publicBatchWrapper89_f2804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14804)) (a (.wire 170 31)) (a (.virt 19409)) (a (.virt 19410)) := by
  have c20706 := (publicBatchWrapper89_copies647 a h).2.2.1
  have c20707 := (publicBatchWrapper89_copies647 a h).2.2.2.1
  have c20708 := (publicBatchWrapper89_copies647 a h).2.2.2.2.1
  have c20709 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.1
  have c20710 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.1
  have c20711 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.1
  have c20712 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.1
  have c20713 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.1
  have c20714 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20715 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20716 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20717 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20718 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20719 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20720 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20721 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20722 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_315_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_3
  simp only [← c20712, ← c20713, ← c20714] at e_315_3
  have e_315_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_4
  simp only [← c20715, ← c20716, ← c20717] at e_315_4
  have e_316_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_3
  simp only [← c20706, k3, ← c20707, k3, ← c20708] at e_316_3
  have e_316_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_4
  simp only [← c20709, ← c20710, k3, ← c20711] at e_316_4
  have e_316_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_5
  simp only [← c20718, ← c20719, k3, ← c20720] at e_316_5
  refine ⟨?_, ?_⟩
  · have hc := e_315_3
    simp only [e_316_4] at hc
    linear_combination c20721.trans k2 - hc
  · have hc := e_316_5
    simp only [e_316_3, e_315_4, e_316_4] at hc
    linear_combination c20722.trans k2 - hc

theorem publicBatchWrapper89_f2805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 15) = bor (a (.wire 72 35)) (a (.virt 19409)) := by
  have c20723 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20724 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20725 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20726 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20727 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20728 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_11
  simp only [← c20723, ← c20724, ← c20725] at e_309_11
  have e_314_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_3
  simp only [← c20726, ← c20727, k3, ← c20728] at e_314_3
  have hr := e_314_3
  simp only [e_309_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 15) = a (.virt 201) := by
  have c20729 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20729

theorem publicBatchWrapper89_f2807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14805)) (a (.wire 169 63)) (a (.virt 19411)) (a (.virt 19412)) := by
  have c20730 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20731 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20732 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20733 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20734 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20735 := (publicBatchWrapper89_copies647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20736 := (publicBatchWrapper89_copies648 a h).1
  have c20737 := (publicBatchWrapper89_copies648 a h).2.1
  have c20738 := (publicBatchWrapper89_copies648 a h).2.2.1
  have c20739 := (publicBatchWrapper89_copies648 a h).2.2.2.1
  have c20740 := (publicBatchWrapper89_copies648 a h).2.2.2.2.1
  have c20741 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.1
  have c20742 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.1
  have c20743 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.1
  have c20744 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.1
  have c20745 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.1
  have c20746 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_315_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_5
  simp only [← c20736, ← c20737, ← c20738] at e_315_5
  have e_315_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_6
  simp only [← c20739, ← c20740, ← c20741] at e_315_6
  have e_316_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_6
  simp only [← c20730, k3, ← c20731, k3, ← c20732] at e_316_6
  have e_316_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_7
  simp only [← c20733, ← c20734, k3, ← c20735] at e_316_7
  have e_316_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_8
  simp only [← c20742, ← c20743, k3, ← c20744] at e_316_8
  refine ⟨?_, ?_⟩
  · have hc := e_315_5
    simp only [e_316_7] at hc
    linear_combination c20745.trans k2 - hc
  · have hc := e_316_8
    simp only [e_316_6, e_315_6, e_316_7] at hc
    linear_combination c20746.trans k2 - hc

theorem publicBatchWrapper89_f2808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14806)) (a (.wire 169 71)) (a (.virt 19413)) (a (.virt 19414)) := by
  have c20747 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20748 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20749 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20750 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20751 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20752 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20753 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20754 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20755 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20756 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20757 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20758 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20759 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20760 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20761 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20762 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20763 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_315_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_7
  simp only [← c20753, ← c20754, ← c20755] at e_315_7
  have e_315_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_8
  simp only [← c20756, ← c20757, ← c20758] at e_315_8
  have e_316_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_9
  simp only [← c20747, k3, ← c20748, k3, ← c20749] at e_316_9
  have e_316_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_10
  simp only [← c20750, ← c20751, k3, ← c20752] at e_316_10
  have e_316_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_11
  simp only [← c20759, ← c20760, k3, ← c20761] at e_316_11
  refine ⟨?_, ?_⟩
  · have hc := e_315_7
    simp only [e_316_10] at hc
    linear_combination c20762.trans k2 - hc
  · have hc := e_316_11
    simp only [e_316_9, e_315_8, e_316_10] at hc
    linear_combination c20763.trans k2 - hc

theorem publicBatchWrapper89_f2809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14807)) (a (.wire 169 79)) (a (.virt 19415)) (a (.virt 19416)) := by
  have c20764 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20765 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20766 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20767 := (publicBatchWrapper89_copies648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20768 := (publicBatchWrapper89_copies649 a h).1
  have c20769 := (publicBatchWrapper89_copies649 a h).2.1
  have c20770 := (publicBatchWrapper89_copies649 a h).2.2.1
  have c20771 := (publicBatchWrapper89_copies649 a h).2.2.2.1
  have c20772 := (publicBatchWrapper89_copies649 a h).2.2.2.2.1
  have c20773 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.1
  have c20774 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.1
  have c20775 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.1
  have c20776 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.1
  have c20777 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.1
  have c20778 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20779 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20780 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_315_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_9
  simp only [← c20770, ← c20771, ← c20772] at e_315_9
  have e_315_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_10
  simp only [← c20773, ← c20774, ← c20775] at e_315_10
  have e_316_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_12
  simp only [← c20764, k3, ← c20765, k3, ← c20766] at e_316_12
  have e_316_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_13
  simp only [← c20767, ← c20768, k3, ← c20769] at e_316_13
  have e_316_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_14
  simp only [← c20776, ← c20777, k3, ← c20778] at e_316_14
  refine ⟨?_, ?_⟩
  · have hc := e_315_9
    simp only [e_316_13] at hc
    linear_combination c20779.trans k2 - hc
  · have hc := e_316_14
    simp only [e_316_12, e_315_10, e_316_13] at hc
    linear_combination c20780.trans k2 - hc

theorem publicBatchWrapper89_f2810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14808)) (a (.wire 170 7)) (a (.virt 19417)) (a (.virt 19418)) := by
  have c20781 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20782 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20783 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20784 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20785 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20786 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20787 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20788 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20789 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20790 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20791 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20792 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20793 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20794 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20795 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20796 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20797 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_315_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_11
  simp only [← c20787, ← c20788, ← c20789] at e_315_11
  have e_315_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_12
  simp only [← c20790, ← c20791, ← c20792] at e_315_12
  have e_316_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_15
  simp only [← c20781, k3, ← c20782, k3, ← c20783] at e_316_15
  have e_316_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_16
  simp only [← c20784, ← c20785, k3, ← c20786] at e_316_16
  have e_316_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_17
  simp only [← c20793, ← c20794, k3, ← c20795] at e_316_17
  refine ⟨?_, ?_⟩
  · have hc := e_315_11
    simp only [e_316_16] at hc
    linear_combination c20796.trans k2 - hc
  · have hc := e_316_17
    simp only [e_316_15, e_315_12, e_316_16] at hc
    linear_combination c20797.trans k2 - hc

theorem publicBatchWrapper89_f2811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 315 55) = band (a (.virt 19411)) (a (.virt 19413)) := by
  have c20798 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20799 := (publicBatchWrapper89_copies649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20800 := (publicBatchWrapper89_copies650 a h).1
  have e_315_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_13
  simp only [← c20798, ← c20799, ← c20800] at e_315_13
  have hr := e_315_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 315 59) = band (a (.virt 19415)) (a (.virt 19417)) := by
  have c20801 := (publicBatchWrapper89_copies650 a h).2.1
  have c20802 := (publicBatchWrapper89_copies650 a h).2.2.1
  have c20803 := (publicBatchWrapper89_copies650 a h).2.2.2.1
  have e_315_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_14
  simp only [← c20801, ← c20802, ← c20803] at e_315_14
  have hr := e_315_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 315 63) = band (a (.wire 315 55)) (a (.wire 315 59)) := by
  have c20804 := (publicBatchWrapper89_copies650 a h).2.2.2.2.1
  have c20805 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.1
  have c20806 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.1
  have e_315_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_15
  simp only [← c20804, ← c20805, ← c20806] at e_315_15
  have hr := e_315_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 19) = bor (a (.wire 72 35)) (a (.wire 315 63)) := by
  have c20807 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.1
  have c20808 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.1
  have c20809 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.1
  have c20810 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20811 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20812 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_12
  simp only [← c20807, ← c20808, ← c20809] at e_309_12
  have e_314_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_4
  simp only [← c20810, ← c20811, k3, ← c20812] at e_314_4
  have hr := e_314_4
  simp only [e_309_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 19) = a (.virt 201) := by
  have c20813 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20813

theorem publicBatchWrapper89_f2816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15003)) (a (.wire 170 23)) (a (.virt 19419)) (a (.virt 19420)) := by
  have c20814 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20815 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20816 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20817 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20818 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20819 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20820 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20821 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20822 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20823 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20824 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20825 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20826 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20827 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20828 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20829 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20830 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_315_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_16
  simp only [← c20820, ← c20821, ← c20822] at e_315_16
  have e_315_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_17
  simp only [← c20823, ← c20824, ← c20825] at e_315_17
  have e_316_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_18
  simp only [← c20814, k3, ← c20815, k3, ← c20816] at e_316_18
  have e_316_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row316 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_316_19
  simp only [← c20817, ← c20818, k3, ← c20819] at e_316_19
  have e_317_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_0
  simp only [← c20826, ← c20827, k3, ← c20828] at e_317_0
  refine ⟨?_, ?_⟩
  · have hc := e_315_16
    simp only [e_316_19] at hc
    linear_combination c20829.trans k2 - hc
  · have hc := e_317_0
    simp only [e_316_18, e_315_17, e_316_19] at hc
    linear_combination c20830.trans k2 - hc

theorem publicBatchWrapper89_f2817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 23) = bor (a (.wire 72 79)) (a (.virt 19419)) := by
  have c20831 := (publicBatchWrapper89_copies650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20832 := (publicBatchWrapper89_copies651 a h).1
  have c20833 := (publicBatchWrapper89_copies651 a h).2.1
  have c20834 := (publicBatchWrapper89_copies651 a h).2.2.1
  have c20835 := (publicBatchWrapper89_copies651 a h).2.2.2.1
  have c20836 := (publicBatchWrapper89_copies651 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_13
  simp only [← c20831, ← c20832, ← c20833] at e_309_13
  have e_314_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_5
  simp only [← c20834, ← c20835, k3, ← c20836] at e_314_5
  have hr := e_314_5
  simp only [e_309_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 23) = a (.virt 201) := by
  have c20837 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.1
  exact c20837

theorem publicBatchWrapper89_f2819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15004)) (a (.wire 170 31)) (a (.virt 19421)) (a (.virt 19422)) := by
  have c20838 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.1
  have c20839 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.1
  have c20840 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.1
  have c20841 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.1
  have c20842 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20843 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20844 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20845 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20846 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20847 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20848 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20849 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20850 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20851 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20852 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20853 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20854 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_315_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_18
  simp only [← c20844, ← c20845, ← c20846] at e_315_18
  have e_315_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row315 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_315_19
  simp only [← c20847, ← c20848, ← c20849] at e_315_19
  have e_317_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_1
  simp only [← c20838, k3, ← c20839, k3, ← c20840] at e_317_1
  have e_317_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_2
  simp only [← c20841, ← c20842, k3, ← c20843] at e_317_2
  have e_317_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_3
  simp only [← c20850, ← c20851, k3, ← c20852] at e_317_3
  refine ⟨?_, ?_⟩
  · have hc := e_315_18
    simp only [e_317_2] at hc
    linear_combination c20853.trans k2 - hc
  · have hc := e_317_3
    simp only [e_317_1, e_315_19, e_317_2] at hc
    linear_combination c20854.trans k2 - hc

theorem publicBatchWrapper89_f2820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 27) = bor (a (.wire 72 79)) (a (.virt 19421)) := by
  have c20855 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20856 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20857 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20858 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20859 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20860 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_14
  simp only [← c20855, ← c20856, ← c20857] at e_309_14
  have e_314_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_6
  simp only [← c20858, ← c20859, k3, ← c20860] at e_314_6
  have hr := e_314_6
  simp only [e_309_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 27) = a (.virt 201) := by
  have c20861 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20861

theorem publicBatchWrapper89_f2822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15005)) (a (.wire 169 63)) (a (.virt 19423)) (a (.virt 19424)) := by
  have c20862 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20863 := (publicBatchWrapper89_copies651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20864 := (publicBatchWrapper89_copies652 a h).1
  have c20865 := (publicBatchWrapper89_copies652 a h).2.1
  have c20866 := (publicBatchWrapper89_copies652 a h).2.2.1
  have c20867 := (publicBatchWrapper89_copies652 a h).2.2.2.1
  have c20868 := (publicBatchWrapper89_copies652 a h).2.2.2.2.1
  have c20869 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.1
  have c20870 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.1
  have c20871 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.1
  have c20872 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.1
  have c20873 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.1
  have c20874 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20875 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20876 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20877 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20878 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_317_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_4
  simp only [← c20862, k3, ← c20863, k3, ← c20864] at e_317_4
  have e_317_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_5
  simp only [← c20865, ← c20866, k3, ← c20867] at e_317_5
  have e_317_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_6
  simp only [← c20874, ← c20875, k3, ← c20876] at e_317_6
  have e_318_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_0
  simp only [← c20868, ← c20869, ← c20870] at e_318_0
  have e_318_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_1
  simp only [← c20871, ← c20872, ← c20873] at e_318_1
  refine ⟨?_, ?_⟩
  · have hc := e_318_0
    simp only [e_317_5] at hc
    linear_combination c20877.trans k2 - hc
  · have hc := e_317_6
    simp only [e_317_4, e_318_1, e_317_5] at hc
    linear_combination c20878.trans k2 - hc

theorem publicBatchWrapper89_f2823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15006)) (a (.wire 169 71)) (a (.virt 19425)) (a (.virt 19426)) := by
  have c20879 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20880 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20881 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20882 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20883 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20884 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20885 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20886 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20887 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20888 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20889 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20890 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20891 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20892 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20893 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20894 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20895 := (publicBatchWrapper89_copies652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_317_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_7
  simp only [← c20879, k3, ← c20880, k3, ← c20881] at e_317_7
  have e_317_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_8
  simp only [← c20882, ← c20883, k3, ← c20884] at e_317_8
  have e_317_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_9
  simp only [← c20891, ← c20892, k3, ← c20893] at e_317_9
  have e_318_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_2
  simp only [← c20885, ← c20886, ← c20887] at e_318_2
  have e_318_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_3
  simp only [← c20888, ← c20889, ← c20890] at e_318_3
  refine ⟨?_, ?_⟩
  · have hc := e_318_2
    simp only [e_317_8] at hc
    linear_combination c20894.trans k2 - hc
  · have hc := e_317_9
    simp only [e_317_7, e_318_3, e_317_8] at hc
    linear_combination c20895.trans k2 - hc

theorem publicBatchWrapper89_f2824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15007)) (a (.wire 169 79)) (a (.virt 19427)) (a (.virt 19428)) := by
  have c20896 := (publicBatchWrapper89_copies653 a h).1
  have c20897 := (publicBatchWrapper89_copies653 a h).2.1
  have c20898 := (publicBatchWrapper89_copies653 a h).2.2.1
  have c20899 := (publicBatchWrapper89_copies653 a h).2.2.2.1
  have c20900 := (publicBatchWrapper89_copies653 a h).2.2.2.2.1
  have c20901 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.1
  have c20902 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.1
  have c20903 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.1
  have c20904 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.1
  have c20905 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.1
  have c20906 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20907 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20908 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20909 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20910 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20911 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20912 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_317_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_10
  simp only [← c20896, k3, ← c20897, k3, ← c20898] at e_317_10
  have e_317_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_11
  simp only [← c20899, ← c20900, k3, ← c20901] at e_317_11
  have e_317_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_12
  simp only [← c20908, ← c20909, k3, ← c20910] at e_317_12
  have e_318_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_4
  simp only [← c20902, ← c20903, ← c20904] at e_318_4
  have e_318_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_5
  simp only [← c20905, ← c20906, ← c20907] at e_318_5
  refine ⟨?_, ?_⟩
  · have hc := e_318_4
    simp only [e_317_11] at hc
    linear_combination c20911.trans k2 - hc
  · have hc := e_317_12
    simp only [e_317_10, e_318_5, e_317_11] at hc
    linear_combination c20912.trans k2 - hc

theorem publicBatchWrapper89_f2825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15008)) (a (.wire 170 7)) (a (.virt 19429)) (a (.virt 19430)) := by
  have c20913 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20914 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20915 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20916 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20917 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20918 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20919 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20920 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20921 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20922 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20923 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20924 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20925 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20926 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20927 := (publicBatchWrapper89_copies653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20928 := (publicBatchWrapper89_copies654 a h).1
  have c20929 := (publicBatchWrapper89_copies654 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_317_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_13
  simp only [← c20913, k3, ← c20914, k3, ← c20915] at e_317_13
  have e_317_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_14
  simp only [← c20916, ← c20917, k3, ← c20918] at e_317_14
  have e_317_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_15
  simp only [← c20925, ← c20926, k3, ← c20927] at e_317_15
  have e_318_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_6
  simp only [← c20919, ← c20920, ← c20921] at e_318_6
  have e_318_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_7
  simp only [← c20922, ← c20923, ← c20924] at e_318_7
  refine ⟨?_, ?_⟩
  · have hc := e_318_6
    simp only [e_317_14] at hc
    linear_combination c20928.trans k2 - hc
  · have hc := e_317_15
    simp only [e_317_13, e_318_7, e_317_14] at hc
    linear_combination c20929.trans k2 - hc

theorem publicBatchWrapper89_f2826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 318 35) = band (a (.virt 19423)) (a (.virt 19425)) := by
  have c20930 := (publicBatchWrapper89_copies654 a h).2.2.1
  have c20931 := (publicBatchWrapper89_copies654 a h).2.2.2.1
  have c20932 := (publicBatchWrapper89_copies654 a h).2.2.2.2.1
  have e_318_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_8
  simp only [← c20930, ← c20931, ← c20932] at e_318_8
  have hr := e_318_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 318 39) = band (a (.virt 19427)) (a (.virt 19429)) := by
  have c20933 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.1
  have c20934 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.1
  have c20935 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.1
  have e_318_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_9
  simp only [← c20933, ← c20934, ← c20935] at e_318_9
  have hr := e_318_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 318 43) = band (a (.wire 318 35)) (a (.wire 318 39)) := by
  have c20936 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.1
  have c20937 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.1
  have c20938 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_318_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_10
  simp only [← c20936, ← c20937, ← c20938] at e_318_10
  have hr := e_318_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 31) = bor (a (.wire 72 79)) (a (.wire 318 43)) := by
  have c20939 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20940 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20941 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20942 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20943 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20944 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_15
  simp only [← c20939, ← c20940, ← c20941] at e_309_15
  have e_314_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_7
  simp only [← c20942, ← c20943, k3, ← c20944] at e_314_7
  have hr := e_314_7
  simp only [e_309_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 31) = a (.virt 201) := by
  have c20945 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20945

theorem publicBatchWrapper89_f2831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15203)) (a (.wire 170 23)) (a (.virt 19431)) (a (.virt 19432)) := by
  have c20946 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20947 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20948 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20949 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20950 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20951 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20952 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20953 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20954 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20955 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20956 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20957 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20958 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20959 := (publicBatchWrapper89_copies654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20960 := (publicBatchWrapper89_copies655 a h).1
  have c20961 := (publicBatchWrapper89_copies655 a h).2.1
  have c20962 := (publicBatchWrapper89_copies655 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_317_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_16
  simp only [← c20946, k3, ← c20947, k3, ← c20948] at e_317_16
  have e_317_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_17
  simp only [← c20949, ← c20950, k3, ← c20951] at e_317_17
  have e_317_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_18
  simp only [← c20958, ← c20959, k3, ← c20960] at e_317_18
  have e_318_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_11
  simp only [← c20952, ← c20953, ← c20954] at e_318_11
  have e_318_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_12
  simp only [← c20955, ← c20956, ← c20957] at e_318_12
  refine ⟨?_, ?_⟩
  · have hc := e_318_11
    simp only [e_317_17] at hc
    linear_combination c20961.trans k2 - hc
  · have hc := e_317_18
    simp only [e_317_16, e_318_12, e_317_17] at hc
    linear_combination c20962.trans k2 - hc

theorem publicBatchWrapper89_f2832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 35) = bor (a (.wire 74 43)) (a (.virt 19431)) := by
  have c20963 := (publicBatchWrapper89_copies655 a h).2.2.2.1
  have c20964 := (publicBatchWrapper89_copies655 a h).2.2.2.2.1
  have c20965 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.1
  have c20966 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.1
  have c20967 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.1
  have c20968 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_16
  simp only [← c20963, ← c20964, ← c20965] at e_309_16
  have e_314_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_8
  simp only [← c20966, ← c20967, k3, ← c20968] at e_314_8
  have hr := e_314_8
  simp only [e_309_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 35) = a (.virt 201) := by
  have c20969 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.1
  exact c20969

theorem publicBatchWrapper89_f2834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15204)) (a (.wire 170 31)) (a (.virt 19433)) (a (.virt 19434)) := by
  have c20970 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.1
  have c20971 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c20972 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20973 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20974 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20975 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20976 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20977 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20978 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20979 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20980 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20981 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20982 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20983 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20984 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20985 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20986 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_317_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row317 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_317_19
  simp only [← c20970, k3, ← c20971, k3, ← c20972] at e_317_19
  have e_318_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_13
  simp only [← c20976, ← c20977, ← c20978] at e_318_13
  have e_318_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_14
  simp only [← c20979, ← c20980, ← c20981] at e_318_14
  have e_319_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_0
  simp only [← c20973, ← c20974, k3, ← c20975] at e_319_0
  have e_319_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_1
  simp only [← c20982, ← c20983, k3, ← c20984] at e_319_1
  refine ⟨?_, ?_⟩
  · have hc := e_318_13
    simp only [e_319_0] at hc
    linear_combination c20985.trans k2 - hc
  · have hc := e_319_1
    simp only [e_317_19, e_318_14, e_319_0] at hc
    linear_combination c20986.trans k2 - hc

theorem publicBatchWrapper89_f2835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 39) = bor (a (.wire 74 43)) (a (.virt 19433)) := by
  have c20987 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20988 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20989 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20990 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20991 := (publicBatchWrapper89_copies655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c20992 := (publicBatchWrapper89_copies656 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_309_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row309 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_309_17
  simp only [← c20987, ← c20988, ← c20989] at e_309_17
  have e_314_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row314 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_9
  simp only [← c20990, ← c20991, k3, ← c20992] at e_314_9
  have hr := e_314_9
  simp only [e_309_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 39) = a (.virt 201) := by
  have c20993 := (publicBatchWrapper89_copies656 a h).2.1
  exact c20993

theorem publicBatchWrapper89_f2837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15205)) (a (.wire 169 63)) (a (.virt 19435)) (a (.virt 19436)) := by
  have c20994 := (publicBatchWrapper89_copies656 a h).2.2.1
  have c20995 := (publicBatchWrapper89_copies656 a h).2.2.2.1
  have c20996 := (publicBatchWrapper89_copies656 a h).2.2.2.2.1
  have c20997 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.1
  have c20998 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.1
  have c20999 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.1
  have c21000 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.1
  have c21001 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.1
  have c21002 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21003 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21004 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21005 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21006 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21007 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21008 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21009 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21010 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_318_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_15
  simp only [← c21000, ← c21001, ← c21002] at e_318_15
  have e_318_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_16
  simp only [← c21003, ← c21004, ← c21005] at e_318_16
  have e_319_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_2
  simp only [← c20994, k3, ← c20995, k3, ← c20996] at e_319_2
  have e_319_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_3
  simp only [← c20997, ← c20998, k3, ← c20999] at e_319_3
  have e_319_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_4
  simp only [← c21006, ← c21007, k3, ← c21008] at e_319_4
  refine ⟨?_, ?_⟩
  · have hc := e_318_15
    simp only [e_319_3] at hc
    linear_combination c21009.trans k2 - hc
  · have hc := e_319_4
    simp only [e_319_2, e_318_16, e_319_3] at hc
    linear_combination c21010.trans k2 - hc

theorem publicBatchWrapper89_f2838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15206)) (a (.wire 169 71)) (a (.virt 19437)) (a (.virt 19438)) := by
  have c21011 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21012 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21013 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21014 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21015 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21016 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21017 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21018 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21019 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21020 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21021 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21022 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21023 := (publicBatchWrapper89_copies656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21024 := (publicBatchWrapper89_copies657 a h).1
  have c21025 := (publicBatchWrapper89_copies657 a h).2.1
  have c21026 := (publicBatchWrapper89_copies657 a h).2.2.1
  have c21027 := (publicBatchWrapper89_copies657 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_318_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_17
  simp only [← c21017, ← c21018, ← c21019] at e_318_17
  have e_318_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row318 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_318_18
  simp only [← c21020, ← c21021, ← c21022] at e_318_18
  have e_319_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_5
  simp only [← c21011, k3, ← c21012, k3, ← c21013] at e_319_5
  have e_319_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_6
  simp only [← c21014, ← c21015, k3, ← c21016] at e_319_6
  have e_319_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row319 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_7
  simp only [← c21023, ← c21024, k3, ← c21025] at e_319_7
  refine ⟨?_, ?_⟩
  · have hc := e_318_17
    simp only [e_319_6] at hc
    linear_combination c21026.trans k2 - hc
  · have hc := e_319_7
    simp only [e_319_5, e_318_18, e_319_6] at hc
    linear_combination c21027.trans k2 - hc

end Plonky2Spec.Generated
